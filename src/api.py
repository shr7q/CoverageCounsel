"""
Week 9: HTTP API wrapping the pipeline, so it can actually be deployed --
Cloud Run (and every other target the plan considered) runs a container
that serves HTTP, not a CLI script you'd SSH in to invoke.

The index is built once at process startup (chunking + embedding + BM25 are
all static over the corpus) and reused across requests, rather than rebuilt
per-request.

Week 10: real Clerk auth replaces the as_user request field this endpoint
used to trust -- that let any caller simply claim to be bob_admin
(compliance_admin) with zero proof, which is a fine trust model for the
CLI's local operator but not for a public HTTP endpoint. An unauthenticated
request is no longer treated as unrestricted either (it was, before this
week, since "logged in vs not" wasn't a real distinction yet) -- it now
gets the same standard-only scope as a freshly-provisioned Clerk account,
so skipping login is never a way to see more, only to see less.

Run locally:
    uvicorn api:app --reload --port 8080
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager

import jwt
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import auth
import db
from query import DATA_DIR, build_index
from query import run_orchestrated_query

state: dict = {}

ANONYMOUS_USER = {"id": None, "username": "anonymous", "role": "anonymous", "access_levels": {"standard"}}

# The frontend (Vercel) is a different origin than this API (Cloud Run), so
# the browser enforces CORS on every request. ALLOWED_ORIGINS narrows the
# wildcard default once a specific frontend domain is known.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Building index from {DATA_DIR}...", file=sys.stderr)
    state["store"] = build_index()
    print("Index ready.", file=sys.stderr)
    yield
    state.clear()


app = FastAPI(title="Healthcare Coverage RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sub_questions: list[str]
    cited_chunk_ids: list[str]
    phantom_citations: list[str]
    unsupported_citations: list[dict]
    viewer_role: str


def _resolve_user(authorization: str | None) -> dict:
    if not authorization:
        return ANONYMOUS_USER
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header must be 'Bearer <token>'")
    token = authorization.removeprefix("Bearer ")
    try:
        clerk_user_id = auth.verify_clerk_token(token)
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid session token: {e}")
    return db.get_or_create_user_by_clerk_id(clerk_user_id)


@app.get("/health")
def health():
    return {"status": "ok", "index_ready": "store" in state}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, authorization: str | None = Header(default=None)):
    if "store" not in state:
        raise HTTPException(status_code=503, detail="Index not ready yet")

    user = _resolve_user(authorization)
    result = run_orchestrated_query(state["store"], req.question, user=user)

    citation_check = result["citation_check"]
    flags = result["faithfulness_flags"]
    return QueryResponse(
        answer=result["final_answer"],
        sub_questions=result["sub_questions"],
        cited_chunk_ids=citation_check.get("cited_ids", []),
        phantom_citations=citation_check.get("phantom_ids", []),
        unsupported_citations=[f for f in flags if not f.get("supported", True)],
        viewer_role=user["role"],
    )
