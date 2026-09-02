"""
Week 9: HTTP API wrapping the pipeline, so it can actually be deployed --
Cloud Run (and every other target the plan considered) runs a container
that serves HTTP, not a CLI script you'd SSH in to invoke.

The index is built once at process startup (chunking + embedding + BM25 are
all static over the corpus) and reused across requests, rather than rebuilt
per-request.

Run locally:
    uvicorn api:app --reload --port 8080
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import db
from query import DATA_DIR, build_index
from query import run_orchestrated_query

state: dict = {}

# The frontend (Vercel) is a different origin than this API (Cloud Run), so
# the browser enforces CORS on every request. No cookies/credentials cross
# this boundary -- as_user is just a JSON body field -- so a wildcard is a
# reasonable default for a public demo; ALLOWED_ORIGINS narrows it once the
# Vercel domain is known.
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
    allow_headers=["Content-Type"],
)


class QueryRequest(BaseModel):
    question: str
    as_user: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sub_questions: list[str]
    cited_chunk_ids: list[str]
    phantom_citations: list[str]
    unsupported_citations: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "index_ready": "store" in state}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if "store" not in state:
        raise HTTPException(status_code=503, detail="Index not ready yet")

    user = None
    if req.as_user:
        try:
            user = db.get_user(req.as_user)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    result = run_orchestrated_query(state["store"], req.question, user=user)

    citation_check = result["citation_check"]
    flags = result["faithfulness_flags"]
    return QueryResponse(
        answer=result["final_answer"],
        sub_questions=result["sub_questions"],
        cited_chunk_ids=citation_check.get("cited_ids", []),
        phantom_citations=citation_check.get("phantom_ids", []),
        unsupported_citations=[f for f in flags if not f.get("supported", True)],
    )
