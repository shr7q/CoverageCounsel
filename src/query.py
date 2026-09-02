"""
The walking skeleton. Run this to prove the full loop works:

    question -> retrieve chunks -> stuff into prompt -> get an answer

No reranking, no query decomposition, no citations parsing, no auth. Those
come later, once you've seen where this naive version breaks.

Usage:
    python src/query.py "Is a total knee replacement covered without physical therapy first?"
"""

import argparse
import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

import db
from chunking import chunk_all_documents
from embed_store import InMemoryStore
from hybrid_store import HybridStore
from llm import generate
from orchestrator import build_graph

# RETRIEVAL_MODE=dense reproduces the Week 1/2 dense-only baseline, for
# direct before/after comparison against the Week 3 hybrid+rerank pipeline.
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "hybrid")

# USE_ORCHESTRATOR=false reproduces the pre-Week-4 single-shot baseline (one
# retrieval call, no decomposition), for direct before/after comparison
# against the LangGraph decompose-and-synthesize pipeline.
USE_ORCHESTRATOR = os.environ.get("USE_ORCHESTRATOR", "true").lower() == "true"

# RAG_DATA_DIR=sample_docs reproduces the original synthetic control set;
# real_docs (the real CMS corpus, Week 1) is the default for anything meant
# to actually answer questions.
DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", os.environ.get("RAG_DATA_DIR", "real_docs")
)

SYSTEM_PROMPT = """You are a healthcare coverage policy assistant. Answer the
user's question using ONLY the policy excerpts provided below. If the excerpts
don't contain enough information to answer confidently, say so explicitly
instead of guessing. Do not use outside knowledge about Medicare policy.

When you answer, mention which excerpt(s) you're relying on."""


def build_index() -> InMemoryStore | HybridStore:
    print("Chunking documents...")
    chunks = chunk_all_documents(DATA_DIR)
    print(f"  {len(chunks)} chunks created")

    print(f"Indexing (retrieval mode: {RETRIEVAL_MODE})...")
    store = HybridStore() if RETRIEVAL_MODE == "hybrid" else InMemoryStore()
    store.index(chunks)
    print("  index ready\n")
    return store


def answer_question(store: InMemoryStore, question: str, top_k: int = 4) -> None:
    """Single-shot baseline: one retrieval call, no decomposition."""
    results = store.search(question, top_k=top_k)

    print(f"Retrieved {len(results)} chunks:")
    context_blocks = []
    for chunk, score in results:
        print(f"  [{score:.3f}] {chunk.chunk_id}")
        context_blocks.append(f"--- {chunk.chunk_id} ---\n{chunk.text}")
    context = "\n\n".join(context_blocks)

    user_content = f"Policy excerpts:\n\n{context}\n\nQuestion: {question}"

    print("\n--- Answer ---")
    print(generate(SYSTEM_PROMPT, user_content))


def run_orchestrated_query(store: InMemoryStore, question: str, user: dict | None = None) -> dict:
    """LangGraph pipeline: decomposes multi-part questions before retrieval,
    then verifies the answer's inline citations before returning it. Returns
    the raw graph result dict (JSON-serializable except for the Chunk
    objects nested in "retrieved") -- both the CLI and api.py's HTTP
    endpoint build their own presentation on top of this shared call so the
    actual pipeline logic (and query logging) isn't duplicated between them.

    user, when given, scopes retrieval to that user's RBAC access_levels
    (Week 6) -- resolved from Postgres via db.get_user(), enforced inside
    store.search() itself, not filtered from the answer afterward. A user
    dict with id=None (Week 10: an unauthenticated web request, resolved to
    the anonymous/standard scope rather than a real Postgres row) still
    scopes retrieval normally but is skipped in query_log -- there's no
    real user row to attach the log entry to."""
    allowed_doc_ids = db.get_allowed_doc_ids(user["access_levels"]) if user else None

    graph = build_graph(store)
    result = graph.invoke(
        {
            "question": question,
            "sub_questions": [],
            "retrieved": {},
            "final_answer": "",
            "citation_check": {},
            "faithfulness_flags": [],
            "allowed_doc_ids": allowed_doc_ids,
        }
    )

    if user and user.get("id") is not None:
        all_retrieved_ids = sorted({c.chunk_id for r in result["retrieved"].values() for c, _ in r})
        db.log_query(user["id"], question, all_retrieved_ids)

    return result


def answer_question_orchestrated(store: InMemoryStore, question: str, user: dict | None = None) -> None:
    """CLI presentation on top of run_orchestrated_query -- see api.py for
    the HTTP presentation of the same underlying call."""
    if user:
        print(f"Running as: {user['username']} (role: {user['role']}, access: {sorted(user['access_levels'])})\n")

    result = run_orchestrated_query(store, question, user=user)

    print(f"Decomposed into {len(result['sub_questions'])} sub-question(s):")
    for sq in result["sub_questions"]:
        results = result["retrieved"].get(sq, [])
        print(f"  - {sq}")
        for chunk, score in results:
            print(f"      [{score:.3f}] {chunk.chunk_id}")

    print("\n--- Answer ---")
    print(result["final_answer"])

    citation_check = result["citation_check"]
    print("\n--- Grounding check ---")
    print(f"Cited chunk IDs: {citation_check.get('cited_ids', [])}")
    phantom_ids = citation_check.get("phantom_ids", [])
    if phantom_ids:
        print(f"  PHANTOM CITATIONS (cited but never retrieved): {phantom_ids}")

    flags = result["faithfulness_flags"]
    unsupported = [f for f in flags if not f.get("supported", True)]
    if unsupported:
        print(f"Faithfulness check flagged {len(unsupported)}/{len(flags)} citation(s) as unsupported:")
        for f in unsupported:
            print(f"  [{f.get('chunk_id')}] {f.get('claim')!r} -- {f.get('reason')}")
    elif flags:
        print(f"Faithfulness check: all {len(flags)} citation(s) supported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument(
        "--as-user",
        help="username from the Postgres users table (Week 6 RBAC) -- omit to run unrestricted",
    )
    args = parser.parse_args()

    user = db.get_user(args.as_user) if args.as_user else None
    store = build_index()
    print(f"Question: {args.question}\n")
    if USE_ORCHESTRATOR:
        answer_question_orchestrated(store, args.question, user=user)
    else:
        answer_question(store, args.question)

