"""
Week 7: run the golden set through the real production pipeline (the same
LangGraph orchestrator query.py uses) and score it with RAGAS.

Metrics: faithfulness (does the answer's claims trace back to the retrieved
context?), answer_relevancy (does the answer actually address the question?),
context_precision (are the retrieved chunks relevant, ranked against the
golden set's reference answer?).

This is the CI gate: run_eval.py exits non-zero if mean faithfulness or mean
answer_relevancy drops below its threshold, so a regression (e.g. reverting
to dense-only retrieval, or breaking the reranker) fails the build instead of
silently shipping worse answers. (Gating on faithfulness alone wasn't enough
-- see the ANSWER_RELEVANCY_THRESHOLD comment below for what that missed.)

Usage (from repo root):
    python eval/run_eval.py                  # full golden set
    python eval/run_eval.py --limit 5         # quick smoke test
    python eval/run_eval.py --retrieval-mode dense   # before/after comparison
"""

import argparse
import json
import os
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

# ragas 0.4.x unconditionally imports ChatVertexAI from langchain_community at
# module load time, but that submodule was removed in langchain-community
# 0.4.x (moved to the separate langchain-google-vertexai package). We never
# use Vertex AI -- pulling in that whole package just to satisfy a dead
# import isn't worth it, and downgrading langchain-community drags
# langchain-core back below 1.0, which breaks langgraph (load-bearing for
# the orchestrator). Stub the one missing symbol instead.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from chunking import chunk_all_documents
from embed_store import InMemoryStore
from hybrid_store import HybridStore
from orchestrator import build_graph

FAITHFULNESS_THRESHOLD = 0.7
# Discovered empirically: reverting hybrid+rerank to dense-only retrieval
# (the Week 3 regression this gate exists to catch) barely moved
# faithfulness (0.708 -> 0.755, within local-model run-to-run noise on 42
# questions) but dropped answer_relevancy substantially (0.716 -> 0.605).
# Faithfulness alone isn't a reliable enough signal for a retrieval-quality
# regression -- gate on both.
ANSWER_RELEVANCY_THRESHOLD = 0.65
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real_docs")
GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


def check_thresholds(means: dict) -> list[str]:
    """Returns a list of failure messages; empty list means the gate passes."""
    failures = []
    if means["faithfulness"] < FAITHFULNESS_THRESHOLD:
        failures.append(
            f"mean faithfulness {means['faithfulness']:.3f} is below the {FAITHFULNESS_THRESHOLD} threshold"
        )
    if means["answer_relevancy"] < ANSWER_RELEVANCY_THRESHOLD:
        failures.append(
            f"mean answer_relevancy {means['answer_relevancy']:.3f} is below the "
            f"{ANSWER_RELEVANCY_THRESHOLD} threshold"
        )
    return failures


def _build_ragas_llm_and_embeddings():
    """RAGAS's metrics are themselves LLM calls and need an embeddings model
    for answer_relevancy -- wire them to whatever this run is already using
    (LLM_PROVIDER/EMBED_PROVIDER), same provider-switching convention as the
    rest of the app, so eval works fully local (Ollama) today and just needs
    the env vars flipped once real Claude/OpenAI keys are in place."""
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    llm_provider = os.environ.get("LLM_PROVIDER", "anthropic")
    embed_provider = os.environ.get("EMBED_PROVIDER", "openai")

    if llm_provider == "ollama":
        llm = ChatOllama(model=os.environ.get("OLLAMA_LLM_MODEL", "llama3.1:8b"), temperature=0)
    else:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model="claude-sonnet-4-6", api_key=os.environ.get("ANTHROPIC_API_KEY"))

    if embed_provider == "ollama":
        embeddings = OllamaEmbeddings(model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    elif embed_provider == "sentence_transformers":
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=os.environ.get("SENTENCE_TRANSFORMERS_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
        )
    else:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.environ.get("OPENAI_API_KEY"))

    return llm, embeddings


def run_pipeline(store, question: str) -> dict:
    """Runs one question through the real orchestrator graph, returns the
    RAGAS-shaped record: question, answer, contexts (retrieved chunk text)."""
    graph = build_graph(store)
    result = graph.invoke(
        {
            "question": question,
            "sub_questions": [],
            "retrieved": {},
            "final_answer": "",
            "citation_check": {},
            "faithfulness_flags": [],
            "allowed_doc_ids": None,
        }
    )
    contexts = []
    for results in result["retrieved"].values():
        contexts.extend(c.text for c, _ in results)
    return {"answer": result["final_answer"], "contexts": contexts or ["(no context retrieved)"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only run the first N golden-set questions")
    parser.add_argument(
        "--retrieval-mode", choices=["hybrid", "dense"], default=os.environ.get("RETRIEVAL_MODE", "hybrid")
    )
    args = parser.parse_args()

    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        golden_set = json.load(f)
    if args.limit:
        golden_set = golden_set[: args.limit]

    print(f"Indexing {DATA_DIR} (retrieval mode: {args.retrieval_mode})...")
    chunks = chunk_all_documents(DATA_DIR)
    store = HybridStore() if args.retrieval_mode == "hybrid" else InMemoryStore()
    store.index(chunks)

    records = []
    for i, item in enumerate(golden_set, start=1):
        print(f"[{i}/{len(golden_set)}] {item['question'][:80]}...")
        start = time.time()
        result = run_pipeline(store, item["question"])
        print(f"    ({time.time() - start:.1f}s)")
        records.append(
            {
                "question": item["question"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": item["ground_truth"],
            }
        )

    dataset = Dataset.from_list(records)
    llm, embeddings = _build_ragas_llm_and_embeddings()

    # RAGAS's default RunConfig fires up to 16 concurrent scoring calls.
    # Originally serialized only for local Ollama (a single instance can't
    # serve that much concurrent load), on the assumption a real hosted API
    # would handle the default concurrency fine in CI. It didn't: a real run
    # against Claude hit the same wall -- every context_precision score came
    # back NaN from timeouts, almost certainly this API tier's rate limits
    # rejecting 16 concurrent requests. Serializing unconditionally is slower
    # (a CI gate that isn't time-critical can afford ~30-60 min) but reliable
    # regardless of what's actually behind LLM_PROVIDER.
    from ragas.run_config import RunConfig

    run_config = RunConfig(max_workers=1, timeout=420)

    print("\nScoring with RAGAS (faithfulness, answer_relevancy, context_precision)...")
    scored = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )
    df = scored.to_pandas()
    df.to_json(RESULTS_PATH, orient="records", indent=2)

    missing = {col: int(df[col].isna().sum()) for col in ("faithfulness", "answer_relevancy", "context_precision")}
    if any(missing.values()):
        print(f"\nWARNING: some scores are missing (likely timeouts): {missing}")
        print("Treat the means below as computed only over the questions that scored successfully.")

    means = {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "context_precision": float(df["context_precision"].mean()),
    }
    print(f"\n=== RAGAS scores (n={len(records)}, retrieval_mode={args.retrieval_mode}) ===")
    for name, value in means.items():
        print(f"  {name}: {value:.3f}")
    print(f"\nPer-question results written to {RESULTS_PATH}")

    failures = check_thresholds(means)
    if failures:
        print("\nFAIL: regression gate tripped --")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)

    print("\nPASS: all thresholds met.")


if __name__ == "__main__":
    main()
