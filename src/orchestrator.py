"""
LangGraph orchestration for the query pipeline: routes a question to either
a direct single-lookup answer or, for genuinely multi-part questions,
decomposes it into sub-questions, retrieves separately for each, and
synthesizes one final answer from all of them.

Decomposition exists because retrieval embeds/searches with one vector per
question -- if a question actually bundles two distinct topics, that one
embedding sits between both in vector space and top-k retrieval can end up
mediocre for both instead of good for either. Splitting it turns one blurred
retrieval problem into N independent ones.

Both terminal paths flow through a `verify` node before END, which checks
every inline [chunk_id] citation the answer makes against what was actually
retrieved (catches phantom citations for free) and runs an LLM faithfulness
check on top (catches claims the citation doesn't actually support -- see
grounding.py's docstring for why that check's own verdicts still need
spot-checking).

State carries allowed_doc_ids, the calling user's RBAC scope, forwarded into
every store.search() call so restricted documents are excluded at the
retrieval layer itself -- never fetched, reranked, or put in front of the
LLM -- rather than filtered out of the final answer afterward.

Graph:

    decompose --(1 sub-question)--> direct_answer ------\
              --(2+ sub-questions)--> retrieve_subquestions --> synthesize --> verify --> END
"""

import json
from typing import TypedDict

from langgraph.graph import END, StateGraph

from chunking import Chunk
from grounding import GROUNDED_ANSWER_INSTRUCTIONS, check_citations, run_faithfulness_check
from llm import extract_json_array, generate

DECOMPOSE_SYSTEM_PROMPT = """You break a user's question into the smallest set \
of independent sub-questions needed to answer it completely, for a healthcare \
coverage policy retrieval system. Most questions are already a single, \
self-contained question -- in that case return exactly one sub-question \
containing the original question, unchanged. Only split into multiple \
sub-questions when the question genuinely asks about two or more distinct \
procedures or policy topics that would need separate policy lookups to \
answer well.

A question about ONE procedure is NOT multi-part just because it has an \
"and" or a qualifier clause in it -- keep it as a single sub-question so \
retrieval keeps the full context (the procedure name, the specific \
condition being asked about). Splitting a one-procedure question loses that \
shared context and hurts retrieval.

Example -- do NOT split (single procedure, one sub-question, unchanged):
Question: "Is a total knee replacement covered without physical therapy first?"
["Is a total knee replacement covered without physical therapy first?"]

Example -- DO split (two distinct procedures):
Question: "What BMI is required for bariatric surgery, and what oxygen saturation level qualifies someone for home oxygen therapy?"
["What BMI is required for bariatric surgery?", "What oxygen saturation level qualifies someone for home oxygen therapy?"]

Respond with ONLY a JSON array of strings, nothing else -- no explanation, no \
markdown fences."""

ANSWER_SYSTEM_PROMPT = (
    """You are a healthcare coverage policy assistant. \
Answer the user's question using ONLY the policy excerpts provided below. If \
the excerpts don't contain enough information to answer confidently, say so \
explicitly instead of guessing. Do not use outside knowledge about Medicare \
policy."""
    + GROUNDED_ANSWER_INSTRUCTIONS
)

SYNTHESIZE_SYSTEM_PROMPT = (
    """You are a healthcare coverage policy assistant. \
The user's original question was broken into sub-questions, and each was \
researched separately against policy excerpts. Combine those findings into \
one coherent answer to the ORIGINAL question, addressing every part. If the \
excerpts for any part don't contain enough information, say so explicitly \
for that part instead of guessing."""
    + GROUNDED_ANSWER_INSTRUCTIONS
)


class GraphState(TypedDict):
    question: str
    sub_questions: list[str]
    retrieved: dict[str, list[tuple[Chunk, float]]]
    final_answer: str
    citation_check: dict
    faithfulness_flags: list[dict]
    allowed_doc_ids: set[str] | None  # RBAC scope of the asking user; None = unrestricted


def _flatten_retrieved(retrieved: dict[str, list[tuple[Chunk, float]]]) -> list[tuple[Chunk, float]]:
    seen: dict[str, tuple[Chunk, float]] = {}
    for results in retrieved.values():
        for chunk, score in results:
            seen[chunk.chunk_id] = (chunk, score)
    return list(seen.values())


def _format_context(results: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"--- {c.chunk_id} ---\n{c.text}" for c, _ in results)


def build_graph(store):
    """store must expose .search(query, top_k) -> list[(Chunk, score)],
    matching both InMemoryStore and HybridStore."""

    def decompose(state: GraphState) -> dict:
        raw = generate(DECOMPOSE_SYSTEM_PROMPT, state["question"], max_tokens=300)
        try:
            parsed = json.loads(extract_json_array(raw))
            if not isinstance(parsed, list) or not parsed:
                raise ValueError("empty or malformed decomposition")
            sub_questions = [str(q) for q in parsed]
        except (json.JSONDecodeError, ValueError):
            sub_questions = [state["question"]]
        return {"sub_questions": sub_questions}

    def route(state: GraphState) -> str:
        return "direct" if len(state["sub_questions"]) <= 1 else "decomposed"

    def direct_answer(state: GraphState) -> dict:
        question = state["sub_questions"][0]
        results = store.search(question, top_k=4, allowed_doc_ids=state.get("allowed_doc_ids"))
        answer = generate(
            ANSWER_SYSTEM_PROMPT,
            f"Policy excerpts:\n\n{_format_context(results)}\n\nQuestion: {question}",
        )
        return {"retrieved": {question: results}, "final_answer": answer}

    def retrieve_subquestions(state: GraphState) -> dict:
        allowed = state.get("allowed_doc_ids")
        retrieved = {sq: store.search(sq, top_k=4, allowed_doc_ids=allowed) for sq in state["sub_questions"]}
        return {"retrieved": retrieved}

    def synthesize(state: GraphState) -> dict:
        blocks = [
            f"## Sub-question: {sq}\n\n{_format_context(results)}"
            for sq, results in state["retrieved"].items()
        ]
        user_content = f"Original question: {state['question']}\n\n" + "\n\n".join(blocks)
        answer = generate(SYNTHESIZE_SYSTEM_PROMPT, user_content, max_tokens=700)
        return {"final_answer": answer}

    def verify(state: GraphState) -> dict:
        all_retrieved = _flatten_retrieved(state["retrieved"])
        citation_check = check_citations(state["final_answer"], all_retrieved)
        faithfulness_flags = run_faithfulness_check(state["final_answer"], all_retrieved)
        return {"citation_check": citation_check, "faithfulness_flags": faithfulness_flags}

    graph = StateGraph(GraphState)
    graph.add_node("decompose", decompose)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("retrieve_subquestions", retrieve_subquestions)
    graph.add_node("synthesize", synthesize)
    graph.add_node("verify", verify)

    graph.set_entry_point("decompose")
    graph.add_conditional_edges(
        "decompose", route, {"direct": "direct_answer", "decomposed": "retrieve_subquestions"}
    )
    graph.add_edge("direct_answer", "verify")
    graph.add_edge("retrieve_subquestions", "synthesize")
    graph.add_edge("synthesize", "verify")
    graph.add_edge("verify", END)

    return graph.compile()
