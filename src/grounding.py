"""
Grounded generation: inline citations + a faithfulness check.

Two independent layers, deliberately kept separate:

1. Citation validation (deterministic, free). Every [chunk_id] the model
   cites is checked against the chunk IDs actually retrieved for this
   question. A citation to an ID that was never retrieved is a "phantom
   citation" -- the model referencing a source it made up, or mangled the
   ID of. This alone catches a real class of hallucination without needing
   another LLM call.

2. Faithfulness check (a second LLM call). For each citation, ask whether
   the cited excerpt's text actually supports the claim attached to it.
   This check is itself an LLM call and can be wrong -- CLAUDE.md's "eval
   circularity" edge case applies here directly. Treat a flagged claim as a
   lead to spot-check by hand, not as ground truth.
"""

import json
import re

from chunking import Chunk
from llm import extract_json_array, generate

CITATION_PATTERN = re.compile(r"\[([^\[\]]+?::\d+)\]")

GROUNDED_ANSWER_INSTRUCTIONS = """

After every sentence that states a fact from the policy excerpts, cite the \
exact excerpt ID it came from in square brackets, using precisely the ID \
shown in the "--- chunk_id ---" marker above that excerpt (e.g. "...covered \
for a BMI of 35 or greater [ncd_57_bariatric_surgery...::7]."). Never invent \
or abbreviate an ID. If a sentence isn't grounded in a specific excerpt, \
don't cite one for it, and don't state it as settled fact either."""

FAITHFULNESS_SYSTEM_PROMPT = """You are a strict fact-checker for a healthcare \
coverage policy system. You will be given policy excerpts and an answer that \
cites them with [chunk_id] markers. For each citation in the answer, check \
whether the specific claim attached to it is actually supported by that \
excerpt's text -- not by outside knowledge, not by an inference the excerpt \
doesn't state.

Respond with ONLY a JSON array, one object per citation found, each with keys \
"chunk_id", "claim" (the sentence or clause citing it), "supported" \
(true/false), and "reason" (one short sentence). Nothing else -- no \
markdown fences, no explanation outside the array."""


def extract_citations(answer: str) -> set[str]:
    return set(CITATION_PATTERN.findall(answer))


def _cited_id_matches(cited: str, retrieved_ids: set[str]) -> bool:
    """Exact match, or -- because a smaller local model sometimes truncates
    a long chunk ID with "..." instead of copying it verbatim despite being
    told not to -- a truncation-tolerant match. Found on real output:
    "ncd_57_bariatric_surgery...::7" for the real ID
    "ncd_57_..._co_morbid_condi.txt::7". Only "..."/"…" get treated as a
    wildcard, so a genuinely fabricated ID still won't match anything."""
    if cited in retrieved_ids:
        return True
    if "..." in cited or "…" in cited:
        pattern = re.escape(cited).replace(r"\.\.\.", ".*").replace("\\…", ".*")
        return any(re.fullmatch(pattern, rid) for rid in retrieved_ids)
    return False


def check_citations(answer: str, retrieved: list[tuple[Chunk, float]]) -> dict:
    """Deterministic check: flags citations to chunk IDs that were never
    actually retrieved for this question."""
    retrieved_ids = {c.chunk_id for c, _ in retrieved}
    cited_ids = extract_citations(answer)
    phantom_ids = {cid for cid in cited_ids if not _cited_id_matches(cid, retrieved_ids)}
    return {
        "cited_ids": sorted(cited_ids),
        "phantom_ids": sorted(phantom_ids),
    }


def run_faithfulness_check(answer: str, retrieved: list[tuple[Chunk, float]]) -> list[dict]:
    """LLM-based check: flags citations whose claim isn't actually supported
    by the cited excerpt's text. See module docstring -- this can itself be
    wrong; don't treat its output as ground truth."""
    if not extract_citations(answer):
        return []

    context = "\n\n".join(f"--- {c.chunk_id} ---\n{c.text}" for c, _ in retrieved)
    user_content = f"Policy excerpts:\n\n{context}\n\nAnswer to fact-check:\n\n{answer}"
    raw = generate(FAITHFULNESS_SYSTEM_PROMPT, user_content, max_tokens=800)
    try:
        parsed = json.loads(extract_json_array(raw))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []
