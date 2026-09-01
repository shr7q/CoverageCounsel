# Bugs found and fixed

This project was built around a specific premise (see `CLAUDE.md`): most RAG
portfolio projects optimize for a happy-path demo question and never
encounter the failure modes that regulatory text actually creates. Every bug
below was found by testing against real CMS coverage policy text and real
questions, not by inspection — each one includes what the system did wrong,
why, and how the fix was verified.

## 1. Fixed-size chunking split the one clause that answered the question

**Symptom.** Asked "Is a total knee replacement covered without physical
therapy first?" against the real LCD 36575 text, the naive fixed-size
chunker answered confidently that physical therapy was "not mentioned in
any of these excerpts" — using four retrieved chunks, none of which
contained the relevant clause.

**Root cause.** The actual eligibility clause — *"history of unsuccessful
conservative therapy (non-surgical medical management)... [including]
physical therapy"* — fell exactly at a fixed-size chunk boundary. The
chunker split it mid-word (`...j` / `erformed)...`), degrading that chunk's
embedding quality enough that it never made the top-4 retrieved results,
even though it directly answered the question.

**Fix.** Replaced fixed-size chunking with a structure-aware chunker
(`src/chunking.py`) that splits on real section/heading boundaries first,
falling back to paragraph- then sentence-boundary splits only when a
section is still too long — never on raw character offsets.

**Verified.** Re-ran the same question: the full eligibility clause is now
one intact 937-character chunk, present in retrieval, and the answer
correctly quotes the actual conditional language ("if appropriate...").

## 2. Dense-only retrieval missed the best chunk even after chunking was fixed

**Symptom.** After fixing (1), a *different* chunk — the one explicitly
listing "supervised physical therapy" as a non-surgical treatment — still
never appeared in the top-4 dense-search results, across three separate
test runs.

**Root cause.** A single embedding of the whole question doesn't
necessarily rank the most specific supporting chunk highest; dense cosine
similarity alone wasn't precise enough.

**Fix.** Added BM25 sparse retrieval alongside dense search, combined via
Reciprocal Rank Fusion, then reranked the fused candidates with a
cross-encoder (`src/hybrid_store.py`).

**Verified.** Same question, same corpus: the previously-missing chunk now
ranks #2, and the generated answer directly engages with its language for
the first time — a controlled before/after on identical input.

## 3. Query decomposition made a single-topic question worse, not better

**Symptom.** While building multi-hop query decomposition, a *single-topic*
question ("Is a total knee replacement covered without physical therapy
first?") got split into two independent sub-questions — one of which lost
the words "knee replacement" entirely and retrieved irrelevant oxygen-LCD
chunks instead.

**Root cause.** The decomposition prompt didn't distinguish "genuinely
multi-topic" from "one topic with an `and`/qualifier clause in it," so a
compound-phrased single question got over-split, and the resulting
sub-question lost shared context the original question carried implicitly.

**Fix.** Added contrasting few-shot examples to the decomposition prompt
(`src/orchestrator.py`) — one showing a single-topic question that should
stay whole, one showing a genuine two-topic question that should split.

**Verified.** Stable across 3 repeated runs each on both a single-topic and
a genuine two-topic test question after the fix — this was caught by
insisting on a real before/after comparison rather than assuming
decomposition is strictly an improvement.

## 4. The model cited a real chunk that didn't actually support its claim

**Symptom.** Asked for the BMI threshold for bariatric surgery, the model
answered correctly (BMI ≥ 35) but cited chunk `::7`, which is actually about
a narrower rule (MAC-discretionary procedures) that happens to also mention
BMI ≥ 35 as a sub-criterion — not the general rule the question asked about.

**Root cause.** The real primary source for that rule is a different chunk
(`::2`, "B. Nationally Covered Indications"). The claim was directionally
true; the citation attached to it wasn't the right evidence.

**Fix.** Added an LLM-based faithfulness check (`src/grounding.py`) that
verifies each cited excerpt actually supports its attached claim, run as a
second pass after generation.

**Verified.** The faithfulness check flagged exactly this mismatch. (Also
documented two cases where the *checker itself* was wrong — a real instance
of the "eval circularity" risk called out in `CLAUDE.md` — confirming its
output needs spot-checking, not blind trust.)

## 5. A broken eval run silently reported a passing score

**Symptom.** The first full RAGAS run against the 42-question golden set
returned a context_precision score of 0.762 — until inspecting the raw
per-question results showed only 12 of 42 questions had actually scored;
the other 30 silently became NaN from timeouts, and `pandas.mean()`
averaged only the survivors with no indication anything was missing.

**Root cause.** RAGAS's default `RunConfig` fires up to 16 concurrent
scoring calls; a single local Ollama instance can't serve that much
concurrent load, so most calls timed out and were dropped rather than
retried to completion.

**Fix.** Serialized scoring to `max_workers=1` for local runs
(`eval/run_eval.py`), and added an explicit NaN-count check that warns
before trusting any reported mean.

**Verified.** Re-ran the full golden set: 39/42 faithfulness scores valid,
42/42 on the other two metrics — a trustworthy number instead of a
plausible-looking one.

## 6. The regression gate didn't actually catch the regression it was built for

**Symptom.** Deliberately reverted retrieval to dense-only (undoing #2
above) and ran the full eval to confirm the CI gate would catch it. It
didn't: mean faithfulness barely moved (0.708 → 0.755).

**Root cause.** Faithfulness measures whether an answer's claims trace back
to *whatever* was retrieved — it doesn't measure whether the *right* things
were retrieved. `answer_relevancy` dropped 0.716 → 0.605 on the same run,
a real signal the single-metric gate missed entirely.

**Fix.** Added an `answer_relevancy` threshold alongside faithfulness in
the CI gate.

**Verified.** Re-checked both real completed runs against the new gate
logic: the hybrid baseline passes, the dense-only regression now correctly
fails — without needing to burn another 40-minute eval run to prove it.

## 7. A bug that was completely invisible in the terminal, found only in traces

**Symptom.** Deliberately dropped `.lower()` from the BM25 tokenizer
(`src/hybrid_store.py`) to test whether LangSmith tracing could actually
catch a real regression. Ran the exact test question from bug #2 — the
final answer still looked correct, same citation as before.

**Root cause.** `re.findall(r"[a-z0-9]+", text)` without lowercasing first
means any capitalized term — "BMI", "Coverage", "Noridian" — is either
mangled (`"Beneficiaries"` → `"eneficiaries"`) or, if fully uppercase,
vanishes from tokenization entirely. BM25 quietly lost the ability to
exact-match on precisely the acronyms and proper nouns it exists to catch.

**Fix path.** The terminal output gave zero indication anything was wrong.
Found it by inspecting the `bm25_search` span in the LangSmith trace —
something the CLI's print statements never surface at all, since they only
print the final fused-and-reranked result — which pointed directly at
`_tokenize()`.

**Verified.** Confirmed via direct test: `_tokenize("What BMI is...")`
dropped "BMI" entirely before the fix, tokenized it correctly after.

---

**Smaller fixes worth a mention:** a `UnicodeEncodeError` crash on Windows
console encoding when real CMS text contains `≥` (never present in the
synthetic sample docs); a Docker image that was 8.8GB instead of 2.3GB
because `sentence-transformers` pulls a CUDA build of PyTorch by default
with no GPU to use it; and the Ollama base URL being hardcoded to
`localhost`, which silently means "the container itself" once the app runs
inside Docker rather than on a laptop.
