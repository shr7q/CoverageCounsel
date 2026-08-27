# Healthcare compliance RAG assistant — project context

## What this is and why

A portfolio project built to demonstrate production grade RAG engineering for
AI Engineer, AI Software Engineer, and Forward Deployed Engineer job
applications. The goal is not a tutorial RAG demo — it's a system that
behaves like something that could actually sit inside a regulated healthcare
organization: grounded answers with citations, access control enforced at
retrieval (not just UI), automated evaluation, and full observability.

It extends an earlier project (Dental Bot, a healthcare RAG chatbot) to build
a coherent two-project healthcare AI narrative for interviews, and ties into
real interview experience at Aegis Health.

The differentiator versus a typical portfolio RAG project: most naive RAG
fails quietly on exactly the failure modes regulatory text creates (see
"known hard edge cases" below). This project is built around catching those
on purpose, with a documented before/after bug fix, rather than optimizing
for a happy-path demo question.

## Target resume bullet (what "done" looks like)

"Production RAG system with hybrid retrieval, reranking, multi-hop query
decomposition, role-based access control, and automated evaluation, deployed
with CI/CD and full observability, covering data ingestion through
deployment."

## Current state

Walking skeleton is done: naive fixed-size chunking -> OpenAI embeddings in
an in-memory store -> top-k retrieval -> Claude generates an answer from
retrieved chunks. No reranking, no decomposition, no auth, no citation
parsing, no eval, no real deployment yet. Sample data is 3 synthetic policy
docs in `data/sample_docs/`, not real CMS data yet.

Known bug already found: the naive chunker splits mid-word/mid-sentence
(confirmed in `bariatric_surgery.txt` chunk 1, "r obstructive sleep apnea").
This is the first concrete "before" example for the eventual bug writeup.

## Architecture (target end state)

```
User query
  -> Hybrid retrieval (dense + BM25 sparse)
  -> Reranking (cross-encoder)
  -> Agent orchestration (LangGraph — routes direct lookup vs. decomposition)
  -> Grounded generation (inline citations + faithfulness check)
  -> Answer to user (with source citations)
```

Cross-cutting, applies to every stage above:
- Auth & RBAC — OAuth (Clerk/Auth0), permissions enforced at retrieval layer,
  Postgres schema for users/roles/document metadata (also the SQL-depth
  practice ground: window functions, CTEs for audit reporting)
- Evaluation — RAGAS (faithfulness, answer relevance, context precision) on
  a ~40-question golden set, gated in GitHub Actions
- Observability — Langfuse or LangSmith, full trace-level cost/latency
- Deployment — Docker, Postgres+pgvector or Qdrant, Railway/Fly.io/ECS,
  Next.js+TypeScript frontend on Vercel, CI/CD gates deploy on eval pass

## Build order (don't skip ahead — each phase should be motivated by a real
failure seen in the phase before it)

1. Real data swap (CMS NCD/LCD bulk XML + Medicare Benefit Policy Manual
   chapters) — do this before anything else below, synthetic data won't
   surface real bugs
2. Structure-aware chunking (split on "Section N." headers, not fixed size)
3. Hybrid retrieval (BM25 + dense)
4. Reranking (cross-encoder)
5. Query decomposition + LangGraph orchestration
6. Grounded generation: inline citations + faithfulness check
7. Auth + RBAC + Postgres schema
8. Eval suite (RAGAS + CI gating)
9. Observability (Langfuse/LangSmith)
10. Deployment (Docker, CI/CD, hosting)

## Known hard edge cases to deliberately test for (don't consider a phase
done until these are checked)

- **Cross-references**: clauses reference other sections/documents ("as
  defined in Section 4.2 of MED-NEC-GEN-01"). A chunk can be technically
  accurate and still misleading in isolation. Test: does retrieval pull the
  referenced clause too, or at least flag that one exists outside context?
- **Effective dates / versioning**: policies get superseded. A confidently
  cited outdated rule is a serious, realistic failure mode. Test: does the
  system ever claim a specific effective date it can't actually verify?
- **Regional variation**: Local Coverage Determinations differ by MAC
  jurisdiction. The "correct" answer can legitimately depend on region.
  Test: does the system silently pick one region's answer, or flag the
  ambiguity and ask?
- **Absence of evidence**: when the corpus has no answer, a naive system
  will still synthesize something plausible. Test: ask something genuinely
  uncovered (see README) and check it admits it doesn't know.
- **Negation / exceptions**: "covered except when condition X" breaks easily
  if retrieval grabs the rule sentence without its exception clause.
- **Table extraction**: coverage criteria are sometimes literally tabular in
  source PDFs; naive text extraction mangles these — needs dedicated
  handling once real PDF data is in use.
- **Cost/latency of decomposition**: multi-part questions trigger multiple
  sub-queries and rerank passes — multiplies token cost and latency. Worth
  measuring, and worth deciding when decomposition is actually triggered vs.
  overkill for a simple lookup.
- **Permission leakage**: RBAC enforced in the UI is not enough — test that
  a restricted user genuinely cannot retrieve a document via the API/
  retrieval layer itself, including through any caching layer.
- **Eval circularity**: the faithfulness checker is itself an LLM and can
  also be wrong — don't treat it as ground truth without spot-checking.

## Tech stack decisions already made

- Generation: Claude (Anthropic API), matches other projects in the
  portfolio (BugSlayer, OneStopJob)
- Embeddings: OpenAI `text-embedding-3-small` (Anthropic has no embeddings
  endpoint)
- Orchestration: LangGraph (same as BugSlayer, for narrative consistency)
- Frontend: Next.js + TypeScript on Vercel (matches OneStopJob stack)
- Auth: Clerk or Auth0
- Vector store: pgvector (simplest credible choice) or Qdrant if wanting to
  show a dedicated vector DB
- Eval: RAGAS + GitHub Actions
- Observability: Langfuse or LangSmith

## What "finished" needs to include for the portfolio writeup

Two or three specific bugs found and fixed, with concrete before/after
examples (the mid-word chunking bug is the first one, already captured
above). This writeup matters as much as the deployed app for interviews —
don't skip documenting it once phases 1-2 surface more of these.