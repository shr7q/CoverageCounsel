# Weekly project plan — healthcare compliance RAG assistant

Ten weeks, part time. Each week builds on a real failure seen in the week
before it — don't skip ahead just because a step sounds easy. Full context,
architecture, and edge case details live in `CLAUDE.md`; this file is just
the week-by-week checklist.

## Week 1 — Real data swap
- [ ] Download CMS National Coverage Determinations (bulk XML) and Local
      Coverage Determinations for at least 2-3 procedure areas
- [ ] Download a few Medicare Benefit Policy Manual chapters (PDF)
- [ ] Replace `data/sample_docs/` with real extracted text
- [ ] Re-run the existing walking skeleton against real data, note what
      breaks (chunking is the most likely first casualty)

**Done when:** the skeleton runs end to end on real CMS text and you have a
list of at least 2-3 concrete things that broke.

## Week 2 — Structure-aware chunking
- [ ] Replace the fixed-size chunker with one that splits on section
      headers / clause boundaries first, sub-chunks only if a section is
      still too long
- [ ] Test against the cross-reference edge case (a clause referencing
      another section) and the negation edge case (rule + exception clause)

**Done when:** chunks stop cutting mid-sentence/mid-clause on real data.

## Week 3 — Hybrid retrieval + reranking
- [ ] Add BM25 sparse retrieval alongside the existing dense search
- [ ] Combine dense + sparse into a single hybrid ranking
- [ ] Add a cross-encoder reranking pass on top of hybrid retrieval
- [ ] Compare hybrid+rerank vs. dense-only on a handful of hand-written
      test questions — confirm it's actually better, don't just assume

**Done when:** you can show retrieval quality improved with before/after
examples, not just "it's hybrid now."

## Week 4 — Query decomposition + orchestration
- [ ] Wrap retrieval + reranking in a LangGraph agent
- [ ] Add routing logic: direct lookup vs. decomposition
- [ ] Add decomposition for multi-part questions (break into sub-questions,
      retrieve for each, synthesize)
- [ ] Test the regional variation edge case (does it ask for region context
      when the answer legitimately depends on it, or guess silently?)

**Done when:** a genuinely multi-part question gets a correct decomposed
answer that a single-shot query would get wrong.

## Week 5 — Grounded generation
- [ ] Add inline citations tracing every claim to a specific chunk
- [ ] Add a faithfulness check flagging claims not traceable to retrieved
      text
- [ ] Test the absence-of-evidence edge case (ask something the corpus
      doesn't cover — does it say so honestly?)
- [ ] Test the versioning/effective-date edge case (does it ever invent or
      assume a specific effective date?)

**Done when:** you have a documented example of the system correctly
refusing to answer instead of hallucinating.

## Week 6 — Access control
- [ ] Add OAuth (Clerk or Auth0)
- [ ] Build the Postgres schema: users, roles, document metadata
- [ ] Enforce permissions at the retrieval layer, not just the UI
- [ ] Test the permission-leakage edge case: confirm a restricted user
      genuinely cannot retrieve a document outside their role via the
      API/retrieval layer itself, including through any caching
- [ ] Write at least one SQL query using window functions or a CTE for an
      audit/usage report

**Done when:** a restricted-role test account is verifiably blocked at the
data layer, not just hidden in the UI.

## Week 7 — Evaluation
- [ ] Build a ~40-question golden set with known correct citations
- [ ] Wire up RAGAS (faithfulness, answer relevance, context precision)
- [ ] Set up GitHub Actions to run the eval suite and fail the build on
      regression

**Done when:** you can show a real RAGAS score, and a deliberate regression
(e.g. reverting reranking) actually fails the CI check.

## Week 8 — Observability
- [ ] Instrument the full pipeline with Langfuse or LangSmith
- [ ] Confirm you can trace a single request end to end: retrieval scores,
      reranker scores, prompt sent, cost, latency per step
- [ ] Deliberately break something and use traces (not print statements) to
      find it — this is the real test of whether observability works

**Done when:** you've debugged a real issue using traces alone.

## Weeks 9-10 — Deployment and polish
- [ ] Containerize retrieval + generation services with Docker
- [ ] Stand up Postgres + pgvector (or Qdrant)
- [ ] Deploy to Railway, Fly.io, or AWS ECS
- [ ] Frontend on Vercel (Next.js + TypeScript)
- [ ] CI/CD: eval suite gates deploy on merge to main
- [ ] Write up 2-3 specific bugs found and fixed, with before/after examples
      (start from the mid-word chunking bug already found in the walking
      skeleton)

**Done when:** the app is live at a real URL, CI/CD is gating deploys, and
the writeup is drafted.