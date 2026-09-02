# CoverageCounsel — Project Deep Dive

A complete reference for explaining this project in an interview: what it is,
why every decision was made, the full tech stack, the numbers, and the
questions an interviewer is likely to ask. See `BUGS.md` for the detailed
before/after writeups of specific bugs found, and `ProjectPlan.md`/
`CLAUDE.md` for the original week-by-week plan this was built against.

## Live

- **App**: https://coverage-counsel.vercel.app
- **API**: https://healthcare-rag-api-1018049989356.us-central1.run.app
- **Source**: https://github.com/shr7q/CoverageCounsel

## 1. What this is, in one paragraph

A production-shaped RAG system that answers questions about real Medicare
coverage policy (real CMS National/Local Coverage Determinations and
Medicare Benefit Policy Manual chapters — not synthetic demo text), with
hybrid retrieval, reranking, multi-hop query decomposition, inline citations
verified by a faithfulness check, role-based access control enforced at the
retrieval layer (not the UI), automated evaluation gating deploys in CI, and
full observability. Deployed end-to-end: Next.js frontend on Vercel, FastAPI
backend on Cloud Run, Postgres+pgvector on Supabase, real Claude for
generation, real OAuth (Clerk) for identity.

## 2. The differentiator (the thing to say if asked "why is this different from every other RAG portfolio project")

Most RAG demos optimize for a happy-path question against clean text. This
one was built around the opposite premise: regulatory/compliance text has
specific, well-known failure modes — dangling cross-references, ambiguous
effective dates, regional variation, negation/exception clauses, silent
hallucination on uncovered topics, permission leakage — and the project was
built to deliberately surface and fix those, with a documented before/after
for each one, rather than to look good on a single demo question. `BUGS.md`
is the evidence: 7 concrete bugs, each with symptom → root cause → fix →
verification, several found only because a test was designed to expose that
exact failure mode.

## 3. Architecture

```
User (browser)
  → Next.js frontend (Vercel)
      -- Clerk sign-in (optional); attaches a session JWT if signed in
  → FastAPI backend (Cloud Run)
      -- verifies the JWT (or treats the request as anonymous/standard)
      -- resolves the caller's RBAC access_levels from Postgres
  → LangGraph orchestrator
      -- decompose: is this one question or several?
      -- (single) → direct_answer   OR   (multi) → retrieve per sub-question → synthesize
      -- verify: check inline citations against what was actually retrieved,
                 then an LLM faithfulness pass
  → Hybrid retrieval (per sub-question)
      -- dense search (pgvector, cosine) -- RBAC-filtered via SQL WHERE
      -- BM25 sparse search -- RBAC-filtered before ranking
      -- Reciprocal Rank Fusion combines both
      -- cross-encoder reranks the fused candidates down to top-k
  → Claude (Anthropic API) generates the grounded answer
  → back to the browser, with citations, sub-questions, and any
    faithfulness warnings
```

Everything is traced end-to-end in LangSmith (prompt, retrieval scores,
reranker scores, token usage, latency per step), and the 42-question golden
set is scored with RAGAS in CI on every change to retrieval/generation code,
gating whether a deploy is even allowed to run.

## 4. Complete tech stack, by category

| Category | Tool/Platform | Why |
|---|---|---|
| Source data | CMS National/Local Coverage Determinations (bulk CSV/XML), Medicare Benefit Policy Manual (PDF) | Real regulatory text, not synthetic — the whole point of the project |
| Backend language/framework | Python, FastAPI | FastAPI for the HTTP layer; async-friendly, typed request/response models via Pydantic |
| Orchestration | LangGraph | Explicit state graph for routing (direct-answer vs. decompose-and-synthesize); chosen partly for portfolio narrative consistency with other projects, partly because it gives per-node tracing for free |
| LLM (generation) | Claude (`claude-sonnet-4-6`), Anthropic API | Documented tech decision; $3/$15 per million input/output tokens |
| Embeddings (production) | `sentence-transformers/all-mpnet-base-v2`, run in-process | Free, no API key, no rate limits, works identically in CI/Cloud Run/local — chosen specifically to eliminate `OPENAI_API_KEY` after weighing the cost/complexity tradeoff |
| Embeddings (local dev alt.) | Ollama + `nomic-embed-text` | Fully free local dev loop before committing to any paid API |
| Sparse retrieval | `rank_bm25` (BM25Okapi) | Exact keyword matching that dense embeddings can underweight (proven necessary — see the Week 3 before/after) |
| Reranking | `sentence-transformers` cross-encoder (`ms-marco-MiniLM-L-6-v2`) | Re-scores each (query, chunk) pair directly instead of relying on one static embedding |
| Vector database | Postgres + `pgvector` extension, hosted on **Supabase** | Real production vector store (HNSW cosine index), replacing the walking skeleton's in-memory numpy array |
| Relational schema | Postgres (`roles`, `users`, `documents`, `query_log`, `chunks`, `embedding_meta`) | Users/roles/RBAC, audit logging, and embedding-provider tracking, on the same database as the vectors |
| Auth | **Clerk** (OAuth), JWT verified server-side via **PyJWT** + JWKS | Real login; backend never trusts a client-supplied identity — verifies an RS256-signed session token against Clerk's public keys |
| Evaluation | **RAGAS** (faithfulness, answer_relevancy, context_precision) | Automated, quantitative answer-quality scoring against a 42-question golden set |
| Observability | **LangSmith** | Full trace tree per request: prompts, retrieval/reranker scores, token usage, latency |
| CI/CD | **GitHub Actions** | Two workflows, path-filtered by cost (see §7) |
| Containerization | **Docker** | Multi-stage-ish build; CPU-only PyTorch pinned explicitly (cut image from 8.8GB to 3.14GB) |
| Container registry | **Google Artifact Registry** | Stores the built image, pulled by Cloud Run |
| Compute | **Google Cloud Run** | Serverless containers, scales to zero, 2Gi memory / 2 vCPU allocated (default 512Mi was insufficient for two ML models in memory) |
| Cloud auth (CI → GCP) | **Workload Identity Federation** | Keyless — GitHub's own OIDC token exchanged for short-lived GCP credentials, no long-lived service-account JSON key stored anywhere |
| Frontend framework | **Next.js 16** (App Router, Turbopack), TypeScript | Documented tech decision; matches another project in the same portfolio |
| Styling | **Tailwind CSS v4** (+ `@tailwindcss/typography`) | Utility-first, CSS-native `@theme`/`@plugin` config (no JS config file) |
| Markdown rendering | `react-markdown` | Claude's answers use markdown formatting (headers, bold, lists) |
| Frontend hosting | **Vercel** | Native Next.js support, free tier, GitHub-integrated deploys |
| Local LLM runtime (dev only) | **Ollama** | Zero-cost local dev loop for the entire pipeline before any real API keys existed |

## 5. What was built, week by week (condensed)

1. **Real data ingestion** — extracted 7 real documents (2 NCDs, 3 LCDs, 2
   Benefit Policy Manual chapter sections) from CMS bulk CSV/PDF downloads
   into clean `.txt`, replacing the original synthetic 3-document walking
   skeleton (kept as a control set in `data/sample_docs/`).
2. **Structure-aware chunking** — replaced fixed-size chunking with one that
   splits on real section/heading boundaries, falling back to
   paragraph→sentence splits only when a section is still too long.
3. **Hybrid retrieval + reranking** — BM25 + dense search combined via
   Reciprocal Rank Fusion, then a cross-encoder reranks the fused
   candidates.
4. **Query decomposition (LangGraph)** — a graph that classifies a question
   as single- or multi-topic, routing to direct-answer or
   decompose-retrieve-synthesize accordingly.
5. **Grounded generation** — inline `[chunk_id]` citations, a deterministic
   phantom-citation check, and an LLM-based faithfulness check as a second
   pass.
6. **Access control** — Postgres-backed RBAC, enforced inside the retrieval
   query itself (a SQL `WHERE doc_id = ANY(...)` for pgvector, a masked
   candidate pool for BM25) rather than filtered from output afterward.
7. **Evaluation** — a 42-question golden set with hand-verified ground-truth
   answers, scored with RAGAS, gating CI.
8. **Observability** — LangSmith tracing on every LLM call and every
   retrieval stage.
9. **Deployment** — pgvector migration, Docker containerization, Cloud Run
   deployment, GitHub Actions CI/CD.
10. **Frontend + real OAuth** — Next.js UI on Vercel, Clerk login replacing
    the CLI-style `as_user` simulation for the public-facing API.

## 6. Key design decisions and the reasoning behind them

**Why hybrid retrieval instead of just dense embeddings?**
Proven necessary, not assumed: after fixing chunking, dense-only search
still missed the single most specific supporting chunk on a real test
question across three separate attempts. Adding BM25 (for exact keyword
matches dense embeddings underweight) + Reciprocal Rank Fusion + a
cross-encoder rerank fixed it, confirmed on the identical question.

**Why enforce RBAC at the retrieval layer instead of the UI?**
Because UI-level filtering doesn't stop a client from calling the API
directly. The `allowed_doc_ids` restriction is a SQL `WHERE` clause for
pgvector and a masked candidate array for BM25 — a restricted document is
never fetched from the database at all, not filtered out of a response
after the fact. Verified with a test that asserts directly on `store.search()`'s
return value, not on rendered UI text.

**Why does an unauthenticated (anonymous) request get "standard" access,
not full access?**
Because once real login exists, "skip the login" cannot be a way to see
*more* — only to see the same as everyone else. This was a real gap the
project caught in itself: before OAuth existed, the CLI/API's default
"no user given" meant "unrestricted," which was fine when there was no real
identity distinction yet. Adding real auth without revisiting that default
would have made anonymous access strictly more privileged than a real
logged-in account, which defeats the purpose of RBAC. Fixed as part of the
OAuth rollout, not left as a known issue.

**Why local `sentence-transformers` embeddings instead of OpenAI's API in
production?**
Cost and portability. It eliminates `OPENAI_API_KEY` entirely, runs
identically in local dev, CI, and Cloud Run (unlike Ollama, which needs a
separately-reachable server), and the model is already a dependency (same
library powers the cross-encoder reranker). The tradeoff is CPU compute at
inference time instead of a network call — irrelevant at this project's
scale.

**Why gate the CI eval on `answer_relevancy` in addition to `faithfulness`?**
Found empirically, not designed in from the start: a deliberate regression
(reverting to dense-only retrieval) barely moved faithfulness but dropped
`answer_relevancy` by ~15% — faithfulness measures whether an answer's
claims trace back to *whatever* was retrieved, not whether the *right*
things were retrieved. A single-metric gate would have let a real
regression through.

**Why Workload Identity Federation instead of a service-account JSON key
for GitHub Actions → GCP auth?**
No long-lived credential to leak or rotate — GitHub's own OIDC identity is
exchanged for a short-lived GCP access token per workflow run.

**Why path-filter the CI workflows instead of running the full eval suite
on every push?**
Real cost control, learned the expensive way: the eval suite makes
~300-500 real Claude API calls per run (42 questions × several pipeline
calls, plus RAGAS's own scoring calls). Early on, unrelated infra commits
(a Dockerfile fix, a GCP auth fix) each re-triggered the full suite. Fixed
by scoping `eval.yml` to only the exact modules `eval/run_eval.py` actually
imports (`chunking`, `embed_store`, `hybrid_store`, `orchestrator` and its
transitive imports), and giving `deploy.yml` its own direct trigger for
serving-layer files (`api.py`, `db.py`, `auth.py`, `query.py`,
`Dockerfile`) that need deploying but can't be validated by an eval run
that never imports them.

**Why Clerk over Auth0?**
Fastest Next.js integration (prebuilt `<SignInButton>`, `<UserButton>`,
`<Show>` components); backend JWT verification is the same generic
JWKS-based approach regardless of provider, since neither has a
first-party polished Python SDK for this.

**Why does a new sign-up default to "clinician" (standard access) with no
self-service admin path?**
Deliberate: promoting an account to `compliance_admin` is a manual
database operation, not an app feature — building a self-service
privilege-escalation UI is exactly the kind of surface RBAC exists to
avoid, and it wasn't asked for.

## 7. Metrics and results

- **Golden set**: 42 hand-verified questions across all 7 real documents,
  plus cross-document and deliberately-unanswerable cases.
- **RAGAS scores, real Claude (production CI run)**: faithfulness **0.838**,
  answer_relevancy **0.932**, context_precision **0.898**.
- **RAGAS scores, local Ollama baseline** (committed in `eval/results.json`,
  used for the dense-vs-hybrid regression demonstration): faithfulness
  0.708, answer_relevancy 0.716, context_precision 0.842 — lower than
  Claude's, consistent with a smaller local model being less reliable.
- **Corpus**: 129 chunks across 7 real CMS/Medicare documents.
- **Docker image**: 3.14GB (down from an initial 8.8GB after pinning
  CPU-only PyTorch — the default install pulled CUDA packages with no GPU
  to use them).
- **Cost per query**: ~$0.015-0.02 for a single-topic question (3 Claude
  calls: decompose, answer, faithfulness-check), roughly double for a
  genuinely multi-part question.
- **Cloud Run config**: 2Gi memory, 2 vCPU, scales to zero.
- **7 real bugs** found and fixed with documented before/after evidence
  (full detail in `BUGS.md`).
- **22 commits** from empty repo to fully deployed, OAuth-enabled
  production app.

## 8. Security / RBAC design specifics

- RBAC boundary is tied to a real distinction in the corpus, not invented:
  DME/oxygen equipment billing is one of CMS's most historically
  fraud-prone benefit categories, so the oxygen LCD is marked `restricted`
  (compliance-admin only); everything else is `standard`.
- Enforcement point: inside `store.search()` itself (SQL `WHERE` for
  pgvector, masked candidate array for BM25) — never a post-hoc filter on
  already-fetched results.
- Identity: real Clerk-issued RS256 JWTs, verified against Clerk's JWKS
  endpoint (no Clerk secret key needed for verification — only the public
  keys). No client-supplied username is ever trusted by the public API.
- New accounts are just-in-time provisioned on first authenticated request,
  defaulting to `clinician` (standard access).
- The original CLI (`query.py --as-user`) keeps its original
  simpler trust model (a local operator can claim any username) since it's
  a trusted local tool, not a public endpoint — the stricter JWT-based
  model applies specifically to the public HTTP API.

## 9. Real bugs found (see `BUGS.md` for full detail)

1. Fixed-size chunking split the exact clause that answered a real test
   question, mid-word.
2. Dense-only retrieval missed the best chunk even after chunking was
   fixed — motivated hybrid retrieval + reranking.
3. Query decomposition over-split a single-topic question, losing shared
   context and producing a worse answer than no decomposition at all.
4. The model cited a real chunk that didn't actually support its specific
   claim (right topic, wrong section) — caught by the faithfulness check.
5. A broken eval run silently reported a passing score — 30/42
   context_precision scores were NaN from timeouts, averaged over the
   survivors with no warning.
6. The CI regression gate didn't catch the regression it was built for —
   faithfulness alone missed a real retrieval-quality drop that
   answer_relevancy caught.
7. A bug (case-sensitive BM25 tokenizer) that was completely invisible in
   the terminal, found only by inspecting a LangSmith trace span the CLI's
   print statements never surface.

Smaller ones: a `UnicodeEncodeError` on Windows console encoding hitting
real CMS text's `≥` symbol; the 8.8GB→3.14GB Docker image fix; an Ollama
base URL hardcoded to `localhost` breaking inside a container; a missing
`DATABASE_URL` in CI silently falling back to `localhost`; Workload
Identity Federation credentials not reaching Docker's push step; Cloud
Run's default 512Mi memory being too small for two ML models in memory at
once; and a Vercel-specific gotcha where a `NEXT_PUBLIC_*` env var marked
as "Secret" type is deliberately excluded from the browser bundle.

## 10. Anticipated interview questions

**"Walk me through what happens when a user asks a question."**
Use the §3 architecture flow. Emphasize: retrieval isn't one lookup, it's
dense + sparse fused and reranked; generation isn't one call, it's
decompose → answer → faithfulness-check; and RBAC is applied inside the
retrieval query, not as a filter afterward.

**"How would you scale this?"**
Cloud Run already scales horizontally (stateless containers, scales to
zero on no traffic). The bottleneck would be Postgres/pgvector at high
query volume — Supabase's free tier is not built for production load;
next step would be a dedicated pgvector instance or a managed vector DB
(e.g., a larger Supabase tier, or Qdrant). The cross-encoder rerank step
is the most CPU-expensive part per request; batching or a smaller reranker
model would be the first optimization.

**"What would you do differently / what's not done yet?"**
No self-service admin role promotion (manual DB operation by design, but a
real product would need an admin UI or invite flow). No rate limiting on
the public API. The eval golden set is 42 questions — a real production
system would want hundreds, ideally sourced from real user queries over
time. CORS is currently a single hardcoded origin — a real multi-environment
setup would need per-environment origin config.

**"How do you know the RAG pipeline actually works, not just that it
compiles?"**
Point to §7's real RAGAS scores against real Claude, and to `BUGS.md`'s
before/after pairs — several of which involved retrieving the exact same
question before and after a fix and showing the output change.

**"Why not just use one big embedding model and call it done?"**
Point to bug #2 — proven, not assumed, that dense-only retrieval misses
real answers even with clean chunking, which is why hybrid+rerank exists.

**"How do you prevent the LLM from hallucinating?"**
Three layers: (1) the system prompt instructs it to say so explicitly when
context is insufficient, verified against real absence-of-evidence
questions ("is heart bypass surgery covered?" → honest refusal); (2)
inline citations are checked against what was actually retrieved
(phantom-citation detection); (3) a second LLM pass checks whether each
citation's claim is actually supported by that excerpt's text (with the
explicit caveat that this checker is itself an LLM and can be wrong —
demonstrated with two real spot-checked false positives).

**"What's the actual cost to run this?"**
See §7 — roughly 1.5-2 cents per single-topic question in Claude API
costs; embeddings are free (local model); the CI eval suite is the biggest
recurring cost driver (~300-500 Claude calls per run), which is why it's
path-filtered to only run when retrieval/generation code actually changes.
