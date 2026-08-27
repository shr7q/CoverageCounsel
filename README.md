# Healthcare compliance RAG — walking skeleton

This is the thinnest possible version of the full project: chunk documents,
embed them, retrieve on a question, generate a grounded-ish answer. No
reranking, no query decomposition, no auth, no citations parsing, no eval
suite yet. The point is to get something real running today, then upgrade
piece by piece once you see where this naive version breaks.

Three synthetic sample policies are included in `data/sample_docs/` so you
can run this immediately without needing the real CMS corpus yet. They're
deliberately written to contain the same problems real regulatory text has:
a cross-reference to a document that doesn't exist in the sample set
(bariatric_surgery.txt references "MED-NEC-GEN-01, not included in this
sample set"), regional variation language, and an effective-dates section
that doesn't give you an actual date — try asking questions that probe these
and see how the naive skeleton handles them.

## Setup

```bash
cd healthcare-rag-skeleton
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your real OPENAI_API_KEY and ANTHROPIC_API_KEY
```

## Run it

```bash
cd src
python query.py "Is a total knee replacement covered without physical therapy first?"
python query.py "What BMI is required for bariatric surgery coverage?"
python query.py "Is bariatric surgery covered for someone who missed a weight management appointment last year?"
python query.py "Is heart bypass surgery covered?"
```

The last question isn't covered by any sample document — watch whether the
model admits that or makes something up. That's your first real test of the
grounding problem.

## What to try breaking

1. Ask a question that needs info from two documents at once (nothing in the
   sample set needs this yet — you'll feel the gap once you add more docs).
2. Ask the bariatric surgery non-adherence exception question and see if the
   model notices the cross-reference to a document it doesn't have.
3. Ask about "the current effective date" and see if it hallucinates one.

## Next steps, in order

1. **Swap in real data.** Download a slice of CMS National Coverage
   Determinations (bulk XML from cms.gov) and a couple of Medicare Benefit
   Policy Manual chapters. Replace `data/sample_docs/` with real `.txt`
   extractions. This alone will surface most of your first real bugs.
2. **Structure-aware chunking.** Replace the fixed-size chunker in
   `chunking.py` with one that splits on `Section N.` headers first.
3. **Hybrid retrieval.** Add BM25 alongside the dense search in
   `embed_store.py` and combine scores.
4. **Reranking.** Add a cross-encoder rerank step after retrieval, before
   the results get passed to `answer_question`.
5. **Query decomposition + LangGraph orchestration.** Wrap this whole flow
   in a LangGraph agent that can route between direct lookup and
   decomposition for multi-part questions.
6. **Citations + faithfulness check.** Make the model cite specific chunk
   IDs inline, then add a second pass that checks the answer's claims are
   actually traceable to the cited chunks.
7. **Auth + RBAC**, then **eval suite (RAGAS)**, then **observability
   (Langfuse/LangSmith)**, then **Docker + deployment**. Each of these can
   be layered on independently once the core loop is solid.

Don't do these in a rush. Get real signal from step 1 (real data breaking
your naive chunker) before building step 2, and so on down the list.
