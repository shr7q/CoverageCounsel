"""
Hybrid retrieval: BM25 sparse search + dense embedding search, combined via
Reciprocal Rank Fusion (RRF), then narrowed to the final top_k with a
cross-encoder reranking pass.

Week 3 addition -- Week 2 fixed chunking, but even with clean, intact chunks
the dense-only top-4 search still missed the single most specific supporting
chunk on a real test question (the knee-replacement chunk that explicitly
lists "supervised physical therapy" never made the cut). BM25 catches exact
keyword overlap that a single embedding's cosine similarity can underweight;
the cross-encoder then re-scores each (query, chunk) pair directly instead of
relying on one pre-computed vector per chunk.

Week 8: BM25 scoring and reranking are pulled into their own @traceable
methods (dense search already traces itself, see embed_store.py) so a
LangSmith trace shows each retrieval stage's own scores as a distinct
nested span, not just the final fused-and-reranked output.

Week 9: the dense backend is swappable (VECTOR_STORE=memory|pgvector) --
BM25 stays in-memory regardless, since migrating it to Postgres full-text
search is a separate, bigger change than "give the dense vectors a real
database," which is what this week's plan item actually asked for.
"""

import os
import re

from langsmith import traceable
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from chunking import Chunk
from embed_store import InMemoryStore

RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RRF_K = 60  # standard RRF damping constant
# pgvector is the default -- Postgres has been a hard dependency since Week 6's
# RBAC anyway. VECTOR_STORE=memory reproduces the original walking-skeleton
# store for comparison, or for running without Docker up.
VECTOR_STORE = os.environ.get("VECTOR_STORE", "pgvector")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridStore:
    def __init__(self):
        if VECTOR_STORE == "pgvector":
            from pgvector_store import PgVectorStore

            self.dense = PgVectorStore()
        else:
            self.dense = InMemoryStore()
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self._reranker: CrossEncoder | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.dense.index(chunks)
        self.bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    @property
    def reranker(self) -> CrossEncoder:
        # loaded lazily so indexing doesn't pay the model-load cost if
        # search() never gets called (e.g. dense-only comparison runs)
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    @traceable(name="bm25_search")
    def _bm25_search(self, query: str, candidate_k: int, allowed_doc_ids: set[str] | None) -> list[str]:
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        if allowed_doc_ids is not None:
            for i, chunk in enumerate(self.chunks):
                if chunk.doc_id not in allowed_doc_ids:
                    bm25_scores[i] = -1
        order = sorted(range(len(self.chunks)), key=lambda i: -bm25_scores[i])[:candidate_k]
        return [self.chunks[i].chunk_id for i in order if bm25_scores[i] != -1]

    @traceable(name="cross_encoder_rerank")
    def _rerank(self, query: str, candidates: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
        pairs = [(query, c.text) for c in candidates]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda cs: -cs[1])
        return [(c, float(s)) for c, s in reranked[:top_k]]

    @traceable(name="hybrid_search")
    def search(
        self,
        query: str,
        top_k: int = 4,
        candidate_k: int = 20,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """allowed_doc_ids, when given, is enforced in both the dense and
        BM25 candidate pools before fusion/reranking -- a disallowed chunk
        never reaches the fused candidate set, let alone the reranker or the
        LLM's context, regardless of how well it would otherwise score."""
        if self.bm25 is None:
            raise RuntimeError("Call .index() before .search()")

        dense_ranked_ids = [
            c.chunk_id for c, _ in self.dense.search(query, top_k=candidate_k, allowed_doc_ids=allowed_doc_ids)
        ]
        bm25_ranked_ids = self._bm25_search(query, candidate_k, allowed_doc_ids)

        fused_scores: dict[str, float] = {}
        for ranked_ids in (dense_ranked_ids, bm25_ranked_ids):
            for rank, chunk_id in enumerate(ranked_ids, start=1):
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        by_id = {c.chunk_id: c for c in self.chunks}
        top_fused = sorted(fused_scores.items(), key=lambda kv: -kv[1])[:candidate_k]
        candidate_chunks = [by_id[chunk_id] for chunk_id, _ in top_fused]

        return self._rerank(query, candidate_chunks, top_k)
