"""
Minimal in-memory vector store.

No pgvector, no Qdrant -- just a numpy array and cosine similarity. This is
intentional: the point of the walking skeleton is to prove the retrieval loop
works before you spend time standing up real infrastructure. Swap this for
pgvector once you've moved to the access-control phase and need a real
database anyway.
"""

import os
import numpy as np
from langsmith import traceable
from openai import OpenAI
from chunking import Chunk

# EMBED_PROVIDER=ollama runs fully local against Ollama's OpenAI-compatible
# endpoint (no API key needed). EMBED_PROVIDER=sentence_transformers runs a
# local model in-process (also free, no API key, and works the same way in
# CI/Cloud Run as it does on a laptop -- unlike ollama, which needs a
# separate server reachable over the network). Set to "openai" for the real
# text-embedding-3-small endpoint.
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "openai")
# "localhost" means the container itself when running in Docker -- override
# to http://host.docker.internal:11434/v1 to reach an Ollama instance
# running on the host machine from inside a container.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

if EMBED_PROVIDER == "ollama":
    EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
elif EMBED_PROVIDER == "sentence_transformers":
    # 768-dim, matching the pgvector schema's VECTOR(768) column -- picking
    # a different model here means a schema migration, not just an env
    # var flip.
    EMBED_MODEL = os.environ.get("SENTENCE_TRANSFORMERS_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
else:
    EMBED_MODEL = "text-embedding-3-small"


class InMemoryStore:
    def __init__(self, api_key: str | None = None):
        self._st_model = None
        if EMBED_PROVIDER == "ollama":
            self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        elif EMBED_PROVIDER == "sentence_transformers":
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        if EMBED_PROVIDER == "sentence_transformers":
            if self._st_model is None:
                # loaded lazily so constructing a store doesn't pay the
                # model-load cost until embedding is actually needed
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(EMBED_MODEL)
            return np.asarray(self._st_model.encode(texts))
        response = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        return np.array([item.embedding for item in response.data])

    def index(self, chunks: list[Chunk], batch_size: int = 50) -> None:
        self.chunks = chunks
        all_vecs = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vecs = self._embed([c.text for c in batch])
            all_vecs.append(vecs)
        self.vectors = np.vstack(all_vecs)
        # normalize once so search is a plain dot product
        self.vectors = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)

    @traceable(name="dense_search")
    def search(
        self, query: str, top_k: int = 4, allowed_doc_ids: set[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        """allowed_doc_ids, when given, restricts the candidate pool to those
        doc_ids *before* ranking -- a chunk outside it can never occupy a
        top_k slot, at the retrieval layer itself rather than being filtered
        out of an already-decided top_k afterward."""
        if self.vectors is None:
            raise RuntimeError("Call .index() before .search()")
        q_vec = self._embed([query])[0]
        q_vec = q_vec / np.linalg.norm(q_vec)
        scores = self.vectors @ q_vec
        if allowed_doc_ids is not None:
            for i, chunk in enumerate(self.chunks):
                if chunk.doc_id not in allowed_doc_ids:
                    scores[i] = -np.inf
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx if scores[i] != -np.inf]
