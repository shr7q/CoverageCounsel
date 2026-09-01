"""
Week 9: pgvector-backed dense store, replacing the in-memory numpy array
from the walking skeleton (embed_store.InMemoryStore) now that there's a
real database to put it in anyway (Week 6 brought Postgres in for RBAC).

Same interface as InMemoryStore -- .index(chunks) / .search(query, top_k,
allowed_doc_ids) -- so HybridStore can use either as its dense backend
without changing anything else. Embedding itself is delegated to an
InMemoryStore instance rather than duplicated, since the OpenAI/Ollama
provider-switching logic already lives there.
"""

import os

import psycopg2
from langsmith import traceable
from pgvector.psycopg2 import register_vector

from chunking import Chunk
from embed_store import InMemoryStore

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://rag:rag_local_dev@localhost:5432/healthcare_rag"
)


class PgVectorStore:
    def __init__(self):
        self._embedder = InMemoryStore()
        self.conn = psycopg2.connect(DATABASE_URL)
        register_vector(self.conn)

    def index(self, chunks: list[Chunk], batch_size: int = 50) -> None:
        """Skips re-embedding entirely if the DB already holds exactly this
        set of chunk_ids. With a real embeddings API (not free local Ollama)
        behind this, re-embedding the whole corpus on every Cloud Run cold
        start -- which happens often once it scales to zero on low traffic
        -- would mean paying OpenAI again for work already done and sitting
        untouched in Supabase. Doesn't detect a chunk whose *text* changed
        while its id stayed the same; fine for a corpus that only changes
        via a new deploy, not a substitute for real content-hash tracking."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id FROM chunks")
            existing_ids = {row[0] for row in cur.fetchall()}
        if existing_ids == {c.chunk_id for c in chunks}:
            return

        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE chunks")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vecs = self._embedder._embed([c.text for c in batch])
            with self.conn.cursor() as cur:
                for chunk, vec in zip(batch, vecs):
                    cur.execute(
                        "INSERT INTO chunks (chunk_id, doc_id, text, embedding) VALUES (%s, %s, %s, %s)",
                        (chunk.chunk_id, chunk.doc_id, chunk.text, vec),
                    )
        self.conn.commit()

    @traceable(name="pgvector_search")
    def search(
        self, query: str, top_k: int = 4, allowed_doc_ids: set[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        """allowed_doc_ids, when given, is a SQL WHERE clause -- the
        restricted rows are never fetched from the database at all, the
        strongest form of "enforced at the retrieval layer" there is."""
        q_vec = self._embedder._embed([query])[0]

        with self.conn.cursor() as cur:
            if allowed_doc_ids is not None:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, text, 1 - (embedding <=> %s) AS score
                    FROM chunks
                    WHERE doc_id = ANY(%s)
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (q_vec, list(allowed_doc_ids), q_vec, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, text, 1 - (embedding <=> %s) AS score
                    FROM chunks
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (q_vec, q_vec, top_k),
                )
            rows = cur.fetchall()

        return [(Chunk(doc_id=doc_id, chunk_id=chunk_id, text=text), float(score)) for chunk_id, doc_id, text, score in rows]
