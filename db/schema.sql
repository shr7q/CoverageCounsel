-- Week 6: access control schema.
--
-- RBAC boundary chosen to map onto a real distinction in the corpus rather
-- than an invented one: DME/oxygen equipment billing is one of CMS's most
-- historically fraud-prone benefit categories, so its detailed coverage and
-- coding criteria (lcd_33797_oxygen_and_oxygen_equipment.txt) are marked
-- 'restricted' -- visible only to the compliance_admin role. Everything else
-- is 'standard', visible to every role.

CREATE TABLE roles (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    access_levels TEXT[] NOT NULL
);

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    role_id INTEGER NOT NULL REFERENCES roles(id)
);

CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,  -- matches the .txt filename used as Chunk.doc_id
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    access_level TEXT NOT NULL DEFAULT 'standard'
);

CREATE TABLE query_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    question TEXT NOT NULL,
    retrieved_doc_ids TEXT[] NOT NULL,
    queried_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Week 9: real vector store, replacing the in-memory numpy array from the
-- walking skeleton. 768 dims matches nomic-embed-text (local Ollama dev);
-- re-embedding + a new column is required if the embedding model ever
-- changes dimension (e.g. OpenAI text-embedding-3-small is 1536) -- that's
-- an accepted migration cost, not something worth engineering around now.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    text TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL
);

CREATE INDEX chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops);
