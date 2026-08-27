"""
Week 6: users/roles/document-metadata lookups against Postgres, backing the
retrieval-layer RBAC enforcement in hybrid_store.py / embed_store.py.

There's no real OAuth login yet -- that needs a frontend to redirect through,
which doesn't exist until Week 9-10. Until then, "who is asking" is simulated
via the --as-user CLI flag in query.py, resolved to a real row in the users
table below. The RBAC enforcement itself (filtering at the retrieval layer)
is real and doesn't change when OAuth is wired in later -- only how the
username gets supplied does.
"""

import os

import psycopg2

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://rag:rag_local_dev@localhost:5432/healthcare_rag"
)


def _connect():
    return psycopg2.connect(DATABASE_URL)


def get_user(username: str) -> dict:
    """Returns {"id", "username", "role", "access_levels"} or raises
    ValueError if the username doesn't exist."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT u.id, u.username, r.name, r.access_levels
            FROM users u JOIN roles r ON r.id = u.role_id
            WHERE u.username = %s
            """,
            (username,),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"no such user: {username!r}")
    user_id, uname, role, access_levels = row
    return {"id": user_id, "username": uname, "role": role, "access_levels": set(access_levels)}


def get_allowed_doc_ids(access_levels: set[str]) -> set[str]:
    """Every document whose access_level is in the given set."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT doc_id FROM documents WHERE access_level = ANY(%s)",
            (list(access_levels),),
        )
        return {row[0] for row in cur.fetchall()}


def log_query(user_id: int, question: str, retrieved_doc_ids: list[str]) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO query_log (user_id, question, retrieved_doc_ids) VALUES (%s, %s, %s)",
            (user_id, question, retrieved_doc_ids),
        )
        conn.commit()
