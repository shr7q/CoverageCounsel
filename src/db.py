"""
Week 6: users/roles/document-metadata lookups against Postgres, backing the
retrieval-layer RBAC enforcement in hybrid_store.py / embed_store.py.

Week 10: real Clerk accounts (get_or_create_user_by_clerk_id) sit alongside
the original --as-user CLI flag (get_user, by username) rather than
replacing it -- the CLI is a trusted local operator tool, so a
client-supplied username is fine there; the public API now requires a
verified Clerk JWT (see auth.py) precisely because it doesn't have that
trust boundary. The RBAC enforcement itself (filtering at the retrieval
layer) is unchanged either way -- only how "who is asking" gets resolved.
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


def get_or_create_user_by_clerk_id(clerk_user_id: str) -> dict:
    """Just-in-time provisioning: a real Clerk account's first authenticated
    request creates its users row here, defaulting to the 'clinician'
    (standard-access) role. There's no sign-up-time admin path -- promoting
    an account to compliance_admin is a manual DB update, not something the
    app exposes, since building that UI wasn't asked for and a wrong click
    there is exactly the kind of mistake RBAC exists to prevent."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT u.id, u.username, r.name, r.access_levels
            FROM users u JOIN roles r ON r.id = u.role_id
            WHERE u.clerk_user_id = %s
            """,
            (clerk_user_id,),
        )
        row = cur.fetchone()
        if row is None:
            username = f"clerk:{clerk_user_id}"
            cur.execute(
                """
                INSERT INTO users (username, role_id, clerk_user_id)
                VALUES (%s, (SELECT id FROM roles WHERE name = 'clinician'), %s)
                RETURNING id
                """,
                (username, clerk_user_id),
            )
            user_id = cur.fetchone()[0]
            conn.commit()
            return {"id": user_id, "username": username, "role": "clinician", "access_levels": {"standard"}}

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
