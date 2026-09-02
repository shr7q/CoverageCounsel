"""
Permission-leakage test: proves RBAC is enforced at the retrieval layer,
not just in whatever the LLM chooses to say.

This does NOT check the LLM's final answer text -- a restricted user could
still leak data if the model happened not to mention it while the forbidden
chunk sat in its context window. Instead it asserts directly on
store.search()'s *return value*: the restricted document must never appear
in the candidate set at all, regardless of how well it would otherwise score
against the question.

lcd_33797_oxygen_and_oxygen_equipment.txt is the one 'restricted' document
(see db/seed.sql) -- restricted to compliance_admin. alice_clinician
(standard-only) asks a question that scores that exact document highest
under no restriction, to make this a real test rather than a question that
would never have hit it anyway.

Run from the repo root: python scripts/test_permission_leakage.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

import db
from chunking import chunk_all_documents
from hybrid_store import HybridStore

RESTRICTED_DOC_ID = "lcd_33797_oxygen_and_oxygen_equipment.txt"
QUESTION = "What are the coverage criteria for oxygen and oxygen equipment?"


def main():
    print("Indexing data/real_docs...")
    chunks = chunk_all_documents(os.path.join(os.path.dirname(__file__), "..", "data", "real_docs"))
    store = HybridStore()
    store.index(chunks)

    print("\n--- Baseline: unrestricted search (no allowed_doc_ids) ---")
    unrestricted = store.search(QUESTION, top_k=4)
    for chunk, score in unrestricted:
        print(f"  [{score:.3f}] {chunk.chunk_id}")
    hit_restricted = any(c.doc_id == RESTRICTED_DOC_ID for c, _ in unrestricted)
    assert hit_restricted, (
        "Test setup problem: the restricted document didn't even show up "
        "unrestricted, so this question doesn't actually exercise the "
        "permission check. Pick a different question."
    )
    print(f"  (confirmed: {RESTRICTED_DOC_ID} is in the unrestricted top-4, as expected)")

    print(f"\n--- alice_clinician (standard-only) searches the same question ---")
    alice = db.get_user("alice_clinician")
    allowed = db.get_allowed_doc_ids(alice["access_levels"])
    restricted_results = store.search(QUESTION, top_k=4, allowed_doc_ids=allowed)
    for chunk, score in restricted_results:
        print(f"  [{score:.3f}] {chunk.chunk_id}")

    leaked = [c for c, _ in restricted_results if c.doc_id == RESTRICTED_DOC_ID]
    assert not leaked, f"PERMISSION LEAK: alice_clinician's search() returned {len(leaked)} chunk(s) from a restricted document"
    print("  PASS: no chunk from the restricted document was returned to alice_clinician")

    print(f"\n--- bob_admin (compliance_admin) searches the same question ---")
    bob = db.get_user("bob_admin")
    allowed = db.get_allowed_doc_ids(bob["access_levels"])
    admin_results = store.search(QUESTION, top_k=4, allowed_doc_ids=allowed)
    for chunk, score in admin_results:
        print(f"  [{score:.3f}] {chunk.chunk_id}")

    admin_saw_it = any(c.doc_id == RESTRICTED_DOC_ID for c, _ in admin_results)
    assert admin_saw_it, "bob_admin should still be able to retrieve the restricted document"
    print("  PASS: bob_admin (compliance_admin) can still retrieve it")

    print("\nAll permission-leakage checks passed.")


if __name__ == "__main__":
    main()
