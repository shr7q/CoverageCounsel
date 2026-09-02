"""
Structure-aware document chunker.

Loads the .txt policy documents in a data directory and splits each one into
retrieval-sized chunks, splitting on real section/heading boundaries instead
of a fixed character count -- a fixed-size split can cut a chunk mid-word,
mid-clause, straight through the one sentence that answers a question, which
knocks it out of the top-k retrieved results entirely.

Heading detection recognizes: the synthetic sample docs' "Section N. Title"
headers, the Benefit Policy Manual's "Section N.N - Title" headers, the field
labels this project's own CMS extraction script writes (e.g. "Indications
and Limitations of Coverage"), and the lettered/numbered subheadings CMS
regulatory text itself uses ("A. Durability", "1. Content:"). A section is
only split further if it's still too long after that, and even then the
split happens on paragraph, then sentence boundaries -- never mid-word.
"""

import os
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str


def load_documents(data_dir: str) -> dict[str, str]:
    """Load all .txt files in data_dir. Returns {filename: full_text}."""
    docs = {}
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".txt"):
            path = os.path.join(data_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                docs[fname] = f.read()
    return docs


KNOWN_FIELD_LABELS = {
    "Indications and Limitations of Coverage",
    "Item/Service Description",
    "Cross-References",
    "Other",
    "Documentation Requirements",
    "ICD-10 Codes That Support Medical Necessity",
    "Coding Guidelines",
}

HEADING_PATTERNS = [
    re.compile(r"^Section \d+(\.\d+)?\s*[-.]\s*\S.*$"),  # "Section 1. Coverage Criteria" / "Section 10.2 - ..."
    re.compile(r"^(?:[A-Z]|\d{1,2})\.\s+[A-Z][^.;:]{2,70}$"),  # "A. Durability" / "1. Necessity for the Equipment"
]


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped in KNOWN_FIELD_LABELS:
        return True
    return any(p.match(stripped) for p in HEADING_PATTERNS)


def split_into_sections(text: str) -> list[str]:
    """Split on heading lines. Each section includes its own heading line."""
    lines = text.split("\n")
    sections = []
    current: list[str] = []
    for line in lines:
        if _is_heading(line) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [s for s in sections if s]


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•])")


def _split_long_section(section: str, max_size: int) -> list[str]:
    """Sub-chunk a too-long section on paragraph, then sentence boundaries."""
    if len(section) <= max_size:
        return [section]

    pieces = [p.strip() for p in section.split("\n\n") if p.strip()]
    if len(pieces) == 1:
        pieces = [s.strip() for s in _SENTENCE_BOUNDARY.split(section) if s.strip()]

    chunks = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > max_size:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)

    # A single paragraph/sentence longer than max_size is kept whole rather
    # than cut mid-word -- an oversized chunk beats a broken one.
    return chunks


def chunk_text(doc_id: str, text: str, max_size: int = 800) -> list[Chunk]:
    """Split text into structure-aware chunks, never mid-word/mid-sentence."""
    sections = split_into_sections(text)
    chunks = []
    idx = 0
    for section in sections:
        for piece in _split_long_section(section, max_size):
            chunks.append(Chunk(doc_id=doc_id, chunk_id=f"{doc_id}::{idx}", text=piece))
            idx += 1
    return chunks


def chunk_all_documents(data_dir: str, max_size: int = 800) -> list[Chunk]:
    docs = load_documents(data_dir)
    all_chunks = []
    for doc_id, text in docs.items():
        all_chunks.extend(chunk_text(doc_id, text, max_size))
    return all_chunks


if __name__ == "__main__":
    # Sanity check: run `python src/chunking.py` from the repo root.
    chunks = chunk_all_documents("data/sample_docs")
    print(f"Loaded {len(chunks)} chunks from data/sample_docs")
    for c in chunks[:3]:
        print(f"\n--- {c.chunk_id} ---\n{c.text[:200]}...")
