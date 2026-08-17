"""RAG helpers: chunk sandbox documents and build memory fact items.

Chunking defaults from Session 7 worked traces:
  400 words per window, 80-word overlap (~11 chunks for attention.md).

index_document is invoked from Action after MCP dispatch; it writes chunk facts
and upserts vectors through memory.index_chunks().
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

ROOT = Path(__file__).parent
SANDBOX = ROOT / "sandbox"

CHUNK_WORDS = 400
CHUNK_OVERLAP = 80


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def chunk_text(text: str, *, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = _words(text)
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + chunk_words])
        if piece:
            chunks.append(piece)
        if start + chunk_words >= len(words):
            break
    return chunks


def resolve_sandbox_path(rel_path: str) -> Path:
    rel = rel_path.strip().lstrip("/")
    full = (SANDBOX / rel).resolve()
    if not str(full).startswith(str(SANDBOX.resolve())):
        raise ValueError(f"path escapes sandbox: {rel_path}")
    return full


def extract_indexable_text(text: str) -> str:
    """Keep title + abstract; drop arXiv nav, submission history, and related lists."""
    start = re.search(r"#\s+Title:", text)
    if not start:
        start = re.search(r">\s*Abstract:", text)
    body = text[start.start() :] if start else text
    end = re.search(r"\n## Submission history|\n## Access Paper", body)
    if end:
        body = body[: end.start()]
    return body.strip()


def read_document(rel_path: str) -> str:
    path = resolve_sandbox_path(rel_path)
    if not path.is_file():
        raise FileNotFoundError(rel_path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    return extract_indexable_text(raw)


def chunk_descriptor(source_path: str, index: int, total: int) -> str:
    return f"[sandbox:{source_path} chunk {index + 1}/{total}]"


def sync_repo_docs_to_sandbox() -> list[str]:
    """Copy repo papers/ and corpus/ into sandbox/ for MCP indexing."""
    import shutil

    synced: list[str] = []
    for name in ("papers", "corpus"):
        src = ROOT / name
        if not src.is_dir():
            continue
        dst = SANDBOX / name
        dst.mkdir(parents=True, exist_ok=True)
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            rel = path.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            synced.append(f"{name}/{rel.as_posix()}")
    return synced


def new_chunk_id() -> str:
    return "mem:" + uuid.uuid4().hex[:10]

