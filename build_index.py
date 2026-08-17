#!/usr/bin/env python3
"""Sync papers/corpus into sandbox and index all documents into FAISS.

Embeddings go through the LLM gateway (Ollama nomic-embed-text, 768-d).

Usage:
  # one-time: pull the embedding model
  ollama pull nomic-embed-text

  # terminal 1
  uv run python llm_gatewayV7/main.py

  # index everything under papers/ and corpus/
  uv run python build_index.py

  # wipe FAISS + chunk facts, then rebuild
  uv run python build_index.py --reset

  # only professor papers or only extra corpus
  uv run python build_index.py --papers
  uv run python build_index.py --corpus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

import memory
import rag
import vector_store
import llm


def _collect_index_paths(*, papers: bool, corpus: bool) -> list[str]:
    rag.SANDBOX.mkdir(parents=True, exist_ok=True)
    synced = rag.sync_repo_docs_to_sandbox()
    if synced:
        print(f"[sync] copied {len(synced)} file(s) into sandbox/")
    else:
        print("[sync] no files under papers/ or corpus/ yet")

    paths: list[str] = []
    if papers:
        base = rag.SANDBOX / "papers"
        if base.is_dir():
            paths.extend(
                f"papers/{p.relative_to(base).as_posix()}"
                for p in sorted(base.rglob("*"))
                if p.is_file() and not p.name.startswith(".") and p.name.lower() != "readme.md"
            )
    if corpus:
        base = rag.SANDBOX / "corpus"
        if base.is_dir():
            paths.extend(
                f"corpus/{p.relative_to(base).as_posix()}"
                for p in sorted(base.rglob("*"))
                if p.is_file() and not p.name.startswith(".") and p.name.lower() != "readme.md"
            )
    return paths


def main() -> int:
    p = argparse.ArgumentParser(description="Index papers/corpus into FAISS (nomic 768-d)")
    p.add_argument("--reset", action="store_true", help="Clear chunk facts + FAISS before indexing")
    p.add_argument("--papers", action="store_true", help="Index sandbox/papers/ only")
    p.add_argument("--corpus", action="store_true", help="Index sandbox/corpus/ only")
    args = p.parse_args()

    papers = args.papers or (not args.papers and not args.corpus)
    corpus = args.corpus or (not args.papers and not args.corpus)

    llm.ensure_gateway()

    if args.reset:
        n = memory.clear_chunks()
        print(f"[reset] removed {n} chunk fact(s); FAISS cleared")

    paths = _collect_index_paths(papers=papers, corpus=corpus)
    if not paths:
        print("No files to index. Add .md files under papers/ or corpus/ first.")
        return 1

    total_chunks = 0
    for rel in paths:
        try:
            out = memory.index_document(rel)
        except FileNotFoundError:
            print(f"[skip] missing: {rel}")
            continue
        n = int(out.get("chunks_indexed", 0))
        total_chunks += n
        print(f"[index] {rel} → {n} chunk(s)")

    print(f"\nDone: {len(paths)} file(s), {total_chunks} chunk(s), {memory.chunk_count()} in memory")
    print(f"FAISS vectors on disk: {vector_store.count_vectors()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
