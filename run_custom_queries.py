#!/usr/bin/env python3
"""Run custom F1 queries with and without index to prove RAG dependency."""

from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

from custom_queries import CUSTOM_QUERIES  # noqa: E402
import agent7  # noqa: E402
import memory  # noqa: E402


async def _run_all(*, with_index: bool) -> None:
    if with_index:
        print("=== WITH INDEX (state preserved) ===\n")
    else:
        print("=== WITHOUT INDEX (state cleared) ===\n")
        agent7.clean_state()
    for item in CUSTOM_QUERIES:
        qid = item["id"]
        query = str(item["query"])
        print(f"--- {qid}: {query[:70]}...")
        answer = await agent7.run(query)
        print(f"ANSWER: {answer[:400]}\n")
        await asyncio.sleep(1.5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--with-index", action="store_true", help="Run after build_index (keep state/)")
    p.add_argument("--without-index", action="store_true", help="Wipe state/ first")
    p.add_argument("--compare", action="store_true", help="Run without then with index")
    args = p.parse_args()
    if args.compare:
        asyncio.run(_run_all(with_index=False))
        # rebuild index for second pass
        import sys

        import build_index

        old_argv = sys.argv
        sys.argv = ["build_index.py", "--reset"]
        try:
            build_index.main()
        finally:
            sys.argv = old_argv
        asyncio.run(_run_all(with_index=True))
        return
    if args.without_index:
        asyncio.run(_run_all(with_index=False))
    else:
        asyncio.run(_run_all(with_index=True))


if __name__ == "__main__":
    main()
