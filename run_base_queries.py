#!/usr/bin/env python3
"""Run the eight verbatim Session 7 base-query demonstrations."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import shutil
import sys
from pathlib import Path

import agent7
import memory

ROOT = Path(__file__).parent
SANDBOX = ROOT / "sandbox"

QUERY_A = (
    "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his "
    "birth date, death date, and three key contributions to information theory."
)
QUERY_B = (
    "Find 3 family-friendly things to do in Tokyo this weekend. "
    "Check Saturday's weather forecast there and tell me which one is most appropriate."
)
QUERY_C1 = (
    "My mom's birthday is 15 May 2026. Remember that and create reminders "
    "for two weeks before and on the day."
)
QUERY_C2 = "When is mom's birthday?"
QUERY_D = (
    'Search for "Python asyncio best practices", read the top 3 results, '
    "and give me a short numbered list of the advice they agree on."
)
QUERY_E = (
    "Index the file papers/attention.md and tell me what the three key "
    "contributions of the Transformer architecture are according to this paper."
)
QUERY_F1 = (
    "Index every .md file under papers/. Confirm how many chunks were indexed in total."
)
QUERY_F2 = (
    "Across the papers I have indexed, what do they say about "
    "chain-of-thought reasoning?"
)
QUERY_G = "Across these papers, how do they handle the credit assignment problem?"
QUERY_H = (
    "Compare how the ReAct paper and the Chain-of-Thought paper differ "
    "in their treatment of intermediate reasoning."
)

QUERIES = {
    "A": QUERY_A,
    "B": QUERY_B,
    "C1": QUERY_C1,
    "C2": QUERY_C2,
    "D": QUERY_D,
    "E": QUERY_E,
    "F1": QUERY_F1,
    "F2": QUERY_F2,
    "G": QUERY_G,
    "H": QUERY_H,
}

ITERATION_BOUNDS = {
    "A": 3,
    "B": 8,
    "C1": 4,
    "C2": 3,
    "D": 6,
    "E": 5,
    "F1": 11,
    "F2": 3,
    "G": 4,
    "H": 3,
}


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def prepare_sandbox(*, clean: bool) -> None:
    if clean:
        memory.clear()
        if SANDBOX.exists():
            shutil.rmtree(SANDBOX)
    (SANDBOX / "papers").mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "papers").glob("*.md")):
        if source.name.lower() == "readme.md":
            continue
        shutil.copy2(source, SANDBOX / "papers" / source.name)


async def _run_one(label: str, *, clean: bool) -> None:
    prepare_sandbox(clean=clean)
    print(f"\n{'#' * 78}\nQUERY {label} — bound {ITERATION_BOUNDS[label]} iterations\n{'#' * 78}")
    answer = await agent7.run(
        QUERIES[label], max_iterations=ITERATION_BOUNDS[label]
    )
    if not answer.strip():
        raise RuntimeError(f"Query {label} completed without a final answer.")


async def _run_selection(selection: str) -> None:
    if selection == "ALL":
        for label in ("A", "B"):
            await _run_one(label, clean=True)
        await _run_one("C1", clean=True)
        await _run_one("C2", clean=False)
        await _run_one("D", clean=True)
        await _run_one("E", clean=True)
        await _run_one("F1", clean=True)
        for label in ("F2", "G", "H"):
            await _run_one(label, clean=False)
        return
    clean = selection not in ("C2", "F2", "G", "H")
    await _run_one(selection, clean=clean)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", choices=["all", *[key.lower() for key in QUERIES]])
    parser.add_argument("--trace", type=Path, help="Also save terminal trace to this path.")
    args = parser.parse_args()
    selection = args.query.upper()
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        with args.trace.open("w", encoding="utf-8") as trace:
            with contextlib.redirect_stdout(_Tee(sys.stdout, trace)):
                asyncio.run(_run_selection(selection))
    else:
        asyncio.run(_run_selection(selection))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
