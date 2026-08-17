#!/usr/bin/env python3
"""Run the five custom Formula One RAG queries with grounded evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from custom_queries import CUSTOM_QUERIES
from gateway import CHAT_PROVIDER, LLM, ensure_gateway
import build_index
import memory


def retrieve(query: str, k: int = 5) -> list:
    return [
        item
        for item in memory.read(query, kinds=["fact"], top_k=k)
        if item.source.startswith("corpus:")
    ]


def answer(query: str) -> tuple[str, list[str]]:
    hits = retrieve(query)
    if not hits:
        return "FAIL — no indexed Formula One evidence was retrieved.", []
    context = "\n\n".join(
        f"SOURCE {item.source}\n{item.value.get('chunk') or item.descriptor}"
        for item in hits
    )
    reply = LLM().chat(
        prompt=(
            f"QUESTION:\n{query}\n\nINDEXED EVIDENCE:\n{context}\n\n"
            "Write a direct factual answer in one to three sentences using only "
            "the indexed evidence. Do not output citations or source labels; the "
            "application adds those. If the evidence is insufficient, say FAIL."
        ),
        provider=CHAT_PROVIDER,
        temperature=0,
        max_tokens=600,
    )
    model_answer = (reply.get("text") or "").strip()
    sources = [item.source for item in hits]
    grounded_answer = model_answer + "\nSources: " + ", ".join(sources)
    return grounded_answer, sources


def run_suite(label: str) -> list[dict]:
    print(f"\n=== {label} ===")
    results: list[dict] = []
    for item in CUSTOM_QUERIES:
        query_id = str(item["id"])
        query = str(item["query"])
        response, sources = answer(query)
        expected = [str(term).lower() for term in item["expected_terms"]]
        answer_only = response.split("\nSources:", 1)[0]
        passed = (
            len(answer_only) >= 20
            and not answer_only.upper().startswith("FAIL")
            and all(term in answer_only.lower() for term in expected)
        )
        result = {
            "id": query_id,
            "query": query,
            "answer": response,
            "sources": sources,
            "pass": passed,
        }
        results.append(result)
        print(f"\n{query_id}: {query}")
        print(f"sources: {sources or '(none)'}")
        print(f"answer: {response}")
        print(f"result: {'PASS' if passed else 'FAIL'}")
    return results


def markdown_report(combined: dict[str, list[dict]]) -> str:
    lines = ["# Custom Formula One RAG comparison", ""]
    for name, suite in combined.items():
        heading = "Without index" if name == "without_index" else "With index"
        lines.extend([f"## {heading}", ""])
        for result in suite:
            lines.extend(
                [
                    f"### {result['id']}",
                    "",
                    f"**Query:** {result['query']}",
                    "",
                    f"**Result:** {'PASS' if result['pass'] else 'FAIL (expected)' if name == 'without_index' else 'FAIL'}",
                    "",
                    f"**Answer:** {result['answer']}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, help="Write a Markdown evidence report.")
    args = parser.parse_args()
    ensure_gateway()
    combined: dict[str, list[dict]] = {}
    if args.compare:
        memory.clear()
        combined["without_index"] = run_suite("WITHOUT INDEX")
        build_index.build(reset=True)
    combined["with_index"] = run_suite("WITH INDEX")
    if args.json:
        print("\n" + json.dumps(combined, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown_report(combined), encoding="utf-8")
        print(f"\nSaved report: {args.output}")
    failures = [
        result
        for name, suite in combined.items()
        for result in suite
        if (name == "with_index" and not result["pass"])
        or (name == "without_index" and result["pass"])
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
