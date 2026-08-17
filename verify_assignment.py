#!/usr/bin/env python3
"""Fast, deterministic checks for the Assignment 7 submission contract."""

from __future__ import annotations

import py_compile
import re
from pathlib import Path

import build_index
import perception
from custom_queries import CUSTOM_QUERIES
from run_base_queries import ITERATION_BOUNDS, QUERIES

ROOT = Path(__file__).parent
TOOL_NAMES = {
    "web_search",
    "fetch_url",
    "get_time",
    "currency_convert",
    "weather_forecast",
    "read_file",
    "list_dir",
    "create_file",
    "update_file",
    "edit_file",
    "index_document",
    "search_knowledge",
}

TRACE_FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "Query did not complete within",
    "completed without a final answer",
    "Error executing tool",
)

TRACE_REQUIRED_FINAL_TERMS = {
    "A": ("1916", "2001", "entropy", "sampling", "noisy channel"),
    "B": ("open-meteo", "activities:", "most appropriate:"),
    "C1": ("15 may 2026", "1 may 2026", "reminders were created"),
    "C2": ("15 may 2026",),
    "D": ("1.", "2.", "3."),
    "E": ("1.", "2.", "3.", "attention mechanisms", "parallel"),
    "F1": ("15 chunks", "5 markdown files"),
    "F2": ("papers/cot.md", "papers/react.md"),
    "G": (
        "papers/react.md",
        "papers/dpo.md",
        "papers/lora.md",
        "papers/attention.md",
        "papers/cot.md",
    ),
    "H": ("papers/react.md", "papers/cot.md"),
}


def check(label: str, condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(f"{label}: {detail}")
    print(f"PASS  {label}: {detail}")


def main() -> int:
    for path in sorted(ROOT.glob("*.py")):
        py_compile.compile(str(path), doraise=True)
    check("python", True, "all top-level Python files compile")

    check(
        "base queries",
        list(QUERIES) == ["A", "B", "C1", "C2", "D", "E", "F1", "F2", "G", "H"],
        "eight demonstrations are present, with C and F split into two runs",
    )
    check(
        "iteration bounds",
        all(bound > 0 for bound in ITERATION_BOUNDS.values()),
        "every base run has the bound stated in the assignment",
    )

    evidence_dir = ROOT / "evidence"
    for label, query in QUERIES.items():
        trace_path = evidence_dir / f"base_{label}.txt"
        check(f"trace {label}", trace_path.exists(), trace_path.name)
        trace = trace_path.read_text(encoding="utf-8")
        iterations = [
            int(match) for match in re.findall(r"─── iter (\d+)", trace)
        ]
        check(
            f"trace {label} query",
            f"query: {query}" in trace,
            "contains the verbatim assigned query",
        )
        check(
            f"trace {label} iterations",
            bool(iterations) and max(iterations) <= ITERATION_BOUNDS[label],
            f"{max(iterations) if iterations else 0}/{ITERATION_BOUNDS[label]}",
        )
        check(
            f"trace {label} completion",
            "[done]" in trace and bool(re.search(r"^FINAL:\s*\S", trace, re.M)),
            "completed with a non-empty final answer",
        )
        final_answer = trace.split("FINAL:", 1)[-1].lower()
        required_terms = TRACE_REQUIRED_FINAL_TERMS[label]
        missing_terms = [
            term for term in required_terms if term.lower() not in final_answer
        ]
        check(
            f"trace {label} content",
            not missing_terms,
            "required answer facts are present",
        )
        failures = [marker for marker in TRACE_FAILURE_MARKERS if marker in trace]
        check(
            f"trace {label} errors",
            not failures,
            "no runtime failure markers",
        )

    tool_contracts = {
        "A": ("fetch_url", 1),
        "B": ("weather_forecast", 1),
        "C1": ("create_file", 2),
        "C2": ("TOOL_CALL:", 0),
        "D": ("fetch_url", 3),
        "E": ("search_knowledge", 1),
        "F1": ("index_document", 5),
        "F2": ("search_knowledge", 1),
        "G": ("search_knowledge", 1),
        "H": ("search_knowledge", 1),
    }
    for label, (tool_name, minimum) in tool_contracts.items():
        trace = (evidence_dir / f"base_{label}.txt").read_text(encoding="utf-8")
        actual = trace.count(f"TOOL_CALL: {tool_name}") if minimum else trace.count(tool_name)
        condition = actual >= minimum if minimum else actual == 0
        check(
            f"trace {label} tool contract",
            condition,
            f"{actual} {tool_name} calls",
        )

    system_lower = perception.SYSTEM.lower()
    leaked = sorted(name for name in TOOL_NAMES if name in system_lower)
    check(
        "Perception gate",
        not leaked,
        "zero MCP tool names occur in Perception.SYSTEM",
    )

    items = build_index.collect_items()
    check("corpus size", len(items) >= 50, f"{len(items)} explicit corpus items")
    check("custom queries", len(CUSTOM_QUERIES) == 5, "exactly five custom queries")
    semantic = [item for item in CUSTOM_QUERIES if item.get("semantic")]
    check("semantic queries", len(semantic) >= 2, f"{len(semantic)} semantic queries")
    for query in semantic:
        probe = str(query["semantic_probe"])
        corpus_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "corpus").glob("*.md"))
        )
        check(
            f"{query['id']} lexical absence",
            probe.lower() not in corpus_text.lower(),
            f"probe phrase {probe!r} is absent from the corpus",
        )

    custom_report = evidence_dir / "custom_comparison.md"
    check("custom trace", custom_report.exists(), custom_report.name)
    custom_text = custom_report.read_text(encoding="utf-8")
    with_index_passes = len(re.findall(r"\*\*Result:\*\* PASS$", custom_text, re.M))
    without_index_failures = len(
        re.findall(r"\*\*Result:\*\* FAIL \(expected\)$", custom_text, re.M)
    )
    check(
        "custom with index",
        with_index_passes == 5,
        f"{with_index_passes}/5 queries pass",
    )
    check(
        "custom without index",
        without_index_failures == 5,
        f"{without_index_failures}/5 queries fail as required",
    )

    print("\nAssignment contract checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
