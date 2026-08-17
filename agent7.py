"""agent7 — Session 6 architecture + Session 7 hybrid RAG memory.

Same loop: Memory → Perception → Decision → Action.
Memory.read now merges FAISS vector hits with keyword hits.
New MCP tools: index_document, search_knowledge.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import action
import artifacts
import decision
import llm
import memory
import perception
import rag
from schemas import AttachedArtifact, Goal, HistoryEvent, ToolSpec

load_dotenv(Path(__file__).parent / ".env")

ROOT = Path(__file__).parent
MAX_ITERATIONS = 16  # Query F run 1 can reach ~11

# --- Session 6 carryover (A–D) ---
QUERY_A = (
    "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his "
    "birth date, death date, and three key contributions to information theory."
)
QUERY_B = (
    "Find 3 family-friendly things to do in Tokyo this weekend. "
    "Check Saturday's weather forecast there and tell me which one is most appropriate."
)
QUERY_C1 = (
    "My mom's birthday is 15 May 2026. Remember that and give me "
    "a calendar reminder for two weeks before and on the day."
)
QUERY_C2 = "When is mom's birthday?"
QUERY_D = (
    "Search for 'Python asyncio best practices', read the top 3 results, "
    "and give me a short numbered list of the advice they agree on."
)

# --- Session 7 RAG queries (E–H) ---
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
QUERY_G = (
    "Across these papers, how do they handle the credit assignment problem?"
)
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


def clean_state(*, keep_index: bool = False) -> None:
    """Wipe runtime state. F2 needs F1's index — use keep_index=True between F1 and F2."""
    for d in (ROOT / "sandbox",):
        if d.exists():
            shutil.rmtree(d)
    if not keep_index:
        for d in (ROOT / "state",):
            if d.exists():
                shutil.rmtree(d)
        memory.reset_index()
    usage = ROOT / "usage.json"
    if usage.exists():
        usage.unlink()
    print("[clean] sandbox cleared" + ("" if keep_index else "; state/ + FAISS reset"))


@asynccontextmanager
async def mcp_session():
    params = StdioServerParameters(
        command="uv",
        args=["run", "python", "mcp_server.py"],
        cwd=str(ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def load_tools(session) -> list:
    listed = await session.list_tools()
    return list(listed.tools)


MAX_ATTACH_CHARS = 120_000
SYNTH_PER_ARTIFACT_CHARS = 35_000


def _to_attached(artifact_id: int, max_chars: int | None = None) -> AttachedArtifact:
    cap = max_chars if max_chars is not None else MAX_ATTACH_CHARS
    raw = artifacts.get_bytes(artifact_id)
    text = raw.decode("utf-8", errors="replace")
    truncated = False
    if len(text) > cap:
        text = text[:cap]
        truncated = True
    return AttachedArtifact(
        artifact_id=artifact_id,
        text=text,
        size_bytes=len(raw),
        truncated=truncated,
    )


def _is_synthesis(goal: Goal) -> bool:
    t = goal.text.lower()
    return any(
        w in t
        for w in (
            "extract",
            "synthes",
            "compare",
            "choose",
            "most appropriate",
            "common advice",
            "numbered",
            "contributions",
        )
    )


def _fetch_artifact_ids(history: list[HistoryEvent]) -> list[int]:
    ids: list[int] = []
    for ev in history:
        if ev.kind == "action" and ev.tool == "fetch_url" and ev.artifact_id:
            ids.append(ev.artifact_id)
    return ids


def _collect_attached(goal: Goal, hits, history: list[HistoryEvent]) -> list[AttachedArtifact]:
    if goal.is_gather():
        return []

    attached: list[AttachedArtifact] = []
    seen: set[int] = set()

    def add(art_id: int | None, max_chars: int | None = None) -> None:
        if not art_id or art_id in seen or not artifacts.exists(art_id):
            return
        seen.add(art_id)
        attached.append(_to_attached(art_id, max_chars=max_chars))

    if _is_synthesis(goal):
        for ev in reversed(history):
            if ev.kind == "action" and ev.tool == "search_knowledge" and ev.artifact_id:
                add(ev.artifact_id, max_chars=SYNTH_PER_ARTIFACT_CHARS)
                return attached
        fetch_ids = _fetch_artifact_ids(history)
        if len(fetch_ids) < 3 and "top 3" not in goal.text.lower():
            pass
        elif len(fetch_ids) >= 2:
            for art_id in fetch_ids[-3:]:
                add(art_id, max_chars=SYNTH_PER_ARTIFACT_CHARS)
            return attached

    add(goal.attach_artifact_id)
    return attached


def final_answer_from(history: list[HistoryEvent]) -> str:
    answers = [e.text for e in history if e.kind == "answer" and e.text]
    if answers:
        return answers[-1]
    bits = [e.result_descriptor for e in history if e.result_descriptor]
    if bits:
        return bits[-1]
    return "No answer was produced."


async def run(query: str) -> str:
    llm.ensure_gateway()
    rag.sync_repo_docs_to_sandbox()
    run_id = uuid.uuid4().hex[:8]
    history: list[HistoryEvent] = []
    prior_goals: list[Goal] = []

    try:
        memory.remember(query, source="user_query", run_id=run_id)
    except Exception as exc:
        print(f"[memory.remember] skipped ({exc})")

    async with mcp_session() as session:
        mcp_tools = await load_tools(session)
        tools: list[ToolSpec] = action.mcp_tools_for_decision(mcp_tools)

        for it in range(1, MAX_ITERATIONS + 1):
            print(f"\n─── iter {it} ───")
            hits = memory.read(query, history)
            print(f"[memory.read]   {len(hits)} hits (vector-first, keyword fallback)")
            obs = perception.observe(query, hits, history, prior_goals, run_id)
            prior_goals = obs.goals
            if obs.all_done:
                break

            goal = obs.next_unfinished()
            attached = _collect_attached(goal, hits, history)
            for a in attached:
                print(f"[attach]        {a.artifact_id} ({a.size_bytes} bytes)")

            out = decision.next_step(goal, hits, attached, history, tools)

            if out.is_answer:
                history.append(
                    HistoryEvent(
                        iter=it,
                        kind="answer",
                        goal_id=goal.id,
                        text=out.answer,
                    )
                )
                continue

            if out.tool_call is None:
                continue

            result_text, art_id = await action.execute(session, out.tool_call)
            memory.record_outcome(
                tool_call=out.tool_call,
                result_text=result_text,
                artifact_id=art_id,
                run_id=run_id,
                goal_id=goal.id,
            )
            desc_cap = 4000 if out.tool_call.name == "web_search" else 300
            history.append(
                HistoryEvent(
                    iter=it,
                    kind="action",
                    goal_id=goal.id,
                    tool=out.tool_call.name,
                    arguments=out.tool_call.arguments,
                    result_descriptor=result_text[:desc_cap],
                    artifact_id=art_id,
                )
            )

    answer = final_answer_from(history)
    print(f"\nFINAL: {answer}")
    return answer


async def _amain(args: argparse.Namespace) -> None:
    if args.clean:
        clean_state()
        if not args.query and not args.all:
            return

    if args.all:
        clean_state()
        for label, q in [("A", QUERY_A), ("B", QUERY_B)]:
            print(f"\n========== QUERY {label} ==========")
            await run(q)
            clean_state()
        print("\n========== QUERY C run 1 ==========")
        await run(QUERY_C1)
        print("\n========== QUERY C run 2 ==========")
        await run(QUERY_C2)
        clean_state()
        for label, q in [("D", QUERY_D), ("E", QUERY_E)]:
            print(f"\n========== QUERY {label} ==========")
            await run(q)
            clean_state()
        print("\n========== QUERY F run 1 ==========")
        await run(QUERY_F1)
        print("\n========== QUERY F run 2 ==========")
        await run(QUERY_F2)
        for label, q in [("G", QUERY_G), ("H", QUERY_H)]:
            print(f"\n========== QUERY {label} ==========")
            await run(q)
        return

    key = (args.query or "").upper()
    if key not in QUERIES:
        raise SystemExit(f"Unknown query {args.query!r}. Use A–H, C1, C2, F1, F2, or --all")
    if key == "F2":
        pass  # must follow F1 in same state/
    elif key != "C2":
        if args.reset:
            clean_state()
    await run(QUERIES[key])


def main() -> None:
    p = argparse.ArgumentParser(description="agent7 — Session 7 RAG agent")
    p.add_argument("query", nargs="?", help="A | B | C1 | C2 | D | E | F1 | F2 | G | H")
    p.add_argument("--all", action="store_true", help="Run base queries A–H in order")
    p.add_argument("--clean", action="store_true", help="Wipe state/ then exit (or then run)")
    p.add_argument("--reset", action="store_true", help="Wipe state/ before this query")
    args = p.parse_args()
    if not args.query and not args.all and not args.clean:
        p.print_help()
        raise SystemExit(2)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
