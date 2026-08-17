"""Session 7 agent orchestrator.

The Session 6 loop is intentionally unchanged:

    Memory.read -> Perception.observe -> Decision.next_step ->
    Action.execute -> Memory.record_outcome

Session 7 extends Memory with embeddings and FAISS, and adds two MCP tools
for document ingestion and indexed search. The orchestrator does not contain
query-specific routing rules.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import action
import artifacts
import decision
import memory
import perception
from gateway import ensure_gateway
from schemas import Goal

MCP_SERVER = Path(__file__).parent / "mcp_server.py"
MAX_ITERATIONS = 20


def _mcp_tools_for_decision(tools) -> list[dict]:
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema
            or {"type": "object", "properties": {}},
        }
        for tool in tools
    ]


async def run(query: str, *, max_iterations: int = MAX_ITERATIONS) -> str:
    ensure_gateway()
    run_id = uuid.uuid4().hex[:8]
    print(f"\n{'═' * 78}")
    print(f"run {run_id}  ─  query: {query}")
    print(f"{'═' * 78}")

    try:
        memory.remember(query, source="user_query", run_id=run_id)
    except Exception as exc:
        print(f"[memory.remember] skipped: {exc}")

    server_parameters = StdioServerParameters(
        command=sys.executable, args=[str(MCP_SERVER)]
    )
    history: list[dict] = []
    prior_goals: list[Goal] = []
    final_answer = ""

    async with stdio_client(server_parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            tools_for_decision = _mcp_tools_for_decision(mcp_tools)
            print(f"[mcp] loaded {len(mcp_tools)} tools: {[t.name for t in mcp_tools]}")

            for iteration in range(1, max_iterations + 1):
                print(f"\n─── iter {iteration} ─────────────────────────────────────────────")

                hits = memory.read(query, history)
                print(f"[memory.read]   {len(hits)} hits")

                observation = perception.observe(
                    query, hits, history, prior_goals, run_id
                )
                prior_goals = observation.goals
                for goal in observation.goals:
                    flag = "✓" if goal.done else "○"
                    attach = (
                        f"  attach={goal.attach_artifact_id}"
                        if goal.attach_artifact_id
                        else ""
                    )
                    print(f"[perception]    {flag} {goal.id} — {goal.text}{attach}")

                if observation.all_done:
                    print(f"\n[done] all {len(observation.goals)} goals satisfied")
                    break

                goal = observation.next_unfinished()
                if goal is None:
                    print("\n[done] no unfinished goal — stopping")
                    break

                attached: list[tuple[str, bytes]] = []
                if goal.attach_artifact_id and artifacts.exists(
                    goal.attach_artifact_id
                ):
                    blob = artifacts.get_bytes(goal.attach_artifact_id)
                    attached.append((goal.attach_artifact_id, blob))
                    print(f"[attach]        {goal.attach_artifact_id} ({len(blob)} bytes)")
                if goal.text.startswith("Synthesize common advice from all fetched sources"):
                    known = {artifact_id for artifact_id, _blob in attached}
                    for event in history:
                        artifact_id = event.get("artifact_id")
                        if not artifact_id or artifact_id in known:
                            continue
                        if event.get("tool") != "fetch_url" or not artifacts.exists(artifact_id):
                            continue
                        blob = artifacts.get_bytes(artifact_id)
                        attached.append((artifact_id, blob))
                        known.add(artifact_id)
                        print(f"[attach]        {artifact_id} ({len(blob)} bytes)")

                output = decision.next_step(
                    goal, hits, attached, history, tools_for_decision
                )
                if output.is_answer:
                    answer = output.answer or ""
                    print(
                        f"[decision]      ANSWER: {answer[:200]}"
                        f"{'...' if len(answer) > 200 else ''}"
                    )
                    history.append(
                        {
                            "iter": iteration,
                            "kind": "answer",
                            "goal_id": goal.id,
                            "text": answer,
                        }
                    )
                    final_answer = answer
                    continue

                tool_call = output.tool_call
                if tool_call is None:
                    print("[decision]      empty response; retrying")
                    continue
                print(
                    f"[decision]      TOOL_CALL: {tool_call.name}"
                    f"({json.dumps(tool_call.arguments)[:120]})"
                )
                result_text, artifact_id = await action.execute(session, tool_call)
                preview = result_text[:200].replace("\n", " ")
                print(
                    f"[action]        → {preview}"
                    f"{'...' if len(result_text) > 200 else ''}"
                    + (f"   +{artifact_id}" if artifact_id else "")
                )
                memory.record_outcome(
                    tool_call=tool_call,
                    result_text=result_text,
                    artifact_id=artifact_id,
                    run_id=run_id,
                    goal_id=goal.id,
                )
                history.append(
                    {
                        "iter": iteration,
                        "kind": "action",
                        "goal_id": goal.id,
                        "tool": tool_call.name,
                        "arguments": tool_call.arguments,
                        "result_descriptor": result_text[:3000],
                        "artifact_id": artifact_id,
                    }
                )
            else:
                raise RuntimeError(
                    f"Query did not complete within {max_iterations} iterations."
                )

    print(f"\n{'═' * 78}")
    print(f"FINAL: {final_answer}")
    print(f"{'═' * 78}\n")
    return final_answer


def main() -> None:
    query = " ".join(sys.argv[1:]) or (
        "What is the current time in Asia/Tokyo and Asia/Kolkata? "
        "Tell me the difference in hours."
    )
    asyncio.run(run(query))


if __name__ == "__main__":
    main()
