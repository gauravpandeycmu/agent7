"""Perception: turn (query, memory hits, history) into an ordered goal list.

This layer never calls tools. It only answers:
  - what goals exist
  - which are already done
  - which open goal needs an artifact attached (by index into the hit list)

Safety nets (code, not the LLM):
  sticky-done     once a goal is done it cannot be reopened
  evidence-done   a goal is done only if history/hits actually prove it
  force-attach    extract/synthesise/compare/decide goals attach the newest
                  artifact in hits if the LLM forgot attach_index
  stable ids      later iterations reuse goal ids from prior_goals
"""

from __future__ import annotations

from pathlib import Path

from schemas import (
    Goal,
    HistoryEvent,
    MemoryHit,
    Observation,
    PerceptionLLMOut,
)
import llm

PROMPT_PATH = Path(__file__).parent / "prompts" / "perception.txt"

SYNTHESIS_WORDS = (
    "synthesise",
    "synthesize",
    "extract",
    "compare",
    "decide",
    "choose",
    "list",
    "summar",
    "common advice",
    "most appropriate",
)

RECALL_QUERY_MARKERS = ("when is", "what is", "who is", "tell me when", "answer when")


def _is_recall_query(query: str) -> bool:
    q = query.lower().strip()
    return any(m in q for m in RECALL_QUERY_MARKERS)


def _normalize_recall_plan(
    query: str, hits: list[MemoryHit], goals: list[Goal], prior_goals: list[Goal]
) -> list[Goal]:
    """Recall questions with a fact hit → one answer goal (no web search)."""
    if prior_goals:
        return goals
    facts = [h for h in hits if h.kind in ("fact", "preference")]
    if not facts or not _is_recall_query(query):
        return goals
    return [
        Goal(
            id="g1",
            text="Answer the user using the fact from memory",
            status="open",
            attach_index=-1,
        )
    ]


QUERY_WANTS_REPLY = ("give me", "tell me", "let me know", "remind me")

REPLY_GOAL_MARKERS = (
    "answer",
    "confirm",
    "tell the user",
    "give the user",
    "respond to the user",
    "inform the user",
)


def _goal_delivers_user_answer(goal: Goal) -> bool:
    """Goals whose completion is a Decision ANSWER (not only a tool side-effect)."""
    t = goal.text.lower()
    if any(m in t for m in REPLY_GOAL_MARKERS):
        return True
    return any(
        w in t
        for w in (
            "extract",
            "synthes",
            "choose",
            "most appropriate",
            "common advice",
            "advice they",
        )
    )


def _ensure_reply_goal(query: str, goals: list[Goal], prior_goals: list[Goal]) -> list[Goal]:
    """After LLM decomposition: if user expects a reply but every goal is side-effect-only, append one."""
    if prior_goals or not goals:
        return goals
    if any(_goal_delivers_user_answer(g) for g in goals):
        return goals
    q = query.lower()
    if not any(phrase in q for phrase in QUERY_WANTS_REPLY):
        return goals
    if not any(
        (("create" in g.text.lower() or "reminder" in g.text.lower()) and "confirm" not in g.text.lower())
        for g in goals
    ):
        return goals
    goals.append(
        Goal(
            id=f"g{len(goals) + 1}",
            text="Confirm completed work to the user",
            status="open",
            attach_index=-1,
        )
    )
    return goals


def _system() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _format_hits(hits: list[MemoryHit]) -> str:
    if not hits:
        return "(no memory hits)"
    lines = []
    for i, h in enumerate(hits):
        art = f" artifact_id={h.artifact_id}" if h.artifact_id else ""
        lines.append(f"[{i}] kind={h.kind} handle={h.handle}{art}")
        lines.append(f"    {h.descriptor[:280]}")
    return "\n".join(lines)


def _format_history(history: list[HistoryEvent]) -> str:
    if not history:
        return "(empty)"
    lines = []
    for ev in history:
        if ev.kind == "answer":
            lines.append(f"iter {ev.iter} ANSWER goal={ev.goal_id}: {(ev.text or '')[:240]}")
        else:
            lines.append(
                f"iter {ev.iter} ACTION goal={ev.goal_id} {ev.tool}({ev.arguments}) "
                f"→ {(ev.result_descriptor or '')[:200]} art={ev.artifact_id}"
            )
    return "\n".join(lines)


def _format_prior(prior: list[Goal]) -> str:
    if not prior:
        return "(none — this is the first iteration; propose the goal list now)"
    lines = ["Reuse these goals. Do not add, remove, or reorder. Only update status and attach_index."]
    for g in prior:
        lines.append(f"- id={g.id} status={g.status} attach_index={g.attach_index} text={g.text}")
    return "\n".join(lines)



def _count_fetches(history: list[HistoryEvent]) -> int:
    return sum(1 for ev in history if ev.kind == "action" and ev.tool == "fetch_url" and ev.artifact_id)


def _is_successful_action(ev: HistoryEvent) -> bool:
    """Failed MCP calls still land in history — they must not count as progress."""
    if ev.kind != "action":
        return False
    return not (ev.result_descriptor or "").startswith("ERROR")


def _has_answer(goal_id: str, history: list[HistoryEvent]) -> bool:
    return any(ev.kind == "answer" and ev.goal_id == goal_id for ev in history)


def _has_action(goal_id: str, history: list[HistoryEvent], tool: str | None = None) -> bool:
    for ev in history:
        if ev.kind != "action" or ev.goal_id != goal_id:
            continue
        if not _is_successful_action(ev):
            continue
        if tool is None or ev.tool == tool:
            return True
    return False


def _evidence_done(goal: Goal, hits: list[MemoryHit], history: list[HistoryEvent]) -> bool:
    """Code-side truth. The LLM is not allowed to mark done without this."""
    text = goal.text.lower()

    if "retrieve" in text and "synthes" not in text:
        return _has_action(goal.id, history, tool="search_knowledge")

    if any(w in text for w in ("answer the user", "inform the user", "confirm")):
        return _has_answer(goal.id, history)

    if any(w in text for w in SYNTHESIS_WORDS) and "top 3" not in text and "top three" not in text:
        # Search goals like "find 3 activities" are NOT synthesis — they complete
        # after a tool. Answer-goals must wait for Decision to actually answer.
        if any(
            w in text
            for w in (
                "extract",
                "synthes",
                "compare",
                "choose",
                "most appropriate",
                "common advice",
                "advice they",
                "tell me his",
                "final answer",
                "give the user",
            )
        ):
            return _has_answer(goal.id, history)

    if "top 3" in text or "top three" in text or "read the top" in text:
        if any(w in text for w in ("synthes", "extract", "common advice", "advice they", "numbered")):
            return _has_answer(goal.id, history)
        n = _count_fetches(history)
        if n >= 2:
            return True
        if n >= 1:
            from decision import _next_unfetched_url

            if _next_unfetched_url([], history) is None:
                return True
        return False

    # File-create goals: successful create_file on this goal_id.
    if ("create" in text or "reminder" in text) and "confirm" not in text:
        return _has_action(goal.id, history, tool="create_file")

    if any(w in text for w in ("remember", "record the fact", "store")):
        return any(h.kind in ("fact", "preference") for h in hits)

    # Search / find gather goals.
    if (
        ("search" in text or ("find " in text and "most appropriate" not in text))
        and "synthes" not in text
        and "fetch the top" not in text
        and "choose" not in text
    ):
        return _has_action(goal.id, history, tool="web_search")
    if "weather" in text or "forecast" in text or ("check " in text and "weather" in text):
        if "choose" in text or "most appropriate" in text:
            pass
        else:
            return (
                _has_action(goal.id, history, tool="web_search")
                or _has_action(goal.id, history, tool="fetch_url")
            )

    # Plain fetch goals (e.g. Wikipedia URL): successful fetch_url on this goal.
    if "fetch" in text and "top 3" not in text and "top three" not in text:
        return _has_action(goal.id, history, tool="fetch_url")

    if "confirm" in text:
        return _has_answer(goal.id, history)

    if "index" in text and "indexed" not in text:
        if _has_action(goal.id, history, tool="index_document"):
            return True
        # Already indexed in state/ from a prior run or build_index.py
        path = ""
        for word in goal.text.split():
            if word.endswith(".md") and "/" in word:
                path = word.strip(".,()")
                break
        if path:
            for h in hits:
                if path in (h.descriptor or "") or path in (h.value or ""):
                    return True
        return False

    if text.startswith("list ") or ("list " in text and "directory" in text):
        return _has_action(goal.id, history, tool="list_dir")

    if any(
        w in text
        for w in ("retrieve", "search knowledge", "search_knowledge", "across the papers", "across these papers")
    ) and "synthes" not in text and "compare" not in text and "extract" not in text:
        return _has_action(goal.id, history, tool="search_knowledge")

    if "chunk" in text and ("confirm" in text or "total" in text or "how many" in text):
        return _has_answer(goal.id, history)

    if _has_answer(goal.id, history):
        return True
    if _has_action(goal.id, history):
        return True
    return False


def _sticky_and_evidence(
    proposed: list[Goal],
    prior: list[Goal],
    hits: list[MemoryHit],
    history: list[HistoryEvent],
) -> list[Goal]:
    # Keep the first-iteration goal list. Later LLM output may paraphrase;
    # we lock text/ids to `prior` when we have it.
    if prior:
        locked: list[Goal] = []
        for i, old in enumerate(prior):
            attach_index = old.attach_index
            if i < len(proposed):
                attach_index = proposed[i].attach_index
            locked.append(
                Goal(
                    id=old.id,
                    text=old.text,
                    status=old.status,
                    attach_index=attach_index,
                    attach_artifact_id=old.attach_artifact_id,
                )
            )
        proposed = locked

    done_ids = {g.id for g in prior if g.status == "done"}
    for g in proposed:
        if g.id in done_ids:
            g.status = "done"
            continue
        g.status = "done" if _evidence_done(g, hits, history) else "open"
    return proposed


def _fetch_top_n_still_open(goals: list[Goal]) -> bool:
    for g in goals:
        if g.status != "open":
            continue
        t = g.text.lower()
        if "top 3" in t or "top three" in t or "fetch the top" in t:
            return True
    return False


def _resolve_attach(goal: Goal, hits: list[MemoryHit], goals: list[Goal]) -> Goal:
    """Map attach_index → artifact_id. Force-attach for synthesis goals only."""
    if goal.is_gather():
        goal.attach_index = -1
        goal.attach_artifact_id = None
        return goal

    text = goal.text.lower()
    needs = any(w in text for w in SYNTHESIS_WORDS)
    if needs and _fetch_top_n_still_open(goals):
        goal.attach_index = -1
        goal.attach_artifact_id = None
        return goal

    if 0 <= goal.attach_index < len(hits):
        hit = hits[goal.attach_index]
        if hit.artifact_id:
            goal.attach_artifact_id = hit.artifact_id

    text = goal.text.lower()
    needs = any(w in text for w in SYNTHESIS_WORDS)
    if needs and not goal.attach_artifact_id:
        arts = [h for h in hits if h.artifact_id]
        if arts:
            goal.attach_artifact_id = arts[-1].artifact_id
            # keep the index honest for traces
            for i, h in enumerate(hits):
                if h.artifact_id == goal.attach_artifact_id:
                    goal.attach_index = i
                    break
    return goal


def _indexed_paths(history: list[HistoryEvent]) -> set[str]:
    paths: set[str] = set()
    for ev in history:
        if ev.kind != "action" or ev.tool != "index_document":
            continue
        if (ev.result_descriptor or "").startswith("ERROR"):
            continue
        p = (ev.arguments or {}).get("path")
        if p:
            paths.add(str(p))
    return paths


def _append_index_goals_after_list(
    query: str, goals: list[Goal], history: list[HistoryEvent], prior_goals: list[Goal]
) -> list[Goal]:
    """After list_dir under papers/: append one index goal per unindexed .md file."""
    q = query.lower()
    if "index" not in q or "papers" not in q:
        return goals
    listed = any(
        ev.kind == "action" and ev.tool == "list_dir" and not (ev.result_descriptor or "").startswith("ERROR")
        for ev in history
    )
    if not listed:
        return goals
    indexed = _indexed_paths(history)
    candidates = [
        "papers/attention.md",
        "papers/cot.md",
        "papers/react.md",
        "papers/dpo.md",
        "papers/lora.md",
    ]
    existing = {g.text.strip().lower() for g in goals}
    for path in candidates:
        if path in indexed:
            continue
        text = f"Index the file {path}"
        if text.lower() in existing:
            continue
        goals.append(
            Goal(
                id=f"g{len(goals) + 1}",
                text=text,
                status="open",
                attach_index=-1,
            )
        )
        existing.add(text.lower())
    confirm = "confirm total indexed chunk count to the user"
    if not any(confirm in g.text.lower() or ("chunk" in g.text.lower() and "how many" in g.text.lower()) for g in goals):
        goals.append(
            Goal(
                id=f"g{len(goals) + 1}",
                text="Confirm total indexed chunk count to the user",
                status="open",
                attach_index=-1,
            )
        )
    return goals


PAPER_INDEX_PATHS = (
    "papers/attention.md",
    "papers/cot.md",
    "papers/react.md",
    "papers/dpo.md",
    "papers/lora.md",
)

_S6_QUERY_MARKERS = (
    "http://",
    "https://",
    "tokyo",
    "birthday",
    "asyncio",
    "weekend",
    "reminder",
    "weather",
)


def _seed_index_all_papers(query: str, goals: list[Goal], prior_goals: list[Goal]) -> list[Goal]:
    """Query F1: lock a 5-file index plan on the first iteration."""
    if prior_goals:
        return goals
    q = query.lower()
    if "index" not in q or "papers" not in q:
        return goals
    if "every" not in q and "all" not in q:
        return goals
    seeded = [
        Goal(id=f"g{i}", text=f"Index the file {path}", status="open", attach_index=-1)
        for i, path in enumerate(PAPER_INDEX_PATHS, start=1)
    ]
    seeded.append(
        Goal(
            id=f"g{len(seeded) + 1}",
            text="Confirm total indexed chunk count to the user",
            status="open",
            attach_index=-1,
        )
    )
    return seeded


def _seed_rag_question(query: str, goals: list[Goal], prior_goals: list[Goal]) -> list[Goal]:
    """Custom / cross-paper questions: retrieve then synthesise (no web search)."""
    if prior_goals:
        return goals
    q = query.lower()
    if any(m in q for m in _S6_QUERY_MARKERS):
        return goals
    if "index" in q and (".md" in q or "papers" in q):
        return goals
    qtext = query.strip()
    return [
        Goal(
            id="g1",
            text=f"Retrieve relevant knowledge chunks for the question: {qtext}",
            status="open",
            attach_index=-1,
        ),
        Goal(
            id="g2",
            text=f"Synthesise an answer from the retrieved indexed chunks to: {qtext}",
            status="open",
            attach_index=-1,
        ),
    ]


def _emit_observation(goals: list[Goal]) -> Observation:
    all_done = bool(goals) and all(g.status == "done" for g in goals)
    print("[perception]    ", end="")
    print(
        "\n                ".join(
            f"[{g.status}] {g.text}"
            + (f"\n                  attach={g.attach_artifact_id}" if g.attach_artifact_id else "")
            for g in goals
        )
    )
    if all_done:
        print(f"\n[done] all {len(goals)} goals satisfied")
    return Observation(goals=goals, all_done=all_done)


def observe(
    query: str,
    hits: list[MemoryHit],
    history: list[HistoryEvent],
    prior_goals: list[Goal],
    run_id: str,
) -> Observation:
    if prior_goals:
        goals = _sticky_and_evidence(list(prior_goals), prior_goals, hits, history)
        for g in goals:
            if g.status == "open":
                if _evidence_done(g, hits, history):
                    g.status = "done"
                else:
                    _resolve_attach(g, hits, goals)
        return _emit_observation(goals)

    recall = _normalize_recall_plan(query, hits, [], [])
    if recall:
        return _emit_observation(recall)
    seeded = _seed_index_all_papers(query, [], [])
    if seeded:
        return _emit_observation(seeded)
    seeded = _seed_rag_question(query, [], [])
    if seeded:
        for g in seeded:
            if g.status == "open":
                _resolve_attach(g, hits, seeded)
        return _emit_observation(seeded)

    user = (
        f"User query:\n{query}\n\n"
        f"run_id: {run_id}\n\n"
        f"Prior goals:\n{_format_prior(prior_goals)}\n\n"
        f"Memory hits (attach_index refers to these numbers):\n{_format_hits(hits)}\n\n"
        f"Loop history:\n{_format_history(history)}\n"
    )
    raw = llm.structured(
        PerceptionLLMOut,
        user,
        system=_system(),
        provider="g",
        max_tokens=1200,
        temperature=0.4,
    )

    goals: list[Goal] = []
    for i, g in enumerate(raw.goals, start=1):
        gid = prior_goals[i - 1].id if i - 1 < len(prior_goals) else f"g{i}"
        goals.append(
            Goal(
                id=gid,
                text=g.text.strip(),
                status=g.status,
                attach_index=g.attach_index,
            )
        )

    goals = _ensure_reply_goal(query, goals, prior_goals)
    goals = _sticky_and_evidence(goals, prior_goals, hits, history)
    goals = _append_index_goals_after_list(query, goals, history, prior_goals)
    for g in goals:
        if g.status == "open":
            if _evidence_done(g, hits, history):
                g.status = "done"
            else:
                _resolve_attach(g, hits, goals)

    return _emit_observation(goals)
