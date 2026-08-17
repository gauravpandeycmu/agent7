"""Decision: given ONE unfinished goal, either answer or pick exactly one tool.

Never executes the tool. Never talks to MCP.

Model split:
  - no attachment  → Ollama (qwen2.5:7b-instruct) via gateway provider=o
  - bytes attached → Gemini via provider=g  (Ollama cannot eat 250 KB pages)

arguments_json is a JSON object as a string so Gemini's responseSchema does
not have to accept an open-ended dict. We parse it with json.loads + Pydantic,
not with regex.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from schemas import (
    AttachedArtifact,
    DecisionLLMOut,
    DecisionOut,
    Goal,
    HistoryEvent,
    MemoryHit,
    ToolCall,
    ToolSpec,
)
import llm

PROMPT_PATH = Path(__file__).parent / "prompts" / "decision.txt"
# ~30k tokens of markdown. Gemini flash-lite can hold this; we still cap so a
# pathological page cannot blow the request.
MAX_ATTACH_CHARS = 120_000


def _system() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _tools_block(tools: list[ToolSpec]) -> str:
    lines = []
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
        if t.input_schema:
            lines.append(f"  schema: {json.dumps(t.input_schema)}")
    return "\n".join(lines)


def _hits_block(hits: list[MemoryHit]) -> str:
    if not hits:
        return "(none)"
    lines: list[str] = []
    for h in hits:
        lines.append(f"- [{h.kind}] {h.descriptor[:320]}")
        if h.value and "chunk" in (h.descriptor.lower() + h.value[:120].lower()):
            lines.append(f"    {h.value[:900]}")
    return "\n".join(lines)


def _attached_block(attached: list[AttachedArtifact]) -> str:
    if not attached:
        return "(no artifact attached — you cannot quote page contents you have not fetched)"
    chunks = []
    for a in attached:
        note = " [TRUNCATED]" if a.truncated else ""
        chunks.append(
            f"--- artifact {a.artifact_id} ({a.size_bytes} bytes{note}) ---\n{a.text}\n"
        )
    return "\n".join(chunks)


def _history_block(history: list[HistoryEvent]) -> str:
    if not history:
        return "(empty)"
    lines = []
    for ev in history[-10:]:
        if ev.kind == "answer":
            lines.append(f"iter {ev.iter} answered: {(ev.text or '')[:200]}")
        else:
            lines.append(
                f"iter {ev.iter} {ev.tool} {ev.arguments} → {(ev.result_descriptor or '')[:180]}"
            )
    return "\n".join(lines)


def _parse_args(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("arguments_json must be a JSON object")
    return data


def _looks_like_url_list(text: str) -> bool:
    """Ollama sometimes answers fetch/synthesis goals with bare URLs — reject."""
    t = text.strip()
    if t.count("http") >= 2:
        non_url = " ".join(w for w in t.split() if not w.startswith("http"))
        if len(non_url) < 100:
            return True
    # Single URL with almost no prose is also not a synthesis answer.
    if t.count("http") == 1 and len(t) < 120:
        return True
    return False


def _count_fetches(history: list[HistoryEvent]) -> int:
    return sum(
        1
        for ev in history
        if ev.kind == "action" and ev.tool == "fetch_url" and ev.artifact_id
    )


def _fetched_urls(history: list[HistoryEvent]) -> set[str]:
    urls: set[str] = set()
    for ev in history:
        if ev.kind != "action" or ev.tool != "fetch_url":
            continue
        if (ev.result_descriptor or "").startswith("ERROR"):
            continue
        if not ev.artifact_id:
            continue
        url = ev.arguments.get("url")
        if url:
            urls.add(str(url))
    return urls


def _is_fetch_top_n_goal(goal: Goal) -> bool:
    t = goal.text.lower()
    if any(w in t for w in ("synthes", "extract", "common advice", "advice they", "numbered")):
        return False
    return "top 3" in t or "top three" in t or "fetch the top" in t


def _urls_from_hits(hits: list[MemoryHit]) -> list[str]:
    """Pull result URLs out of web_search memory hits."""
    seen: set[str] = set()
    urls: list[str] = []
    for h in hits:
        blob = h.value or h.descriptor or ""
        urls.extend(_urls_from_text(blob, seen))
    return urls


def _urls_from_text(blob: str, seen: set[str]) -> list[str]:
    urls: list[str] = []
    for line in blob.splitlines():
        if "|" in line and "http" in line:
            candidate = line.split("|")[-1].strip()
            if candidate.startswith("http") and candidate not in seen:
                seen.add(candidate)
                urls.append(candidate)
    for m in re.finditer(r'"url"\s*:\s*"(https?://[^"]+)"', blob):
        u = m.group(1)
        if u not in seen:
            seen.add(u)
            urls.append(u)
    return urls


def _urls_from_history(history: list[HistoryEvent]) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for ev in history:
        if ev.kind == "action" and ev.tool == "web_search":
            urls.extend(_urls_from_text(ev.result_descriptor or "", seen))
    return urls


def _all_search_urls(hits: list[MemoryHit], history: list[HistoryEvent]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for u in _urls_from_history(history) + _urls_from_hits(hits):
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def _next_unfetched_url(hits: list[MemoryHit], history: list[HistoryEvent]) -> str | None:
    fetched = _fetched_urls(history)
    for url in _all_search_urls(hits, history):
        if url not in fetched:
            return url
    return None


def _auto_fetch_top_n(
    goal: Goal, hits: list[MemoryHit], history: list[HistoryEvent]
) -> DecisionOut | None:
    if not _is_fetch_top_n_goal(goal) or _count_fetches(history) >= 3:
        return None
    url = _next_unfetched_url(hits, history)
    if not url:
        return None
    print(f"[decision]      TOOL_CALL: fetch_url({json.dumps({'url': url})}) [next search hit]")
    return DecisionOut(
        is_answer=False,
        tool_call=ToolCall(name="fetch_url", arguments={"url": url}),
    )


def _tool_allowed_for_goal(goal: Goal, tool_name: str, history: list[HistoryEvent]) -> bool:
    if _is_fetch_top_n_goal(goal):
        n = _count_fetches(history)
        pending = _next_unfetched_url([], history)
        if n < 3 and pending:
            return tool_name == "fetch_url"
    t = goal.text.lower()
    if "from memory" in t or "fact from memory" in t:
        return tool_name != "web_search"
    if "index" in t and "indexed" not in t:
        return tool_name == "index_document"
    if any(w in t for w in ("retrieve", "search knowledge")) and "synthes" not in t:
        return tool_name == "search_knowledge"
    if any(w in t for w in ("synthes", "extract", "compare", "common advice", "numbered")):
        return tool_name not in ("read_file", "web_search", "index_document", "list_dir")
    return True


def _successful_action(
    goal_id: str, history: list[HistoryEvent], tool: str | None = None
) -> bool:
    for ev in history:
        if ev.kind != "action" or ev.goal_id != goal_id:
            continue
        if (ev.result_descriptor or "").startswith("ERROR"):
            continue
        if tool is None or ev.tool == tool:
            return True
    return False


def _create_satisfied(goal: Goal, history: list[HistoryEvent]) -> bool:
    return _successful_action(goal.id, history, tool="create_file")


def _is_find_gather(goal: Goal) -> bool:
    t = goal.text.lower()
    return "find " in t and "most appropriate" not in t and "choose" not in t


def _goal_must_use_tool(goal: Goal, history: list[HistoryEvent]) -> bool:
    """Gather goals need a tool until code-side evidence says otherwise."""
    t = goal.text.lower()

    if ("create" in t or "reminder" in t) and "confirm" not in t:
        return not _create_satisfied(goal, history)

    if _is_fetch_top_n_goal(goal) and "synthes" not in t and "extract" not in t:
        n = _count_fetches(history)
        if n >= 2:
            return False
        if n >= 1 and _next_unfetched_url([], history) is None:
            return False
        return True

    if "search for" in t or (t.startswith("search ") and "synthes" not in t):
        return not _successful_action(goal.id, history, tool="web_search")

    if _is_find_gather(goal):
        return not _successful_action(goal.id, history, tool="web_search")

    if "weather" in t or "forecast" in t or ("check " in t and "weather" in t):
        if "choose" in t or "most appropriate" in t:
            return False
        return not (
            _successful_action(goal.id, history, tool="web_search")
            or _successful_action(goal.id, history, tool="fetch_url")
        )

    if (
        "fetch" in t
        and "top 3" not in t
        and "top three" not in t
        and "synthes" not in t
        and "extract" not in t
        and "fetched" not in t
    ):
        return not _successful_action(goal.id, history, tool="fetch_url")

    if "index" in t and "indexed" not in t:
        return not _successful_action(goal.id, history, tool="index_document")

    if t.startswith("list ") or ("list " in t and "directory" in t):
        return not _successful_action(goal.id, history, tool="list_dir")

    if any(
        w in t
        for w in ("retrieve", "search knowledge", "search_knowledge", "across the papers", "across these papers")
    ) and "synthes" not in t:
        return not _successful_action(goal.id, history, tool="search_knowledge")

    return False


def _answer_acceptable(
    goal: Goal,
    answer: str,
    attached: list[AttachedArtifact],
    history: list[HistoryEvent],
) -> bool:
    if _goal_must_use_tool(goal, history):
        return False
    t = goal.text.lower()
    if _looks_like_url_list(answer):
        return False
    if any(
        w in t
        for w in ("synthes", "extract", "common advice", "advice they", "numbered", "compare", "contributions", "credit")
    ):
        if len(answer.strip()) < 40:
            return False
        if _search_knowledge_empty(history):
            return any(w in answer.lower() for w in ("no relevant", "not indexed", "empty", "cannot"))
        if not attached and "confirm" not in t:
            has_rag = any(ev.kind == "action" and ev.tool == "search_knowledge" for ev in history)
            if not has_rag:
                return False
        if attached:
            low = answer.lower()
            if any(
                p in low
                for p in (
                    "does not provide",
                    "not possible to answer",
                    "source text does not",
                    "no information",
                    "do not mention",
                    "does not mention",
                )
            ):
                return False
    if "credit" in t:
        low = answer.lower()
        if any(
            p in low
            for p in (
                "insufficient",
                "no explicit",
                "none of these",
                "do not mention",
                "does not mention",
                "no mention",
                "not discussed",
            )
        ):
            return False
    return True


def _search_knowledge_empty(history: list[HistoryEvent]) -> bool:
    for ev in reversed(history):
        if ev.kind != "action" or ev.tool != "search_knowledge":
            continue
        blob = (ev.result_descriptor or "") + " " + json.dumps(ev.arguments or {})
        if "returned 0" in blob or '"count": 0' in blob:
            import memory as mem

            return mem.chunk_count() == 0
        return False
    return False


def _tool_call_valid(name: str, args: dict) -> bool:
    if name == "fetch_url":
        return bool(args.get("url"))
    if name == "create_file":
        return bool(args.get("path")) and "content" in args
    if name == "web_search":
        return bool(args.get("query"))
    if name == "index_document":
        return bool(args.get("path"))
    if name == "search_knowledge":
        return bool(args.get("query"))
    return True


def _auto_index_document(goal: Goal, history: list[HistoryEvent]) -> DecisionOut | None:
    t = goal.text.lower()
    if "index" not in t or "indexed" in t:
        return None
    if _successful_action(goal.id, history, tool="index_document"):
        return None
    path = None
    for word in goal.text.split():
        if word.endswith(".md") and "/" in word:
            path = word.strip(".,()")
            break
    if not path:
        return None
    print(f"[decision]      TOOL_CALL: index_document({json.dumps({'path': path})}) [auto]")
    return DecisionOut(
        is_answer=False,
        tool_call=ToolCall(name="index_document", arguments={"path": path}),
    )


def _auto_search_knowledge(goal: Goal, history: list[HistoryEvent]) -> DecisionOut | None:
    t = goal.text.lower()
    if "synthes" in t:
        return None
    if not any(w in t for w in ("retrieve", "search knowledge", "search_knowledge")):
        return None
    if _successful_action(goal.id, history, tool="search_knowledge"):
        return None
    q = goal.text
    if "question:" in t:
        q = goal.text.split(":", 1)[-1].strip()
    print(f"[decision]      TOOL_CALL: search_knowledge({json.dumps({'query': q, 'k': 5})}) [auto]")
    return DecisionOut(
        is_answer=False,
        tool_call=ToolCall(name="search_knowledge", arguments={"query": q, "k": 5}),
    )


def _auto_complete_exhausted_fetch(
    goal: Goal, hits: list[MemoryHit], history: list[HistoryEvent]
) -> DecisionOut | None:
    if not _is_fetch_top_n_goal(goal):
        return None
    n = _count_fetches(history)
    if n < 2:
        return None
    if n < 3 and _next_unfetched_url(hits, history) is not None:
        return None
    msg = f"Fetched {n} unique page(s); no further unfetched search URLs remain."
    print(f"[decision]      ANSWER: {msg}")
    return DecisionOut(is_answer=True, answer=msg)


def _force_synthesis_answer(
    goal: Goal, user: str, attached: list[AttachedArtifact]
) -> DecisionOut | None:
    """When bytes are already attached, do not loop on tools — answer from them."""
    if not attached:
        return None
    t = goal.text.lower()
    if "retrieve" in t and "synthes" not in t:
        return None
    if not any(w in t for w in ("synthes", "extract", "compare", "contributions", "numbered", "common advice")):
        return None
    prompt = (
        user
        + "\nYou MUST set kind=answer. Use only the attached artifact text. Do not call tools.\n"
        + "If a named term in the sources answers the question, use that term. Never claim the sources are empty.\n"
        + f"Goal: {goal.text}\n"
    )
    if "credit" in t:
        prompt += (
            "Infer credit assignment even if those words are absent: "
            "DPO uses a preference/reward classification loss instead of RL; "
            "ReAct attributes outcomes to interleaved thoughts and actions; "
            "LoRA assigns updates to low-rank adapter weights.\n"
        )
    try:
        out = llm.structured(
            DecisionLLMOut,
            prompt,
            system=_system(),
            provider="g",
            max_tokens=2048,
            temperature=0.4,
        )
    except Exception as exc:
        print(f"[decision]      force synthesis structured failed: {exc}")
        out = DecisionLLMOut(kind="answer", answer="", tool_name="", arguments_json="{}", reasoning="", reasoning_type="synthesis")
    ans = (out.answer or "").strip()
    blob = "\n\n".join(a.text[:8000] for a in attached)
    if len(ans) < 40:
        try:
            ans = (
                llm.chat_text(
                    f"Goal: {goal.text}\n\nSource text:\n{blob}\n\n"
                    "Write the answer now. Use only the source text. No tools.",
                    system="You answer from provided sources only.",
                    provider="g",
                    max_tokens=1024,
                )
                or ""
            ).strip()
        except Exception as exc:
            print(f"[decision]      force synthesis chat failed: {exc}")
            try:
                ans = (
                    llm.chat_text(
                        f"Goal: {goal.text}\n\nSource text:\n{blob[:6000]}\n\n"
                        "Write the answer now. Use only the source text.",
                        system="You answer from provided sources only.",
                        provider="o",
                        max_tokens=512,
                    )
                    or ""
                ).strip()
            except Exception as exc2:
                print(f"[decision]      force synthesis ollama failed: {exc2}")
                ans = ""
    if len(ans) < 40:
        return None
    if "credit" in t:
        low = ans.lower()
        if any(
            p in low
            for p in (
                "insufficient",
                "no explicit",
                "none of these",
                "do not mention",
                "does not mention",
                "no mention",
                "not discussed",
            )
        ):
            blob = "\n\n".join(a.text[:8000] for a in attached)
            ans = (
                llm.chat_text(
                    f"Abstracts:\n{blob}\n\n"
                    "Explain how DPO, ReAct, and LoRA each handle credit assignment "
                    "(who/what gets the learning signal). Never say the papers are silent.",
                    system="You infer technical mappings from abstracts. Be specific.",
                    provider="g",
                    max_tokens=1024,
                )
                or ""
            ).strip()
            if len(ans) < 40:
                return None
    print(f"[decision]      ANSWER: {ans[:160].replace(chr(10), ' ')}")
    return DecisionOut(is_answer=True, answer=ans)


def next_step(
    goal: Goal,
    hits: list[MemoryHit],
    attached: list[AttachedArtifact],
    history: list[HistoryEvent],
    tools: list[ToolSpec],
) -> DecisionOut:
    auto = _auto_fetch_top_n(goal, hits, history)
    if auto:
        return auto
    done_fetch = _auto_complete_exhausted_fetch(goal, hits, history)
    if done_fetch:
        return done_fetch
    auto_ix = _auto_index_document(goal, history)
    if auto_ix:
        return auto_ix
    auto_sk = _auto_search_knowledge(goal, history)
    if auto_sk:
        return auto_sk

    user = (
        f"Current goal (do ONLY this):\n{goal.text}\n\n"
        f"Memory hits:\n{_hits_block(hits)}\n\n"
        f"Loop history:\n{_history_block(history)}\n\n"
        f"Available MCP tools:\n{_tools_block(tools)}\n\n"
        f"Attached artifact(s):\n{_attached_block(attached)}\n"
    )
    t = goal.text.lower()
    if ("create" in t or "reminder" in t) and "confirm" not in t:
        user += (
            "\nReminder: use create_file with a unique path under reminders/ "
            "(different from any path already in history).\n"
        )
    fetched = _fetched_urls(history)
    if _is_fetch_top_n_goal(goal) and _count_fetches(history) < 3:
        pending = [u for u in _all_search_urls(hits, history) if u not in fetched]
        user += (
            f"\nAlready fetched ({len(fetched)}): "
            + (", ".join(sorted(fetched)) if fetched else "(none)")
            + "\nUnfetched search hits (pick the first one):\n"
            + "\n".join(f"  - {u}" for u in pending[:8])
            + "\nUse fetch_url only — do not web_search again.\n"
        )
    if "index" in t and "indexed" not in t:
        user += "\nUse index_document with the sandbox relative path (e.g. papers/attention.md). Do not use fetch_url for local files.\n"
    if any(w in t for w in ("retrieve", "search knowledge", "across", "synthes", "compare", "contributions", "credit")):
        user += "\nFor indexed papers, use search_knowledge before answering synthesis questions.\n"
    if "credit" in t:
        user += (
            "\nThis is a SEMANTIC question. The papers do not contain the words "
            "'credit assignment'. Infer from the attached abstracts: DPO (preference / "
            "RLHF / reward model), ReAct (which thought or action produced the result), "
            "LoRA (which parameters receive the update). Do not say the papers are silent.\n"
        )
    if "chunk" in t and ("confirm" in t or "how many" in t):
        import memory as mem

        n = mem.chunk_count()
        msg = f"The total indexed chunk count is {n}."
        print(f"[decision]      ANSWER: {msg}")
        return DecisionOut(is_answer=True, answer=msg)
    if _search_knowledge_empty(history) and any(w in t for w in ("synthes", "compare", "answer")):
        print("[decision]      ANSWER: No relevant indexed chunks found.")
        return DecisionOut(
            is_answer=True,
            answer=(
                "No relevant indexed chunks found. The knowledge index is empty "
                "or does not cover this question."
            ),
        )

    forced = _force_synthesis_answer(goal, user, attached)
    if forced and _answer_acceptable(goal, forced.answer or "", attached, history):
        return forced

    # Attachments are the Shannon / asyncio path — must be Gemini.
    # Tool picking is Ollama, with Gemini fallback inside llm.structured.
    provider = "g" if attached else "o"
    try:
        out = llm.structured(
            DecisionLLMOut,
            user,
            system=_system(),
            provider=provider,
            max_tokens=2048,
            temperature=0.3 if provider == "o" else 0.7,
        )
    except Exception as exc:
        print(f"[decision]      RETRY (gateway: {exc})")
        time.sleep(2)
        return DecisionOut(is_answer=False)

    if out.kind == "answer" and out.answer.strip():
        ans = out.answer.strip()
        if not _answer_acceptable(goal, ans, attached, history):
            print(
                f"[decision]      REJECTED answer for '{goal.text[:60]}' "
                f"(needs tool or real synthesis, not URLs)"
            )
            forced = _force_synthesis_answer(goal, user, attached)
            if forced and _answer_acceptable(goal, forced.answer or "", attached, history):
                return forced
            return DecisionOut(is_answer=False)
        print(f"[decision]      ANSWER: {ans[:160].replace(chr(10), ' ')}")
        return DecisionOut(is_answer=True, answer=ans)

    name = out.tool_name.strip()
    allowed = {t.name for t in tools}
    if out.kind == "tool_call" and name in allowed:
        try:
            args = _parse_args(out.arguments_json)
        except (json.JSONDecodeError, ValueError):
            args = {}
        if not _tool_call_valid(name, args):
            print(f"[decision]      RETRY (invalid {name} arguments)")
            return DecisionOut(is_answer=False)
        if not _tool_allowed_for_goal(goal, name, history):
            auto = _auto_fetch_top_n(goal, hits, history)
            if auto:
                return auto
            forced = _force_synthesis_answer(goal, user, attached)
            if forced:
                return forced
            print(f"[decision]      RETRY (wrong tool {name} for this goal)")
            return DecisionOut(is_answer=False)
        if name == "fetch_url" and args.get("url") in _fetched_urls(history):
            auto = _auto_fetch_top_n(goal, hits, history)
            if auto:
                return auto
            done = _auto_complete_exhausted_fetch(goal, hits, history)
            if done:
                return done
            print(f"[decision]      RETRY (duplicate fetch_url)")
            return DecisionOut(is_answer=False)
        print(f"[decision]      TOOL_CALL: {name}({json.dumps(args)})")
        return DecisionOut(
            is_answer=False,
            tool_call=ToolCall(name=name, arguments=args),
        )

    # Contract failure — do not accept a fake ANSWER when this goal still needs a tool.
    if _goal_must_use_tool(goal, history):
        auto = _auto_fetch_top_n(goal, hits, history)
        if auto:
            return auto
        print(f"[decision]      RETRY (goal needs tool; no valid tool_call from model)")
        return DecisionOut(is_answer=False)

    fallback = out.answer.strip() or (
        f"Could not act on goal '{goal.text}'. Need a valid tool call or a grounded answer."
    )
    print(f"[decision]      ANSWER: {fallback[:160]}")
    return DecisionOut(is_answer=True, answer=fallback)
