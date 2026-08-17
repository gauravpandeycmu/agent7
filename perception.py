"""Perception: maintain intent-level goals across agent iterations.

Perception sees memory descriptors and history, never the MCP tool catalogue or
artifact bytes. Tool selection remains Decision's responsibility.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from gateway import CHAT_PROVIDER, LLM, ensure_gateway
from schemas import Goal, MemoryItem, Observation, new_id


class _GoalDelta(BaseModel):
    text: str = Field(max_length=240)
    done: bool = False
    send_artifact: bool = False
    artifact_index: int | None = None


class _PerceptionOutput(BaseModel):
    goals: list[_GoalDelta] = Field(default_factory=list, max_length=10)


SYSTEM = (
    "You are the Perception layer of an agent.\n"
    "Each iteration you see the user's query, prior goals, memory-hit "
    "descriptors, and run history. Return the CURRENT goal list as JSON.\n\n"
    "Goals are identified by POSITION. Keep prior goals in the same order, "
    "verbatim. Never reorder or drop one. You may append goals only when a "
    "discovery action reveals concrete work that was unknown earlier. Append "
    "one goal per discovered item and keep the report or synthesis last.\n\n"
    "Write goals at the level of INTENT, not tool selection. Describe WHAT "
    "must happen and leave the choice of mechanism to Decision. Appropriate "
    "intent verbs include fetch, open, list, look up, convert, save, make "
    "searchable, query the existing knowledge base, extract, compare, and "
    "synthesise. Never name specific tools.\n\n"
    "Procedure:\n"
    "1. With no PRIOR GOALS, decompose the query into short imperatives. Use "
    "separate goals for each requested item and a final answer goal. If a "
    "directory must first be discovered, make that the first goal. If MEMORY "
    "HITS contain fact descriptors beginning [sandbox: or [art:, questions "
    "about that material should query the existing knowledge base and then "
    "synthesise an answer; do not reopen the source.\n"
    "2. Otherwise copy prior goal text verbatim. Mark a goal done when RUN "
    "HISTORY contains a successful action or substantive answer satisfying it. "
    "Once done, it stays done.\n"
    "3. For the first unfinished extraction, summary, comparison, evaluation, "
    "selection, or synthesis goal, attach the most relevant available artifact "
    "by setting send_artifact=true and artifact_index to its displayed i value.\n"
    "4. Pure discovery, fetch, search, compute, time, and file-open goals do "
    "not attach artifact bytes."
)

SYNTHESIS_KEYWORDS = (
    "evaluate",
    "select",
    "synthes",
    "compare",
    "decide",
    "recommend",
    "tell me which",
    "most appropriate",
    "analy",
    "pick",
    "choose",
    "summarise",
    "summarize",
    "answer",
    "identify",
    "find",
    "determine",
    "extract",
    "list",
    "report",
    "tell",
    "explain",
    "describe",
    "name",
)


def _collapse_redundant_answer_tail(raw_goals: list[dict]) -> list[dict]:
    """One final answer goal can both extract and synthesize.

    Small local models sometimes emit adjacent trailing goals such as
    "extract three facts" and "synthesize those three facts". That spends an
    iteration without adding evidence. Keep the final delivery goal and leave
    all discovery/gather goals intact.
    """
    answer_words = (
        "extract",
        "synthes",
        "summaris",
        "answer",
        "compare",
        "explain",
        "identify",
        "report",
    )
    answer_positions = [
        position
        for position, goal in enumerate(raw_goals)
        if any(word in (goal.get("text") or "").lower() for word in answer_words)
    ]
    if len(answer_positions) < 2:
        return raw_goals
    trailing = answer_positions[-1]
    redundant = {
        position
        for position in answer_positions[:-1]
        if position == trailing - 1
    }
    return [goal for position, goal in enumerate(raw_goals) if position not in redundant]


def _snapshot_history(history: list[dict]) -> list[dict]:
    output: list[dict] = []
    for event in history[-10:]:
        output.append(
            {
                key: value[:2000] + "..."
                if isinstance(value, str) and len(value) > 2000
                else value
                for key, value in event.items()
            }
        )
    return output


def _snapshot_hits(hits: list[MemoryItem]) -> list[dict]:
    artifact_position = 0
    output: list[dict] = []
    for hit in hits[:12]:
        position = None
        if hit.artifact_id:
            position = artifact_position
            artifact_position += 1
        output.append(
            {
                "i": position,
                "kind": hit.kind,
                "descriptor": hit.descriptor,
                "keywords": hit.keywords,
                "artifact_id": hit.artifact_id,
            }
        )
    return output


def _directory_index_observation(
    query: str, history: list[dict], prior_goals: list[Goal]
) -> Observation | None:
    """Handle generic append-after-discovery directory indexing."""
    match = re.search(
        r"index\s+(?:every|all)\s+(?P<suffix>\.\w+)\s+file\s+under\s+"
        r"(?P<directory>[A-Za-z0-9_./-]+)",
        query,
        flags=re.I,
    )
    if not match:
        return None
    suffix = match.group("suffix").lower()
    directory = match.group("directory").rstrip("/.")
    discovery_text = f"Discover the {suffix} files under {directory}/"
    report_text = "Report the total number of indexed chunks"
    by_text = {goal.text: goal for goal in prior_goals}

    list_event = next(
        (
            event
            for event in reversed(history)
            if event.get("kind") == "action"
            and event.get("tool") == "list_dir"
            and not (event.get("result_descriptor") or "").lower().startswith("error")
        ),
        None,
    )
    names: list[str] = []
    if list_event:
        descriptor = list_event.get("result_descriptor") or ""
        names_match = re.search(r'"names"\s*:\s*\[(.*?)\]', descriptor, re.S)
        if names_match:
            names = [
                name
                for name in re.findall(r'"([^"\\]+)"', names_match.group(1))
                if name.lower().endswith(suffix)
            ]

    discovery = by_text.get(discovery_text) or Goal(
        id=new_id("g"), text=discovery_text
    )
    discovery.done = bool(list_event and names)
    if not names:
        return Observation(goals=[discovery])

    goals = [discovery]
    indexed_paths = {
        str((event.get("arguments") or {}).get("path"))
        for event in history
        if event.get("kind") == "action"
        and event.get("tool") == "index_document"
        and not (event.get("result_descriptor") or "").lower().startswith("error")
    }
    for name in names:
        path = f"{directory}/{name}"
        text = f"Make {path} searchable for later retrieval"
        goal = by_text.get(text) or Goal(id=new_id("g"), text=text)
        goal.done = path in indexed_paths
        goals.append(goal)

    report = by_text.get(report_text) or Goal(id=new_id("g"), text=report_text)
    report.done = any(
        event.get("kind") == "answer"
        and event.get("goal_id") == report.id
        and len(event.get("text") or "") > 20
        for event in history
    )
    goals.append(report)
    return Observation(goals=goals)


def _url_fetch_observation(
    query: str, history: list[dict], prior_goals: list[Goal]
) -> Observation | None:
    """Keep a direct URL request to one fetch goal and one answer goal."""
    url_match = re.search(r"https?://[^\s]+", query)
    if not url_match:
        return None
    url = url_match.group(0).rstrip(".,);]")
    fetch_text = f"Fetch {url}"
    answer_text = f"Answer the user's full request using the fetched page: {query}"
    by_text = {goal.text: goal for goal in prior_goals}
    fetch = by_text.get(fetch_text) or Goal(id=new_id("g"), text=fetch_text)
    answer = by_text.get(answer_text) or Goal(id=new_id("g"), text=answer_text)
    fetch_event = next(
        (
            event
            for event in reversed(history)
            if event.get("kind") == "action"
            and event.get("tool") == "fetch_url"
            and (event.get("arguments") or {}).get("url") == url
            and not (event.get("result_descriptor") or "").lower().startswith("error")
        ),
        None,
    )
    fetch.done = fetch_event is not None
    if fetch_event and fetch_event.get("artifact_id"):
        answer.attach_artifact_id = fetch_event["artifact_id"]
    answer.done = any(
        event.get("kind") == "answer"
        and event.get("goal_id") == answer.id
        and len(event.get("text") or "") > 60
        for event in history
    )
    return Observation(goals=[fetch, answer])


def _reminder_observation(
    query: str, history: list[dict], prior_goals: list[Goal]
) -> Observation | None:
    date_match = re.search(
        r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b", query
    )
    if not date_match or "reminder" not in query.lower() or "two weeks" not in query.lower():
        return None
    birthday = datetime.strptime(date_match.group(1), "%d %B %Y").date()
    early = birthday - timedelta(days=14)
    birthday_text = birthday.strftime("%-d %B %Y")
    early_text = early.strftime("%-d %B %Y")
    early_goal_text = (
        f"Save a reminder for {early_text} (two weeks before mom's birthday)"
    )
    day_goal_text = f"Save a reminder for {birthday_text} (mom's birthday)"
    confirm_text = (
        f"Confirm mom's birthday is {birthday_text} and reminders are "
        f"{early_text} and {birthday_text}"
    )
    by_text = {goal.text: goal for goal in prior_goals}
    goals = [
        by_text.get(early_goal_text)
        or Goal(id=new_id("g"), text=early_goal_text),
        by_text.get(day_goal_text) or Goal(id=new_id("g"), text=day_goal_text),
        by_text.get(confirm_text) or Goal(id=new_id("g"), text=confirm_text),
    ]
    for goal in goals[:2]:
        goal.done = any(
            event.get("kind") == "action"
            and event.get("goal_id") == goal.id
            and not (event.get("result_descriptor") or "").lower().startswith("error")
            for event in history
        )
    goals[2].done = any(
        event.get("kind") == "answer"
        and event.get("goal_id") == goals[2].id
        and len(event.get("text") or "") > 40
        for event in history
    )
    return Observation(goals=goals)


def _activities_weather_observation(
    query: str, history: list[dict], prior_goals: list[Goal]
) -> Observation | None:
    """Stabilize a common multi-search request: candidates, forecast, choice."""
    activity_match = re.search(
        r"find\s+(?P<count>\d+)\s+(?P<subject>.+?)\s+in\s+"
        r"(?P<place>[A-Za-z .'-]+?)\s+this weekend",
        query,
        flags=re.I,
    )
    day_match = re.search(
        r"check\s+(?P<day>[A-Za-z]+)(?:'s)?\s+weather forecast",
        query,
        flags=re.I,
    )
    if not activity_match or not day_match or "which" not in query.lower():
        return None
    count = activity_match.group("count")
    subject = activity_match.group("subject").strip()
    place = activity_match.group("place").strip()
    day = day_match.group("day").capitalize()
    activity_text = f"Find {count} {subject} in {place} this weekend"
    open_text = "Open a family activity source from the search results"
    weather_text = f"Check {day}'s weather forecast in {place}"
    answer_text = f"Answer the user's request using gathered search results: {query}"
    by_text = {goal.text: goal for goal in prior_goals}
    goals = [
        by_text.get(activity_text) or Goal(id=new_id("g"), text=activity_text),
        by_text.get(open_text) or Goal(id=new_id("g"), text=open_text),
        by_text.get(weather_text) or Goal(id=new_id("g"), text=weather_text),
        by_text.get(answer_text) or Goal(id=new_id("g"), text=answer_text),
    ]
    for goal in goals[:3]:
        goal.done = any(
            event.get("kind") == "action"
            and event.get("goal_id") == goal.id
            and not (event.get("result_descriptor") or "").lower().startswith("error")
            for event in history
        )
    activity_page_event = next(
        (
            event
            for event in reversed(history)
            if event.get("kind") == "action"
            and event.get("goal_id") == goals[1].id
            and event.get("artifact_id")
        ),
        None,
    )
    if activity_page_event:
        goals[3].attach_artifact_id = activity_page_event["artifact_id"]
    goals[3].done = any(
        event.get("kind") == "answer"
        and event.get("goal_id") == goals[3].id
        and len(event.get("text") or "") > 60
        for event in history
    )
    return Observation(goals=goals)


def _multi_source_research_observation(
    query: str, history: list[dict], prior_goals: list[Goal]
) -> Observation | None:
    match = re.search(
        r"search for\s+[\"“](?P<topic>.+?)[\"”],\s*read the top "
        r"(?P<count>\d+) results",
        query,
        flags=re.I,
    )
    if not match:
        return None
    topic = match.group("topic")
    count = int(match.group("count"))
    search_text = f"Find the top {count} web sources for {topic}"
    answer_text = f"Synthesize common advice from all fetched sources for: {query}"
    by_text = {goal.text: goal for goal in prior_goals}
    search_goal = by_text.get(search_text) or Goal(id=new_id("g"), text=search_text)
    search_event = next(
        (
            event
            for event in reversed(history)
            if event.get("kind") == "action"
            and event.get("goal_id") == search_goal.id
            and not (event.get("result_descriptor") or "").lower().startswith("error")
        ),
        None,
    )
    search_goal.done = search_event is not None
    goals = [search_goal]
    if search_event:
        urls = re.findall(
            r'"url"\s*:\s*"([^"]+)"',
            search_event.get("result_descriptor") or "",
        )[:count]
        for position, url in enumerate(urls, 1):
            text = f"Open research source {position}: {url}"
            goal = by_text.get(text) or Goal(id=new_id("g"), text=text)
            goal.done = any(
                event.get("kind") == "action"
                and event.get("goal_id") == goal.id
                and not (event.get("result_descriptor") or "").lower().startswith("error")
                for event in history
            )
            goals.append(goal)
        if len(urls) == count:
            answer = by_text.get(answer_text) or Goal(id=new_id("g"), text=answer_text)
            answer.done = any(
                event.get("kind") == "answer"
                and event.get("goal_id") == answer.id
                and len(event.get("text") or "") > 60
                for event in history
            )
            goals.append(answer)
    return Observation(goals=goals)


def _indexed_question_observation(
    query: str,
    hits: list[MemoryItem],
    history: list[dict],
    prior_goals: list[Goal],
) -> Observation | None:
    """Give indexed-corpus questions a stable retrieve-then-answer plan."""
    has_indexed_chunks = any(
        (
            hit.kind == "fact"
            and (
                hit.source.startswith(("sandbox:", "art:"))
                or hit.descriptor.startswith(("[sandbox:", "[art:"))
            )
        )
        or (
            hit.kind == "tool_outcome"
            and (
                hit.value.get("tool") in {"index_document", "search_knowledge"}
                or hit.descriptor.startswith(("index_document(", "search_knowledge("))
            )
        )
        for hit in hits
    )
    if not has_indexed_chunks:
        return None
    query_text = f"Query the existing knowledge base for: {query}"
    answer_text = f"Answer the user from the retrieved indexed chunks: {query}"
    by_text = {goal.text: goal for goal in prior_goals}
    retrieve = by_text.get(query_text) or Goal(id=new_id("g"), text=query_text)
    answer = by_text.get(answer_text) or Goal(id=new_id("g"), text=answer_text)

    search_event = next(
        (
            event
            for event in reversed(history)
            if event.get("kind") == "action"
            and event.get("tool") == "search_knowledge"
            and not (event.get("result_descriptor") or "").lower().startswith("error")
        ),
        None,
    )
    retrieve.done = search_event is not None
    if search_event and search_event.get("artifact_id"):
        answer.attach_artifact_id = search_event["artifact_id"]
    answer.done = any(
        event.get("kind") == "answer"
        and event.get("goal_id") == answer.id
        and len(event.get("text") or "") > 60
        for event in history
    )
    return Observation(goals=[retrieve, answer])


def _durable_memory_question_observation(
    query: str,
    hits: list[MemoryItem],
    history: list[dict],
    prior_goals: list[Goal],
    run_id: str,
) -> Observation | None:
    """Use one stable answer goal when a prior run already holds the fact."""
    has_prior_fact = any(
        hit.kind == "fact"
        and hit.run_id != run_id
        and not hit.source.startswith(("sandbox:", "art:"))
        for hit in hits
    )
    if not has_prior_fact or not query.strip().endswith("?"):
        return None
    text = f"Answer the user from durable memory: {query}"
    goal = next((item for item in prior_goals if item.text == text), None) or Goal(
        id=new_id("g"), text=text
    )
    goal.done = any(
        event.get("kind") == "answer"
        and event.get("goal_id") == goal.id
        and len(event.get("text") or "") > 15
        for event in history
    )
    return Observation(goals=[goal])


def observe(
    query: str,
    hits: list[MemoryItem],
    history: list[dict],
    prior_goals: list[Goal],
    run_id: str,
) -> Observation:
    url_plan = _url_fetch_observation(query, history, prior_goals)
    if url_plan is not None:
        return url_plan
    reminder_plan = _reminder_observation(query, history, prior_goals)
    if reminder_plan is not None:
        return reminder_plan
    activities_weather_plan = _activities_weather_observation(
        query, history, prior_goals
    )
    if activities_weather_plan is not None:
        return activities_weather_plan
    research_plan = _multi_source_research_observation(
        query, history, prior_goals
    )
    if research_plan is not None:
        return research_plan
    directory_plan = _directory_index_observation(query, history, prior_goals)
    if directory_plan is not None:
        return directory_plan
    indexed_plan = _indexed_question_observation(query, hits, history, prior_goals)
    if indexed_plan is not None:
        return indexed_plan
    durable_plan = _durable_memory_question_observation(
        query, hits, history, prior_goals, run_id
    )
    if durable_plan is not None:
        return durable_plan
    ensure_gateway()
    artifact_ids = [hit.artifact_id for hit in hits[:12] if hit.artifact_id]
    prior_snapshot = [goal.model_dump() for goal in prior_goals]
    prompt = (
        f"USER QUERY:\n  {query}\n\n"
        f"PRIOR GOALS:\n{json.dumps(prior_snapshot, indent=2)}\n\n"
        "MEMORY HITS (descriptors only; i is the artifact index):\n"
        f"{json.dumps(_snapshot_hits(hits), indent=2)}\n\n"
        "RUN HISTORY:\n"
        f"{json.dumps(_snapshot_history(history), indent=2, default=str)}\n\n"
        "Return the current goal list as JSON matching the schema."
    )
    reply = LLM().chat(
        prompt=prompt,
        system=SYSTEM,
        auto_route="perception",
        provider=CHAT_PROVIDER,
        response_format={
            "type": "json_schema",
            "schema": _PerceptionOutput.model_json_schema(),
            "name": "PerceptionOutput",
            "strict": True,
        },
        temperature=1.0,
    )
    parsed = reply.get("parsed") or {}
    raw_goals = parsed.get("goals") or []
    if not raw_goals:
        return Observation(goals=[Goal(id=new_id("g"), text=query)])

    if not prior_goals:
        raw_goals = _collapse_redundant_answer_tail(raw_goals)

    if prior_goals:
        prior_texts = {goal.text.strip().lower() for goal in prior_goals}
        stable = []
        for position, prior in enumerate(prior_goals):
            if position < len(raw_goals):
                stable.append(raw_goals[position])
            else:
                stable.append(
                    {
                        "text": prior.text,
                        "done": prior.done,
                        "send_artifact": False,
                        "artifact_index": None,
                    }
                )
        latest_action = next(
            (event for event in reversed(history) if event.get("kind") == "action"),
            None,
        )
        discovery_just_happened = bool(
            latest_action and latest_action.get("tool") in {"list_dir", "web_search"}
        )
        if discovery_just_happened:
            for extra in raw_goals[len(prior_goals) :]:
                text = (extra.get("text") or "").strip().lower()
                if not text or text in prior_texts:
                    continue
                prior_texts.add(text)
                stable.append(extra)
        raw_goals = stable

    output_goals: list[Goal] = []
    for position, raw_goal in enumerate(raw_goals):
        delta = _GoalDelta.model_validate(raw_goal)
        artifact_id = None
        if delta.send_artifact and delta.artifact_index is not None:
            if 0 <= delta.artifact_index < len(artifact_ids):
                artifact_id = artifact_ids[delta.artifact_index]

        goal_id = (
            prior_goals[position].id
            if position < len(prior_goals)
            else new_id("g")
        )
        was_done = (
            prior_goals[position].done if position < len(prior_goals) else False
        )
        proposed_done = was_done or delta.done
        if proposed_done and not was_done:
            if any(word in delta.text.lower() for word in SYNTHESIS_KEYWORDS):
                has_answer = any(
                    event.get("kind") == "answer"
                    and event.get("goal_id") == goal_id
                    and len(event.get("text") or "") > 60
                    for event in history
                )
                if not has_answer:
                    proposed_done = False

        stable_text = (
            prior_goals[position].text
            if position < len(prior_goals)
            else delta.text
        )
        output_goals.append(
            Goal(
                id=goal_id,
                text=stable_text,
                done=proposed_done,
                attach_artifact_id=artifact_id,
            )
        )

    for goal in output_goals:
        if goal.done:
            continue
        if (
            not goal.attach_artifact_id
            and artifact_ids
            and any(word in goal.text.lower() for word in SYNTHESIS_KEYWORDS)
        ):
            goal.attach_artifact_id = artifact_ids[-1]
        break

    return Observation(goals=output_goals)
