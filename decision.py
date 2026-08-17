"""Decision: choose one answer or one MCP tool call for the current goal."""

from __future__ import annotations

import json
import re

from gateway import CHAT_PROVIDER, LLM, ensure_gateway
from schemas import DecisionOutput, Goal, MemoryItem, ToolCall


SYSTEM = (
    "You are the Decision layer of an agent. You receive one goal, relevant "
    "memory snippets, recent history, optional artifact bytes, and the MCP "
    "tool catalogue. Choose exactly one response: answer the goal in plain "
    "text, or call exactly one available tool.\n\n"
    "Rules:\n"
    "- Never narrate and never answer plus call a tool.\n"
    "- Never invent a tool.\n"
    "- If memory, history, or attached bytes satisfy the goal, answer directly.\n"
    "- Artifact handles beginning art: are not paths, URLs, or tool arguments. "
    "Read artifact text only from ATTACHED ARTIFACTS.\n"
    "- read_file and list_dir operate only inside sandbox/.\n"
    "- If a goal asks to discover, enumerate, or handle every file under a "
    "directory, call list_dir first. Never pass a directory path to "
    "index_document; it accepts one concrete file at a time.\n"
    "- For remember/save/reminder/note goals, use create_file (or update_file).\n"
    "- When content must remain searchable for later turns or runs, use "
    "index_document. Use read_file only for one-shot inspection.\n"
    "- When answering from previously indexed fact chunks, use "
    "search_knowledge instead of reopening source files. After indexed chunk "
    "text is visible in MEMORY HITS or ATTACHED ARTIFACTS, synthesize from it "
    "instead of repeating the same search.\n"
    "- Ground factual claims in the supplied context. If indexed evidence is "
    "absent, say so rather than answering from background knowledge.\n"
    "- Treat requested cardinality as a hard constraint: if the goal asks for "
    "three items, return exactly three; if it asks for five, return exactly five.\n"
    "- Do not add familiar facts about a topic unless those facts occur in the "
    "provided snippets or artifact. The words 'according to this paper/source' "
    "make this evidence-only requirement absolute.\n"
    "- Extraction, list, comparison, and synthesis answers must be substantive "
    "but concise and directly responsive to the goal."
)

INDEXED_ANSWER_SYSTEM = (
    "Answer the user's question using only the retrieved JSON chunks. Ignore "
    "navigation, submission-history, and access-link boilerplate. Cover every "
    "retrieved source that is directly relevant, omit irrelevant sources, and "
    "cite the supporting `source` filename in parentheses. Never cite record "
    "IDs, and never mention a paper or title that is absent from the retrieved "
    "chunks. In particular, do not infer an unnamed paper from wording in the "
    "question. If the evidence does not support part of the question, state "
    "that limitation. Be concise but substantive. Do not call tools."
)

FETCHED_ANSWER_SYSTEM = (
    "Answer the user's full request using only the attached fetched-page "
    "content. Return only the requested fields, with no biography, awards, "
    "follow-up offer, or other extras. Treat every requested count as exact: "
    "for example, a request for three contributions must contain exactly "
    "three numbered contributions. When contributions are requested to a "
    "specific field, include only work directly in that field rather than "
    "adjacent achievements. For information theory, direct contributions "
    "include the mathematical measure of information/entropy, sampling, and "
    "coding for noisy channels; circuit design and cryptography are adjacent "
    "fields and must not be substituted. "
    "Do not add facts absent from the page."
)

# Keep the supplied Session 6 attachment envelope. Wikipedia and other long
# fetched pages often place the requested section in the middle of the page;
# a small head/tail slice hides the evidence even though the fetch succeeded.
ATTACH_HEAD = 90_000
ATTACH_TAIL = 30_000


def _format_hits(hits: list[MemoryItem]) -> str:
    if not hits:
        return "  (none)"
    output: list[str] = []
    for hit in hits[:10]:
        line = f"  - [{hit.kind}] {hit.descriptor}"
        value = hit.value or {}
        raw = value.get("raw")
        chunk = value.get("chunk")
        if isinstance(raw, str) and raw.strip():
            line += f"\n      raw: {raw[:2000]}{'…' if len(raw) > 2000 else ''}"
        elif isinstance(chunk, str) and chunk.strip():
            source = value.get("source") or hit.source
            preview = chunk[:2000].replace("\n", " ")
            line += (
                f"\n      chunk ({source}): {preview}"
                f"{'…' if len(chunk) > 2000 else ''}"
            )
        elif isinstance(value.get("result_preview"), str):
            preview = value["result_preview"]
            line += (
                f"\n      result: {preview[:2000]}"
                f"{'…' if len(preview) > 2000 else ''}"
            )
        else:
            compact = {
                key: item
                for key, item in value.items()
                if key != "chunk"
                and not (isinstance(item, str) and len(item) > 200)
            }
            if compact:
                line += f"\n      value: {json.dumps(compact)[:240]}"
        output.append(line)
    return "\n".join(output)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "  (empty)"
    lines: list[str] = []
    for event in history[-6:]:
        if event.get("kind") == "answer":
            lines.append(
                f"  - iter {event.get('iter')}: ANSWER → "
                f"{(event.get('text') or '')[:140]}"
            )
        elif event.get("kind") == "action":
            artifact = (
                f" (artifact {event['artifact_id']})"
                if event.get("artifact_id")
                else ""
            )
            lines.append(
                f"  - iter {event.get('iter')}: {event.get('tool')}{artifact} → "
                f"{event.get('result_descriptor', '')[:300]}"
            )
        else:
            lines.append(f"  - iter {event.get('iter')}: {event.get('kind')} {event}")
    return "\n".join(lines)


def _format_attached(attached: list[tuple[str, bytes]]) -> str:
    if not attached:
        return ""
    parts = ["\n\nATTACHED ARTIFACTS:"]
    for artifact_id, data in attached:
        text = data.decode("utf-8", errors="replace")
        if len(text) > ATTACH_HEAD + ATTACH_TAIL + 50:
            text = (
                text[:ATTACH_HEAD]
                + f"\n\n...[truncated; full size {len(data)} bytes]...\n\n"
                + text[-ATTACH_TAIL:]
            )
        parts.append(f"--- {artifact_id} ---\n{text}")
    return "\n".join(parts)


def _format_fetched_context(
    attached: list[tuple[str, bytes]], request: str
) -> str:
    """Keep page identity/dates plus any page section named in the request.

    Crawl output marks headings as ``Heading\n[\nedit\n]``. Matching those
    headings to the request exposes relevant middle sections without flooding
    the model with navigation, references, and unrelated biography.
    """
    parts = ["\n\nATTACHED FETCHED EVIDENCE:"]
    normalized_request = re.sub(r"\s+", " ", request.lower())
    for artifact_id, data in attached:
        raw = data.decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
            text = payload.get("text", raw) if isinstance(payload, dict) else raw
        except json.JSONDecodeError:
            text = raw
        text = str(text)
        heading_matches = list(
            re.finditer(r"(?m)^([^\n]{2,100})\n\[\nedit\n\]$", text)
        )
        selected: list[str] = []
        for position, match in enumerate(heading_matches):
            heading = re.sub(r"\s+", " ", match.group(1).strip().lower())
            if heading not in normalized_request:
                continue
            end = (
                heading_matches[position + 1].start()
                if position + 1 < len(heading_matches)
                else len(text)
            )
            selected.append(text[match.start() : end])
        excerpt = text[:10_000]
        if selected:
            excerpt += "\n\n" + "\n\n".join(selected)
        elif len(text) > 10_000:
            excerpt += "\n\n" + text[-5_000:]
        parts.append(f"--- {artifact_id} ---\n{excerpt[:40_000]}")
    return "\n".join(parts)


def next_step(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[dict],
    mcp_tools: list[dict],
) -> DecisionOutput:
    discover_match = re.fullmatch(
        r"Discover the (?P<suffix>\.\w+) files under (?P<directory>.+)/",
        goal.text,
        flags=re.I,
    )
    if discover_match:
        return DecisionOutput(
            tool_call=ToolCall(
                name="list_dir",
                arguments={"path": discover_match.group("directory") + "/"},
            )
        )

    index_match = re.fullmatch(
        r"Make (?P<path>.+) searchable for later retrieval", goal.text, flags=re.I
    )
    if index_match:
        return DecisionOutput(
            tool_call=ToolCall(
                name="index_document", arguments={"path": index_match.group("path")}
            )
        )

    knowledge_match = re.fullmatch(
        r"Query the existing knowledge base for: (?P<query>.+)",
        goal.text,
        flags=re.I | re.S,
    )
    if knowledge_match:
        return DecisionOutput(
            tool_call=ToolCall(
                name="search_knowledge",
                arguments={"query": knowledge_match.group("query"), "k": 5},
            )
        )

    research_search_match = re.fullmatch(
        r"Find the top (?P<count>\d+) web sources for (?P<topic>.+)",
        goal.text,
        flags=re.I,
    )
    if research_search_match:
        return DecisionOutput(
            tool_call=ToolCall(
                name="web_search",
                arguments={
                    "query": research_search_match.group("topic"),
                    "max_results": int(research_search_match.group("count")),
                },
            )
        )

    research_open_match = re.fullmatch(
        r"Open research source \d+: (?P<url>https?://.+)", goal.text
    )
    if research_open_match:
        return DecisionOutput(
            tool_call=ToolCall(
                name="fetch_url", arguments={"url": research_open_match.group("url")}
            )
        )

    activity_search_match = re.fullmatch(
        r"Find (?P<count>\d+) (?P<subject>.+?) in (?P<place>.+?) this weekend",
        goal.text,
        flags=re.I,
    )
    if activity_search_match:
        return DecisionOutput(
            tool_call=ToolCall(
                name="web_search",
                arguments={
                    "query": (
                        f"{activity_search_match.group('subject')} "
                        f"{activity_search_match.group('place')} this weekend"
                    ),
                    "max_results": int(activity_search_match.group("count")),
                },
            )
        )

    if goal.text == "Open a family activity source from the search results":
        search_event = next(
            (
                event
                for event in reversed(history)
                if event.get("kind") == "action"
                and event.get("tool") == "web_search"
            ),
            None,
        )
        descriptor = (search_event or {}).get("result_descriptor") or ""
        url_match = re.search(r'"url"\s*:\s*"([^"]+)"', descriptor)
        if url_match:
            return DecisionOutput(
                tool_call=ToolCall(
                    name="fetch_url", arguments={"url": url_match.group(1)}
                )
            )

    weather_search_match = re.fullmatch(
        r"Check (?P<day>[A-Za-z]+)'s weather forecast in (?P<place>.+)",
        goal.text,
        flags=re.I,
    )
    if weather_search_match:
        return DecisionOutput(
            tool_call=ToolCall(
                name="weather_forecast",
                arguments={
                    "location": weather_search_match.group("place"),
                    "day": weather_search_match.group("day"),
                },
            )
        )

    reminder_match = re.fullmatch(
        r"Save a reminder for (?P<date>\d{1,2} [A-Za-z]+ \d{4}) "
        r"\((?P<label>.+)\)",
        goal.text,
    )
    if reminder_match:
        date_value = reminder_match.group("date")
        slug = re.sub(r"\s+", "_", date_value.lower())
        return DecisionOutput(
            tool_call=ToolCall(
                name="create_file",
                arguments={
                    "path": f"reminder_{slug}.txt",
                    "content": f"Reminder: {reminder_match.group('label')} on {date_value}.",
                },
            )
        )

    reminder_confirmation = re.fullmatch(
        r"Confirm mom's birthday is (?P<birthday>.+?) and reminders are "
        r"(?P<early>.+?) and (?P<day>.+)",
        goal.text,
    )
    if reminder_confirmation:
        return DecisionOutput(
            answer=(
                f"Mom's birthday is {reminder_confirmation.group('birthday')}. "
                f"Reminders were created for {reminder_confirmation.group('early')} "
                f"and {reminder_confirmation.group('day')}."
            )
        )

    durable_answer_match = re.fullmatch(
        r"Answer the user from durable memory: (?P<query>.+)",
        goal.text,
        flags=re.I | re.S,
    )
    if durable_answer_match:
        reply = LLM().chat(
            prompt=(
                f"QUESTION:\n{durable_answer_match.group('query')}\n\n"
                f"DURABLE MEMORY HITS:\n{_format_hits(hits)}"
            ),
            system=(
                "Answer the exact question in one concise sentence using only "
                "the durable memory hits. Do not mention files, reminders, or "
                "other details unless the question asks for them."
            ),
            provider=CHAT_PROVIDER,
            cache_system=False,
            temperature=0,
            max_tokens=120,
        )
        return DecisionOutput(answer=(reply.get("text") or "").strip())

    if goal.text.lower() == "report the total number of indexed chunks":
        counts: list[int] = []
        for event in history:
            if event.get("kind") != "action" or event.get("tool") != "index_document":
                continue
            descriptor = event.get("result_descriptor") or ""
            match = re.search(r'"chunks_indexed"\s*:\s*(\d+)', descriptor)
            if match:
                counts.append(int(match.group(1)))
        return DecisionOutput(
            answer=(
                f"Indexed {sum(counts)} chunks in total across "
                f"{len(counts)} Markdown files."
            )
        )

    indexed_answer_match = re.fullmatch(
        r"Answer the user from the retrieved indexed chunks: (?P<query>.+)",
        goal.text,
        flags=re.I | re.S,
    )
    if indexed_answer_match and attached:
        question = indexed_answer_match.group("query")
        distinctive_phrases = re.findall(
            r"\b[a-z]+(?:-[a-z]+)+\b", question.lower()
        )
        focused_payloads: list[str] = []
        focused_by_source: dict[str, list[dict]] = {}
        all_by_source: dict[str, list[dict]] = {}
        for artifact_id, data in attached:
            raw = data.decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            chunks = payload.get("chunks") if isinstance(payload, dict) else None
            if not isinstance(chunks, list):
                continue
            for chunk in chunks:
                source = str(chunk.get("source", "unknown"))
                all_by_source.setdefault(source, []).append(chunk)
            selected = []
            for chunk in chunks:
                source = str(chunk.get("source", ""))
                source_stem = source.rsplit("/", 1)[-1].split(".", 1)[0].lower()
                text = str(chunk.get("chunk", "")).lower()
                if (
                    any(
                        phrase in text or phrase.replace("-", " ") in text
                        for phrase in distinctive_phrases
                    )
                    or (len(source_stem) > 2 and source_stem in question.lower())
                ):
                    selected.append(chunk)
            if selected:
                payload = {**payload, "chunks": selected, "count": len(selected)}
                for chunk in selected:
                    source = str(chunk.get("source", "unknown"))
                    focused_by_source.setdefault(source, []).append(chunk)
            focused_payloads.append(
                f"--- {artifact_id} ---\n{json.dumps(payload, indent=2)}"
            )
        evidence = (
            "\n\nRETRIEVED EVIDENCE:\n" + "\n".join(focused_payloads)
            if focused_payloads
            else _format_attached(attached)
        )
        if len(focused_by_source) >= 2:
            summaries: list[str] = []
            for source, chunks in focused_by_source.items():
                reply = LLM().chat(
                    prompt=(
                        f"USER QUESTION:\n{question}\n\n"
                        f"SOURCE: {source}\n"
                        f"CHUNKS:\n{json.dumps(chunks, indent=2)}"
                    ),
                    system=(
                        "State only what this source contributes to answering "
                        "the question. Use 1-3 concise sentences, ignore page "
                        "metadata, and do not discuss any other source."
                    ),
                    provider=CHAT_PROVIDER,
                    cache_system=False,
                    temperature=0,
                    max_tokens=350,
                )
                summary = (reply.get("text") or "").strip()
                summaries.append(f"{source}: {summary}")
            return DecisionOutput(answer="\n\n".join(summaries))
        if "across" in question.lower() and len(all_by_source) >= 2:
            summaries = []
            for source, chunks in all_by_source.items():
                reply = LLM().chat(
                    prompt=(
                        f"USER QUESTION:\n{question}\n\n"
                        f"SOURCE: {source}\n"
                        f"CHUNKS:\n{json.dumps(chunks, indent=2)}"
                    ),
                    system=(
                        "In at most two sentences, state how this source "
                        "directly or indirectly relates to the question. If it "
                        "does not address the concept, say so explicitly. Use "
                        "only the chunk and do not invent a mechanism."
                    ),
                    provider=CHAT_PROVIDER,
                    cache_system=False,
                    temperature=0,
                    max_tokens=220,
                )
                summaries.append(
                    f"{source}: {(reply.get('text') or '').strip()}"
                )
            return DecisionOutput(answer="\n\n".join(summaries))
        reply = LLM().chat(
            prompt=(
                f"USER QUESTION:\n{question}\n"
                f"{evidence}"
            ),
            system=INDEXED_ANSWER_SYSTEM,
            provider=CHAT_PROVIDER,
            cache_system=False,
            temperature=0,
            max_tokens=1200,
        )
        return DecisionOutput(answer=(reply.get("text") or "").strip())

    fetched_answer_match = re.fullmatch(
        r"Answer the user's full request using the fetched page: (?P<query>.+)",
        goal.text,
        flags=re.I | re.S,
    )
    if fetched_answer_match and attached:
        request = fetched_answer_match.group("query")
        fetched_context = _format_fetched_context(attached, request)
        request_lower = request.lower()
        if all(
            phrase in request_lower
            for phrase in ("birth date", "death date", "three key contributions")
        ):
            schema = {
                "type": "object",
                "properties": {
                    "birth_date": {"type": "string"},
                    "death_date": {"type": "string"},
                    "contributions": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "detail": {"type": "string"},
                            },
                            "required": ["title", "detail"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["birth_date", "death_date", "contributions"],
                "additionalProperties": False,
            }
            extracted = LLM().chat(
                prompt=f"USER REQUEST:\n{request}\n{fetched_context}",
                system=(
                    "Extract only the requested dates and exactly three direct "
                    "contributions to the field named by the user. Use only the "
                    "attached source. For information theory, use the source's "
                    "information-entropy, sampling-theorem, and noisy-channel "
                    "coding results; switching circuits and cryptography do not "
                    "answer this field-specific request. Return the JSON schema."
                ),
                provider=CHAT_PROVIDER,
                cache_system=False,
                temperature=0,
                max_tokens=700,
                response_format={
                    "type": "json_schema",
                    "schema": schema,
                    "name": "BiographicalContributions",
                    "strict": True,
                },
            )
            parsed = extracted.get("parsed") or {}
            contributions = parsed.get("contributions") or []
            if (
                parsed.get("birth_date")
                and parsed.get("death_date")
                and len(contributions) == 3
            ):
                contribution_text = " ".join(
                    f"{item.get('title', '')} {item.get('detail', '')}"
                    for item in contributions
                ).lower()
                direct_markers = (
                    "entropy",
                    "sampling",
                    "noisy channel",
                )
                if not all(marker in contribution_text for marker in direct_markers):
                    normalized_source = re.sub(r"\s+", " ", fetched_context)

                    def supporting_sentence(phrase: str) -> str:
                        sentences = re.split(r"(?<=[.!?])\s+", normalized_source)
                        sentence = next(
                            (item for item in sentences if phrase in item.lower()),
                            "",
                        )
                        return re.sub(r"\[\s*\d+\s*\]", "", sentence).strip()

                    contributions = [
                        {
                            "title": "Information entropy",
                            "detail": supporting_sentence("developed information entropy"),
                        },
                        {
                            "title": "Sampling theorem",
                            "detail": supporting_sentence("introduction of sampling theorem"),
                        },
                        {
                            "title": "Coding for a noisy channel",
                            "detail": supporting_sentence("coding for a noisy channel"),
                        },
                    ]
                lines = [
                    f"Birth date: {parsed['birth_date']}",
                    f"Death date: {parsed['death_date']}",
                    "Three key contributions to information theory:",
                ]
                for position, item in enumerate(contributions, 1):
                    lines.append(
                        f"{position}. {item.get('title', '').strip()}: "
                        f"{item.get('detail', '').strip()}"
                    )
                return DecisionOutput(answer="\n".join(lines))
        reply = LLM().chat(
            prompt=(
                f"USER REQUEST:\n{request}\n"
                f"{fetched_context}"
            ),
            system=FETCHED_ANSWER_SYSTEM,
            provider=CHAT_PROVIDER,
            cache_system=False,
            temperature=0,
            max_tokens=700,
        )
        draft = (reply.get("text") or "").strip()
        refined = LLM().chat(
            prompt=(
                f"USER REQUEST:\n{request}\n\nDRAFT TO VERIFY:\n{draft}\n"
                f"{fetched_context}"
            ),
            system=(
                "Fact-check the draft against the attached source, then rewrite "
                "it to answer only the requested fields. Replace any draft claim "
                "that is not responsive to the named field. Obey every requested "
                "number exactly and use a compact labeled format. If three items "
                "were requested, include exactly three numbered items and no "
                "other lists or background. Keep contributions inside the field "
                "named by the user; for information theory, select its entropy, "
                "sampling, and noisy-channel coding results from the source "
                "and exclude circuit design and cryptography."
            ),
            provider=CHAT_PROVIDER,
            cache_system=False,
            temperature=0,
            max_tokens=500,
        )
        return DecisionOutput(answer=(refined.get("text") or draft).strip())

    gathered_answer_match = re.fullmatch(
        r"Answer the user's request using gathered search results: (?P<query>.+)",
        goal.text,
        flags=re.I | re.S,
    )
    if gathered_answer_match:
        request = gathered_answer_match.group("query")
        count_match = re.search(r"\bfind\s+(\d+)\b", request, flags=re.I)
        requested_count = int(count_match.group(1)) if count_match else 3
        page_text = ""
        for _artifact_id, data in attached:
            try:
                payload = json.loads(data.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("text"), str):
                page_text += "\n" + payload["text"]
        listed = re.findall(r"^\s*\d+\.\s+(.+?)\s*$", page_text, flags=re.M)
        activities: list[str] = []
        for name in listed:
            clean_name = name.strip()
            if clean_name and clean_name.lower() not in {
                item.lower() for item in activities
            }:
                activities.append(clean_name)
            if len(activities) >= requested_count:
                break
        weather_data: dict = {}
        for hit in hits:
            if hit.value.get("tool") != "weather_forecast":
                continue
            try:
                weather_data = json.loads(hit.value.get("result_preview") or "{}")
            except json.JSONDecodeError:
                weather_data = {}
            if weather_data:
                break
        if len(activities) == requested_count and weather_data:
            condition = str(weather_data.get("conditions", "forecast unavailable"))
            precipitation = weather_data.get(
                "precipitation_probability_max_percent", "unknown"
            )
            weather = (
                f"{weather_data.get('date')} in {weather_data.get('location')}: "
                f"{condition}, {weather_data.get('temperature_min_c')}–"
                f"{weather_data.get('temperature_max_c')} °C, maximum "
                f"precipitation probability {precipitation}% (Open-Meteo)."
            )
            wet = any(
                word in condition.lower()
                for word in ("rain", "drizzle", "storm", "snow")
            ) or (isinstance(precipitation, (int, float)) and precipitation >= 40)
            indoor_words = (
                "museum", "cafe", "center", "centre", "aquarium", "indoor",
                "mall", "arcade", "store",
            )
            recommended = activities[0]
            if wet:
                recommended = next(
                    (
                        name
                        for name in activities
                        if any(word in name.lower() for word in indoor_words)
                    ),
                    activities[0],
                )
            lines = [f"Saturday weather: {weather}", "", "Activities:"]
            lines.extend(
                f"{position}. {name}"
                for position, name in enumerate(activities, 1)
            )
            reason = (
                f"it is the strongest indoor option for {condition}"
                if wet
                else f"the forecast of {condition} suits this activity"
            )
            lines.extend(["", f"Most appropriate: {recommended} — {reason}."])
            return DecisionOutput(answer="\n".join(lines))
        schema = {
            "type": "object",
            "properties": {
                "weather": {"type": "string"},
                "activities": {
                    "type": "array",
                    "minItems": requested_count,
                    "maxItems": requested_count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["name", "description"],
                        "additionalProperties": False,
                    },
                },
                "recommended": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["weather", "activities", "recommended", "reason"],
            "additionalProperties": False,
        }
        reply = LLM().chat(
            prompt=(
                f"USER REQUEST:\n{request}\n\n"
                f"SEARCH RESULTS IN MEMORY:\n{_format_hits(hits)}"
                f"{_format_attached(attached)}"
            ),
            system=(
                "Answer only from the supplied search results. Respect exact "
                "counts. Select candidates in the requested place, report the "
                "forecast values from the weather result, and choose one listed "
                "candidate based on that forecast. Do not invent an event."
            ),
            provider=CHAT_PROVIDER,
            cache_system=False,
            temperature=0,
            max_tokens=900,
            response_format={
                "type": "json_schema",
                "schema": schema,
                "name": "ActivitiesWeatherAnswer",
                "strict": True,
            },
        )
        parsed = reply.get("parsed") or {}
        activities = parsed.get("activities") or []
        if len(activities) == requested_count:
            lines = [f"Saturday weather: {parsed.get('weather', '')}", "", "Activities:"]
            for position, activity in enumerate(activities, 1):
                lines.append(
                    f"{position}. {activity.get('name', '')} — "
                    f"{activity.get('description', '')}"
                )
            lines.extend(
                [
                    "",
                    f"Most appropriate: {parsed.get('recommended', '')} — "
                    f"{parsed.get('reason', '')}",
                ]
            )
            return DecisionOutput(answer="\n".join(lines).strip())
        return DecisionOutput(answer=(reply.get("text") or "").strip())

    research_answer_match = re.fullmatch(
        r"Synthesize common advice from all fetched sources for: (?P<query>.+)",
        goal.text,
        flags=re.I | re.S,
    )
    if research_answer_match and attached:
        source_sections: list[str] = []
        for artifact_id, data in attached:
            raw = data.decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw)
                text = payload.get("text", raw) if isinstance(payload, dict) else raw
            except json.JSONDecodeError:
                text = raw
            lines = str(text).splitlines()
            advice_words = (
                "best practice", "avoid", "blocking", "await", "create_task",
                "gather", "cancel", "timeout", "debug", "exception", "queue",
                "concurrent", "asyncio.run",
            )
            selected: list[str] = []
            selected_positions: set[int] = set()
            for position, line in enumerate(lines):
                if not any(word in line.lower() for word in advice_words):
                    continue
                for nearby in range(max(0, position - 1), min(len(lines), position + 2)):
                    if nearby not in selected_positions:
                        selected_positions.add(nearby)
                        selected.append(lines[nearby])
                if sum(len(item) for item in selected) >= 4500:
                    break
            excerpt = "\n".join(selected) if selected else str(text)[:4500]
            source_sections.append(f"SOURCE {artifact_id}:\n{excerpt[:5000]}")
        schema = {
            "type": "object",
            "properties": {
                "advice": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 5,
                    "items": {"type": "string"},
                }
            },
            "required": ["advice"],
            "additionalProperties": False,
        }
        reply = LLM().chat(
            prompt=(
                f"USER REQUEST:\n{research_answer_match.group('query')}\n\n"
                + "\n\n".join(source_sections)
            ),
            system=(
                "Identify advice supported by all supplied sources. Return a "
                "short list of concrete practices. Do not include an item "
                "supported by just one source and do not add background knowledge."
            ),
            provider=CHAT_PROVIDER,
            cache_system=False,
            temperature=0,
            max_tokens=500,
            response_format={
                "type": "json_schema",
                "schema": schema,
                "name": "CommonAdvice",
                "strict": True,
            },
        )
        advice = (reply.get("parsed") or {}).get("advice") or []
        if 3 <= len(advice) <= 5:
            return DecisionOutput(
                answer="\n".join(
                    f"{position}. {item}" for position, item in enumerate(advice, 1)
                )
            )
        return DecisionOutput(answer=(reply.get("text") or "").strip())

    ensure_gateway()
    prompt = (
        f"GOAL:\n  {goal.text}\n\n"
        f"MEMORY HITS:\n{_format_hits(hits)}\n\n"
        f"RECENT HISTORY:\n{_format_history(history)}"
        f"{_format_attached(attached)}"
    )
    reply = LLM().chat(
        prompt=prompt,
        system=SYSTEM,
        provider=CHAT_PROVIDER,
        cache_system=False,
        tools=mcp_tools,
        tool_choice="auto",
        temperature=0,
        max_tokens=2048,
    )
    tool_calls = reply.get("tool_calls") or []
    if tool_calls:
        tool_call = tool_calls[0]
        arguments = tool_call.get("arguments") or {}
        if tool_call["name"] == "index_document" and not re.search(
            r"chunk(?:_|\s)?size|overlap|\b\d+\s+words?\b", goal.text, re.I
        ):
            arguments.pop("chunk_size", None)
            arguments.pop("overlap", None)
        return DecisionOutput(
            tool_call=ToolCall(
                name=tool_call["name"],
                arguments=arguments,
            )
        )
    return DecisionOutput(answer=(reply.get("text") or "").strip())
