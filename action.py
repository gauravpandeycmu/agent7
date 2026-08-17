"""Action layer: the only place that talks to MCP.

Decision never executes anything. It returns a ToolCall; this module runs it
on the stdio MCP session and turns the result into:
  - a short descriptor (what Memory stores)
  - an optional artifact_id (what Perception may later attach)

Safety net — artifact-handle guard:
  If Decision passes an integer artifact id (or a string like "3" / "artifact 3")
  as a URL or path, we do NOT call MCP. Those are store handles, not locations.
"""

from __future__ import annotations

import json
from typing import Any

import artifacts
from schemas import ToolCall, ToolSpec

# fetch_url pages larger than this become artifacts instead of inline text.
ARTIFACT_MIN_BYTES = 1500


def _looks_like_handle(value: Any) -> bool:
    if isinstance(value, int) and artifacts.exists(value):
        return True
    if isinstance(value, str):
        s = value.strip().lower()
        if s.startswith("mem:"):
            return True
        if s.startswith("artifact "):
            tail = s.removeprefix("artifact ").strip()
            return tail.isdigit() and artifacts.exists(int(tail))
        if s.isdigit() and artifacts.exists(int(s)):
            return True
    return False


def _flatten_mcp_result(result: Any) -> str:
    """MCP returns a list of content blocks. We want one UTF-8 string."""
    parts: list[str] = []
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    raw = "\n".join(parts).strip()
    if not raw:
        return str(result)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return json.dumps(parsed, indent=2) if not isinstance(parsed, str) else parsed


def _search_descriptor(text: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text[:800]
    if isinstance(data, list):
        lines = [f"[{len(data)} results returned, descriptors recorded]"]
        for i, item in enumerate(data, 1):
            if isinstance(item, dict):
                title = item.get("title") or ""
                url = item.get("url") or ""
                snippet = (item.get("snippet") or "")[:160]
                lines.append(f"{i}. {title} | {url}")
                if snippet:
                    lines.append(f"   {snippet}")
            else:
                lines.append(f"{i}. {item}")
        return "\n".join(lines)
    return text[:800]


async def execute(session: Any, tool_call: ToolCall) -> tuple[str, int | None]:
    """Run one MCP tool. Returns (descriptor, artifact_id | None)."""

    for key, val in tool_call.arguments.items():
        if key not in ("url", "path"):
            continue
        if _looks_like_handle(val):
            msg = (
                f"ERROR: argument '{key}' is an artifact handle ({val}), not a "
                f"URL or filesystem path. Do not pass artifact integer ids to tools. "
                f"Ask Perception to attach the artifact instead."
            )
            print(f"[action]        → {msg}")
            return msg, None

    result = await session.call_tool(tool_call.name, tool_call.arguments)
    is_error = bool(getattr(result, "isError", False))
    text = _flatten_mcp_result(result)
    if is_error:
        desc = f"ERROR from {tool_call.name}: {text[:500]}"
        print(f"[action]        → {desc}")
        return desc, None

    if tool_call.name == "web_search":
        desc = _search_descriptor(text)
        print(f"[action]        → {desc.splitlines()[0]}")
        return desc, None

    if tool_call.name == "index_document":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"raw": text[:400]}
        n = data.get("chunks_indexed", "?")
        p = tool_call.arguments.get("path", data.get("path", "?"))
        desc = f"indexed {p} → {n} chunks"
        print(f"[action]        → {desc}")
        return desc, None

    if tool_call.name == "search_knowledge":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = []
        if isinstance(data, dict):
            chunks = data.get("chunks", [])
        elif isinstance(data, list):
            chunks = data
        else:
            chunks = []
        count = len(chunks)
        desc = f"search_knowledge returned {count} chunks"
        print(f"[action]        → {desc}")
        body = "\n\n---\n\n".join(
            f"[{c.get('descriptor', '')}]\n{c.get('text', '')}"
            for c in chunks
            if isinstance(c, dict) and c.get("text")
        )
        payload = {"count": count, "chunks": chunks}
        if body.strip():
            art_id = artifacts.put(body.encode("utf-8"))
            return json.dumps(payload, indent=2)[:4000], art_id
        return json.dumps(payload, indent=2)[:4000], None

    if tool_call.name == "fetch_url":
        raw = text.encode("utf-8")
        # crawl4ai returns a dict with a 'text' field — unwrap if needed.
        body = text
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "text" in parsed:
                body = str(parsed["text"])
                raw = body.encode("utf-8")
        except json.JSONDecodeError:
            pass
        if len(raw) >= ARTIFACT_MIN_BYTES:
            art_id = artifacts.put(raw)
            preview = body[:240].replace("\n", " ")
            desc = f"[artifact id={art_id}, {len(raw)} bytes] preview: {preview}"
            print(f"[action]        → {desc[:160]}")
            return desc, art_id
        desc = body[:800]
        print(f"[action]        → {desc[:160]}")
        return desc, None

    desc = text[:800]
    print(f"[action]        → {desc[:160]}")
    return desc, None


def mcp_tools_for_decision(mcp_tools: list[Any]) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for t in mcp_tools:
        schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None) or {}
        specs.append(
            ToolSpec(
                name=t.name,
                description=t.description or "",
                input_schema=schema,
            )
        )
    return specs
