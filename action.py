"""Action layer: pure MCP dispatch plus artifact storage for large results."""

from __future__ import annotations

import json
from typing import Any

from mcp import ClientSession

import artifacts
from schemas import ToolCall

ARTIFACT_THRESHOLD_BYTES = 4096


def _result_to_text(result: Any) -> str:
    if not getattr(result, "content", None):
        return ""
    parts: list[str] = []
    for content in result.content:
        item_text = getattr(content, "text", None)
        parts.append(item_text if item_text is not None else str(content))
    return "\n".join(parts)


async def execute(
    session: ClientSession, tool_call: ToolCall
) -> tuple[str, str | None]:
    for argument_name in ("path", "url"):
        value = tool_call.arguments.get(argument_name)
        if isinstance(value, str) and value.startswith("art:"):
            return (
                f"ERROR: {argument_name}={value!r} is an artifact handle, not a "
                "path or URL. Answer from ATTACHED ARTIFACTS instead.",
                None,
            )

    result = await session.call_tool(tool_call.name, arguments=tool_call.arguments)
    result_text = _result_to_text(result)
    byte_count = len(result_text.encode("utf-8"))
    if byte_count > ARTIFACT_THRESHOLD_BYTES:
        artifact_id = artifacts.put(
            result_text.encode("utf-8"),
            content_type="text/plain",
            source=f"mcp:{tool_call.name}",
            descriptor=(
                f"{tool_call.name}({json.dumps(tool_call.arguments)[:80]}) "
                f"→ {byte_count} bytes"
            ),
        )
        descriptor = (
            f"[artifact {artifact_id}, {byte_count} bytes] preview: "
            + result_text[:240].replace("\n", " ")
            + ("..." if byte_count > 240 else "")
        )
        return descriptor, artifact_id
    return result_text, None
