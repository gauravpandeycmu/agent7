"""Pydantic v2 contracts for every role boundary.

Nothing passes between Memory / Perception / Decision / Action as a raw dict.
Tool *arguments* are a dict because that is what MCP JSON-Schema tools accept;
they still travel inside a ToolCall model, not as a loose mapping.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    """One MCP tool, as Decision is allowed to see it."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Goal(BaseModel):
    id: str
    text: str
    status: Literal["open", "done"] = "open"
    # Position in the numbered memory-hit list (0-based). -1 = attach nothing.
    # The LLM only picks attach_index; code resolves it to attach_artifact_id.
    attach_index: int = -1
    # Integer assigned by artifacts.put() — never LLM-generated.
    attach_artifact_id: Optional[int] = None

    def is_gather(self) -> bool:
        """Tool-pick goals: Decision uses hits/history, not attached page bytes."""
        t = self.text.lower()
        if "retrieve" in t and "synthes" not in t:
            return True
        if any(w in t for w in ("synthes", "extract", "choose", "common advice")):
            return False
        if "compare" in t and "retrieve" not in t:
            return False
        if any(x in t for x in ("top 3", "top three", "fetch the top")):
            return True
        if "search for" in t or (t.startswith("search ") and "synthes" not in t):
            return True
        if "find " in t and "most appropriate" not in t and "choose" not in t:
            return True
        if ("create" in t or "reminder" in t) and "confirm" not in t:
            return True
        if ("weather" in t or "forecast" in t) and "choose" not in t and "most appropriate" not in t:
            return True
        if "check saturday" in t or ("check " in t and "weather" in t):
            return True
        if (
            "fetch" in t
            and "top 3" not in t
            and "top three" not in t
            and "synthes" not in t
            and "extract" not in t
        ):
            return True
        if "index" in t and "indexed" not in t:
            return True
        if t.startswith("list ") or "list " in t and "directory" in t:
            return True
        if "search_knowledge" in t or (
            ("retrieve" in t or "search " in t)
            and "synthes" not in t
            and "compare" not in t
            and "across" not in t
        ):
            return True
        return False


class Observation(BaseModel):
    goals: list[Goal]
    all_done: bool = False

    def next_unfinished(self) -> Goal:
        for g in self.goals:
            if g.status == "open":
                return g
        raise RuntimeError("next_unfinished() called but every goal is done")


class MemoryHit(BaseModel):
    """What Memory returns each iteration: a handle + a short descriptor.

    Full page bodies live in the artifact store, never in these hits.
    """

    handle: str
    kind: Literal["fact", "preference", "outcome", "artifact"]
    descriptor: str
    keywords: list[str] = Field(default_factory=list)
    artifact_id: Optional[int] = None
    value: str = ""


class MemoryRecord(BaseModel):
    """One durable row in state/memory.json."""

    id: str
    kind: Literal["fact", "preference", "outcome", "artifact"]
    value: str
    keywords: list[str] = Field(default_factory=list)
    source: str = ""
    run_id: str = ""
    goal_id: Optional[str] = None
    artifact_id: Optional[int] = None
    embedding: Optional[list[float]] = None
    created_ts: float = 0.0


class AttachedArtifact(BaseModel):
    artifact_id: int
    text: str
    size_bytes: int
    truncated: bool = False


class HistoryEvent(BaseModel):
    iter: int
    kind: Literal["answer", "action"]
    goal_id: str
    text: Optional[str] = None
    tool: Optional[str] = None
    arguments: Optional[dict[str, Any]] = None
    result_descriptor: Optional[str] = None
    artifact_id: Optional[int] = None


class DecisionOut(BaseModel):
    is_answer: bool
    answer: str = ""
    tool_call: Optional[ToolCall] = None


# --- LLM output shapes (gateway response_format / json_schema) ---

class PerceptionGoalOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    status: Literal["open", "done"]
    attach_index: int = -1


class PerceptionLLMOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goals: list[PerceptionGoalOut]
    all_done: bool


class DecisionLLMOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["answer", "tool_call"]
    reasoning_type: Literal[
        "planning",
        "lookup",
        "extraction",
        "synthesis",
        "recall",
        "verification",
    ]
    reasoning: str
    answer: str = ""
    tool_name: str = ""
    arguments_json: str = "{}"


class MemoryClassifyOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    store: bool
    kind: Literal["fact", "preference", "none"]
    value: str = ""
    keywords: list[str] = Field(default_factory=list)
