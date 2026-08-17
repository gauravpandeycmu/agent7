"""Typed memory with vector-first retrieval and keyword fallback.

This is the supplied Session 7 Memory service: writes for durable kinds are
embedded through Gateway V7 and added to a persisted FAISS index. Scratchpad
items skip embeddings. Reads try FAISS first and use Session 6 keyword overlap
only when the vector path returns no results.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from gateway import CHAT_PROVIDER, LLM, embed as gateway_embed, ensure_gateway
from schemas import MemoryItem, ToolCall, new_id
from vector_index import VectorIndex

STATE_PATH = Path(__file__).parent / "state" / "memory.json"
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
EMBEDDABLE_KINDS = {"fact", "preference", "tool_outcome"}


def _load() -> list[MemoryItem]:
    if not STATE_PATH.exists():
        return []
    raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return [MemoryItem.model_validate(record) for record in raw]


def _save(items: list[MemoryItem]) -> None:
    STATE_PATH.write_text(
        json.dumps([item.model_dump(mode="json") for item in items], indent=2),
        encoding="utf-8",
    )


def _index() -> VectorIndex:
    """Reload every call so agent and MCP subprocess writes stay in sync."""
    index = VectorIndex(STATE_PATH.parent)
    if index.size == 0:
        for item in _load():
            if item.embedding is not None:
                index.add(item.id, item.embedding)
        if index.size:
            index.persist()
    return index


def _try_embed(text: str, task_type: str) -> list[float] | None:
    # A 400-word document chunk can exceed nomic-embed-text's token context
    # when it contains dense URLs or LaTeX. Embed the complete chunk as small
    # windows and mean-pool them; queries and short descriptors remain one call.
    words = text.split()
    parts = (
        [" ".join(words[start : start + 180]) for start in range(0, len(words), 180)]
        if task_type == "retrieval_document" and len(words) > 180
        else [text]
    )
    try:
        vectors = [
            list(gateway_embed(part, task_type=task_type)["embedding"])
            for part in parts
        ]
        if len(vectors) == 1:
            return vectors[0]
        return [sum(values) / len(vectors) for values in zip(*vectors)]
    except Exception as exc:
        print(f"[memory] embedding failed ({exc!r}); item written without vector")
        return None


STOPWORDS = {
    "the", "is", "a", "an", "of", "to", "and", "or", "in", "on", "for",
    "at", "with", "by", "from", "what", "how", "when", "where", "why",
    "this", "that", "it", "be", "as", "are", "was", "were", "i", "you",
    "me", "my", "your",
}


def _tokens(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"\w+", text.lower())
        if word not in STOPWORDS and len(word) > 2
    }


def _keyword_search(
    query: str,
    history: list[dict] | None,
    *,
    kinds: list[str] | None,
    top_k: int,
) -> list[MemoryItem]:
    items = _load()
    if kinds:
        items = [item for item in items if item.kind in kinds]
    query_tokens = _tokens(query)
    for event in (history or [])[-3:]:
        query_tokens |= _tokens(json.dumps(event, default=str))
    scored: list[tuple[int, MemoryItem]] = []
    for item in items:
        item_tokens = {word.lower() for word in item.keywords} | _tokens(
            item.descriptor
        )
        score = len(query_tokens & item_tokens)
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: -pair[0])
    return [item for _, item in scored[:top_k]]


def _vector_search(
    query: str, *, kinds: list[str] | None, top_k: int
) -> list[MemoryItem]:
    query_vector = _try_embed(query, task_type="retrieval_query")
    if query_vector is None:
        return []
    index = _index()
    if index.size == 0:
        return []
    matches = index.search(query_vector, k=top_k * 2 if kinds else top_k)
    by_id = {item.id: item for item in _load()}
    output: list[MemoryItem] = []
    for item_id, _score in matches:
        item = by_id.get(item_id)
        if item is None or (kinds and item.kind not in kinds):
            continue
        output.append(item)
        if len(output) >= top_k:
            break
    return output


def read(
    query: str,
    history: list[dict] | None = None,
    *,
    kinds: list[str] | None = None,
    top_k: int = 8,
) -> list[MemoryItem]:
    """Return FAISS hits, or keyword hits only when FAISS has none."""
    vector_hits = _vector_search(query, kinds=kinds, top_k=top_k)
    if vector_hits:
        return vector_hits
    return _keyword_search(query, history, kinds=kinds, top_k=top_k)


class _Classification(BaseModel):
    kind: str
    descriptor: str
    keywords: list[str] = Field(default_factory=list)
    value: dict = Field(default_factory=dict)


def _persist_item(item: MemoryItem) -> MemoryItem:
    items = _load()
    items.append(item)
    _save(items)
    if item.embedding is not None and item.kind in EMBEDDABLE_KINDS:
        index = _index()
        index.add(item.id, item.embedding)
        index.persist()
    return item


def _fallback_remember(
    raw_text: str, *, source: str, run_id: str, goal_id: str | None
) -> MemoryItem:
    descriptor = raw_text[:200]
    return _persist_item(
        MemoryItem(
            id=new_id("mem"),
            kind="fact",
            keywords=list(_tokens(raw_text))[:10],
            descriptor=descriptor,
            value={"raw": raw_text},
            embedding=_try_embed(descriptor, task_type="retrieval_document"),
            source=source,
            run_id=run_id,
            goal_id=goal_id,
        )
    )


def remember(
    raw_text: str,
    *,
    source: str,
    run_id: str,
    goal_id: str | None = None,
) -> MemoryItem:
    """Classify a free-form write, preserving the raw text as a fallback."""
    ensure_gateway()
    schema = _Classification.model_json_schema()
    try:
        reply = _llm_classify(raw_text, schema)
    except Exception as exc:
        print(f"[memory.remember] classifier failed ({exc!r}); using fact fallback")
        return _fallback_remember(
            raw_text, source=source, run_id=run_id, goal_id=goal_id
        )
    parsed = reply.get("parsed") or {}
    classification = _Classification.model_validate(
        {
            "kind": parsed.get("kind", "fact"),
            "descriptor": parsed.get("descriptor", raw_text[:120]),
            "keywords": parsed.get("keywords") or list(_tokens(raw_text))[:10],
            "value": parsed.get("value") or {"raw": raw_text},
        }
    )
    embedding = None
    if classification.kind in EMBEDDABLE_KINDS:
        embedding = _try_embed(
            classification.descriptor, task_type="retrieval_document"
        )
    return _persist_item(
        MemoryItem(
            id=new_id("mem"),
            kind=classification.kind,  # type: ignore[arg-type]
            keywords=[word.lower() for word in classification.keywords],
            descriptor=classification.descriptor,
            value=classification.value,
            embedding=embedding,
            source=source,
            run_id=run_id,
            goal_id=goal_id,
        )
    )


def _llm_classify(raw_text: str, schema: dict) -> dict:
    return LLM().chat(
        prompt=(
            "Classify this content into a JSON memory record.\n\n"
            f"CONTENT: {raw_text!r}\n\n"
            "Return kind (fact, preference, tool_outcome, or scratchpad), "
            "a short descriptor containing every concrete name/date/number, "
            "3-8 lowercase keywords, and a non-empty value dict. If no better "
            "structure applies, set value.raw to the original content."
        ),
        auto_route="memory",
        provider=CHAT_PROVIDER,
        response_format={
            "type": "json_schema",
            "schema": schema,
            "name": "Classification",
            "strict": True,
        },
        temperature=1.0,
    )


def record_outcome(
    *,
    tool_call: ToolCall,
    result_text: str,
    artifact_id: str | None,
    run_id: str,
    goal_id: str | None,
) -> MemoryItem:
    argument_tokens: list[str] = []
    for value in tool_call.arguments.values():
        if isinstance(value, str):
            argument_tokens += _tokens(value)
        elif isinstance(value, (int, float)):
            argument_tokens.append(str(value))
    descriptor = f"{tool_call.name}({json.dumps(tool_call.arguments)[:80]}) -> "
    descriptor += (
        f"artifact {artifact_id}"
        if artifact_id
        else result_text[:120].replace("\n", " ")
    )
    return _persist_item(
        MemoryItem(
            id=new_id("mem"),
            kind="tool_outcome",
            keywords=list({tool_call.name.lower(), *argument_tokens})[:10],
            descriptor=descriptor,
            value={
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
                "result_preview": result_text[:2000],
            },
            artifact_id=artifact_id,
            embedding=_try_embed(descriptor, task_type="retrieval_document"),
            source="action",
            run_id=run_id,
            goal_id=goal_id,
        )
    )


def add_fact(
    descriptor: str,
    *,
    value: dict | None = None,
    keywords: list[str] | None = None,
    source: str,
    run_id: str,
    goal_id: str | None = None,
) -> MemoryItem:
    """Direct fact write for document ingestion (no classifier call).

    Document indexing must not report success for a fact that has no vector.
    Ordinary agent memories retain their keyword fallback, but an indexed fact
    is useful only when it can participate in semantic retrieval.
    """
    item_value = value or {}
    chunk_text = item_value.get("chunk")
    embedding_text = (
        chunk_text if isinstance(chunk_text, str) and chunk_text.strip() else descriptor
    )
    embedding = _try_embed(embedding_text, task_type="retrieval_document")
    if embedding is None:
        raise RuntimeError(
            "Could not embed an indexed fact; no incomplete index entry was written."
        )
    return _persist_item(
        MemoryItem(
            id=new_id("mem"),
            kind="fact",
            keywords=list(
                {word.lower() for word in (keywords or list(_tokens(descriptor))[:10])}
            ),
            descriptor=descriptor,
            value=item_value,
            embedding=embedding,
            source=source,
            run_id=run_id,
            goal_id=goal_id,
        )
    )


def clear() -> None:
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    VectorIndex(STATE_PATH.parent).clear()
