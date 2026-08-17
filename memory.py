"""Session 7 Memory: vector-first FAISS read, keyword fallback.

Interface preserved from agent6: remember(), read(), record_outcome().

New:
  index_document(path, chunk_size, overlap) — chunk, embed, persist facts + vectors
  search_knowledge(query, k) — vector search over indexed chunks
  chunk_count() — for Query F reporting

Read path (course spec): embed query → FAISS.search. If any vector hits, return
those. If the index is empty or the gateway is down, fall back to Session 6
keyword overlap. This is NOT hybrid RRF (that waits for a later session).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

import rag
import vector_store
from schemas import HistoryEvent, MemoryClassifyOut, MemoryHit, MemoryRecord, ToolCall
import llm

ROOT = Path(__file__).parent
MEM_PATH = ROOT / "state" / "memory.json"

_STOP = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "is", "it",
    "me", "my", "we", "you", "that", "this", "with", "from", "at", "as", "be",
    "are", "was", "were", "by", "if", "do", "tell", "give", "find", "check",
    "when", "what", "which", "who", "how", "across", "they", "them", "their",
}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP and len(w) >= 2]


def _load() -> list[MemoryRecord]:
    if not MEM_PATH.exists():
        return []
    raw = json.loads(MEM_PATH.read_text(encoding="utf-8"))
    return [MemoryRecord.model_validate(x) for x in raw]


def _save(rows: list[MemoryRecord]) -> None:
    MEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEM_PATH.write_text(
        json.dumps([r.model_dump() for r in rows], indent=2),
        encoding="utf-8",
    )


def _append(row: MemoryRecord) -> None:
    rows = _load()
    rows.append(row)
    _save(rows)


def _row_by_id(mem_id: str) -> MemoryRecord | None:
    for row in _load():
        if row.id == mem_id:
            return row
    return None


def _record_to_hit(row: MemoryRecord, *, score_note: str = "") -> MemoryHit:
    desc = row.value[:400]
    if row.source.startswith("chunk:"):
        desc = row.value.split("\n", 1)[0][:400]
    return MemoryHit(
        handle=row.id,
        kind=row.kind,
        descriptor=desc + score_note,
        keywords=row.keywords,
        artifact_id=row.artifact_id,
        value=row.value,
    )


def _try_embed(text: str, *, task_type: str = "retrieval_document") -> list[float] | None:
    last = None
    for attempt in range(8):
        try:
            return llm.embed(text, task_type=task_type)
        except Exception as exc:
            last = exc
            if attempt < 7:
                time.sleep(2.0 * (attempt + 1))
                continue
            break
    print(f"[memory] embed failed ({task_type}): {last}")
    return None


def _persist_item(row: MemoryRecord) -> MemoryRecord:
    """Append memory.json, then FAISS, then write the index. Reload-from-disk
    on the next read picks up MCP-subprocess writes."""
    if row.embedding:
        vector_store.upsert(row.id, row.embedding)
    _append(row)
    return row


def remember(query: str, source: str, run_id: str) -> MemoryRecord | None:
    system = (
        "You extract durable personal facts from a user message. "
        "Store a fact only if the user stated something that should survive "
        "into a later conversation (birthday, name, preference, deadline). "
        "Research questions, URLs to fetch, and 'search for X' are NOT facts. "
        "keywords: 3-8 lowercase tokens the later search will use."
    )
    out = llm.structured(
        MemoryClassifyOut,
        f"User message:\n{query}",
        system=system,
        provider="g",
        max_tokens=400,
    )
    if not out.store or out.kind == "none" or not out.value.strip():
        return None
    keywords = out.keywords or _tokens(out.value + " " + query)
    row = MemoryRecord(
        id="mem:" + uuid.uuid4().hex[:10],
        kind="fact" if out.kind == "fact" else "preference",
        value=out.value.strip(),
        keywords=[k.lower() for k in keywords],
        source=source,
        run_id=run_id,
        embedding=_try_embed(out.value.strip(), task_type="retrieval_document"),
        created_ts=time.time(),
    )
    _persist_item(row)
    print(f"[memory.remember]  classified as {row.kind}")
    print(f"                   keywords: {row.keywords}")
    return row


def _keyword_hits(query: str, history: list[HistoryEvent], top_k: int) -> list[MemoryHit]:
    rows = _load()
    q_terms = set(_tokens(query))
    for ev in history[-8:]:
        blob = " ".join(
            str(x)
            for x in (ev.text, ev.tool, ev.result_descriptor, ev.artifact_id)
            if x is not None
        )
        q_terms.update(_tokens(blob))

    scored: list[tuple[int, MemoryRecord]] = []
    for row in rows:
        overlap = len(q_terms.intersection(set(row.keywords + _tokens(row.value))))
        if row.kind == "artifact":
            overlap = max(overlap, 1)
        if overlap > 0:
            scored.append((overlap, row))
    scored.sort(key=lambda x: (x[0], x[1].created_ts), reverse=True)

    if any(m in query.lower() for m in ("when is", "what is", "who is", "tell me when")):
        seen = {row.id for _, row in scored}
        for row in rows:
            if row.kind not in ("fact", "preference") or row.id in seen:
                continue
            overlap = len(q_terms.intersection(set(row.keywords + _tokens(row.value))))
            if overlap >= 1:
                scored.append((overlap + 10, row))
        scored.sort(key=lambda x: (x[0], x[1].created_ts), reverse=True)

    return [_record_to_hit(row) for _, row in scored[:top_k]]


def _vector_hits(query: str, top_k: int) -> list[MemoryHit]:
    if vector_store.count_vectors() == 0:
        return []
    q_vec = _try_embed(query, task_type="retrieval_query")
    if not q_vec:
        return []
    hits: list[MemoryHit] = []
    for mem_id, dist in vector_store.search(q_vec, top_k=top_k):
        row = _row_by_id(mem_id)
        if not row:
            continue
        hits.append(_record_to_hit(row, score_note=f" [vector ip={dist:.3f}]"))
    return hits


def read(query: str, history: list[HistoryEvent], *, top_k: int = 8) -> list[MemoryHit]:
    """Vector first. Keyword overlap only when FAISS returns nothing."""
    try:
        vec = _vector_hits(query, top_k=top_k)
    except Exception as exc:
        print(f"[memory.read] vector path failed, keyword fallback: {exc}")
        vec = []
    if vec:
        return vec
    return _keyword_hits(query, history, top_k=top_k)


def record_outcome(
    tool_call: ToolCall,
    result_text: str,
    artifact_id: int | None,
    run_id: str,
    goal_id: str,
) -> MemoryRecord:
    preview = result_text.replace("\n", " ").strip()[:400]
    arg_blob = " ".join(str(v) for v in tool_call.arguments.values())
    keywords = _tokens(f"{tool_call.name} {arg_blob} {preview}")
    kind: str = "artifact" if artifact_id else "outcome"
    value = preview
    if tool_call.name == "web_search":
        value = result_text[:4000]
    if artifact_id is not None:
        value = f"[artifact {artifact_id}] {preview}"
    embedding = None
    if not preview.startswith("ERROR"):
        embedding = _try_embed(preview, task_type="retrieval_document")
    row = MemoryRecord(
        id="mem:" + uuid.uuid4().hex[:10],
        kind=kind,  # type: ignore[arg-type]
        value=value,
        keywords=keywords,
        source=f"tool:{tool_call.name}",
        run_id=run_id,
        goal_id=goal_id,
        artifact_id=artifact_id,
        embedding=embedding,
        created_ts=time.time(),
    )
    return _persist_item(row)


def index_document(
    rel_path: str,
    *,
    chunk_size: int = 400,
    overlap: int = 80,
    run_id: str = "",
    goal_id: str | None = None,
) -> dict:
    """Read sandbox file or artifact, chunk, embed, persist to memory + FAISS."""
    if rel_path.startswith("art:"):
        import artifacts

        art_id = int(rel_path.split(":", 1)[1].strip())
        text = artifacts.get_bytes(art_id).decode("utf-8", errors="replace")
        source_label = rel_path
    else:
        text = rag.read_document(rel_path)
        source_label = rel_path
    chunks = rag.chunk_text(text, chunk_words=chunk_size, overlap=overlap)
    rows: list[MemoryRecord] = []
    for i, chunk in enumerate(chunks):
        descriptor = rag.chunk_descriptor(source_label, i, len(chunks))
        body = f"{descriptor}\n{chunk}"
        embedding = _try_embed(chunk, task_type="retrieval_document")
        if embedding is None:
            time.sleep(4)
            embedding = _try_embed(chunk, task_type="retrieval_document")
        row = MemoryRecord(
            id=rag.new_chunk_id(),
            kind="fact",
            value=body,
            keywords=_tokens(rel_path + " " + chunk[:200]),
            source=f"chunk:{rel_path}",
            run_id=run_id,
            goal_id=goal_id,
            embedding=embedding,
            created_ts=time.time(),
        )
        _persist_item(row)
        rows.append(row)
        time.sleep(0.35)
    return {
        "ok": True,
        "path": rel_path,
        "chunks_indexed": len(rows),
        "chunk_ids": [r.id for r in rows],
    }


def search_knowledge(query: str, *, k: int = 5, top_k: int | None = None) -> dict:
    n = top_k if top_k is not None else k
    hits = [h for h in _vector_hits(query, top_k=n * 2) if h.kind == "fact"][:n]
    chunks = [
        {
            "mem_id": h.handle,
            "descriptor": h.descriptor,
            "text": h.value[:1200],
            "source": "fact",
        }
        for h in hits
    ]
    return {"query": query, "count": len(chunks), "chunks": chunks}


def chunk_count() -> int:
    return sum(1 for r in _load() if r.source.startswith("chunk:"))


def reset_index() -> None:
    vector_store.reset()


def clear_chunks() -> int:
    """Drop all indexed document chunks from memory.json and FAISS."""
    rows = _load()
    kept = [r for r in rows if not r.source.startswith("chunk:")]
    removed = len(rows) - len(kept)
    _save(kept)
    reset_index()
    return removed
