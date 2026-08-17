"""FAISS-backed vector index persisted under state/.

Session 7 layout:
  state/index.faiss      — IndexFlatIP, dim=768, L2-normalized (cosine = inner product)
  state/index_ids.json   — mem id strings in insertion order (parallel to FAISS rows)

Reloaded from disk on every call so MCP-subprocess writes are visible to the agent.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).parent
STATE = ROOT / "state"
INDEX_PATH = STATE / "index.faiss"
IDS_PATH = STATE / "index_ids.json"
EMBED_DIM = 768


def _load_ids() -> list[str]:
    if not IDS_PATH.exists():
        return []
    return json.loads(IDS_PATH.read_text(encoding="utf-8"))


def _save_ids(ids: list[str]) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    IDS_PATH.write_text(json.dumps(ids, indent=2), encoding="utf-8")


def _as_normed(vector: list[float]) -> np.ndarray:
    vec = np.array([vector], dtype=np.float32)
    if vec.shape[1] != EMBED_DIM:
        raise ValueError(f"expected dim {EMBED_DIM}, got {vec.shape[1]}")
    faiss.normalize_L2(vec)
    return vec


def _load_index() -> faiss.IndexFlatIP:
    if INDEX_PATH.exists():
        return faiss.read_index(str(INDEX_PATH))
    return faiss.IndexFlatIP(EMBED_DIM)


def _save_index(index: faiss.IndexFlatIP) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))


def count_vectors() -> int:
    if not INDEX_PATH.exists():
        return 0
    return _load_index().ntotal


def upsert(mem_id: str, vector: list[float]) -> None:
    """Append one L2-normalized vector. Duplicate mem_ids are ignored."""
    ids = _load_ids()
    if mem_id in ids:
        return
    vec = _as_normed(vector)
    index = _load_index()
    index.add(vec)
    ids.append(mem_id)
    _save_index(index)
    _save_ids(ids)


def search(vector: list[float], top_k: int = 8) -> list[tuple[str, float]]:
    """Return (mem_id, inner-product score) pairs best-first."""
    if not INDEX_PATH.exists() or count_vectors() == 0:
        return []
    index = _load_index()
    ids = _load_ids()
    q = _as_normed(vector)
    k = min(top_k, index.ntotal)
    scores, indices = index.search(q, k)
    out: list[tuple[str, float]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(ids):
            continue
        out.append((ids[int(idx)], float(score)))
    return out


def reset() -> None:
    for p in (INDEX_PATH, IDS_PATH):
        if p.exists():
            p.unlink()
