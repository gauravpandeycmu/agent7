"""FAISS-backed vector index for the Session 7 Memory service.

The index uses inner product over L2-normalized vectors, which is cosine
similarity, and keeps the application memory ids in a parallel JSON list.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore[import-untyped]
except ImportError as exc:
    raise SystemExit("faiss-cpu is required. Run: uv add faiss-cpu") from exc


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


class VectorIndex:
    """Small persistent wrapper around ``faiss.IndexFlatIP``."""

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "index.faiss"
        self.ids_path = self.store_dir / "index_ids.json"
        self._index: faiss.IndexFlatIP | None = None
        self._ids: list[str] = []
        self._dim: int | None = None
        self._load()

    def _load(self) -> None:
        if self.index_path.exists() and self.ids_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            self._ids = json.loads(self.ids_path.read_text(encoding="utf-8"))
            self._dim = self._index.d

    def persist(self) -> None:
        if self._index is None:
            return
        faiss.write_index(self._index, str(self.index_path))
        self.ids_path.write_text(json.dumps(self._ids), encoding="utf-8")

    def clear(self) -> None:
        self._index = None
        self._ids = []
        self._dim = None
        for path in (self.index_path, self.ids_path):
            if path.exists():
                path.unlink()

    def add(self, item_id: str, embedding: list[float]) -> None:
        vector = _l2_normalize(np.array(embedding, dtype=np.float32))
        if self._index is None:
            self._dim = vector.shape[0]
            self._index = faiss.IndexFlatIP(self._dim)
        elif vector.shape[0] != self._dim:
            raise ValueError(
                f"Embedding dim {vector.shape[0]} does not match index dim "
                f"{self._dim}. Rebuild the index after changing embedding models."
            )
        self._index.add(vector.reshape(1, -1))
        self._ids.append(item_id)

    def search(
        self, query_embedding: list[float], k: int = 5
    ) -> list[tuple[str, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        vector = _l2_normalize(np.array(query_embedding, dtype=np.float32))
        scores, positions = self._index.search(
            vector.reshape(1, -1), min(k, self._index.ntotal)
        )
        return [
            (self._ids[position], float(score))
            for score, position in zip(scores[0].tolist(), positions[0].tolist())
            if position >= 0
        ]

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    @property
    def dim(self) -> int | None:
        return self._dim
