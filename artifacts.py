"""Byte store for tool results that are too big to live in Memory.

Memory keeps a short descriptor plus an integer handle (1, 2, 3, …).
When Perception sets attach on a goal, the loop calls get_bytes() and only
then does Decision see the full page.

IDs are assigned sequentially by this module — the LLM never mints them.
Perception picks an attach_index into the numbered memory-hit list; code
maps that to the integer artifact_id stored on the hit.

Files: state/artifacts/<id>.md
Counter: state/artifacts/counter.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
ART_DIR = ROOT / "state" / "artifacts"
COUNTER_PATH = ART_DIR / "counter.json"


def _path(artifact_id: int) -> Path:
    return ART_DIR / f"{artifact_id}.md"


def _next_id() -> int:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    if COUNTER_PATH.exists():
        n = int(json.loads(COUNTER_PATH.read_text(encoding="utf-8"))["next"])
    else:
        # Recover from existing files if counter was lost but blobs remain.
        existing = [
            int(p.stem)
            for p in ART_DIR.glob("*.md")
            if p.stem.isdigit()
        ]
        n = max(existing, default=0) + 1
    COUNTER_PATH.write_text(json.dumps({"next": n + 1}), encoding="utf-8")
    return n


def put(data: bytes | str) -> int:
    artifact_id = _next_id()
    raw = data.encode("utf-8") if isinstance(data, str) else data
    _path(artifact_id).write_bytes(raw)
    return artifact_id


def exists(artifact_id: int | None) -> bool:
    if artifact_id is None or artifact_id < 1:
        return False
    return _path(artifact_id).is_file()


def get_bytes(artifact_id: int) -> bytes:
    return _path(artifact_id).read_bytes()


def size(artifact_id: int) -> int:
    p = _path(artifact_id)
    return p.stat().st_size if p.is_file() else 0
