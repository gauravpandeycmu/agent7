"""Content-addressable storage for large tool results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from schemas import Artifact

STORE = Path(__file__).parent / "state" / "artifacts"
STORE.mkdir(parents=True, exist_ok=True)


def put(blob: bytes, *, content_type: str, source: str, descriptor: str) -> str:
    digest = hashlib.sha256(blob).hexdigest()[:16]
    artifact_id = f"art:{digest}"
    binary_path = STORE / f"{digest}.bin"
    metadata_path = STORE / f"{digest}.json"
    if not binary_path.exists():
        binary_path.write_bytes(blob)
        metadata = Artifact(
            id=artifact_id,
            content_type=content_type,
            size_bytes=len(blob),
            source=source,
            descriptor=descriptor,
        )
        metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return artifact_id


def get_bytes(artifact_id: str) -> bytes:
    digest = artifact_id.removeprefix("art:")
    return (STORE / f"{digest}.bin").read_bytes()


def get_meta(artifact_id: str) -> Artifact:
    digest = artifact_id.removeprefix("art:")
    raw = json.loads((STORE / f"{digest}.json").read_text(encoding="utf-8"))
    return Artifact.model_validate(raw)


def exists(artifact_id: str) -> bool:
    digest = artifact_id.removeprefix("art:")
    return (STORE / f"{digest}.bin").exists()
