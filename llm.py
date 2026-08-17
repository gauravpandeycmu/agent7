"""Single door to LLM Gateway V7 (chat + embed).

V7 is V3 wire-compatible for /v1/chat and adds /v1/embed (nomic-embed-text, 768-d).
Runs on port 8107 by default.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TypeVar

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT / "llm_gatewayV7"))
from client import LLM  # noqa: E402

T = TypeVar("T", bound=BaseModel)

GATEWAY_URL = os.getenv("LLM_GATEWAY_V7_URL", "http://localhost:8107")
_llm = LLM(base_url=GATEWAY_URL, timeout=180)


def gateway_up() -> bool:
    try:
        r = httpx.get(f"{GATEWAY_URL}/v1/providers", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def ensure_gateway() -> None:
    """Start llm_gatewayV7 on :8107 if it is not already serving."""
    if gateway_up():
        return
    log = ROOT / "state"
    log.mkdir(parents=True, exist_ok=True)
    out = open(log / "gateway.log", "ab")
    subprocess.Popen(
        ["uv", "run", "--project", str(ROOT), "python", "main.py"],
        cwd=str(ROOT / "llm_gatewayV7"),
        stdout=out,
        stderr=out,
    )
    deadline = time.time() + 40
    while time.time() < deadline:
        if gateway_up():
            return
        time.sleep(0.4)
    raise RuntimeError(
        "LLM Gateway V7 did not start on http://localhost:8107. "
        "Try: uv run python llm_gatewayV7/main.py"
    )


def structured(
    model: type[T],
    prompt: str,
    *,
    system: str,
    provider: str,
    auto_route: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 2048,
) -> T:
    schema = model.model_json_schema()
    use_provider = provider
    temp = temperature if temperature is not None else (1.0 if provider in ("g", "gemini") else 0.2)

    def _call(prov: str) -> T:
        result = _llm.chat(
            prompt,
            system=system,
            provider=prov,
            temperature=temp,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "schema": schema,
                "name": model.__name__,
                "strict": True,
            },
            auto_route=None if prov else auto_route,
        )
        data = result.get("parsed")
        if data is None:
            data = json.loads(result["text"])
        return model.model_validate(data)

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            return _call(use_provider)
        except (ValidationError, json.JSONDecodeError, httpx.HTTPError) as first:
            last_exc = first
            if "502" in str(first) or "503" in str(first) or "429" in str(first):
                time.sleep(2.0 * (attempt + 1))
                continue
            break
    fallback = "g" if use_provider in ("o", "ollama") else None
    if fallback:
        try:
            return _call(fallback)
        except Exception:
            pass
    raise RuntimeError(f"gateway structured({model.__name__}) failed: {last_exc}") from last_exc


def chat_text(
    prompt: str,
    *,
    system: str,
    provider: str = "g",
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """Unstructured chat — used only as a synthesis fallback."""
    ensure_gateway()
    last: Exception | None = None
    for attempt in range(2):
        try:
            result = _llm.chat(
                prompt,
                system=system,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            break
        except httpx.HTTPError as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))
    else:
        raise RuntimeError(f"gateway chat_text failed: {last}") from last
    if isinstance(result, dict):
        text = result.get("text") or ""
        if not text and isinstance(result.get("parsed"), dict):
            text = str(result["parsed"].get("answer") or "")
        return str(text)
    return str(result or "")


def embed(text: str, *, task_type: str = "retrieval_document") -> list[float]:
    """768-d embedding via Gateway V7 /v1/embed (Ollama nomic-embed-text)."""
    text = (text or "").strip() or " "
    ensure_gateway()
    data = _llm.embed(text, task_type=task_type)
    vec = data.get("embedding")
    dim = data.get("dim", 768)
    if not isinstance(vec, list) or len(vec) != dim:
        raise RuntimeError(f"gateway /v1/embed returned invalid vector: {data}")
    return [float(x) for x in vec]
