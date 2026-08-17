"""Bridge to the supplied LLM Gateway V7.

The assignment keeps all model and embedding calls behind the gateway.
This file is the reference Session 7 bridge with only the local gateway path
adapted to this repository layout.
"""

from __future__ import annotations

import importlib.util as _importlib_util
import os
import subprocess
import time
from pathlib import Path

import httpx

GATEWAY_V7_DIR = Path(__file__).resolve().parent / "llm_gatewayV7"
GATEWAY_URL = "http://localhost:8107"
CHAT_PROVIDER = os.getenv("AGENT_CHAT_PROVIDER", "ollama")


def _is_up() -> bool:
    try:
        response = httpx.get(f"{GATEWAY_URL}/v1/routers", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def ensure_gateway() -> None:
    """Start Gateway V7 when it is not already running."""
    if _is_up():
        return
    if not GATEWAY_V7_DIR.exists():
        raise RuntimeError(
            f"Gateway V7 directory not found at {GATEWAY_V7_DIR}. "
            "Restore the supplied llm_gatewayV7 folder before running the agent."
        )
    print(f"[gateway] launching llm_gatewayV7 from {GATEWAY_V7_DIR}")
    subprocess.Popen(
        ["uv", "run", "main.py"],
        cwd=str(GATEWAY_V7_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(45):
        time.sleep(1)
        if _is_up():
            print(f"[gateway] up on {GATEWAY_URL}")
            return
    raise RuntimeError(
        f"Gateway V7 failed to start within 45 seconds. Check {GATEWAY_V7_DIR}."
    )


_client_path = GATEWAY_V7_DIR / "client.py"
if _client_path.exists():
    _spec = _importlib_util.spec_from_file_location(
        "llm_gatewayV7_client", _client_path
    )
    assert _spec is not None and _spec.loader is not None
    _mod = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    LLM = _mod.LLM
else:
    LLM = None


def embed(text: str, task_type: str = "retrieval_document") -> dict:
    """Compute an embedding locally through Gateway V7 and Ollama.

    Pinning the provider prevents the gateway's configured external fallback
    from receiving corpus text. If local Ollama is unavailable, the call fails
    visibly and Memory uses its documented keyword fallback.
    """
    ensure_gateway()
    if LLM is None:
        raise RuntimeError(
            "Gateway V7 client unavailable. Confirm llm_gatewayV7/client.py exists."
        )
    return LLM().embed(text, task_type=task_type, provider="ollama")


__all__ = [
    "ensure_gateway",
    "LLM",
    "GATEWAY_URL",
    "GATEWAY_V7_DIR",
    "CHAT_PROVIDER",
    "embed",
]
