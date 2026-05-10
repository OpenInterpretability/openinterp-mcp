"""HTTP client wrapper for the Colab backend. Stateless; reads session state on each call."""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from openinterp_mcp.sessions import require_session


class BackendError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"backend {status}: {detail}")
        self.status = status
        self.detail = detail


def _client(timeout: float = 60.0) -> httpx.Client:
    return httpx.Client(timeout=timeout, follow_redirects=True)


def _post(path: str, json_body: Dict[str, Any], session_name: str = "default", timeout: float = 60.0) -> Dict[str, Any]:
    sess = require_session(session_name)
    url = sess.endpoint.rstrip("/") + path
    with _client(timeout) as c:
        r = c.post(url, json=json_body)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise BackendError(r.status_code, detail)
    return r.json()


def _get(path: str, session_name: str = "default", timeout: float = 30.0) -> Dict[str, Any]:
    sess = require_session(session_name)
    url = sess.endpoint.rstrip("/") + path
    with _client(timeout) as c:
        r = c.get(url)
    if r.status_code >= 400:
        raise BackendError(r.status_code, r.text)
    return r.json()


def health(session_name: str = "default") -> Dict[str, Any]:
    return _get("/health", session_name)


def list_probes(session_name: str = "default") -> Dict[str, Any]:
    return _get("/list-probes", session_name)


def capture(prompt: str, layers: list[int], positions: list[str], session_name: str = "default") -> Dict[str, Any]:
    return _post("/capture", {"prompt": prompt, "layers": layers, "positions": positions}, session_name, timeout=120.0)


def probe(probe_id: str, capture_id: str, labels: Optional[list[int]] = None, layer: Optional[int] = None,
          session_name: str = "default") -> Dict[str, Any]:
    body: Dict[str, Any] = {"probe_id": probe_id, "capture_id": capture_id}
    if labels is not None: body["labels"] = labels
    if layer is not None: body["layer"] = layer
    return _post("/probe", body, session_name)


def steer(prompt: str, layer: int, direction_id: str, alpha: float, max_new_tokens: int = 32,
          control_tokens: Optional[list[str]] = None, session_name: str = "default") -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "prompt": prompt, "layer": layer, "direction_id": direction_id,
        "alpha": alpha, "max_new_tokens": max_new_tokens,
    }
    if control_tokens: body["control_tokens"] = control_tokens
    return _post("/steer", body, session_name, timeout=120.0)


def sae_lookup(sae_id: str, capture_id: str, layer: int, top_k: int = 32,
               session_name: str = "default") -> Dict[str, Any]:
    return _post("/sae-lookup", {"sae_id": sae_id, "capture_id": capture_id, "layer": layer, "top_k": top_k},
                 session_name, timeout=180.0)


def causality_protocol(probe_id: str, capture_id: str, labels: list[int],
                       alpha_sweep: Optional[list[float]] = None, n_random_seeds: int = 5,
                       session_name: str = "default") -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "probe_id": probe_id, "capture_id": capture_id, "labels": labels,
        "n_random_seeds": n_random_seeds,
    }
    if alpha_sweep: body["alpha_sweep"] = alpha_sweep
    return _post("/causality-protocol", body, session_name, timeout=300.0)
