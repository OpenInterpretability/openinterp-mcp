"""Phase 1 smoke tests: package imports, app boots, /health responds, stubs return 501.

Does NOT load a model or open ngrok — those need GPU + auth and are tested manually on Colab.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from openinterp_mcp.colab.endpoints import router
from openinterp_mcp.colab.manifest import build_manifest
from openinterp_mcp.colab.state import STATE


def _make_client():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def setup_function():
    STATE.reset()


def test_package_version_present():
    import openinterp_mcp

    assert openinterp_mcp.__version__


def test_health_endpoint():
    client = _make_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["model_id"] is None
    assert body["probes_loaded"] == 0


def test_list_probes_empty():
    client = _make_client()
    resp = client.get("/list-probes")
    assert resp.status_code == 200
    assert resp.json()["probes"] == []


def test_capture_503_without_model():
    client = _make_client()
    resp = client.post(
        "/capture",
        json={"prompt": "hi", "layers": [10], "positions": ["end_question"]},
    )
    assert resp.status_code == 503


def test_probe_404_unknown_id():
    client = _make_client()
    resp = client.post(
        "/probe",
        json={"probe_id": "nope", "capture_id": "nope"},
    )
    assert resp.status_code == 404


def test_steer_503_without_loaded_model():
    # steer is implemented now (no longer a Phase-1 501 stub); with no model loaded it must
    # refuse with 503 Service Unavailable rather than attempt a forward pass.
    client = _make_client()
    resp = client.post(
        "/steer",
        json={
            "prompt": "x",
            "layer": 10,
            "direction_id": "any",
            "alpha": 1.0,
        },
    )
    assert resp.status_code == 503


def test_manifest_sha_stable():
    m1 = build_manifest("capture", {"prompt": "x", "layers": [1]}, {"shape": [1, 4]})
    m2 = build_manifest("capture", {"prompt": "x", "layers": [1]}, {"shape": [1, 4]})
    # call_id and timestamp differ, but the payload-derived hash must be deterministic
    assert m1.input_hash == m2.input_hash
    assert m1.output_hash == m2.output_hash


def test_manifest_sha_differs_on_input_change():
    m1 = build_manifest("capture", {"prompt": "x"}, {"out": 1})
    m2 = build_manifest("capture", {"prompt": "y"}, {"out": 1})
    assert m1.input_hash != m2.input_hash
