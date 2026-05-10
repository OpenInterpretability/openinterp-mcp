"""FastAPI route handlers — Phase 1 final: substantive implementations of all 7 primitives."""
from __future__ import annotations

import base64
import io
import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from openinterp_mcp import __version__
from openinterp_mcp.colab.causality import run_protocol
from openinterp_mcp.colab.manifest import build_manifest
from openinterp_mcp.colab.positions import resolve_positions
from openinterp_mcp.colab.probes import apply_probe
from openinterp_mcp.colab.state import STATE
from openinterp_mcp.colab.steering import control_token_ids, control_token_normalize

router = APIRouter()


# ---------- request / response models ----------


class HealthResponse(BaseModel):
    ok: bool
    version: str
    model_id: Optional[str]
    device: str
    dtype: str
    probes_loaded: int
    captures_in_memory: int


class CaptureRequest(BaseModel):
    prompt: str
    layers: List[int]
    positions: List[Union[str, int]] = Field(default_factory=lambda: ["end_question"])
    max_new_tokens: int = 0


class CaptureResponse(BaseModel):
    capture_id: str
    layers: List[int]
    positions: List[Union[str, int]]
    position_indices: List[int]
    n_input_tokens: int
    d_model: int
    manifest_sha256: str


class ProbeEvalRequest(BaseModel):
    probe_id: str
    capture_id: str
    layer: Optional[int] = None
    labels: Optional[List[int]] = None


class SteerRequest(BaseModel):
    prompt: str
    layer: int
    direction_id: str
    alpha: float
    control_tokens: Optional[List[str]] = None
    max_new_tokens: int = 32


class CausalityProtocolRequest(BaseModel):
    probe_id: str
    capture_id: Optional[str] = None
    labels: Optional[List[int]] = None
    alpha_sweep: List[float] = Field(default_factory=lambda: [-200.0, -100.0, -50.0, 50.0, 100.0, 200.0])
    n_random_seeds: int = 5


class SAELookupRequest(BaseModel):
    sae_id: str
    capture_id: str
    layer: int
    top_k: int = 32


class TrainProbeRequest(BaseModel):
    probe_id: str = Field(..., description="Unique id for the trained probe")
    prompts: List[str]
    labels: List[int]
    layer: int
    position: str = "end_question"
    c: float = Field(default=1.0, description="LogisticRegression C parameter")
    max_iter: int = 2000


class LoadProbeRequest(BaseModel):
    probe_id: str
    layer: int
    position: str = "end_question"
    direction: List[float]
    bias: float = 0.0
    source_url: Optional[str] = None
    license: str = "apache-2.0"


# ---------- routes ----------


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        version=__version__,
        model_id=STATE.model_id,
        device=STATE.device,
        dtype=STATE.dtype,
        probes_loaded=len(STATE.probes),
        captures_in_memory=len(STATE.captures),
    )


@router.get("/list-probes")
def list_probes() -> Dict[str, Any]:
    return {
        "model_id": STATE.model_id,
        "probes": [
            {k: v for k, v in p.items() if k not in ("direction",)}
            for p in STATE.probes.values()
        ],
    }


@router.post("/capture", response_model=CaptureResponse)
def capture(req: CaptureRequest) -> CaptureResponse:
    if STATE.model is None or STATE.tokenizer is None:
        raise HTTPException(503, "Model not loaded. Call launch(model=...) first.")

    from openinterp_mcp.colab.hooks import capture_layers

    tok = STATE.tokenizer
    model = STATE.model
    enc = tok(req.prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(next(model.parameters()).device)
    n_input_tokens = int(input_ids.shape[1])

    position_indices = resolve_positions(req.positions, n_input_tokens)

    with capture_layers(model, req.layers) as buf:
        import torch

        with torch.no_grad():
            model(input_ids=input_ids)

    extracted: Dict[int, np.ndarray] = {}
    for layer_idx, tensor in buf.items():
        rows = tensor[0, position_indices, :].to(dtype=__import__("torch").float32).cpu().numpy()
        extracted[layer_idx] = rows

    capture_id = str(uuid.uuid4())
    d_model = int(next(iter(extracted.values())).shape[-1])

    STATE.captures[capture_id] = {
        "prompt": req.prompt,
        "layers": req.layers,
        "positions": req.positions,
        "position_indices": position_indices,
        "n_input_tokens": n_input_tokens,
        "tensors": extracted,
    }

    m = build_manifest(
        "capture",
        inputs=req.model_dump(),
        outputs={"capture_id": capture_id, "d_model": d_model, "n_input_tokens": n_input_tokens},
        config={"model_id": STATE.model_id},
    )
    STATE.manifests.append(m.to_dict())

    return CaptureResponse(
        capture_id=capture_id,
        layers=req.layers,
        positions=req.positions,
        position_indices=position_indices,
        n_input_tokens=n_input_tokens,
        d_model=d_model,
        manifest_sha256=m.sha256,
    )


@router.post("/probe")
def probe(req: ProbeEvalRequest) -> Dict[str, Any]:
    if req.probe_id not in STATE.probes:
        raise HTTPException(404, f"Unknown probe `{req.probe_id}`. See /list-probes.")
    if req.capture_id not in STATE.captures:
        raise HTTPException(404, f"Unknown capture_id `{req.capture_id}`.")

    p = STATE.probes[req.probe_id]
    cap = STATE.captures[req.capture_id]

    layer = req.layer if req.layer is not None else p.get("layer")
    if layer not in cap["tensors"]:
        raise HTTPException(400, f"capture_id has no activations at layer {layer}.")

    acts = cap["tensors"][layer]
    direction = np.asarray(p["direction"])
    if direction.shape[0] != acts.shape[1]:
        raise HTTPException(400, {
            "error": "d_model_mismatch",
            "probe_d_model": int(direction.shape[0]),
            "capture_d_model": int(acts.shape[1]),
            "hint": "Probe direction dimensionality must match capture d_model. "
                    "Probe is for a different model or layer than the capture.",
        })
    scores = apply_probe(acts, direction, p["bias"])

    result: Dict[str, Any] = {
        "probe_id": req.probe_id,
        "capture_id": req.capture_id,
        "layer": layer,
        "n_samples": int(scores.shape[0]),
        "scores": scores.tolist(),
        "mean_score": float(scores.mean()),
    }

    if req.labels is not None and len(req.labels) == scores.shape[0]:
        from openinterp_mcp.colab.probes import auroc

        result["auroc"] = auroc(scores, np.array(req.labels))

    m = build_manifest("probe", req.model_dump(), result, config={"model_id": STATE.model_id})
    STATE.manifests.append(m.to_dict())
    result["manifest_sha256"] = m.sha256
    return result


@router.post("/steer")
def steer(req: SteerRequest) -> Dict[str, Any]:
    if STATE.model is None or STATE.tokenizer is None:
        raise HTTPException(503, "Model not loaded.")
    if req.direction_id not in STATE.probes:
        raise HTTPException(404, f"Unknown direction `{req.direction_id}`. See /list-probes.")

    from openinterp_mcp.colab.hooks import steer_layer
    import torch

    p = STATE.probes[req.direction_id]
    direction_arr = np.asarray(p["direction"])
    d_model = int(STATE.model.config.hidden_size) if hasattr(STATE.model, "config") else direction_arr.shape[0]
    if direction_arr.shape[0] != d_model:
        raise HTTPException(400, {
            "error": "d_model_mismatch",
            "probe_d_model": int(direction_arr.shape[0]),
            "model_d_model": d_model,
            "hint": "Steering direction dimensionality must match model hidden size.",
        })
    direction = torch.from_numpy(direction_arr)

    tok = STATE.tokenizer
    model = STATE.model
    device = next(model.parameters()).device

    enc = tok(req.prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]

    with torch.no_grad():
        base_out = model(input_ids=input_ids)
    base_logits = base_out.logits[0, -1, :]

    with steer_layer(model, req.layer, direction, req.alpha):
        with torch.no_grad():
            steered_out = model(input_ids=input_ids)
    steered_logits = steered_out.logits[0, -1, :]

    target_id = int(base_logits.argmax().item())
    control_ids = control_token_ids(tok, req.control_tokens)
    norm = control_token_normalize(base_logits, steered_logits, target_id, control_ids)

    if req.max_new_tokens > 0:
        with torch.no_grad():
            base_gen = model.generate(
                input_ids, max_new_tokens=req.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id
            )
        with steer_layer(model, req.layer, direction, req.alpha):
            with torch.no_grad():
                steered_gen = model.generate(
                    input_ids, max_new_tokens=req.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id
                )
        base_text = tok.decode(base_gen[0, input_ids.shape[1] :], skip_special_tokens=True)
        steered_text = tok.decode(steered_gen[0, input_ids.shape[1] :], skip_special_tokens=True)
        flipped = base_text.strip() != steered_text.strip()
    else:
        base_text = steered_text = ""
        flipped = False

    result = {
        "layer": req.layer,
        "alpha": req.alpha,
        "direction_id": req.direction_id,
        "target_token_id": target_id,
        "target_token": tok.decode([target_id]),
        "normalization": norm,
        "base_generation": base_text,
        "steered_generation": steered_text,
        "flipped": bool(flipped),
    }
    m = build_manifest("steer", req.model_dump(), result, config={"model_id": STATE.model_id})
    STATE.manifests.append(m.to_dict())
    result["manifest_sha256"] = m.sha256
    return result


@router.post("/sae-lookup")
def sae_lookup(req: SAELookupRequest) -> Dict[str, Any]:
    from openinterp_mcp.colab.sae import load_sae, top_k_features

    if req.capture_id not in STATE.captures:
        raise HTTPException(404, f"Unknown capture_id `{req.capture_id}`.")
    cap = STATE.captures[req.capture_id]
    if req.layer not in cap["tensors"]:
        raise HTTPException(400, f"capture_id has no activations at layer {req.layer}.")

    sae_key = f"{req.sae_id}#L{req.layer}"
    if sae_key not in STATE.probes:
        try:
            sae = load_sae(req.sae_id, req.layer)
        except FileNotFoundError as e:
            raise HTTPException(404, f"SAE not found for L{req.layer} in `{req.sae_id}`: {e}")
        except Exception as e:
            raise HTTPException(502, {
                "error": "sae_load_failed",
                "sae_id": req.sae_id,
                "layer": req.layer,
                "detail": str(e),
                "hint": "Make sure the SAE is published on HF and contains an L{layer} encoder shard. "
                        "For Qwen3.6-27B use `caiovicentino1/qwen36-27b-sae-fullstack`; for other models "
                        "you'll need to train your own (no general-purpose Qwen2.5-3B SAE is published yet).",
            })
        STATE.probes[sae_key] = sae

    sae = STATE.probes[sae_key]
    activation_row = cap["tensors"][req.layer][0]
    if activation_row.shape[0] != sae["encoder"].shape[1]:
        raise HTTPException(400, {
            "error": "d_model_mismatch",
            "capture_d_model": int(activation_row.shape[0]),
            "sae_d_model": int(sae["encoder"].shape[1]),
            "hint": "Loaded SAE was trained for a different model. Use an SAE matching the active model.",
        })
    features = top_k_features(activation_row, sae["encoder"], req.top_k, sae.get("feature_descriptions"))

    result = {
        "sae_id": req.sae_id,
        "layer": req.layer,
        "capture_id": req.capture_id,
        "d_sae": sae["d_sae"],
        "top_features": features,
    }
    m = build_manifest("sae_lookup", req.model_dump(), result, config={"model_id": STATE.model_id})
    STATE.manifests.append(m.to_dict())
    result["manifest_sha256"] = m.sha256
    return result


@router.post("/causality-protocol")
def causality_protocol(req: CausalityProtocolRequest) -> Dict[str, Any]:
    if req.probe_id not in STATE.probes:
        raise HTTPException(404, f"Unknown probe `{req.probe_id}`.")
    if req.capture_id and req.capture_id not in STATE.captures:
        raise HTTPException(404, f"Unknown capture_id `{req.capture_id}`.")

    p = STATE.probes[req.probe_id]
    layer = p["layer"]

    if req.capture_id is None:
        raise HTTPException(
            400,
            "Phase 1 requires a capture_id with stored activations. "
            "Run /capture first on your probe-eval dataset.",
        )

    cap = STATE.captures[req.capture_id]
    activations = cap["tensors"][layer]
    if req.labels is None or len(req.labels) != activations.shape[0]:
        raise HTTPException(400, "labels list required and must match capture row count.")

    labels = np.array(req.labels)

    steering_results: List[Dict[str, Any]] = []
    for alpha in req.alpha_sweep:
        try:
            res = steer(
                SteerRequest(
                    prompt=cap["prompt"],
                    layer=layer,
                    direction_id=req.probe_id,
                    alpha=alpha,
                    max_new_tokens=24,
                )
            )
            steering_results.append({
                "alpha": alpha,
                "delta_rel": res["normalization"]["delta_rel"],
                "flip_rate": 1.0 if res["flipped"] else 0.0,
            })
        except HTTPException:
            steering_results.append({"alpha": alpha, "delta_rel": 0.0, "flip_rate": 0.0})

    report = run_protocol(
        probe_activations=activations,
        labels=labels,
        probe_direction=p["direction"],
        probe_bias=p["bias"],
        steering_results=steering_results,
        n_random_seeds=req.n_random_seeds,
    )
    report["alpha_sweep"] = req.alpha_sweep
    report["steering_results"] = steering_results

    m = build_manifest("causality_protocol", req.model_dump(), report, config={"model_id": STATE.model_id})
    STATE.manifests.append(m.to_dict())
    report["manifest_sha256"] = m.sha256
    return report


@router.get("/manifests")
def list_manifests(limit: int = 50) -> Dict[str, Any]:
    return {"count": len(STATE.manifests), "manifests": STATE.manifests[-limit:]}


@router.post("/train-probe")
def train_probe(req: TrainProbeRequest) -> Dict[str, Any]:
    """Capture activations across labeled prompts, train a linear probe, register it in STATE.

    Convenience endpoint: lets an agent bootstrap a probe + combined capture in one round-trip,
    without needing direct access to the Colab Python kernel.

    Returns probe_id, capture_id (with all N rows + labels), train_accuracy.
    """
    if STATE.model is None or STATE.tokenizer is None:
        raise HTTPException(503, "Model not loaded.")
    if len(req.prompts) != len(req.labels):
        raise HTTPException(400, "prompts and labels must have equal length")
    if len(set(req.labels)) < 2:
        raise HTTPException(400, "labels must contain at least two distinct values")

    from openinterp_mcp.colab.hooks import capture_layers
    from sklearn.linear_model import LogisticRegression
    import torch

    model, tok = STATE.model, STATE.tokenizer
    device = next(model.parameters()).device

    rows: List[np.ndarray] = []
    for prompt in req.prompts:
        enc = tok(prompt, return_tensors="pt").to(device)
        n_input = int(enc["input_ids"].shape[1])
        idx = resolve_positions([req.position], n_input)[0]
        with capture_layers(model, [req.layer]) as buf:
            with torch.no_grad():
                model(input_ids=enc["input_ids"])
        act = buf[req.layer][0, idx, :].to(dtype=torch.float32).cpu().numpy()
        rows.append(act)

    X = np.stack(rows)
    y = np.array(req.labels)

    clf = LogisticRegression(max_iter=req.max_iter, C=req.c).fit(X, y)
    direction = clf.coef_[0].astype(np.float32)
    bias = float(clf.intercept_[0])
    train_acc = float(clf.score(X, y))

    STATE.probes[req.probe_id] = {
        "id": req.probe_id,
        "model_id": STATE.model_id,
        "layer": req.layer,
        "position": req.position,
        "license": "apache-2.0",
        "source_url": "in-session via /train-probe",
        "direction": direction,
        "bias": bias,
        "training": {"n": len(req.prompts), "C": req.c, "train_accuracy": train_acc},
    }

    capture_id = f"{req.probe_id}__train_capture"
    STATE.captures[capture_id] = {
        "prompt": req.prompts[0],  # representative for steering test inside causality_protocol
        "layers": [req.layer],
        "positions": [req.position] * len(req.prompts),
        "position_indices": list(range(len(req.prompts))),
        "n_input_tokens": len(req.prompts),
        "tensors": {req.layer: X},
        "labels": req.labels,
    }

    result = {
        "probe_id": req.probe_id,
        "capture_id": capture_id,
        "layer": req.layer,
        "position": req.position,
        "n_samples": len(req.prompts),
        "d_model": int(X.shape[1]),
        "train_accuracy": train_acc,
        "direction_norm": float(np.linalg.norm(direction)),
    }
    m = build_manifest("train_probe", req.model_dump(), result, config={"model_id": STATE.model_id})
    STATE.manifests.append(m.to_dict())
    result["manifest_sha256"] = m.sha256
    return result


@router.post("/load-probe")
def load_probe_endpoint(req: LoadProbeRequest) -> Dict[str, Any]:
    """Register a pre-trained probe (direction + bias) without running model forward passes."""
    direction = np.array(req.direction, dtype=np.float32)
    STATE.probes[req.probe_id] = {
        "id": req.probe_id,
        "model_id": STATE.model_id,
        "layer": req.layer,
        "position": req.position,
        "license": req.license,
        "source_url": req.source_url or "loaded via /load-probe",
        "direction": direction,
        "bias": req.bias,
        "training": {},
    }
    return {"probe_id": req.probe_id, "layer": req.layer, "d_model": int(direction.shape[0])}
