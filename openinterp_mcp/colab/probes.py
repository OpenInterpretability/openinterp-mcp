"""Probe loading + application.

A probe is a linear classifier on residual-stream activations:
    score(act) = sigmoid(act @ direction + bias)

Probes are loaded from HuggingFace as a small JSON manifest + a safetensors file holding the
direction vector. The manifest captures provenance: which model, which layer, which position,
training data summary, license, source URL.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _hf_download_probe(repo_id: str, filename: str) -> Path:
    """Download a single file from a HF repo and return the local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError("Probe loading needs huggingface_hub. `pip install huggingface_hub`") from e
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def load_probe(probe_id: str, source: Optional[str] = None) -> Dict[str, Any]:
    """Load a probe from a local path or from HF.

    `source` formats:
        "hf://<repo_id>"                          → loads manifest.json + direction.safetensors
        "hf://<repo_id>#<probe_name>"             → loads <probe_name>.json + <probe_name>.safetensors
        "/local/path/probe.json"                  → loads sibling .safetensors
        None                                      → assume probe_id is "hf://openinterp/<probe_id>"
    """
    src = source or f"hf://openinterp-community/probes#{probe_id}"

    if src.startswith("hf://"):
        body = src[len("hf://") :]
        if "#" in body:
            repo_id, name = body.split("#", 1)
        else:
            repo_id, name = body, "probe"
        manifest_path = _hf_download_probe(repo_id, f"{name}.json")
        weights_path = _hf_download_probe(repo_id, f"{name}.safetensors")
    else:
        manifest_path = Path(src)
        weights_path = manifest_path.with_suffix(".safetensors")

    manifest = json.loads(manifest_path.read_text())
    direction, bias = _load_safetensors(weights_path)

    return {
        "id": probe_id,
        "model_id": manifest.get("model_id"),
        "layer": manifest.get("layer"),
        "position": manifest.get("position"),
        "license": manifest.get("license", "apache-2.0"),
        "source_url": manifest.get("source_url", src),
        "training": manifest.get("training", {}),
        "direction": direction,
        "bias": float(bias),
    }


def _load_safetensors(path: Path):
    try:
        from safetensors.numpy import load_file
    except ImportError as e:
        raise RuntimeError("Probe weights need safetensors. `pip install safetensors`") from e
    blobs = load_file(str(path))
    direction = blobs["direction"]
    bias = float(blobs.get("bias", np.array([0.0]))[0])
    return direction, bias


def apply_probe(activations: np.ndarray, direction: np.ndarray, bias: float) -> np.ndarray:
    """Compute probe scores. `activations` shape (N, d_model). Returns shape (N,)."""
    logits = activations @ direction + bias
    return 1.0 / (1.0 + np.exp(-logits))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Macro AUROC for binary labels. Returns float in [0, 1]."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def random_feature_baseline(
    n_samples: int, d_model: int, labels: np.ndarray, n_seeds: int = 5
) -> Dict[str, float]:
    """Construct random Gaussian probes of dimension d_model and report their mean AUROC.

    Use this as a sanity check when N is small: real probe AUROC should significantly exceed
    the random baseline. Per paper-6, mandatory at N<100.
    """
    aurocs = []
    rng_seeds = list(range(n_seeds))
    for s in rng_seeds:
        rng = np.random.default_rng(s)
        random_dir = rng.standard_normal(d_model).astype(np.float32)
        random_dir /= np.linalg.norm(random_dir) + 1e-8
        random_acts = rng.standard_normal((n_samples, d_model)).astype(np.float32)
        scores = apply_probe(random_acts, random_dir, 0.0)
        aurocs.append(auroc(scores, labels))
    return {"mean": float(np.mean(aurocs)), "std": float(np.std(aurocs)), "n_seeds": n_seeds}
