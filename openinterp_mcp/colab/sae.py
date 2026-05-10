"""SAE feature lookup. Loads a TopK SAE from HF and decomposes activations into top-K features.

Supports `caiovicentino1/qwen36-27b-sae-fullstack` layout out of the box; other layouts via the
`SAELoader` adapter interface.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _hf_snapshot(repo_id: str, allow_patterns: Optional[List[str]] = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns))


def load_sae(sae_id: str, layer: int, source: Optional[str] = None) -> Dict[str, Any]:
    """Load SAE encoder weights for a specific layer. Returns dict with `encoder` matrix (d_sae, d_model)."""
    src = source or f"hf://caiovicentino1/qwen36-27b-sae-fullstack"
    if not src.startswith("hf://"):
        raise NotImplementedError("Only hf:// sources in Phase 1")
    repo_id = src[len("hf://") :]

    snapshot = _hf_snapshot(repo_id, allow_patterns=[f"*L{layer}*", "*.json"])

    encoder_files = list(snapshot.glob(f"*L{layer}*encoder*")) + list(snapshot.glob(f"*L{layer}*W_enc*"))
    if not encoder_files:
        raise FileNotFoundError(
            f"No encoder weights found for L{layer} in {repo_id}. Looked for *L{layer}*encoder* / *L{layer}*W_enc*."
        )

    encoder = _load_array(encoder_files[0])
    feature_descriptions = _load_descriptions(snapshot, layer)

    return {
        "id": sae_id,
        "layer": layer,
        "d_model": int(encoder.shape[1]),
        "d_sae": int(encoder.shape[0]),
        "encoder": encoder,
        "feature_descriptions": feature_descriptions,
        "source_url": src,
    }


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file
        blobs = load_file(str(path))
        return next(iter(blobs.values()))
    return np.load(str(path))


def _load_descriptions(snapshot: Path, layer: int) -> Dict[int, str]:
    candidates = list(snapshot.glob(f"*L{layer}*descriptions*.json"))
    if not candidates:
        return {}
    raw = json.loads(candidates[0].read_text())
    return {int(k): v for k, v in raw.items()}


def top_k_features(
    activation: np.ndarray,
    encoder: np.ndarray,
    top_k: int = 32,
    descriptions: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Decompose `activation` into top-K SAE features.

    Args:
        activation: shape (d_model,)
        encoder:    shape (d_sae, d_model)
        top_k:      how many features to return
        descriptions: optional feature_id → auto-interp string

    Returns: list of {feature_id, activation, description?} sorted by descending activation.
    """
    feature_activations = encoder @ activation
    idx = np.argsort(feature_activations)[::-1][:top_k]
    out = []
    for i in idx:
        item = {
            "feature_id": int(i),
            "activation": float(feature_activations[i]),
        }
        if descriptions and int(i) in descriptions:
            item["description"] = descriptions[int(i)]
        out.append(item)
    return out
