"""Steering with mandatory control-token normalization (paper-6 methodology).

Δrel = Δlogit(target) − mean(Δlogit(controls))

If Δrel ≈ 0 while raw Δlogit(target) is large, the probe direction is producing a uniform
softmax-temperature shift, not a directed semantic intervention — paper-6 epiphenomenal class 1.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

DEFAULT_CONTROL_TOKENS = [
    " the", " and", " of", " to", " a", " in", " is", " that", " for", " on",
    " with", " as", " was", " at", " by", " be", " this", " not", " or", " an",
    " from", " but", " they", " we", " he", " his", " her", " their", " its", " can",
]


def control_token_ids(tokenizer, controls: Optional[Sequence[str]] = None) -> List[int]:
    controls = controls or DEFAULT_CONTROL_TOKENS
    ids = []
    for tok in controls:
        encoded = tokenizer.encode(tok, add_special_tokens=False)
        if encoded:
            ids.append(encoded[0])
    return ids


def control_token_normalize(
    base_logits: "torch.Tensor",
    steered_logits: "torch.Tensor",
    target_token_id: int,
    control_ids: List[int],
) -> dict:
    """Compute Δraw and Δrel for a single (target, control_set) pair.

    Inputs are 1-D logit vectors over the vocabulary at the position of interest
    (typically the last input token, i.e. the position whose argmax determines the next token).
    """
    import torch.nn.functional as F
    import torch

    base = base_logits.to(dtype=__import__("torch").float32)
    steered = steered_logits.to(dtype=base.dtype)
    delta = steered - base

    delta_target = delta[target_token_id].item()
    delta_controls = delta[control_ids].mean().item() if control_ids else 0.0
    delta_rel = delta_target - delta_controls

    base_logprobs = F.log_softmax(base, dim=-1)
    steer_logprobs = F.log_softmax(steered, dim=-1)
    target_logprob_delta = (steer_logprobs[target_token_id] - base_logprobs[target_token_id]).item()

    return {
        "delta_raw_logit_target": float(delta_target),
        "delta_raw_logit_controls_mean": float(delta_controls),
        "delta_rel": float(delta_rel),
        "delta_logprob_target": float(target_logprob_delta),
        "n_controls": len(control_ids),
        "epiphenomenal_flag": abs(delta_rel) < 0.1 and abs(delta_target) > 0.3,
    }
