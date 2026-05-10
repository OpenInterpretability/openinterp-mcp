"""Causality Protocol: the three mandatory checks for probe causality (paper-6 + paper-5).

Check 1 — Random-feature baseline (paper-6)
    Compare probe AUROC against random Gaussian probes of the same dimensionality.
    If gap ≤ 0.1, the probe is N-overparameterized noise. Mandatory at N<100.

Check 2 — Control-token normalization (paper-6)
    During steering, compare Δlogit(target) against mean Δlogit(controls).
    If Δrel ≈ 0 while raw Δ is large, the intervention is a uniform softmax-temperature shift
    — epiphenomenal class 1 (softmax-temp).

Check 3 — Structural-rigidity α-sweep (paper-6)
    Sweep α from −‖residual‖×k to +‖residual‖×k. If zero behavioral change at any α, the
    decision is in input tokens or vocab attractor — epiphenomenal class 2 (template-lock).

The protocol returns a verdict in {causal, weak-causal, epiphenomenal-softmax, epiphenomenal-template, undetermined}.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from openinterp_mcp.colab.probes import auroc, random_feature_baseline


def causality_verdict(
    real_auroc: float,
    random_auroc_mean: float,
    delta_rel_max: float,
    flip_rate_at_max_alpha: float,
) -> Dict[str, Any]:
    """Synthesize a verdict from the three signals.

    Heuristic thresholds derived from paper-5 + paper-6 empirical findings.
    """
    gap = real_auroc - random_auroc_mean

    if gap < 0.10:
        return {
            "verdict": "undetermined",
            "reason": "Random-feature baseline ate the signal (gap < 0.10). Need larger N.",
        }

    if abs(delta_rel_max) < 0.10 and flip_rate_at_max_alpha == 0.0:
        return {
            "verdict": "epiphenomenal-template",
            "reason": "Probe direction has zero behavioral effect at supra-norm α. Decision lives in input tokens.",
        }

    if abs(delta_rel_max) < 0.05:
        return {
            "verdict": "epiphenomenal-softmax",
            "reason": "Δrel ≈ 0 — raw steering effect is a uniform softmax-temperature shift.",
        }

    if flip_rate_at_max_alpha < 0.10:
        return {
            "verdict": "weak-causal",
            "reason": f"Δrel = {delta_rel_max:.3f}, flip rate {flip_rate_at_max_alpha:.0%} at α_max. Saturation-direction lever.",
        }

    return {
        "verdict": "causal",
        "reason": f"AUROC gap +{gap:.2f}, Δrel = {delta_rel_max:.3f}, flip rate {flip_rate_at_max_alpha:.0%}.",
    }


def run_protocol(
    probe_activations: np.ndarray,
    labels: np.ndarray,
    probe_direction: np.ndarray,
    probe_bias: float,
    steering_results: Optional[List[Dict[str, Any]]] = None,
    n_random_seeds: int = 5,
) -> Dict[str, Any]:
    """Execute all three checks and return a structured report."""
    from openinterp_mcp.colab.probes import apply_probe

    real_scores = apply_probe(probe_activations, probe_direction, probe_bias)
    real_auroc = auroc(real_scores, labels)

    n_samples, d_model = probe_activations.shape
    random = random_feature_baseline(n_samples, d_model, labels, n_seeds=n_random_seeds)

    delta_rels: List[float] = []
    flip_rates: List[float] = []
    if steering_results:
        delta_rels = [r.get("delta_rel", 0.0) for r in steering_results]
        flip_rates = [r.get("flip_rate", 0.0) for r in steering_results]

    delta_rel_max = max(delta_rels, key=abs) if delta_rels else 0.0
    flip_rate_at_max_alpha = max(flip_rates) if flip_rates else 0.0

    verdict = causality_verdict(real_auroc, random["mean"], delta_rel_max, flip_rate_at_max_alpha)

    return {
        "real_auroc": real_auroc,
        "random_baseline": random,
        "auroc_gap": real_auroc - random["mean"],
        "delta_rel_max": delta_rel_max,
        "flip_rate_at_max_alpha": flip_rate_at_max_alpha,
        "verdict": verdict["verdict"],
        "reason": verdict["reason"],
        "checks_run": {
            "random_feature_baseline": True,
            "control_token_normalization": bool(steering_results),
            "structural_rigidity_alpha_sweep": bool(steering_results),
        },
    }
