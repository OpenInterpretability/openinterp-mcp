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

from openinterp_mcp.colab.probes import (
    auroc, random_feature_baseline, random_direction_on_real_baseline, shuffled_label_baseline,
)


def _required_gap_for_n(n_samples: int) -> float:
    """N-adaptive minimum AUROC gap before a probe can be called causal.

    Paper-6 + Phase 6c finding: at small N with high d_model, even random directions / shuffled
    labels can show apparent signal in residual activations. The gap threshold must scale with N.
    """
    if n_samples < 20: return 0.40
    if n_samples < 50: return 0.25
    if n_samples < 100: return 0.15
    return 0.10


def causality_verdict(
    real_auroc: float,
    strongest_baseline_mean: float,
    delta_rel_max: float,
    flip_rate_at_max_alpha: float,
    n_samples: int = 100,
) -> Dict[str, Any]:
    """Synthesize a verdict from three signals + N-adaptive thresholds.

    `strongest_baseline_mean` is max of (random-direction-random-acts, random-direction-real-acts,
    shuffled-label) AUROCs — the hardest baseline to beat.
    """
    gap = real_auroc - strongest_baseline_mean
    required = _required_gap_for_n(n_samples)

    if gap < required:
        return {
            "verdict": "undetermined",
            "reason": f"AUROC gap +{gap:.3f} below N={n_samples} threshold ({required:.2f}). "
                      f"Baselines reach {strongest_baseline_mean:.3f}; real {real_auroc:.3f}. "
                      "Either need more data or the probe is memorizing.",
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
        "reason": f"AUROC gap +{gap:.3f} (vs N={n_samples} threshold {required:.2f}), "
                  f"Δrel = {delta_rel_max:.3f}, flip rate {flip_rate_at_max_alpha:.0%}.",
    }


def run_protocol(
    probe_activations: np.ndarray,
    labels: np.ndarray,
    probe_direction: np.ndarray,
    probe_bias: float,
    steering_results: Optional[List[Dict[str, Any]]] = None,
    n_random_seeds: int = 5,
) -> Dict[str, Any]:
    """Execute all checks and return a structured report.

    Three independent baselines compared; verdict uses the STRONGEST (= max baseline AUROC) as
    threshold to beat. This prevents small-N false positives from passing under a single weak
    baseline.
    """
    from openinterp_mcp.colab.probes import apply_probe

    real_scores = apply_probe(probe_activations, probe_direction, probe_bias)
    real_auroc = auroc(real_scores, labels)

    n_samples, d_model = probe_activations.shape
    random_floor = random_feature_baseline(n_samples, d_model, labels, n_seeds=n_random_seeds)
    random_real = random_direction_on_real_baseline(probe_activations, labels, n_seeds=n_random_seeds)
    shuffled = shuffled_label_baseline(probe_activations, labels, n_seeds=n_random_seeds)

    strongest_mean = max(random_floor["mean"], random_real["mean"], shuffled["mean"])

    delta_rels: List[float] = []
    flip_rates: List[float] = []
    if steering_results:
        delta_rels = [r.get("delta_rel", 0.0) for r in steering_results]
        flip_rates = [r.get("flip_rate", 0.0) for r in steering_results]

    delta_rel_max = max(delta_rels, key=abs) if delta_rels else 0.0
    flip_rate_at_max_alpha = max(flip_rates) if flip_rates else 0.0

    verdict = causality_verdict(
        real_auroc, strongest_mean, delta_rel_max, flip_rate_at_max_alpha, n_samples=n_samples,
    )

    return {
        "real_auroc": real_auroc,
        "baselines": {
            "random_direction_random_acts": random_floor,
            "random_direction_real_acts": random_real,
            "shuffled_labels": shuffled,
            "strongest_baseline_mean": strongest_mean,
        },
        "auroc_gap": real_auroc - strongest_mean,
        "n_samples": int(n_samples),
        "delta_rel_max": delta_rel_max,
        "flip_rate_at_max_alpha": flip_rate_at_max_alpha,
        "verdict": verdict["verdict"],
        "reason": verdict["reason"],
        "checks_run": {
            "random_feature_baseline": True,
            "random_direction_real_acts_baseline": True,
            "shuffled_label_baseline": True,
            "control_token_normalization": bool(steering_results),
            "structural_rigidity_alpha_sweep": bool(steering_results),
        },
    }
