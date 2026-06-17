"""Causality Protocol: the four mandatory checks for probe causality (paper-6 + paper-5 + paper-11).

Check 1 — Random-feature baseline (paper-6)
    Compare probe AUROC against random Gaussian probes of the same dimensionality.
    If gap ≤ threshold, the probe is N-overparameterized noise. Mandatory at N<100.

Check 2 — Control-token normalization (paper-6)
    During steering, compare Δlogit(target) against mean Δlogit(controls).
    If Δrel ≈ 0 while raw Δ is large, the intervention is a uniform softmax-temperature shift
    — epiphenomenal class 1 (softmax-temp).

Check 3 — Structural-rigidity α-sweep (paper-6)
    Sweep α from −‖residual‖×k to +‖residual‖×k. If zero behavioral change at any α, the
    decision is in input tokens or vocab attractor — epiphenomenal class 2 (template-lock).

Check 4 — Structure-matched control + naming gate (paper-11, "Form, Not Granted")
    A high AUROC can be a STRUCTURAL/LEXICAL/PUNCTUATION confound, not the concept the probe
    claims to read. Paper-11 was published twice on a direction with AUROC 0.838 that collapsed
    to 0.08 once the conversational structure (and a stray em-dash) was balanced. So:
      (a) re-evaluate the SAME probe on a `structure_matched` set where the confound is balanced.
          If the above-chance signal does not survive → `confounded-structural`.
      (b) require the feature to be NAMED (max-activating context) before any causal claim.
          High cross-family AUROC without naming is a NON-result — the separator can be scaffold,
          lexicon, or punctuation. Without (a) AND a name, the best verdict is `inconclusive-unverified`.

The protocol returns a verdict in
    {causal, weak-causal, confounded-structural, epiphenomenal-softmax, epiphenomenal-template,
     inconclusive-unverified, undetermined}.
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


def _structure_confound_collapses(real_auroc: float, structure_matched_auroc: float) -> bool:
    """True if more than half of the above-chance signal disappears when the confound is balanced.

    Paper-11 case: 0.838 -> 0.08. original = 0.338, retained = -0.42 < 0.169 -> collapses.
    A direction that reads the concept (not the structure) keeps most of its margin over chance.
    """
    original = real_auroc - 0.5
    retained = structure_matched_auroc - 0.5
    return original > 0 and retained < 0.5 * original


def causality_verdict(
    real_auroc: float,
    strongest_baseline_mean: float,
    delta_rel_max: float,
    flip_rate_at_max_alpha: float,
    n_samples: int = 100,
    structure_matched_auroc: Optional[float] = None,
    feature_named: bool = False,
) -> Dict[str, Any]:
    """Synthesize a verdict from four signals + N-adaptive thresholds.

    `strongest_baseline_mean` is max of (random-direction-random-acts, random-direction-real-acts,
    shuffled-label) AUROCs — the hardest baseline to beat.

    `structure_matched_auroc` is the SAME probe's AUROC on a confound-balanced set (Check 4a).
    `feature_named` is whether the feature has a max-activating-context name (Check 4b).
    A `causal` verdict requires BOTH: the signal survives the structure-matched control AND the
    feature is named. Otherwise the strongest honest verdict is `inconclusive-unverified`.
    """
    gap = real_auroc - strongest_baseline_mean
    required = _required_gap_for_n(n_samples)

    # 1. Not above the N-adaptive floor — too little data or memorization.
    if gap < required:
        return {
            "verdict": "undetermined",
            "reason": f"AUROC gap +{gap:.3f} below N={n_samples} threshold ({required:.2f}). "
                      f"Baselines reach {strongest_baseline_mean:.3f}; real {real_auroc:.3f}. "
                      "Either need more data or the probe is memorizing.",
        }

    # 2. Check 4a — the signal does not survive a confound-balanced set.
    if structure_matched_auroc is not None and _structure_confound_collapses(real_auroc, structure_matched_auroc):
        return {
            "verdict": "confounded-structural",
            "reason": f"AUROC {real_auroc:.3f} collapses to {structure_matched_auroc:.3f} on the "
                      "structure-matched set — the direction reads conversation structure / lexicon / "
                      "punctuation, not the claimed concept (paper-11 failure mode). Do NOT publish as causal.",
        }

    # 3 & 4. Steering says the residual stream is not the causal locus.
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

    # 5. Verification gate (Check 4b) — statistics + steering pass, but the controls that
    #    separate a real concept from a confound were not supplied. Honest non-claim.
    missing: List[str] = []
    if not feature_named:
        missing.append("name the feature (max-activating context) — high AUROC without naming is a non-result")
    if structure_matched_auroc is None:
        missing.append("run the structure-matched control (balance structure/lexicon/punctuation, re-evaluate)")
    if missing:
        return {
            "verdict": "inconclusive-unverified",
            "reason": "Statistics and steering pass, but a causal claim is not yet earned: "
                      + "; ".join(missing) + ". Until then this is NOT causal and NOT null.",
        }

    # 6 & 7. Both controls passed.
    if flip_rate_at_max_alpha < 0.10:
        return {
            "verdict": "weak-causal",
            "reason": f"Survives structure-matched control ({structure_matched_auroc:.3f}) and named; "
                      f"Δrel = {delta_rel_max:.3f}, flip rate {flip_rate_at_max_alpha:.0%} at α_max. "
                      "Saturation-direction lever.",
        }

    return {
        "verdict": "causal",
        "reason": f"AUROC gap +{gap:.3f} (vs N={n_samples} threshold {required:.2f}); survives "
                  f"structure-matched control ({structure_matched_auroc:.3f}); feature named; "
                  f"Δrel = {delta_rel_max:.3f}, flip rate {flip_rate_at_max_alpha:.0%}.",
    }


def run_protocol(
    probe_activations: np.ndarray,
    labels: np.ndarray,
    probe_direction: np.ndarray,
    probe_bias: float,
    steering_results: Optional[List[Dict[str, Any]]] = None,
    n_random_seeds: int = 5,
    structure_matched_activations: Optional[np.ndarray] = None,
    structure_matched_labels: Optional[np.ndarray] = None,
    feature_named: bool = False,
    naming_evidence: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute all four checks and return a structured report.

    Three independent baselines compared; verdict uses the STRONGEST (= max baseline AUROC) as
    threshold to beat. This prevents small-N false positives from passing under a single weak
    baseline.

    Check 4 (paper-11): if `structure_matched_activations` + `structure_matched_labels` are given,
    the SAME probe is re-evaluated on that confound-balanced set; collapse => `confounded-structural`.
    `feature_named` / `naming_evidence` record whether the feature has been named — required before
    any causal claim.
    """
    from openinterp_mcp.colab.probes import apply_probe

    real_scores = apply_probe(probe_activations, probe_direction, probe_bias)
    real_auroc = auroc(real_scores, labels)

    n_samples, d_model = probe_activations.shape
    random_floor = random_feature_baseline(n_samples, d_model, labels, n_seeds=n_random_seeds)
    random_real = random_direction_on_real_baseline(probe_activations, labels, n_seeds=n_random_seeds)
    shuffled = shuffled_label_baseline(probe_activations, labels, n_seeds=n_random_seeds)

    strongest_mean = max(random_floor["mean"], random_real["mean"], shuffled["mean"])

    # Check 4a — structure-matched control with the SAME probe.
    structure_matched_auroc: Optional[float] = None
    if structure_matched_activations is not None and structure_matched_labels is not None:
        sm_scores = apply_probe(structure_matched_activations, probe_direction, probe_bias)
        structure_matched_auroc = auroc(sm_scores, np.asarray(structure_matched_labels))

    feature_named = bool(feature_named or naming_evidence)

    delta_rels: List[float] = []
    flip_rates: List[float] = []
    if steering_results:
        delta_rels = [r.get("delta_rel", 0.0) for r in steering_results]
        flip_rates = [r.get("flip_rate", 0.0) for r in steering_results]

    delta_rel_max = max(delta_rels, key=abs) if delta_rels else 0.0
    flip_rate_at_max_alpha = max(flip_rates) if flip_rates else 0.0

    verdict = causality_verdict(
        real_auroc, strongest_mean, delta_rel_max, flip_rate_at_max_alpha, n_samples=n_samples,
        structure_matched_auroc=structure_matched_auroc, feature_named=feature_named,
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
        "structure_matched_auroc": structure_matched_auroc,
        "feature_named": feature_named,
        "naming_evidence": naming_evidence,
        "verdict": verdict["verdict"],
        "reason": verdict["reason"],
        "checks_run": {
            "random_feature_baseline": True,
            "random_direction_real_acts_baseline": True,
            "shuffled_label_baseline": True,
            "control_token_normalization": bool(steering_results),
            "structural_rigidity_alpha_sweep": bool(steering_results),
            "structure_matched_control": structure_matched_auroc is not None,
            "feature_naming": feature_named,
        },
    }
