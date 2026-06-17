"""Unit tests for the paper-11 Check 4 additions to the causality verdict.

These exercise the pure `causality_verdict` function — no GPU / backend required.
"""
from openinterp_mcp.colab.causality import causality_verdict, _structure_confound_collapses


# Signals that, pre-paper-11, would have returned `causal`: strong gap, real steering, flips.
PASSING = dict(
    real_auroc=0.84,
    strongest_baseline_mean=0.55,
    delta_rel_max=0.30,
    flip_rate_at_max_alpha=0.40,
    n_samples=100,
)


def test_collapse_helper_matches_paper11_case():
    # 0.838 -> 0.08 must register as a collapse; 0.84 -> 0.80 must not.
    assert _structure_confound_collapses(0.838, 0.08) is True
    assert _structure_confound_collapses(0.84, 0.80) is False


def test_high_auroc_without_naming_or_control_is_inconclusive_not_causal():
    # The exact paper-11 mistake: high AUROC + steering, but no structure-matched control, no name.
    v = causality_verdict(**PASSING)
    assert v["verdict"] == "inconclusive-unverified", v
    assert "structure-matched" in v["reason"] and "name the feature" in v["reason"]


def test_structure_matched_collapse_is_confounded_structural():
    v = causality_verdict(**PASSING, structure_matched_auroc=0.08, feature_named=True)
    assert v["verdict"] == "confounded-structural", v


def test_named_and_surviving_control_is_causal():
    v = causality_verdict(**PASSING, structure_matched_auroc=0.80, feature_named=True)
    assert v["verdict"] == "causal", v


def test_survives_control_but_unnamed_is_inconclusive():
    v = causality_verdict(**PASSING, structure_matched_auroc=0.80, feature_named=False)
    assert v["verdict"] == "inconclusive-unverified", v
    assert "name the feature" in v["reason"]


def test_named_but_no_control_is_inconclusive():
    v = causality_verdict(**PASSING, feature_named=True)  # structure_matched_auroc=None
    assert v["verdict"] == "inconclusive-unverified", v
    assert "structure-matched" in v["reason"]


def test_low_flip_with_full_controls_is_weak_causal():
    signals = dict(PASSING, flip_rate_at_max_alpha=0.05)
    v = causality_verdict(**signals, structure_matched_auroc=0.80, feature_named=True)
    assert v["verdict"] == "weak-causal", v


def test_small_n_still_undetermined_regardless_of_controls():
    signals = dict(PASSING, real_auroc=0.60, n_samples=12)  # gap 0.05 << 0.40 floor
    v = causality_verdict(**signals, structure_matched_auroc=0.59, feature_named=True)
    assert v["verdict"] == "undetermined", v
