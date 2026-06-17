---
name: causality-protocol
description: Run the four mandatory checks (random-feature baseline, control-token norm, structural-rigidity α-sweep, and the paper-11 structure-matched control + naming gate) and return a verdict in {causal, weak-causal, confounded-structural, epiphenomenal-softmax, epiphenomenal-template, inconclusive-unverified, undetermined}.
---

# causality-protocol

The complete paper-6 + paper-5 + paper-11 audit, packaged. Use this before claiming a probe is causal in any output the researcher will publish.

## When to use

- A probe just got a high AUROC and the researcher wants to know if it's real
- Reviewing a paper claim and the researcher wants to audit it
- Before submitting an atlas entry as "gold-tier"
- When the researcher asks "is this causal?" or "is this probe lever-able?"

Always before publication. Never skip just because AUROC looks good.

## How to use

Call the MCP tool **`causality_protocol`** with:
- `probe_id`: loaded probe
- `capture_id`: stored activations + the original prompt (which must be one of the labeled examples)
- `labels`: binary labels for the captured rows
- `alpha_sweep` (default `[-200, -100, -50, 50, 100, 200]`): α values to try
- `n_random_seeds` (default 5): random-feature baseline seeds
- `structure_matched_capture_id` + `structure_matched_labels` (**Check 4a**): a capture where the
  confound (conversation structure, lexicon, punctuation, length, referent specificity) is balanced
  against the label. The SAME probe is re-evaluated on it. **Supply this whenever the claim is about
  what the feature *means*** — without it the tool cannot return `causal`.
- `feature_name` (**Check 4b**): the feature's max-activating-context name. Required before any causal
  claim — high AUROC without naming is a non-result.

## The four checks

1. **Random-feature baseline** (paper-6) — beat random Gaussian probes by an N-adaptive gap.
2. **Control-token normalization** (paper-6) — Δrel ≈ 0 ⇒ uniform softmax-temp shift (epiphenomenal).
3. **Structural-rigidity α-sweep** (paper-6) — zero flips at supra-norm α ⇒ decision is in input tokens.
4. **Structure-matched control + naming** (paper-11, "Form, Not Granted") — a high AUROC can be a
   *structural / lexical / punctuation* confound, not the concept. Paper-11 shipped twice on a
   direction with AUROC **0.838 that collapsed to 0.08** once structure (and a stray em-dash) was
   balanced. So the signal must survive a confound-balanced set **and** the feature must be named.

## Interpreting the verdict

| Verdict | What it means | What to recommend |
|---|---|---|
| `causal` | Gap over baselines + Δrel + flips > 10%, **survives the structure-matched control, and is named** | Safe to publish as causal in this regime. |
| `weak-causal` | Survives Check 4 + Δrel signal but flip rate < 10% | Paper-5 saturation-direction lever. Publish with the class label. |
| `confounded-structural` | AUROC collapses on the structure-matched set | **The direction reads structure/lexicon/punctuation, not the concept. Do NOT publish as causal.** (paper-11 failure mode) |
| `epiphenomenal-softmax` | Δrel ≈ 0 across all α | Uniform softmax-temp shifts. Do NOT publish as causal. |
| `epiphenomenal-template` | Δrel ≈ 0 AND zero flips even at supra-norm α | Decision is in input tokens. Do NOT publish as causal. |
| `inconclusive-unverified` | Stats + steering pass, but **no structure-matched control and/or no name** | NOT causal and NOT null. Run the missing control(s) before any claim. |
| `undetermined` | AUROC gap below the N-adaptive floor | N too small / memorization. Need more data. |

## Output to surface

Lead with the verdict and reason in two lines:

> **Verdict**: `{verdict}`
> {reason}
>
> Real AUROC {real:.3f} vs random {rand:.3f} (gap +{gap:.3f}). Structure-matched {sm:.3f}. Δrel max {drm:.3f}, flip rate {fr:.0%} at α_max.

If the verdict is `inconclusive-unverified`, surface the exact missing control(s) from `reason` and offer to run them — do **not** present the result as causal. Then offer `/publish` only if the verdict is `causal` or `weak-causal`.

## Do not

- Do not run with `labels` that don't match the captured rows. The tool will 400 you.
- Do not call a probe `causal` on AUROC alone — that is exactly the paper-11 mistake. Without a
  structure-matched control and a feature name, the honest verdict is `inconclusive-unverified`.
- Do not infer "more α" can rescue an `epiphenomenal-template` verdict. That class means the residual
  stream is not the causal locus regardless of magnitude.
- Do not treat a `confounded-structural` verdict as fixable by more data — the signal is real but it
  is the wrong signal. Re-pose the contrast so structure/lexicon/punctuation no longer track the label.
