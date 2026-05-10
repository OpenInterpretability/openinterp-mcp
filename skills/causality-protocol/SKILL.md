---
name: causality-protocol
description: Run the three mandatory checks (random-feature baseline, control-token norm, structural-rigidity α-sweep) on a probe and return a verdict in {causal, weak-causal, epiphenomenal-softmax, epiphenomenal-template, undetermined}.
---

# causality-protocol

The complete paper-6 + paper-5 audit, packaged. Use this before claiming a probe is causal in any output the researcher will publish.

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

## Interpreting the verdict

| Verdict | What it means | What to recommend |
|---|---|---|
| `causal` | AUROC gap > 0.1 over random, Δrel non-trivial, flip rate > 10% | Safe to publish as causal in this regime. |
| `weak-causal` | AUROC + Δrel signal but flip rate < 10% | Paper-5 saturation-direction lever. Publish with the class label. |
| `epiphenomenal-softmax` | Δrel ≈ 0 across all α | Probe direction produces uniform softmax-temp shifts. Do NOT publish as causal. |
| `epiphenomenal-template` | Δrel ≈ 0 AND zero flips even at supra-norm α | Decision is in input tokens. Do NOT publish as causal. |
| `undetermined` | AUROC gap < 0.1 over random | N too small. Need more data before any causality claim. |

## Output to surface

Lead with the verdict and reason in two lines:

> **Verdict**: `{verdict}`
> {reason}
>
> Real AUROC {real:.3f} vs random {rand:.3f} (gap +{gap:.3f}). Δrel max {drm:.3f}, flip rate {fr:.0%} at α_max.

Then offer to publish the result via `/publish` skill if the verdict supports it.

## Do not

- Do not run with `labels` that don't match the captured rows. The tool will 400 you.
- Do not infer "more α" can rescue an `epiphenomenal-template` verdict. That class means the residual stream is not the causal locus regardless of magnitude.
