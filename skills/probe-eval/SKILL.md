---
name: probe-eval
description: Apply a loaded linear probe to a stored capture. Returns per-sample scores and AUROC (when labels are provided).
---

# probe-eval

Score residual-stream activations with a published probe direction. Probes are loaded into the Colab backend via the launch script or via separate probe-loading utilities; this skill only evaluates.

## When to use

After a `capture-acts` call, when the researcher wants to know "does this probe fire on these activations" or "what's the AUROC of probe X on these labels".

## How to use

Call the MCP tool **`probe_eval`** with:
- `probe_id`: ID of a probe already loaded (run `list_probes` first if unsure)
- `capture_id`: from a prior `capture-acts` call
- `labels` (optional): list of 0/1 labels for the captured rows. If provided, return AUROC.
- `layer` (optional): override the probe's declared layer

## Methodology guard rails

If the researcher reports AUROC > 0.9 with N < 50 and asks "is this a real result?":
- Recommend running `causality-protocol` to apply the three mandatory checks (random-feature baseline, control-token normalization, structural-rigidity α-sweep).
- AUROC alone proves nothing at small N — `causality-protocol` separates signal from over-parameterization.

## Output to surface

Compact:
> "Probe {probe_id}@L{layer}: AUROC {auroc:.3f} on {n_samples} samples. Mean score {mean:.3f}. Manifest {sha[:12]}."

Avoid claims like "this is causal" or "this is the real direction" — `probe_eval` is detection-only; causality needs the protocol.
