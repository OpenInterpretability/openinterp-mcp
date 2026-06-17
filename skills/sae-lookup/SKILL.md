---
name: sae-lookup
description: Decompose a captured activation into its top-K sparse-autoencoder features and read their auto-interp names — the bridge from a residual-stream vector to human-readable concepts.
---

# sae-lookup

Turn a raw activation into a short list of named SAE features. This is the entry point to the
full-stack SAE on Qwen3.6-27B (`caiovicentino1/qwen36-27b-sae-fullstack`, 11 layers, d_sae=40960,
k=128) — the rare asset that lets you say *which concepts are active*, not just *how active a
direction is*.

## When to use

- A capture is interesting and the researcher wants to know **what concepts fired**, not just a probe score
- Naming a feature for the causality-protocol Check 4b (a probe claim needs a named feature)
- Exploring a layer to find candidate directions before training a probe
- Comparing two prompts: decompose both, diff the top features

## How to use

Call the MCP tool **`sae_lookup`** with:
- `sae_id`: an SAE matching the active model and the layer (e.g. a layer-59 SAE for an L59 capture)
- `capture_id`: a stored capture (from `capture_acts`)
- `layer`: the layer whose activation to decompose — must be one the capture stored and the SAE was trained for
- `top_k` (default 32): how many features to return

Returns `top_features`: `feature_id`, `activation`, and an auto-interp `description` when available,
plus the manifest SHA256.

## Output to surface

List the top features with their names and activations, highest first:

> Layer {layer}: #{fid} "{description}" ({act:.2f}) · #{fid} "{description}" ({act:.2f}) · …

When the researcher is naming a feature for a probe, surface the **max-activating description** of the
relevant feature and hand it to `causality_protocol` as `feature_name` — that is Check 4b.

## Do not

- Do not use an SAE trained for a different model or layer — the tool 400s on a `d_model_mismatch`.
  Match the SAE to the active model and the capture's layer.
- Do not treat a high single-feature activation as a causal claim. A name is necessary but not
  sufficient — run `causality-protocol` for the causal verdict.
