---
name: capture-acts
description: Capture residual-stream activations at specified layers and token positions during a forward pass on the attached model. Returns a capture_id for downstream tools.
---

# capture-acts

Run a forward pass through the researcher's loaded model, hooking specified layers, extracting activations at named positions. Activations stay in the Colab session memory; only metadata (shape, capture_id, manifest SHA256) crosses the wire.

## When to use

Before any of: `probe-eval`, `causality-protocol`, `sae-lookup`. These all need a `capture_id`.

Also when the researcher asks "what are the activations at L20 for this prompt?" or "store the residual at L11/L31/L55 for {prompt}".

## How to use

Call the MCP tool **`capture_acts`** with:
- `prompt`: the text to run through the model
- `layers`: list of layer indices (e.g. `[10, 20, 27]`)
- `positions`: list of position specs. Defaults to `["end_question"]`.
  - `end_question` / `end_prompt` / `last_token` → last input token
  - `first_token` → token 0
  - integer `<int>` → absolute index (negative counts from end)
  - `"end_question:-2"` → 2 tokens before end

## Output to surface

> "Captured L{layers} × {positions} → capture_id `{id[:8]}…` (d_model={d}, manifest sha {sha[:12]})"

Hand the capture_id to whatever comes next. Don't try to read the raw tensors; they live in the Colab session.

## Cost / time

A single forward pass on Qwen2.5-7B at bf16 on a T4 is ~2 s; on A100 ~0.5 s. Larger models scale. Don't batch hundreds of prompts in a tight loop without warning the researcher.
