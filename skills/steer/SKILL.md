---
name: steer
description: Inject a direction*alpha into the residual stream during a forward pass and observe the behavioral effect. Returns both raw and control-token-normalized deltas.
---

# steer

Causal-intervention skill. Adds a probe direction to layer L with magnitude α during generation. Returns base output, steered output, raw Δlogit on the originally-most-likely next token, and the control-token-normalized Δrel (paper-6).

## When to use

When the researcher asks "is this probe causal" or "what happens if we push this direction up/down" or "sweep α from −200 to +200 on probe X".

For a single intervention, call `steer` directly. For a full sweep + verdict, call `causality-protocol` instead — it orchestrates the sweep and applies the verdict heuristics.

## How to use

Call the MCP tool **`steer`** with:
- `prompt`: the input to steer over
- `layer`: where to inject
- `direction_id`: the probe whose direction to inject (same id as in `probe_eval`)
- `alpha`: scalar magnitude. Sign matters — positive pushes in the probe direction, negative pushes against.
- `max_new_tokens` (default 32): how much to generate for the behavioral check

## Interpretation rules (paper-6 — critical)

Always report `normalization.delta_rel`, NOT raw `delta_raw_logit_target`.

If `delta_rel ≈ 0` (|Δrel| < 0.1) while raw Δ is large:
> "Δrel ≈ 0 — the raw effect is a uniform softmax-temperature shift across the vocabulary. Probe direction is **epiphenomenal-softmax** at this α."

If the tool returns `epiphenomenal_flag: true`, surface it prominently.

If `flipped: false` at every α tested but `delta_rel ≠ 0`:
> "Probe shifts logits but never flips the argmax — weak causal signal, paper-5 saturation-direction lever class."

If `flipped: false` and `delta_rel ≈ 0` even at α >> ‖residual‖:
> "Structural-rigidity null. Decision lives in input tokens (e.g. chat-template gating), not the residual stream — **epiphenomenal-template**."

## Do not

- Do not interpret `flipped: true` at large α as "causal" by itself. Confirm Δrel is non-trivial.
- Do not skip control-token normalization. The tool always runs it; you always surface it.
