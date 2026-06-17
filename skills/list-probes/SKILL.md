---
name: list-probes
description: List the probes currently loaded in the Colab backend — their model, layer, position, and source — so you know what is available to evaluate or steer.
---

# list-probes

A quick inventory of the probes in the attached session's memory. Use it to discover what you can
pass to `probe_eval`, `steer`, or `causality_protocol`.

## When to use

- Right after `colab_attach`, to see what is already loaded
- Before `probe_eval` / `steer` / `causality_protocol`, to get a valid `probe_id` and its declared layer
- When a tool 404s on an unknown `probe_id` — list to find the correct id

## How to use

Call the MCP tool **`list_probes`** (no arguments beyond `session_name`). Returns each probe's
`probe_id`, `model_id`, `layer`, `position`, and `source`.

## Output to surface

A short table, one row per probe:

> `{probe_id}` — {model_id} · L{layer} · {position} · {source}

Then suggest the natural next step (evaluate with `probe_eval`, or audit with `causality_protocol`).

## Do not

- Do not assume a probe's layer — read it from `list_probes`. Passing the wrong layer to `probe_eval`
  silently scores the wrong activations.
