---
name: colab-status
description: Report the health of the active session — loaded model, probes, and captures in memory — to confirm the backend is alive before a run and to debug a broken attach.
---

# colab-status

The `/health` of your attached Colab/vast.ai session. Use it to confirm the GPU backend is up and
holding the state you expect before you spend a forward pass.

## When to use

- After `colab_attach`, to confirm the model loaded and the endpoint is live
- Before a long run, to verify the captures/probes you need are still in memory
- When a tool errors and you are not sure whether the session died or the request was wrong

## How to use

Call the MCP tool **`colab_status`** (no arguments beyond `session_name`). Returns the active
`model_id`, `probes_loaded`, and `captures_in_memory`.

## Output to surface

One line:

> Session `{session_name}`: {model_id} · {probes_loaded} probes · {captures_in_memory} captures in memory.

If the call fails, the session is not attached or the tunnel is down — re-run `colab_attach` with the
current public HTTPS URL.

## Do not

- Do not poll this in a tight loop; it is a forward-free health check, not a progress bar.
