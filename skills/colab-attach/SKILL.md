---
name: colab-attach
description: Attach to a researcher's running Colab/vast.ai openinterp session via its public HTTPS URL. Run this once at session start before any other openinterp skill.
---

# colab-attach

Connects the OpenInterp MCP server to the researcher's compute session (Colab Pro / Free / vast.ai / runpod / local).

## When to use

The researcher pastes a URL like `https://abc123.ngrok-free.app` and asks you to attach, OR you see no `default` session active in `~/.openinterp/sessions.json` and a later openinterp skill is about to run.

## How to use

Call the MCP tool **`colab_attach`** with:
- `endpoint`: the HTTPS URL the researcher provided
- `session_name`: defaults to `default`; only override if user explicitly names a session

If the tool returns `{"ok": true, "health": {...}}`, report the loaded model_id and number of probes available. If it returns an error, the URL is unreachable or the backend isn't running — ask the researcher to verify their Colab cell completed.

## Output to surface

Compact one-liner:
> "Attached to {model_id} at {endpoint}. {probes_loaded} probes available."

If the researcher wants details, call `colab_status` afterward.

## Do not

- Do not invent URLs. The researcher provides them.
- Do not store the URL anywhere except via the tool itself (the tool writes to `~/.openinterp/sessions.json`).
- Do not assume the session persists between sittings — Colab dies in 12-24h. Re-attach when stale.
