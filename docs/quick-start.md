# Quick start

Three steps. Five minutes if you have a Colab Pro account.

## 1. Spin up a research session in Colab

Open the template notebook: [research.ipynb](../templates/research.ipynb) (or [Open in Colab](https://colab.research.google.com/github/OpenInterpretability/openinterp-mcp/blob/main/templates/research.ipynb)).

In Colab → Secrets (🔑 in left sidebar) → add:
- `NGROK_AUTHTOKEN` — required. Free at https://dashboard.ngrok.com
- `HF_TOKEN` — recommended (for private probes/SAEs)
- `ANTHROPIC_API_KEY` — required only if you plan to use `judge_reproduce`
- `OPENAI_API_KEY` — required only if you plan to use embeddings search

Run the three cells. The third prints a `/colab-attach https://abc.ngrok-free.app` line.

## 2. Install the MCP server on your laptop

```bash
pip install 'openinterp-mcp[server]'
```

## 3. Wire it into your agent harness

Pick your harness:

- [Claude Code](./integration-claude-code.md)
- [Cursor](./integration-cursor.md)
- [Cline](./integration-cline.md)

All three are one JSON edit. Restart your editor. The 8 openinterp tools appear in the agent's tool list.

## 4. First experiment

In the agent, paste the line you got from Colab. Then:

```
/colab-attach https://abc.ngrok-free.app
# Then ask the agent:
> Capture L20 activations at end_question for the prompt "Solve x^2 = 4". Report capture_id.
> List the loaded probes.
> Run causality_protocol on the saturation-direction-L20 probe over capture {capture_id}.
```

Total time from Colab attach to first verdict: ~30 seconds.

## What lives where

| | Your laptop | Colab session |
|---|---|---|
| MCP server | ✓ runs locally | |
| Session metadata | ~/.openinterp/sessions.json | |
| Model weights | | ✓ |
| Activations | | ✓ |
| Probe weights | | ✓ |
| Your API keys | (only if you use judge or embeddings) | ✓ Colab Secrets |
| Audit manifests | | ✓ (until you `/publish`) |

Nothing crosses openinterp.org infrastructure. The site is for *publication*, not inference.
