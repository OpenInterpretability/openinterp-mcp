# openinterp-mcp

> MCP server + Colab backend for mechanistic interpretability research.
> **Bring your own agent.** Works with Claude Code, Cursor, Cline, OpenHands, Aider, or any harness that speaks MCP.
> **Privacy-first.** We do not host inference. We do not custody your keys. Your Colab session, your model, your data.

## What this is

A research toolkit that turns probe-causality and SAE-feature experiments into agent-callable primitives. Researchers run the model on their own compute (Colab Pro recommended), expose an HTTPS endpoint over ngrok, and let an LLM agent drive experiments via 7 typed tools.

The 7 primitives:

| Tool | What it does |
|---|---|
| `capture_acts` | Run a forward pass with hooks, return activations at specified layers/positions |
| `probe_eval` | Apply a published probe to captured activations, return AUROC + per-sample scores |
| `steer` | Inject a direction at layer L with magnitude alpha, capture output + control-token-normalized delta |
| `sae_lookup` | Decompose an activation into top-K SAE features with auto-interp descriptions |
| `causality_protocol` | Run the three mandatory checks (random-feature baseline, control-token norm, structural-rigidity alpha-sweep) automatically |
| `publish` | Submit experiment result to the public Atlas (HF Datasets + Zenodo DOI), opt-in |
| `judge_reproduce` | Spawn Claude-Code-as-judge to re-execute and verify a published claim |

## Architecture (privacy-first)

```
USER'S MACHINE (laptop)              USER'S COMPUTE (Colab/vast.ai/runpod)
├── Claude Code / Cursor / Cline     ├── Colab Secrets (HF/OAI/Anthropic keys)
├── openinterp-mcp (stateless)       ├── FastAPI + 7 endpoints
└── ~/.openinterp/sessions.json      ├── Qwen3.6-27B + probes loaded
    (URLs cached, no secrets)        └── ngrok / cloudflared tunnel
                                          │
              ←── HTTPS (ngrok URL) ──────┘

DOES NOT EXIST ANYWHERE:
✗ api.openinterp.org inference endpoint
✗ a server custodying your keys
✗ telemetry / logs traversing our infra
✗ a database of your queries
```

## Quick start (researchers)

### 1. In a Colab notebook (one cell)
```python
%pip install openinterp-mcp[colab] -q
from google.colab import userdata
import os
for k in ['HF_TOKEN', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'NGROK_AUTHTOKEN']:
    try: os.environ[k] = userdata.get(k)
    except: pass

from openinterp_mcp.colab import launch
url = launch(model="Qwen/Qwen2.5-7B-Instruct")
print(f"\n✓ OpenInterp session ready.\n  Paste in Claude:  /colab-attach {url}\n")
```

### 2. In Claude Code / Cursor / Cline
```
/colab-attach https://abc123.ngrok-free.app
✓ Connected. Qwen2.5-7B loaded. 5 probes available.

/capture-acts "Solve x^2 = 4" --layers L11,L20,L27 --positions end_question
/probe-eval saturation-direction-L20 --acts last_capture
/causality-protocol L20_pre_tool
```

## Install (agent-side)

```bash
pip install openinterp-mcp
```

Add to `claude_desktop_config.json` (or equivalent for Cursor/Cline):
```json
{
  "mcpServers": {
    "openinterp": {
      "command": "openinterp-mcp",
      "args": ["serve"]
    }
  }
}
```

## Status

**v0.0.1 alpha** — Phase 1 of an 11-phase build documented at [openinterp.org/mcp](https://openinterp.org/mcp).
Track progress at [github.com/OpenInterpretability/openinterp-mcp/issues](https://github.com/OpenInterpretability/openinterp-mcp/issues).

## License

Apache-2.0. See [LICENSE](LICENSE).
