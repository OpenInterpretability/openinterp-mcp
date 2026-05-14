# openinterp-mcp

> **v0.1.0 beta · API may shift before v1.0**
>
> MCP server + Colab backend for mechanistic interpretability research.
> **Bring your own agent.** Works with Claude Code, Cursor, Cline, OpenHands, Aider, or any harness that speaks MCP.
> **Privacy-first.** We do not host inference. We do not custody your keys. Your Colab session, your model, your data.

<a href="https://www.producthunt.com/products/openinterpretability?embed=true&utm_source=badge-featured&utm_medium=badge&utm_campaign=badge-openinterpretability" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1146555&theme=light&t=1778728126905" alt="OpenInterpretability - Open-source toolkit to audit what your LLM knows | Product Hunt" width="250" height="54" /></a>

## What this is

A research toolkit that turns probe-causality and SAE-feature experiments into agent-callable primitives. Researchers run the model on their own compute (Colab Pro recommended), expose an HTTPS endpoint over ngrok, and let an LLM agent drive experiments via **8 typed MCP tools**.

The 8 MCP primitives:

| Tool | What it does |
|---|---|
| `colab_attach` | Attach to a running Colab/vast.ai/runpod session via its public HTTPS URL. Validates `/health`, caches the endpoint locally. |
| `colab_status` | Health check — loaded model, probes in memory, captures held. |
| `list_probes` | List probes currently loaded in the backend (model_id, layer, position, source). |
| `capture_acts` | Run a forward pass with hooks, extract activations at specified layers/positions. Returns `capture_id`. |
| `probe_eval` | Apply a loaded probe to a stored capture, return AUROC + per-sample scores. |
| `steer` | Inject direction×α at layer L. Returns base + steered generation + control-token-normalized Δrel (paper-6 protocol). |
| `sae_lookup` | Decompose a stored activation into top-K SAE features with auto-interp descriptions. |
| `causality_protocol` | Run the three mandatory checks (random-feature baseline, control-token norm, structural-rigidity α-sweep) and emit a verdict in {causal, weak-causal, epiphenomenal-softmax, epiphenomenal-template, undetermined}. |

> **Publish + judge primitives are Python modules**, not MCP tools.
> Use `from openinterp_mcp.publish import publish` to submit to the Atlas (HF Dataset + Zenodo DOI + registry PR), and `from openinterp_mcp.judge import reproduce` for Claude-Code-as-judge replication. These run outside the MCP request/response loop because they take minutes (long-running side effects).

## Architecture (privacy-first)

```
USER'S MACHINE (laptop)              USER'S COMPUTE (Colab/vast.ai/runpod)
├── Claude Code / Cursor / Cline     ├── Colab Secrets (HF/OAI/Anthropic keys)
├── openinterp-mcp (stateless)       ├── FastAPI + 8 endpoints
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
