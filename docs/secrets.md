# Secrets

OpenInterp-mcp uses four secrets, all optional except `NGROK_AUTHTOKEN`.

| Secret | Required for | Where to add it |
|---|---|---|
| `NGROK_AUTHTOKEN` | Public HTTPS endpoint from Colab | https://dashboard.ngrok.com → free tier |
| `HF_TOKEN` | Private probe/SAE downloads | https://huggingface.co/settings/tokens |
| `ANTHROPIC_API_KEY` | `judge_reproduce` and adversarial tools | https://console.anthropic.com |
| `OPENAI_API_KEY` | Atlas embeddings search (optional) | https://platform.openai.com |

## In Colab (recommended)

Colab → 🔑 Secrets in left sidebar → add each. The notebook template auto-loads them via `userdata.get(k)`.

**Never paste a secret into a notebook cell.** It will appear in cell output, get auto-saved to Drive, and persist forever in your share/copy history.

## On vast.ai / runpod / local

Export as environment variables in the shell before running `launch`:

```bash
export NGROK_AUTHTOKEN=...
export HF_TOKEN=hf_...
python launch.py
```

Or use a `.env` file + `python-dotenv` if you prefer. The `openinterp_mcp.colab.secrets.get_secret(key)` reader checks Colab userdata first, then environment.

## What we never receive

- Your Anthropic / OpenAI / HF tokens. They live on your compute. The agent only sees endpoint URLs.
- Your queries, captures, generations, or activations — unless you explicitly call `/publish` on a result you've decided to share.

The openinterp.org domain hosts:
- Documentation (this file)
- Published atlas entries (Apache-2.0 contributions from researchers who chose to share)
- ProbeBench leaderboard (opt-in submissions)

Nothing else.
