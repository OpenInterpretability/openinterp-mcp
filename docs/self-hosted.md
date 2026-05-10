# Self-hosted compute (vast.ai / runpod / local)

Colab is the default. When you need bigger models or longer sessions, run the same backend on a self-hosted GPU.

## vast.ai

1. Rent an A100 80GB instance with PyTorch + CUDA pre-baked (template: "PyTorch (cuDNN Devel)").
2. SSH in.
3. Install:

```bash
pip install 'openinterp-mcp[colab]'   # the `colab` extra includes torch, transformers, etc.
```

4. Start the backend:

```python
# Save as launch.py on the instance
import os
os.environ['NGROK_AUTHTOKEN'] = '<your-token>'
os.environ['HF_TOKEN'] = '<your-hf-token>'

from openinterp_mcp.colab import launch
url = launch(model='Qwen/Qwen3.6-27B-Instruct', device='cuda', dtype='bfloat16')
print(url)
```

```bash
python launch.py
```

5. Copy the printed URL, paste into your agent (`/colab-attach <url>`).

## runpod

Same as vast.ai. Pick an A100 80GB or H100 template; install; launch.

## Local GPU (Mac M-series with MPS)

Works for small models (Qwen2.5-3B at bf16 fits on 32GB unified memory). For research-scale models, rent.

```python
from openinterp_mcp.colab import launch
url = launch(model='Qwen/Qwen2.5-3B-Instruct', device='mps', dtype='float16')
```

## When self-hosted beats Colab

| | Colab Pro | vast.ai A100 80GB |
|---|---|---|
| Cost | $10/mo flat | ~$0.30-0.80/hr |
| Session timeout | 24h | until you stop the instance |
| Qwen3.6-27B bf16 | tight (40GB A100) | comfortable |
| Reproducibility | session-bound | instance-bound |
| Setup time | 0 (web) | ~10 min (SSH) |

Recommendation: Colab Pro for iterative work, vast.ai when you need a 6-12h run that you can leave unattended.

## Privacy unchanged

The architecture is identical regardless of host. The backend reads secrets from environment variables (Colab Secrets is just one source), exposes endpoints over ngrok, and runs the model on user-owned hardware. We never see your traffic.
