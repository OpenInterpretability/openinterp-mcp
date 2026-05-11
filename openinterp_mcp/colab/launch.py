"""Researcher entry point. One call from a Colab cell brings up:

  - FastAPI app with 8 typed endpoints
  - HTTPS ngrok tunnel
  - Loaded HF model + tokenizer
  - Probe registry (empty by default; researchers add via /list-probes or HF refs)

Usage in Colab:

    from openinterp_mcp.colab import launch
    url = launch(model="Qwen/Qwen2.5-7B-Instruct", probes=["L20_pre_tool"])
    print(url)  # → https://xxx.ngrok-free.app — paste in Claude Code

Privacy: this function never sends data to openinterp.org. Everything runs in your
Colab session. Tunnel URL is public but ephemeral; close the notebook and it dies.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional

import uvicorn
from fastapi import FastAPI

from openinterp_mcp import __version__
from openinterp_mcp.colab.endpoints import router
from openinterp_mcp.colab.state import STATE
from openinterp_mcp.colab.tunnel import open_tunnel


def _build_app() -> FastAPI:
    app = FastAPI(
        title="openinterp-mcp Colab backend",
        version=__version__,
        description="Privacy-first mechanistic interpretability primitives. "
        "User-owned compute, agent-driven experiments.",
    )
    app.include_router(router)
    return app


def _load_model(model_id: str, device: str, dtype: str) -> None:
    """Load HF model + tokenizer into STATE. Heavy import — only runs when launch() is called
    so the package stays light for harness-side install."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "Model loading requires `openinterp-mcp[colab]`. "
            "Run: `pip install 'openinterp-mcp[colab]'`"
        ) from e

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        dtype
    ]

    print(f"[openinterp-mcp] loading {model_id} ({dtype}, {device})...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    STATE.model = model
    STATE.tokenizer = tok
    STATE.model_id = model_id
    STATE.device = device
    STATE.dtype = dtype
    print(f"[openinterp-mcp] ✓ model ready ({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params)")


def _serve_in_background(app: FastAPI, port: int) -> threading.Thread:
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for uvicorn to bind. Worst-case 5s on cold Colab.
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    return thread


def launch(
    model: Optional[str] = None,
    probes: Optional[List[str]] = None,
    port: int = 8000,
    device: str = "cuda",
    dtype: str = "bfloat16",
    region: str = "us",
) -> str:
    """Bring up the Colab backend. Returns the public ngrok URL.

    Parameters
    ----------
    model : HF model id (e.g. "Qwen/Qwen2.5-7B-Instruct"). Required for /capture, /probe, /steer.
            Pass None to skip model load (e.g. for testing the route surface).
    probes : list of probe ids to preload. Phase 1 ships with placeholder registry; real probe
             loaders land in Phase 1 final commits.
    port : local port. Default 8000.
    device : "cuda" | "cpu". Default "cuda".
    dtype : "bfloat16" | "float16" | "float32". Default "bfloat16".
    region : ngrok region. Default "us".
    """
    if model:
        _load_model(model, device, dtype)
    else:
        print("[openinterp-mcp] no model specified — route surface only mode (good for testing)")

    if probes:
        print(f"[openinterp-mcp] probe loading deferred to Phase 1 final commits ({len(probes)} requested)")

    app = _build_app()
    _serve_in_background(app, port)
    url = open_tunnel(port=port, region=region)

    print(f"\n  ✓ OpenInterp session ready at {url}")
    print(f"  Paste in Claude Code / Cursor / Cline:")
    print(f"      /colab-attach {url}\n")

    return url
