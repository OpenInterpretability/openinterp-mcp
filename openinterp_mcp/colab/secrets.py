"""Colab Secrets reader with environment-variable fallback.

Priority order:
1. Colab `userdata.get(key)` — preferred, never appears in notebook output
2. `os.environ[key]` — fallback for vast.ai / runpod / local
3. None — caller decides whether to error or proceed

Keys we look for: HF_TOKEN, OPENAI_API_KEY, ANTHROPIC_API_KEY, NGROK_AUTHTOKEN.
"""
from __future__ import annotations

import os
from typing import Optional


def _try_colab_userdata(key: str) -> Optional[str]:
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        return None
    try:
        return userdata.get(key)
    except Exception:
        return None


def get_secret(key: str) -> Optional[str]:
    """Read a secret from Colab userdata first, then env vars, else None."""
    val = _try_colab_userdata(key)
    if val:
        return val
    return os.environ.get(key)


def require_secret(key: str) -> str:
    """Read a secret or raise with a researcher-friendly message."""
    val = get_secret(key)
    if not val:
        raise RuntimeError(
            f"Missing secret `{key}`. Add it via Colab → Secrets (🔑 icon in left sidebar) "
            f"or `export {key}=...` in your shell. See https://openinterp.org/mcp/secrets"
        )
    return val
