"""ngrok HTTPS tunnel wrapper. Single public URL for the FastAPI app."""
from __future__ import annotations

from typing import Optional

from openinterp_mcp.colab.secrets import get_secret


def open_tunnel(port: int = 8000, region: str = "us") -> str:
    """Open an HTTPS ngrok tunnel to localhost:port. Returns the public URL.

    Requires NGROK_AUTHTOKEN in Colab Secrets or environment. Free-tier tokens work
    fine for research use (1 concurrent tunnel, 40 conns/min).
    """
    from pyngrok import conf, ngrok

    token = get_secret("NGROK_AUTHTOKEN")
    if token:
        conf.get_default().auth_token = token

    conf.get_default().region = region

    existing = ngrok.get_tunnels()
    for t in existing:
        if t.config.get("addr", "").endswith(f":{port}"):
            ngrok.disconnect(t.public_url)

    tunnel = ngrok.connect(addr=port, proto="http", bind_tls=True)
    return tunnel.public_url


def close_tunnels() -> None:
    """Close all open ngrok tunnels for this process."""
    from pyngrok import ngrok

    for t in ngrok.get_tunnels():
        ngrok.disconnect(t.public_url)
