"""MCP server entry point — registers 8 typed tools and dispatches to the Colab backend.

Usage (from any MCP-compatible harness):

    pip install openinterp-mcp[server]
    openinterp-mcp serve

Then in ~/.cursor/mcp.json or claude_desktop_config.json:
    {"mcpServers": {"openinterp": {"command": "openinterp-mcp", "args": ["serve"]}}}

The server is STATELESS at the MCP layer. All compute state lives in the user's Colab
session. The server only holds session metadata (active endpoint URL) on disk in
~/.openinterp/sessions.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any, Dict, List

from openinterp_mcp import __version__, client
from openinterp_mcp.client import BackendError
from openinterp_mcp.sessions import (
    Session,
    get_session,
    list_sessions,
    put_session,
    remove_session,
)


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "colab_attach",
        "description": "Attach to a running Colab/vast.ai openinterp-mcp session via its public HTTPS URL. Validates /health, caches the endpoint locally. Run this once per session before calling other tools.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "Public HTTPS URL (e.g. https://abc.ngrok-free.app)"},
                "session_name": {"type": "string", "default": "default"},
            },
            "required": ["endpoint"],
        },
    },
    {
        "name": "colab_status",
        "description": "Return /health for the active session (loaded model, probes, captures in memory).",
        "inputSchema": {
            "type": "object",
            "properties": {"session_name": {"type": "string", "default": "default"}},
        },
    },
    {
        "name": "list_probes",
        "description": "List probes currently loaded in the Colab backend (model_id, layer, position, source).",
        "inputSchema": {
            "type": "object",
            "properties": {"session_name": {"type": "string", "default": "default"}},
        },
    },
    {
        "name": "capture_acts",
        "description": "Run a forward pass with hooks on specified layers, extract activations at named positions. Returns capture_id for downstream tools. Activations stay in Colab memory (privacy-first); only metadata + manifest SHA256 returns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "layers": {"type": "array", "items": {"type": "integer"}},
                "positions": {"type": "array", "items": {"type": "string"}, "default": ["end_question"],
                              "description": "Position specs: end_question | end_prompt | last_token | first_token | <int>"},
                "session_name": {"type": "string", "default": "default"},
            },
            "required": ["prompt", "layers"],
        },
    },
    {
        "name": "probe_eval",
        "description": "Apply a loaded probe to a stored capture. Returns per-sample scores + AUROC (if labels provided) + manifest SHA256.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "probe_id": {"type": "string"},
                "capture_id": {"type": "string"},
                "labels": {"type": "array", "items": {"type": "integer"}, "description": "Optional binary labels for AUROC"},
                "layer": {"type": "integer", "description": "Override probe's declared layer if needed"},
                "session_name": {"type": "string", "default": "default"},
            },
            "required": ["probe_id", "capture_id"],
        },
    },
    {
        "name": "steer",
        "description": "Inject a direction*alpha at layer L during forward pass. Returns base + steered generation + control-token-normalized delta (Δrel from paper-6). Δrel ≈ 0 with non-zero raw Δ = epiphenomenal-softmax-temp warning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "layer": {"type": "integer"},
                "direction_id": {"type": "string", "description": "Probe id whose direction to inject"},
                "alpha": {"type": "number"},
                "max_new_tokens": {"type": "integer", "default": 32},
                "control_tokens": {"type": "array", "items": {"type": "string"},
                                   "description": "Override default 30-token control set"},
                "session_name": {"type": "string", "default": "default"},
            },
            "required": ["prompt", "layer", "direction_id", "alpha"],
        },
    },
    {
        "name": "sae_lookup",
        "description": "Decompose a captured activation into top-K SAE features. Loads SAE on demand if not in memory. Returns feature_ids + activations + auto-interp descriptions when available.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sae_id": {"type": "string"},
                "capture_id": {"type": "string"},
                "layer": {"type": "integer"},
                "top_k": {"type": "integer", "default": 32},
                "session_name": {"type": "string", "default": "default"},
            },
            "required": ["sae_id", "capture_id", "layer"],
        },
    },
    {
        "name": "causality_protocol",
        "description": "Run the three mandatory checks (random-feature baseline, control-token norm, structural-rigidity α-sweep) and return a verdict in {causal, weak-causal, epiphenomenal-softmax, epiphenomenal-template, undetermined}. Requires a capture_id with stored activations + binary labels.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "probe_id": {"type": "string"},
                "capture_id": {"type": "string"},
                "labels": {"type": "array", "items": {"type": "integer"}},
                "alpha_sweep": {"type": "array", "items": {"type": "number"},
                                "default": [-200, -100, -50, 50, 100, 200]},
                "n_random_seeds": {"type": "integer", "default": 5},
                "session_name": {"type": "string", "default": "default"},
            },
            "required": ["probe_id", "capture_id", "labels"],
        },
    },
]


def _format_result(result: Dict[str, Any]) -> str:
    return json.dumps(result, indent=2, default=str)


def _dispatch(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Pure dispatch — used by both the MCP server (async wrapper) and the standalone judge."""
    session_name = arguments.get("session_name", "default")

    if name == "colab_attach":
        endpoint = arguments["endpoint"].rstrip("/")
        sess = Session(name=session_name, endpoint=endpoint, attached_at=time.time())
        put_session(sess)
        try:
            h = client.health(session_name)
            sess.model_id = h.get("model_id")
            sess.probes_loaded = h.get("probes_loaded", 0)
            sess.captures_in_memory = h.get("captures_in_memory", 0)
            sess.last_health = time.time()
            put_session(sess)
            return {"ok": True, "endpoint": endpoint, "health": h}
        except Exception as e:
            remove_session(session_name)
            raise RuntimeError(f"Could not attach to {endpoint}: {e}") from e

    if name == "colab_status":
        return client.health(session_name)
    if name == "list_probes":
        return client.list_probes(session_name)
    if name == "capture_acts":
        return client.capture(
            arguments["prompt"], arguments["layers"], arguments.get("positions", ["end_question"]),
            session_name,
        )
    if name == "probe_eval":
        return client.probe(
            arguments["probe_id"], arguments["capture_id"],
            labels=arguments.get("labels"), layer=arguments.get("layer"),
            session_name=session_name,
        )
    if name == "steer":
        return client.steer(
            arguments["prompt"], arguments["layer"], arguments["direction_id"], arguments["alpha"],
            max_new_tokens=arguments.get("max_new_tokens", 32),
            control_tokens=arguments.get("control_tokens"),
            session_name=session_name,
        )
    if name == "sae_lookup":
        return client.sae_lookup(
            arguments["sae_id"], arguments["capture_id"], arguments["layer"],
            top_k=arguments.get("top_k", 32), session_name=session_name,
        )
    if name == "causality_protocol":
        return client.causality_protocol(
            arguments["probe_id"], arguments["capture_id"], arguments["labels"],
            alpha_sweep=arguments.get("alpha_sweep"),
            n_random_seeds=arguments.get("n_random_seeds", 5),
            session_name=session_name,
        )

    raise ValueError(f"Unknown tool: {name}")


async def _run_mcp_stdio() -> None:
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except ImportError as e:
        raise RuntimeError(
            "MCP SDK missing. Install with: pip install 'openinterp-mcp[server]'"
        ) from e

    server = Server("openinterp")

    @server.list_tools()
    async def _list() -> List["Tool"]:
        return [Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
                for t in TOOL_DEFINITIONS]

    @server.call_tool()
    async def _call(name: str, arguments: Dict[str, Any]) -> List["TextContent"]:
        try:
            result = _dispatch(name, arguments)
            return [TextContent(type="text", text=_format_result(result))]
        except BackendError as e:
            return [TextContent(type="text",
                                text=_format_result({"error": "backend_error", "status": e.status, "detail": e.detail}))]
        except Exception as e:
            return [TextContent(type="text", text=_format_result({"error": "tool_error", "detail": str(e)}))]

    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def main() -> int:
    parser = argparse.ArgumentParser(prog="openinterp-mcp", description=f"OpenInterp MCP server v{__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="Run the MCP server over stdio (use from claude_desktop_config.json / mcp.json)")
    p_status = sub.add_parser("status", help="Print session info without starting the server")
    p_status.add_argument("--name", default="default")
    sub.add_parser("sessions", help="List all known sessions")
    p_detach = sub.add_parser("detach", help="Remove a session")
    p_detach.add_argument("--name", default="default")

    args = parser.parse_args()
    if args.cmd == "serve":
        asyncio.run(_run_mcp_stdio())
        return 0
    if args.cmd == "status":
        s = get_session(args.name)
        print(json.dumps({"session": args.name, "exists": s is not None,
                          **({"endpoint": s.endpoint, "model_id": s.model_id} if s else {})}, indent=2))
        return 0
    if args.cmd == "sessions":
        print(json.dumps({n: vars(s) for n, s in list_sessions().items()}, indent=2, default=str))
        return 0
    if args.cmd == "detach":
        ok = remove_session(args.name)
        print(json.dumps({"detached": ok, "name": args.name}))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
