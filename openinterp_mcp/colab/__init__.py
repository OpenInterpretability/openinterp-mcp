"""Colab backend — researcher runs `launch(...)` in a Colab cell, gets a public ngrok URL,
pastes the URL into an MCP-enabled agent (Claude Code / Cursor / Cline / ...) to drive
experiments via 8 typed primitives."""

from openinterp_mcp.colab.launch import launch

__all__ = ["launch"]
