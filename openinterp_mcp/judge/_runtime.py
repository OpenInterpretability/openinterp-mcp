"""Shared agent-loop runtime used by all four judge entry points.

Runs an Anthropic API tool-use loop with the openinterp-mcp tools. Returns the final assistant
text. Caller decides how to parse it (typically asks the agent to emit a JSON verdict in a
specific schema).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openinterp_mcp.server import TOOL_DEFINITIONS, _dispatch


def _anthropic_tool_schemas() -> List[Dict[str, Any]]:
    """Convert our MCP tool defs into Anthropic API tool schemas."""
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["inputSchema"]}
        for t in TOOL_DEFINITIONS
    ]


def run_agent_loop(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-6",
    max_turns: int = 25,
    max_tokens: int = 4096,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run an Anthropic tool-use agent loop. Returns dict with `transcript`, `final_text`, `n_turns`.

    Requires ANTHROPIC_API_KEY in env. The agent has access to all 8 openinterp tools.
    """
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("Judge requires the anthropic SDK. `pip install anthropic`.") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "Set ANTHROPIC_API_KEY in your environment to run the judge. The key is read locally and never crosses openinterp.org."
        )

    client = anthropic.Anthropic()
    tools = _anthropic_tool_schemas()
    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    transcript: List[Dict[str, Any]] = []

    for turn in range(max_turns):
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )
        transcript.append({"turn": turn, "stop_reason": resp.stop_reason, "content": [c.model_dump() for c in resp.content]})

        if resp.stop_reason == "tool_use":
            tool_uses = [c for c in resp.content if c.type == "tool_use"]
            tool_results = []
            for tu in tool_uses:
                try:
                    result = _dispatch(tu.name, dict(tu.input))
                    content = json.dumps(result, default=str)
                except Exception as e:
                    content = json.dumps({"error": str(e)})
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": content})
                if verbose:
                    print(f"[turn {turn}] {tu.name}({tu.input}) → {content[:200]}")
            messages.append({"role": "assistant", "content": [c.model_dump() for c in resp.content]})
            messages.append({"role": "user", "content": tool_results})
            continue

        final_text = "".join(c.text for c in resp.content if c.type == "text")
        return {"transcript": transcript, "final_text": final_text, "n_turns": turn + 1}

    return {"transcript": transcript, "final_text": "(max_turns exhausted)", "n_turns": max_turns}
