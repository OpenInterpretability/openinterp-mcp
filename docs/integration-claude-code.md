# Integration — Claude Code

## Install the MCP server

```bash
pip install 'openinterp-mcp[server]'
```

## Register the server

Edit `~/.claude/mcp.json` (or `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS for Claude Desktop):

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

Restart Claude Code. You should see `openinterp` listed in your active MCP servers (`/mcp` slash command).

## Drop in the skills (optional but recommended)

```bash
mkdir -p ~/.claude/skills
ln -s "$(python -c 'import openinterp_mcp, os; print(os.path.join(os.path.dirname(openinterp_mcp.__file__), \"..\", \"skills\"))')" ~/.claude/skills/openinterp
```

Or copy them manually from the `skills/` directory in this repo. Claude Code picks them up automatically; reference them as `/colab-attach`, `/capture-acts`, `/probe-eval`, `/steer`, `/causality-protocol` in conversation.

## Verify

In Claude Code:

```
/mcp
> openinterp ● connected · 8 tools

# Now spin up a Colab session (see quick-start.md), grab the URL, and:

/colab-attach https://abc.ngrok-free.app
```

If the agent responds with the loaded model_id, you're done.

## Troubleshooting

**"openinterp-mcp: command not found"** — the pip install put the entry point in a directory not on PATH. Try `python -m openinterp_mcp.server serve` in the `command` field instead.

**"backend 503: Model not loaded"** — the Colab cell that calls `launch(model=...)` hasn't run yet or the model load failed (often OOM on Free-tier T4 with 7B+ models — switch to 4-bit quantization or Colab Pro A100).

**"No active session named `default`"** — you haven't called `/colab-attach` yet, or your Colab session died (12-24h timeout). Re-attach.
