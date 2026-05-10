# Integration — Cursor

## Install

```bash
pip install 'openinterp-mcp[server]'
```

## Register the server

Edit `~/.cursor/mcp.json` (create if missing):

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

Restart Cursor. In the chat panel, the openinterp tools appear under the MCP servers list.

## Verify

Open a chat in Cursor, type:

> "Attach to https://abc.ngrok-free.app and tell me what model is loaded."

The agent should invoke `colab_attach` and report back.

## Notes

- Cursor's agent uses the same Anthropic models as Claude Code, so methodology guard rails inside the skills carry over.
- Tools are invoked silently in Cursor (no `/` prefix needed). The agent decides when to call them based on your natural-language requests.
