# Integration — Cline

## Install

```bash
pip install 'openinterp-mcp[server]'
```

## Register the server

In VS Code (with Cline extension installed), open Settings → Cline → MCP Servers, or edit `~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`:

```json
{
  "mcpServers": {
    "openinterp": {
      "command": "openinterp-mcp",
      "args": ["serve"],
      "disabled": false,
      "autoApprove": ["colab_status", "list_probes"]
    }
  }
}
```

`autoApprove` lets read-only tools run without confirmation. Compute-bound tools (`capture_acts`, `steer`, `causality_protocol`) still require approval — desirable since they can take seconds-to-minutes.

Reload Cline. Tool list appears in the MCP panel.

## Use

The openinterp skills work as natural-language requests. Example:

> "I have a Qwen2.5-7B Colab running at https://abc.ngrok-free.app. Attach, then run causality_protocol on the L20 saturation-direction probe using the prompts 'Solve x^2=4', 'What is 2+2?', '3+5=' with labels [1,1,1]."

Cline will sequence the calls (attach → capture → causality-protocol) and report.

## Notes

- Cline supports approval-policy customization per tool. Strict approval is good for steering experiments — the researcher sees the α value and prompt before each forward pass.
- Cline can use either Anthropic or OpenAI models; both work with the openinterp tool schemas.
