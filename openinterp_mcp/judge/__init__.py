"""Claude-Code-as-judge: spawn an Anthropic-API agent loop that uses the MCP tools to
re-execute experiments and produce verifiable verdicts. Distinct from passive LLM-as-judge —
the judge HERE actually runs the protocol, doesn't just opine on text."""

from openinterp_mcp.judge.reproduce import judge_reproduce
from openinterp_mcp.judge.methodology import judge_methodology
from openinterp_mcp.judge.adversarial import judge_adversarial

__all__ = ["judge_reproduce", "judge_methodology", "judge_adversarial"]
