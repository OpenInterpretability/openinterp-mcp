"""judge_reproduce: take a paper-claim spec, re-execute the experiment via MCP tools, return PASS/FAIL with diff."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from openinterp_mcp.judge._runtime import run_agent_loop


REPRODUCE_SYSTEM = """You are an experimental-reproducibility judge for mechanistic interpretability research.
You have access to the openinterp MCP tools, which control a researcher's Colab/vast.ai backend.

Your job:
1. Read the claim spec (probe id, layer, prompts, expected AUROC or expected verdict).
2. Use capture_acts + probe_eval (and causality_protocol if needed) to re-run the experiment.
3. Compare your observed numbers to the claim.
4. Emit a final JSON object in this exact format (no preamble, no markdown fence):

{
  "verdict": "verified" | "drift" | "failed_to_run",
  "claim": <the claim spec as given>,
  "observed": {"auroc": <float or null>, "gap": <float or null>, "verdict": <string or null>},
  "diff": {"auroc_delta": <float or null>, "verdict_match": <bool or null>},
  "notes": <one-line natural-language summary>
}

Tolerances: AUROC within ±0.03 = verified. Larger = drift.

Do not invent numbers. Always pull from tool outputs. If the backend is unreachable, return failed_to_run."""


def judge_reproduce(
    claim: Dict[str, Any],
    model: str = "claude-sonnet-4-6",
    max_turns: int = 25,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Reproduce a paper-claim spec via the agent loop.

    Claim format (example):
        {
            "paper_id": "saturation-direction",
            "probe_id": "L20_pre_tool",
            "layer": 20,
            "prompts": [...],
            "labels": [...],
            "expected_auroc": 0.83,
            "expected_verdict": "weak-causal"
        }
    """
    user_prompt = (
        "Reproduce this claim using the openinterp MCP tools. "
        "Attach to the active session, capture activations, evaluate the probe, run the causality protocol "
        "if a verdict claim is included, then emit the final JSON.\n\n"
        f"Claim:\n```json\n{json.dumps(claim, indent=2)}\n```"
    )
    out = run_agent_loop(REPRODUCE_SYSTEM, user_prompt, model=model, max_turns=max_turns, verbose=verbose)

    verdict = _extract_json(out["final_text"]) or {"verdict": "failed_to_run", "notes": out["final_text"][:500]}
    return {**verdict, "_transcript_turns": out["n_turns"]}


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None
