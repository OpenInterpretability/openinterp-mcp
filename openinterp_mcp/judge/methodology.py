"""judge_methodology: given a probe id + a capture id, auto-apply the three mandatory checks
and emit a structured verdict. Thin wrapper around causality_protocol with judge-grade reporting."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openinterp_mcp.judge._runtime import run_agent_loop


METHODOLOGY_SYSTEM = """You are a methodology auditor for mechanistic interpretability research.
You have access to openinterp MCP tools. Your job:

1. Verify the named probe exists (list_probes).
2. Verify the named capture exists (colab_status implicitly via probe_eval).
3. Run causality_protocol with the provided labels and alpha sweep.
4. Inspect the verdict and emit JSON in this format:

{
  "verdict": <one of: causal, weak-causal, epiphenomenal-softmax, epiphenomenal-template, undetermined>,
  "real_auroc": <float>,
  "random_baseline_mean": <float>,
  "auroc_gap": <float>,
  "delta_rel_max": <float>,
  "flip_rate_at_max_alpha": <float>,
  "publishable_as_causal": <bool — true ONLY for verdict==causal>,
  "recommendation": <one-line natural-language guidance>
}

Be conservative. weak-causal is NOT publishable as causal — only as a saturation-direction lever class.
epiphenomenal-* is NEVER publishable as causal."""


def judge_methodology(
    probe_id: str,
    capture_id: str,
    labels: List[int],
    alpha_sweep: Optional[List[float]] = None,
    model: str = "claude-sonnet-4-6",
    max_turns: int = 12,
    verbose: bool = False,
) -> Dict[str, Any]:
    spec = {
        "probe_id": probe_id,
        "capture_id": capture_id,
        "labels": labels,
        "alpha_sweep": alpha_sweep or [-200, -100, -50, 50, 100, 200],
    }
    user_prompt = (
        "Audit this probe via the three mandatory checks. Emit the structured JSON verdict.\n\n"
        f"Spec:\n```json\n{json.dumps(spec, indent=2)}\n```"
    )
    out = run_agent_loop(METHODOLOGY_SYSTEM, user_prompt, model=model, max_turns=max_turns, verbose=verbose)
    from openinterp_mcp.judge.reproduce import _extract_json

    result = _extract_json(out["final_text"]) or {
        "verdict": "undetermined",
        "recommendation": "Could not parse judge output. Raw: " + out["final_text"][:300],
    }
    return {**result, "_transcript_turns": out["n_turns"]}
