"""CLI entry point for use in CI (GitHub Actions) — runs judge_reproduce on a YAML claim file
and exits 0 on `verified`, 1 on `drift`, 2 on `failed_to_run`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openinterp_mcp.judge.reproduce import judge_reproduce


def _load_claim(path: Path):
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError("YAML claims need pyyaml: `pip install pyyaml`") from e
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def main() -> int:
    p = argparse.ArgumentParser(prog="openinterp-judge", description="Run reproducibility judge in CI")
    p.add_argument("claim_path", type=Path, help="Path to a claim spec (JSON or YAML)")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--max-turns", type=int, default=25)
    p.add_argument("--out", type=Path, help="Where to write the verdict JSON")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    claim = _load_claim(args.claim_path)
    verdict = judge_reproduce(claim, model=args.model, max_turns=args.max_turns, verbose=args.verbose)

    if args.out:
        args.out.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))

    return {"verified": 0, "drift": 1, "failed_to_run": 2}.get(verdict.get("verdict"), 3)


if __name__ == "__main__":
    sys.exit(main())
