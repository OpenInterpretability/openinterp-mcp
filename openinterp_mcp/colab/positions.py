"""Resolve a position spec into an integer token index.

Position specs:
    "end_question"  → last token of the input (before any generated tokens)
    "end_prompt"    → alias for end_question
    "last_token"    → alias for end_question
    "first_token"   → token index 0
    "<int>"         → absolute token index (negative allowed: -1 = last)
    "<str:int>"     → "end_question:-1" subtracts from end_question

Multiple positions can be resolved together; we return a list of ints.
"""
from __future__ import annotations

from typing import List, Union

PositionSpec = Union[str, int]


def resolve_position(spec: PositionSpec, n_input_tokens: int) -> int:
    """Resolve a single position spec into an integer index in [0, n_input_tokens)."""
    if isinstance(spec, int):
        idx = spec if spec >= 0 else n_input_tokens + spec
        if not 0 <= idx < n_input_tokens:
            raise ValueError(f"Position {spec} out of range for {n_input_tokens} tokens")
        return idx

    s = spec.strip().lower()
    base_aliases = {
        "end_question": n_input_tokens - 1,
        "end_prompt": n_input_tokens - 1,
        "last_token": n_input_tokens - 1,
        "first_token": 0,
    }
    if s in base_aliases:
        return base_aliases[s]

    if ":" in s:
        base, offset = s.split(":", 1)
        if base not in base_aliases:
            raise ValueError(f"Unknown position base `{base}`")
        return base_aliases[base] + int(offset)

    try:
        return resolve_position(int(s), n_input_tokens)
    except ValueError:
        pass
    raise ValueError(f"Unknown position spec: {spec!r}")


def resolve_positions(specs: List[PositionSpec], n_input_tokens: int) -> List[int]:
    return [resolve_position(s, n_input_tokens) for s in specs]
