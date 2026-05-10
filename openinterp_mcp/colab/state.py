"""Singleton state for the Colab backend: holds the loaded model, tokenizer, and probe registry.

Populated by `launch()`. Read by the FastAPI endpoint handlers. No persistence — when the
Colab session ends, state dies with it (privacy-first).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BackendState:
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    model_id: Optional[str] = None
    device: str = "cpu"
    dtype: str = "bfloat16"
    probes: Dict[str, Any] = field(default_factory=dict)
    captures: Dict[str, Any] = field(default_factory=dict)
    manifests: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_id = None
        self.probes.clear()
        self.captures.clear()
        self.manifests.clear()


STATE = BackendState()
