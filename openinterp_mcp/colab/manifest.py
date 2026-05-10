"""Audit manifest: every tool call produces a SHA256-anchored record.

Researchers can prove reproducibility by including the manifest hash in publications.
Manifests live in memory; the researcher decides whether to dump or publish.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class Manifest:
    call_id: str
    tool: str
    timestamp: float
    input_hash: str
    output_hash: str
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            {
                "tool": self.tool,
                "timestamp": self.timestamp,
                "input_hash": self.input_hash,
                "output_hash": self.output_hash,
                "config": self.config,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sha256"] = self.sha256
        return d


def _hash_payload(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def build_manifest(
    tool: str,
    inputs: Any,
    outputs: Any,
    config: Dict[str, Any] | None = None,
) -> Manifest:
    return Manifest(
        call_id=str(uuid.uuid4()),
        tool=tool,
        timestamp=time.time(),
        input_hash=_hash_payload(inputs),
        output_hash=_hash_payload(outputs),
        config=config or {},
    )
