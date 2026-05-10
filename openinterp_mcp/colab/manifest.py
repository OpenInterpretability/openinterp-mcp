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


_VOLATILE_KEYS = {"capture_id", "manifest_sha256", "call_id", "timestamp", "audit_sha256"}


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
        """Content hash: deterministic across calls with identical (tool, inputs, outputs, config).

        Excludes timestamp + call_id so the SAME experiment always produces the SAME hash.
        This is what papers cite. For unique-per-call tracking use `audit_sha256`.
        """
        payload = json.dumps(
            {
                "tool": self.tool,
                "input_hash": self.input_hash,
                "output_hash": self.output_hash,
                "config": self.config,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def audit_sha256(self) -> str:
        """Audit hash: unique per call (includes timestamp + call_id). For log integrity."""
        payload = json.dumps(
            {
                "tool": self.tool,
                "call_id": self.call_id,
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
        d["audit_sha256"] = self.audit_sha256
        return d


def _hash_payload(payload: Any) -> str:
    """Stable content hash: strips volatile keys (UUIDs, timestamps) before hashing."""
    if isinstance(payload, dict):
        clean = {k: v for k, v in payload.items() if k not in _VOLATILE_KEYS}
    else:
        clean = payload
    serialized = json.dumps(clean, sort_keys=True, default=str)
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
