"""Publication manifest schema. Every /publish call produces one of these, which is what
the openinterp.org indexer reads to populate the Atlas / ProbeBench / Replications pages.

The schema is intentionally minimal and additive. New entry types can extend it without
breaking the indexer.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


EntryType = Literal["probe-result", "atlas-entry", "replication", "adversarial-finding", "sae-feature"]


@dataclass
class PublicationManifest:
    """Schema v1 for atlas publications."""

    title: str
    author: str  # GitHub handle (preferred) or X handle
    type: EntryType
    license: str = "apache-2.0"
    model_id: Optional[str] = None
    claim: Optional[str] = None  # one-line plain-language summary
    numbers: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)  # filenames in the HF dataset
    methodology_check: Optional[Dict[str, Any]] = None  # causality_protocol output
    reproduces: Optional[str] = None  # paper id this replicates, if type == replication
    schema_version: int = 1
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    manifest_sha256: Optional[str] = None  # filled at finalize()

    def finalize(self) -> "PublicationManifest":
        payload = asdict(self)
        payload["manifest_sha256"] = None
        encoded = json.dumps(payload, sort_keys=True, default=str).encode()
        self.manifest_sha256 = hashlib.sha256(encoded).hexdigest()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_publication_manifest(
    *,
    title: str,
    author: str,
    type: EntryType,
    license: str = "apache-2.0",
    model_id: Optional[str] = None,
    claim: Optional[str] = None,
    numbers: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[str]] = None,
    methodology_check: Optional[Dict[str, Any]] = None,
    reproduces: Optional[str] = None,
) -> PublicationManifest:
    m = PublicationManifest(
        title=title,
        author=author,
        type=type,
        license=license,
        model_id=model_id,
        claim=claim,
        numbers=numbers or {},
        artifacts=artifacts or [],
        methodology_check=methodology_check,
        reproduces=reproduces,
    )
    return m.finalize()


def validate(manifest: PublicationManifest) -> List[str]:
    """Return a list of validation errors. Empty list = valid."""
    errs: List[str] = []
    if not manifest.title or len(manifest.title) > 200:
        errs.append("title must be 1-200 chars")
    if not manifest.author:
        errs.append("author required (github handle preferred)")
    if manifest.license not in {"apache-2.0", "mit", "cc-by-4.0", "cc0"}:
        errs.append(f"license `{manifest.license}` not in allowed set")
    if manifest.type == "probe-result" and not manifest.model_id:
        errs.append("probe-result entries must include model_id")
    if manifest.type == "replication" and not manifest.reproduces:
        errs.append("replication entries must name what they reproduce (paper id)")
    return errs
