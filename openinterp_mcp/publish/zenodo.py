"""Zenodo DOI deposit. Zenodo (CERN-hosted, free) gives us permanent academic-grade citations
for atlas publications. Each publish() call creates a Zenodo deposit in parallel with the HF upload.

The researcher provides ZENODO_TOKEN (env or Colab Secret). We don't host anything.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from openinterp_mcp.publish.manifest import PublicationManifest


ZENODO_API = os.environ.get("ZENODO_API", "https://zenodo.org/api")


def _token() -> str:
    t = os.environ.get("ZENODO_TOKEN")
    if not t:
        raise RuntimeError(
            "ZENODO_TOKEN missing. Get one at https://zenodo.org/account/settings/applications/tokens/new/"
        )
    return t


def deposit_zenodo(manifest: PublicationManifest, hf_url: str, sandbox: bool = False) -> Dict[str, Any]:
    """Create a Zenodo deposit, attach the manifest JSON, publish, return DOI + URL.

    `sandbox=True` uses sandbox.zenodo.org for testing (no real DOI, no permanence).
    """
    api = "https://sandbox.zenodo.org/api" if sandbox else ZENODO_API
    headers = {"Authorization": f"Bearer {_token()}"}

    metadata = _build_metadata(manifest, hf_url)

    with httpx.Client(timeout=60.0) as c:
        # 1. create empty deposition
        r = c.post(f"{api}/deposit/depositions", json={}, headers=headers)
        r.raise_for_status()
        dep = r.json()
        dep_id = dep["id"]
        bucket_url = dep["links"]["bucket"]

        # 2. upload manifest as the deposit file
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(manifest.to_dict(), f, indent=2, default=str)
            f.flush()
            local = Path(f.name)
        try:
            with open(local, "rb") as fp:
                r = c.put(f"{bucket_url}/manifest.json", content=fp.read(), headers=headers)
                r.raise_for_status()
        finally:
            local.unlink(missing_ok=True)

        # 3. attach metadata
        r = c.put(f"{api}/deposit/depositions/{dep_id}", json={"metadata": metadata}, headers=headers)
        r.raise_for_status()

        # 4. publish to mint a DOI
        r = c.post(f"{api}/deposit/depositions/{dep_id}/actions/publish", headers=headers)
        r.raise_for_status()
        published = r.json()

    return {
        "id": dep_id,
        "doi": published.get("doi"),
        "doi_url": published.get("doi_url"),
        "html_url": published.get("links", {}).get("html"),
    }


def _build_metadata(manifest: PublicationManifest, hf_url: str) -> Dict[str, Any]:
    keywords = ["mechanistic-interpretability", "openinterp", manifest.type]
    if manifest.model_id:
        keywords.append(manifest.model_id)

    related = [
        {"identifier": hf_url, "relation": "isVersionOf", "scheme": "url"},
        {"identifier": "https://openinterp.org", "relation": "isPartOf", "scheme": "url"},
    ]
    if manifest.reproduces:
        related.append({"identifier": manifest.reproduces, "relation": "isReplicationOf", "scheme": "url"})

    return {
        "title": manifest.title,
        "upload_type": "dataset",
        "description": (manifest.claim or "OpenInterp Atlas publication.") + f"<br/><br/>Manifest SHA256: <code>{manifest.manifest_sha256}</code>",
        "creators": [{"name": manifest.author, "affiliation": "OpenInterpretability"}],
        "keywords": keywords,
        "access_right": "open",
        "license": manifest.license,
        "related_identifiers": related,
        "communities": [{"identifier": "openinterp"}],
    }
