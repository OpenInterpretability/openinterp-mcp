"""Top-level publish() entry — wraps manifest build → HF upload → registry PR → Zenodo DOI.

Each step is best-effort; failure in one does not block the others. Returns whatever succeeded.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from openinterp_mcp.publish.hf_dataset import upload_publication
from openinterp_mcp.publish.manifest import build_publication_manifest, validate
from openinterp_mcp.publish.registry_pr import open_registry_pr


def publish(
    result_payload: Dict[str, Any],
    *,
    title: str,
    author: str,
    type: str = "atlas-entry",
    license: str = "apache-2.0",
    model_id: Optional[str] = None,
    claim: Optional[str] = None,
    reproduces: Optional[str] = None,
    methodology_check: Optional[Dict[str, Any]] = None,
    numbers: Optional[Dict[str, Any]] = None,
    skip_zenodo: bool = False,
    skip_registry_pr: bool = False,
) -> Dict[str, Any]:
    """Publish a result to the public atlas. All optional steps; researcher controls each."""
    manifest = build_publication_manifest(
        title=title, author=author, type=type, license=license, model_id=model_id,
        claim=claim, reproduces=reproduces, methodology_check=methodology_check, numbers=numbers,
    )
    errs = validate(manifest)
    if errs:
        return {"ok": False, "validation_errors": errs}

    out: Dict[str, Any] = {"ok": True, "manifest_sha256": manifest.manifest_sha256}

    try:
        hf_repo_id, hf_url = upload_publication(manifest, result_payload)
        out["hf_repo_id"] = hf_repo_id
        out["hf_url"] = hf_url
    except Exception as e:
        out["hf_error"] = str(e)
        return out

    if not skip_registry_pr:
        try:
            pr = open_registry_pr(manifest, hf_repo_id, hf_url)
            out["registry_pr"] = pr
        except Exception as e:
            out["registry_pr_error"] = str(e)

    if not skip_zenodo:
        try:
            from openinterp_mcp.publish.zenodo import deposit_zenodo
            from openinterp_mcp.publish.bibtex import to_bibtex

            zenodo = deposit_zenodo(manifest, hf_url)
            out["doi"] = zenodo.get("doi")
            out["zenodo_url"] = zenodo.get("html_url")
            out["bibtex"] = to_bibtex(manifest, doi=zenodo.get("doi"))
        except Exception as e:
            out["zenodo_error"] = str(e)
            try:
                from openinterp_mcp.publish.bibtex import to_bibtex
                out["bibtex"] = to_bibtex(manifest, doi=None)
            except Exception:
                pass

    out["atlas_url"] = f"https://openinterp.org/atlas/{manifest.manifest_sha256[:10]}"
    return out
