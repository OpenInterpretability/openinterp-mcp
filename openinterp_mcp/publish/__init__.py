"""Publication pipeline — turn an in-memory experiment result into a citable atlas entry.

User flow:
  result_dict = {...}              # from /probe-eval, /causality-protocol, etc.
  out = publish(result_dict,
                title="L20 saturation-direction on BigCodeBench",
                author="caiovicentino",
                license="apache-2.0")
  # out = {"hf_repo_id": "openinterp-community/atlas-...",
  #        "atlas_url": "https://openinterp.org/atlas/...",
  #        "doi": "10.5281/zenodo.7891234",  # populated in Fase 7
  #        "registry_pr_url": "...",          # populated in Fase 6
  #        "bibtex": "..."}                   # populated in Fase 7

Everything is opt-in. Nothing is uploaded unless the researcher calls `publish()` explicitly.
"""
from openinterp_mcp.publish.api import publish
from openinterp_mcp.publish.manifest import build_publication_manifest, PublicationManifest

__all__ = ["publish", "build_publication_manifest", "PublicationManifest"]
