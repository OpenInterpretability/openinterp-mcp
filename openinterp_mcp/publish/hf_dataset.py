"""Create / update an HF dataset for a single atlas publication.

Layout per dataset:
  caiovicentino1/openinterp-community-atlas-2026-05-10-<uuid>/
    manifest.json           ← PublicationManifest serialized
    result.json             ← the raw experiment result the researcher chose to publish
    README.md               ← auto-generated, human-readable summary
    artifacts/              ← optional: probe weights, capture tensors, plots
"""
from __future__ import annotations

import json
import os
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from openinterp_mcp.publish.manifest import PublicationManifest, validate


HF_NAMESPACE = "openinterp-community"


def _render_readme(manifest: PublicationManifest) -> str:
    body = f"""---
license: {manifest.license}
language: en
tags:
  - mechanistic-interpretability
  - openinterp
  - {manifest.type}
{f'  - {manifest.model_id}' if manifest.model_id else ''}
size_categories:
  - n<1K
---

# {manifest.title}

**Author:** [@{manifest.author}](https://github.com/{manifest.author})
**Type:** `{manifest.type}`
**Model:** `{manifest.model_id or 'n/a'}`
**License:** {manifest.license}
**Manifest SHA256:** `{manifest.manifest_sha256}`
**Created:** {manifest.created_at}

## Claim
{manifest.claim or '_(not provided)_'}

## Numbers
```json
{json.dumps(manifest.numbers, indent=2)}
```

## Methodology check
{json.dumps(manifest.methodology_check, indent=2) if manifest.methodology_check else '_(no causality protocol verdict attached)_'}

## Reproduces
{f'This replicates: `{manifest.reproduces}`' if manifest.reproduces else '_(original work)_'}

## Reproduce locally

```bash
pip install 'openinterp-mcp[server]'
# In a Colab session running openinterp_mcp.colab.launch(...):
openinterp-judge ./this-manifest.json
```

---
Submitted via OpenInterp MCP — https://openinterp.org/mcp
"""
    return textwrap.dedent(body).strip() + "\n"


def upload_publication(
    manifest: PublicationManifest,
    result_payload: Dict[str, Any],
    artifacts: Optional[Dict[str, bytes]] = None,
    hf_token: Optional[str] = None,
) -> Tuple[str, str]:
    """Create a new HF dataset and upload manifest + result + readme + artifacts.

    Returns: (hf_repo_id, hf_url)
    """
    errs = validate(manifest)
    if errs:
        raise ValueError(f"Invalid manifest: {errs}")

    try:
        from huggingface_hub import HfApi, login
    except ImportError as e:
        raise RuntimeError("Publication needs huggingface_hub. `pip install huggingface_hub`") from e

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required to publish. Set in Colab Secrets or env.")
    login(token=token, add_to_git_credential=False)

    slug = _slugify(manifest.title, manifest.manifest_sha256)
    repo_id = f"{HF_NAMESPACE}/atlas-{slug}"

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

    tmpdir = Path(f"/tmp/openinterp-publish-{uuid.uuid4().hex[:8]}")
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        (tmpdir / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2, default=str))
        (tmpdir / "result.json").write_text(json.dumps(result_payload, indent=2, default=str))
        (tmpdir / "README.md").write_text(_render_readme(manifest))
        if artifacts:
            (tmpdir / "artifacts").mkdir(exist_ok=True)
            for name, blob in artifacts.items():
                (tmpdir / "artifacts" / name).write_bytes(blob)

        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"publish: {manifest.title}",
        )
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    return repo_id, f"https://huggingface.co/datasets/{repo_id}"


def _slugify(title: str, sha: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
    date = time.strftime("%Y-%m-%d")
    return f"{date}-{s}-{sha[:8]}"
