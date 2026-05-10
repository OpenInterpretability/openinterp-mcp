"""Open a PR to openinterp/registry (the GitHub index that openinterp.org reads).

Registry layout:
  registry/
    atlas/<year>/<slug>.json     ← one file per publication, manifest contents
    replications/<paper-id>/<slug>.json
    probebench/<slug>.json
    contributors/<github-handle>.json   ← auto-generated, indexes a user's submissions
"""
from __future__ import annotations

import json
import os
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, Optional

from openinterp_mcp.publish.manifest import PublicationManifest


REGISTRY_REPO = os.environ.get("OPENINTERP_REGISTRY_REPO", "OpenInterpretability/registry")


def open_registry_pr(
    manifest: PublicationManifest,
    hf_repo_id: str,
    hf_url: str,
    github_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Open a PR to the registry index repo with a new entry. Returns dict with PR url + branch."""
    try:
        from github import Github, GithubException
    except ImportError as e:
        raise RuntimeError("Registry PRs need PyGithub. `pip install PyGithub`") from e

    token = github_token or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN required to open registry PR. Set in env.")

    gh = Github(token)
    repo = gh.get_repo(REGISTRY_REPO)

    branch_name = f"publish/{manifest.manifest_sha256[:10]}"
    base = repo.get_branch("main")

    try:
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base.commit.sha)
    except GithubException as e:
        if e.status != 422:  # 422 = already exists
            raise

    year = time.strftime("%Y")
    if manifest.type == "replication":
        path = f"replications/{manifest.reproduces}/{year}/{manifest.manifest_sha256[:10]}.json"
    elif manifest.type == "probe-result":
        path = f"probebench/{year}/{manifest.manifest_sha256[:10]}.json"
    else:
        path = f"atlas/{year}/{manifest.manifest_sha256[:10]}.json"

    entry = {
        **manifest.to_dict(),
        "hf_repo_id": hf_repo_id,
        "hf_url": hf_url,
    }
    content = json.dumps(entry, indent=2, default=str)

    try:
        existing = repo.get_contents(path, ref=branch_name)
        repo.update_file(path, f"update {manifest.title}", content, existing.sha, branch=branch_name)
    except GithubException as e:
        if e.status == 404:
            repo.create_file(path, f"add {manifest.title}", content, branch=branch_name)
        else:
            raise

    body = textwrap.dedent(f"""
        Automated atlas submission.

        - **Title:** {manifest.title}
        - **Author:** @{manifest.author}
        - **Type:** `{manifest.type}`
        - **HF dataset:** {hf_url}
        - **Manifest SHA256:** `{manifest.manifest_sha256}`

        Auto-merge once `openinterp-judge` CI passes (causality_protocol replication on a clean Colab session).
    """).strip()

    pr = repo.create_pull(
        title=f"publish: {manifest.title}",
        body=body,
        head=branch_name,
        base="main",
    )
    return {"pr_url": pr.html_url, "pr_number": pr.number, "branch": branch_name, "path": path}
