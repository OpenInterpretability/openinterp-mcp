"""Discord webhook posters.

Set `DISCORD_WEBHOOK_ATLAS` / `DISCORD_WEBHOOK_CITATIONS` env vars to the channel webhook URLs.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx


def _post(webhook_env: str, payload: Dict[str, Any]) -> bool:
    url = os.environ.get(webhook_env)
    if not url:
        return False
    with httpx.Client(timeout=15.0) as c:
        r = c.post(url, json=payload)
    return r.status_code in (200, 204)


def post_atlas_announce(
    title: str,
    author: str,
    atlas_url: str,
    entry_type: str,
    hf_url: str,
    doi: str | None = None,
) -> bool:
    """Post a new atlas publication announcement to the #atlas channel."""
    fields = [
        {"name": "Author", "value": f"[@{author}](https://github.com/{author})", "inline": True},
        {"name": "Type", "value": f"`{entry_type}`", "inline": True},
        {"name": "HuggingFace", "value": f"[dataset]({hf_url})", "inline": True},
    ]
    if doi:
        fields.append({"name": "DOI", "value": f"[{doi}](https://doi.org/{doi})", "inline": True})

    embed = {
        "title": title,
        "url": atlas_url,
        "color": 0xF59E0B,
        "fields": fields,
    }
    return _post("DISCORD_WEBHOOK_ATLAS", {"embeds": [embed]})


def post_citation_summary(new_citations: List[Dict[str, Any]]) -> bool:
    """Post the daily citation digest to the #citations channel."""
    if not new_citations:
        return False
    bucket: Dict[str, List[Dict[str, Any]]] = {}
    for c in new_citations:
        bucket.setdefault(c["atlas_slug"], []).append(c)

    lines = [f"**{len(new_citations)} new citation(s)** in the last 24h:", ""]
    for slug, hits in bucket.items():
        lines.append(f"_atlas/{slug[:10]}_ — {len(hits)} mention(s)")
        for h in hits[:3]:
            lines.append(f"  • [{h.get('source')}] {h.get('title', '')[:80]}")
        if len(hits) > 3:
            lines.append(f"  …and {len(hits) - 3} more")

    return _post("DISCORD_WEBHOOK_CITATIONS", {"content": "\n".join(lines)[:1900]})
