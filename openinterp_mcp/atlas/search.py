"""Atlas search facade — embed a natural-language query and return top-K related entries."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import httpx

from openinterp_mcp.atlas.embeddings import embed_text
from openinterp_mcp.atlas.vector_store import AtlasEntry, VectorStore


REGISTRY_INDEX_URL = os.environ.get(
    "OPENINTERP_REGISTRY_INDEX",
    "https://raw.githubusercontent.com/OpenInterpretability/registry/main/index.json",
)


def search_atlas(query: str, top_k: int = 10, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Embed query, search local vector store, return top-K with similarity scores."""
    store = VectorStore()
    if store.count() == 0:
        raise RuntimeError("Atlas index is empty. Run `refresh_index()` first.")
    q_emb = np.array(embed_text(query))
    hits = store.search(q_emb, top_k=top_k, filter_type=filter_type)
    return [
        {
            "similarity": sim,
            "slug": entry.slug,
            "title": entry.title,
            "author": entry.author,
            "type": entry.type,
            "model_id": entry.model_id,
            "claim": entry.claim,
            "hf_url": entry.hf_url,
            "doi": entry.doi,
            "url": f"https://openinterp.org/atlas/{entry.manifest_sha256[:10]}",
        }
        for sim, entry in hits
    ]


def refresh_index(registry_index_url: str = REGISTRY_INDEX_URL) -> int:
    """Pull the latest atlas index from the registry repo and re-embed any new entries.

    The registry repo maintains a single `index.json` listing every atlas entry. This function
    fetches it, computes embeddings for entries not yet in the local sqlite store, and writes them.
    Returns the number of new entries added.
    """
    store = VectorStore()
    with httpx.Client(timeout=30.0) as c:
        r = c.get(registry_index_url)
    if r.status_code == 404:
        return 0
    r.raise_for_status()
    index = r.json()

    existing = {row[0] for row in store.conn.execute("SELECT slug FROM entries").fetchall()}
    new_entries = [e for e in index.get("entries", []) if e["slug"] not in existing]
    if not new_entries:
        return 0

    texts = [_embed_text_for(e) for e in new_entries]
    from openinterp_mcp.atlas.embeddings import embed_batch
    embeddings = embed_batch(texts)

    for entry_dict, emb in zip(new_entries, embeddings):
        store.upsert(AtlasEntry(
            slug=entry_dict["slug"],
            title=entry_dict["title"],
            author=entry_dict["author"],
            type=entry_dict["type"],
            model_id=entry_dict.get("model_id"),
            claim=entry_dict.get("claim"),
            hf_url=entry_dict.get("hf_url"),
            doi=entry_dict.get("doi"),
            manifest_sha256=entry_dict["manifest_sha256"],
            created_at=entry_dict["created_at"],
            embedding=np.array(emb),
        ))
    return len(new_entries)


def _embed_text_for(entry: Dict[str, Any]) -> str:
    parts = [entry.get("title", ""), entry.get("claim", "") or "", entry.get("model_id", "") or "",
             entry.get("type", ""), entry.get("author", "")]
    return " | ".join(p for p in parts if p)
