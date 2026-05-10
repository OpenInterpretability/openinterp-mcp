"""Citation tracking — scan external sources for mentions of atlas DOIs / URLs.

Sources queried:
  - Semantic Scholar API (free, ~100 req/5min unauthenticated)
  - arXiv full-text search via export.arxiv.org
  - GitHub code search for openinterp.org/atlas/ + DOI strings

For each atlas entry, we record:
  citing_paper_id, source, snippet, found_at, url

Output is a JSON-lines file that the registry repo commits back to itself nightly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx


S2_API = "https://api.semanticscholar.org/graph/v1"
ARXIV_API = "http://export.arxiv.org/api/query"


def scan_semantic_scholar(query: str, since_year: int = 2026) -> List[Dict[str, Any]]:
    """Search Semantic Scholar for papers mentioning `query` (typically a DOI or atlas URL)."""
    with httpx.Client(timeout=30.0, headers={"User-Agent": "openinterp-citation-bot/0.1"}) as c:
        r = c.get(
            f"{S2_API}/paper/search",
            params={"query": query, "year": f"{since_year}-", "fields": "paperId,title,authors,year,url,abstract"},
        )
    if r.status_code != 200:
        return []
    hits = r.json().get("data", [])
    return [
        {
            "source": "semantic_scholar",
            "paper_id": h.get("paperId"),
            "title": h.get("title"),
            "authors": [a.get("name") for a in h.get("authors", [])],
            "year": h.get("year"),
            "url": h.get("url"),
            "snippet": (h.get("abstract") or "")[:300],
        }
        for h in hits
    ]


def scan_arxiv(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """Full-text-like search via arXiv. Note: arXiv search API only matches metadata not body —
    use Semantic Scholar for citation-discovery; arXiv for new-listing alerts."""
    with httpx.Client(timeout=30.0) as c:
        r = c.get(ARXIV_API, params={"search_query": f"all:{query}", "max_results": max_results})
    if r.status_code != 200:
        return []
    # Minimal XML parsing without external deps
    text = r.text
    entries = []
    for chunk in text.split("<entry>")[1:]:
        end = chunk.find("</entry>")
        body = chunk[:end] if end != -1 else chunk
        entries.append({
            "source": "arxiv",
            "title": _xml_extract(body, "title"),
            "url": _xml_extract(body, "id"),
            "snippet": _xml_extract(body, "summary")[:300],
            "year": _xml_extract(body, "published")[:4],
        })
    return entries


def _xml_extract(text: str, tag: str) -> str:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    s = text.find(open_tag)
    if s == -1: return ""
    s += len(open_tag)
    e = text.find(close_tag, s)
    if e == -1: return ""
    return text[s:e].strip()


def scan_all_for_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Query all sources for an atlas entry. Returns flattened citation hits."""
    queries = []
    if entry.get("doi"):
        queries.append(entry["doi"])
    queries.append(f"openinterp.org/atlas/{entry['manifest_sha256'][:10]}")

    hits: List[Dict[str, Any]] = []
    for q in queries:
        try:
            hits.extend(scan_semantic_scholar(q))
        except Exception:
            pass
        try:
            hits.extend(scan_arxiv(q, max_results=10))
        except Exception:
            pass
        time.sleep(1.0)  # be polite

    seen = set()
    deduped = []
    for h in hits:
        key = (h.get("source"), h.get("paper_id") or h.get("url"))
        if key not in seen:
            seen.add(key)
            h["found_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            h["atlas_slug"] = entry["slug"]
            deduped.append(h)
    return deduped


def scan_registry(registry_index_url: str, output_path: Path) -> int:
    """Walk every entry in the registry index, scan citations, append to output JSONL.

    Designed to be idempotent — if a (paper_id, atlas_slug) pair is already in output, it's skipped.
    """
    with httpx.Client(timeout=30.0) as c:
        r = c.get(registry_index_url)
    if r.status_code != 200:
        return 0
    index = r.json()

    existing_keys = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            try:
                obj = json.loads(line)
                existing_keys.add((obj.get("source"), obj.get("paper_id") or obj.get("url"), obj.get("atlas_slug")))
            except Exception:
                continue

    new_count = 0
    with output_path.open("a") as f:
        for entry in index.get("entries", []):
            hits = scan_all_for_entry(entry)
            for h in hits:
                key = (h.get("source"), h.get("paper_id") or h.get("url"), h.get("atlas_slug"))
                if key in existing_keys:
                    continue
                f.write(json.dumps(h, default=str) + "\n")
                new_count += 1
                existing_keys.add(key)
    return new_count
