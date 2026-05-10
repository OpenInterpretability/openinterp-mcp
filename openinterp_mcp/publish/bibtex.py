"""Auto-generate BibTeX citations for atlas publications.

Output format follows the @misc convention used in ML papers for non-peer-reviewed
artifacts. When a Zenodo DOI is available, it's included.
"""
from __future__ import annotations

import re
import time
from typing import Optional

from openinterp_mcp.publish.manifest import PublicationManifest


def to_bibtex(manifest: PublicationManifest, doi: Optional[str] = None) -> str:
    key = _make_key(manifest)
    fields = [
        ("author", manifest.author),
        ("title", manifest.title),
        ("year", time.strftime("%Y")),
        ("howpublished", "OpenInterp Atlas — https://openinterp.org"),
        ("note", f"Manifest SHA256: {manifest.manifest_sha256}"),
    ]
    if doi:
        fields.append(("doi", doi.replace("https://doi.org/", "")))
        fields.append(("url", f"https://doi.org/{doi.replace('https://doi.org/', '')}"))
    else:
        fields.append(("url", f"https://openinterp.org/atlas/{manifest.manifest_sha256[:10]}"))

    lines = [f"@misc{{{key},"]
    for k, v in fields:
        lines.append(f"  {k} = {{{v}}},")
    lines[-1] = lines[-1].rstrip(",")  # no trailing comma on last field
    lines.append("}")
    return "\n".join(lines)


def _make_key(manifest: PublicationManifest) -> str:
    """Construct a BibTeX key: <author><year><first-word-of-title>."""
    year = time.strftime("%Y")
    first_word = re.split(r"[\s\-_:]+", manifest.title.strip())[0].lower()
    first_word = re.sub(r"[^a-z0-9]", "", first_word)[:12]
    author = re.sub(r"[^a-zA-Z0-9]", "", manifest.author).lower()
    return f"{author}{year}{first_word}"
