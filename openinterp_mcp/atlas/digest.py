"""Email digest of new citations for atlas contributors.

Sends one email per contributor with a summary of new citations in the last N days, via
caio@openinterp.org SMTP (ImprovMX). Opt-in — only contributors who explicitly registered
an email in their contributor profile receive digests.
"""
from __future__ import annotations

import json
import os
import smtplib
from collections import defaultdict
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List


SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.improvmx.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_FROM = os.environ.get("SMTP_FROM", "caio@openinterp.org")
SMTP_USER = os.environ.get("SMTP_USER", SMTP_FROM)
SMTP_PASS = os.environ.get("SMTP_PASS")


def build_digest_text(handle: str, hits: List[dict]) -> str:
    lines = [
        f"Hi @{handle},",
        "",
        f"Your atlas contributions were cited {len(hits)} time(s) since the last digest.",
        "",
    ]
    by_slug: Dict[str, List[dict]] = defaultdict(list)
    for h in hits:
        by_slug[h["atlas_slug"]].append(h)
    for slug, slug_hits in by_slug.items():
        lines.append(f"## openinterp.org/atlas/{slug[:10]}")
        for h in slug_hits:
            t = h.get("title", "(untitled)")
            url = h.get("url", "")
            src = h.get("source")
            lines.append(f"  - [{src}] {t}")
            if url:
                lines.append(f"    {url}")
        lines.append("")
    lines.append("Unsubscribe: reply STOP. Manage at https://openinterp.org/community/" + handle)
    return "\n".join(lines)


def send_digest(to_email: str, handle: str, hits: List[dict]) -> bool:
    if not SMTP_PASS:
        raise RuntimeError("SMTP_PASS not set. Aborting digest send.")
    msg = MIMEText(build_digest_text(handle, hits), "plain", "utf-8")
    msg["Subject"] = f"OpenInterp Atlas: {len(hits)} new citation(s)"
    msg["From"] = SMTP_FROM
    msg["To"] = to_email

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_FROM, [to_email], msg.as_string())
    return True


def run_digest_for_all(
    contributors_path: Path,
    citations_jsonl: Path,
    since_iso: str,
) -> int:
    """Send digests for all opted-in contributors. Returns number of emails sent."""
    contributors = json.loads(contributors_path.read_text())  # {handle: {email, ...}}
    citations_by_slug: Dict[str, List[dict]] = defaultdict(list)
    if not citations_jsonl.exists():
        return 0
    for line in citations_jsonl.read_text().splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("found_at", "") >= since_iso:
            citations_by_slug[obj["atlas_slug"]].append(obj)

    handle_to_hits: Dict[str, List[dict]] = defaultdict(list)
    for handle, contrib in contributors.items():
        if not contrib.get("email") or not contrib.get("digest_optin"):
            continue
        for slug in contrib.get("slugs", []):
            handle_to_hits[handle].extend(citations_by_slug.get(slug, []))

    sent = 0
    for handle, hits in handle_to_hits.items():
        if not hits:
            continue
        email = contributors[handle]["email"]
        try:
            send_digest(email, handle, hits)
            sent += 1
        except Exception as e:
            print(f"failed for @{handle}: {e}")
    return sent
