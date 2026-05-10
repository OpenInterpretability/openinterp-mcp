"""Discord bot — announces atlas publishes, citation alerts, challenge milestones.

Architecture:
  - GitHub Actions in the registry repo POSTs to a Discord webhook on `git merge atlas/...`
  - Citation tracker workflow POSTs a daily summary to the #citations channel
  - Challenge events posted manually or via /challenge slash command (future)

We deliberately do NOT build a full bot daemon — webhooks suffice for v1 and need no
always-on infra.
"""
from openinterp_mcp.discord.webhook import post_atlas_announce, post_citation_summary

__all__ = ["post_atlas_announce", "post_citation_summary"]
