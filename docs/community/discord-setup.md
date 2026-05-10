# Discord server setup

Manual one-time setup. Discord doesn't let bots create servers programmatically; the founder
creates the server in the Discord UI and then the bot fills it.

## 1. Create the server

- Name: **OpenInterp**
- Icon: openinterp logo (use existing favicon)
- Region: auto

## 2. Create channels

Categories and channels:

```
INFO
├── #welcome           — pinned: README + code of conduct + /start link
├── #announcements     — read-only, webhook-only
└── #rules

RESEARCH
├── #general           — open chat
├── #methodology       — paper-6 protocol discussion, debugging causality_protocol verdicts
├── #saes              — SAE training, feature interpretation
├── #probes            — probe training, deployment
└── #replications      — Sleeper Agents et al. — coordinate replications

PRODUCTS
├── #mcp-help          — openinterp-mcp issues / install / Colab woes
├── #atlas             — webhook from registry: every new publish lands here
└── #citations         — webhook from citation-tracker: daily digest

EVENTS
├── #q3-2026-challenge — current quarterly challenge
└── #office-hours      — synchronous Q&A schedule
```

## 3. Roles

- **Founder** (Caio): full admin
- **Contributor** — auto-assigned to anyone who's published an atlas entry (bot syncs from
  registry contributors.json)
- **Researcher** — opt-in via reaction, gives access to RESEARCH channels
- **Bot** — for the webhook-posters

## 4. Webhooks

Create webhooks for #atlas, #citations, #q3-2026-challenge. Save URLs as GitHub secrets:

```
DISCORD_WEBHOOK_ATLAS
DISCORD_WEBHOOK_CITATIONS
DISCORD_WEBHOOK_CHALLENGE
```

Add to:
- registry repo CI (post on merge to main)
- openinterp-mcp citation-tracker workflow

## 5. Invite link

Permanent, no expiration. Add to:
- openinterp.org footer
- README of all openinterp repos
- mcp landing page (`/mcp`)

## 6. First seeds

Manually invite 8-12 people in week 1 before listing publicly. Target list:
- 2-3 MATS Winter 2026 alumni
- 2-3 Apart Research fellows
- 2-3 AI Safety Camp participants
- 1-2 PhD students in interp (via Twitter DM)

Aim for active conversation in #methodology and #replications by week 2 before any public push.
