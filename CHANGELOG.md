# Changelog

All notable changes to `openinterp-mcp` are documented here. Versioning follows [SemVer](https://semver.org/).

## [0.2.0] — 2026-06-17

### Added — `causality_protocol` Check 4 (paper-11, "Form, Not Granted")

A high AUROC can be a **structural / lexical / punctuation** confound, not the concept a probe
claims to read. The v0.1.0 three-check protocol would have stamped a direction with AUROC 0.838
that collapsed to 0.08 under structure-matching as `causal`. v0.2.0 adds the fourth mandatory check:

- **`structure_matched_capture_id` + `structure_matched_labels`** — re-evaluate the *same* probe on
  a confound-balanced set. If the above-chance signal does not survive → new verdict
  **`confounded-structural`**.
- **`feature_name`** — a causal claim now *requires* a named feature (max-activating context). High
  AUROC without naming + a structure-matched control → new verdict **`inconclusive-unverified`**
  (neither causal nor null; lists the missing control).

Verdict set is now `{causal, weak-causal, confounded-structural, epiphenomenal-softmax,
epiphenomenal-template, inconclusive-unverified, undetermined}`. `causal` is only returned when the
signal survives the structure-matched control **and** the feature is named. The new `causality-protocol`
SKILL.md documents the four checks and the failure mode.

### Added — skills for the remaining typed tools (8 tools ↔ 8 skills)

- **`sae-lookup`** — decompose a captured activation into named SAE features (full-stack SAE on Qwen3.6-27B).
- **`list-probes`** — inventory the probes loaded in the backend.
- **`colab-status`** — session health (model, probes, captures in memory).

### Changed / Fixed

- `Citation tracker` workflow is gated behind `vars.ENABLE_CITATION_TRACKER` (off by default → skipped,
  not failed) until a populated registry index + push rights are configured.
- Fixed a stale Phase-1 scaffold test (`/steer` is implemented; returns 503 without a loaded model, not a 501 stub).

## [0.1.0] — 2026-05-10

Initial release: MCP server + Colab backend, 8 typed tools, the three-check causality protocol,
atlas publishing with Zenodo DOIs, and the first five skills.
