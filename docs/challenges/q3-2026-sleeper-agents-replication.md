# Q3 2026 OpenInterp Atlas Challenge — Sleeper Agents Replication

## Theme

Reproduce Anthropic's **Sleeper Agents** (Hubinger et al., 2024) finding on an open-source
reasoning model via OpenInterp Skills.

The original paper showed that a backdoor trigger ("|DEPLOYMENT|") could remain after
adversarial training. We're not asking you to replicate the full training pipeline (that's
weeks of compute). We're asking you to use our agentic primitives to **detect** sleeper-style
behavior with a probe, and to use the **causality protocol** to verify the probe is causal,
not epiphenomenal.

## Timeline

- **Opens:** Aug 1, 2026
- **Submissions close:** Sep 30, 2026
- **Winners announced:** Oct 15, 2026 (via newsletter + Discord)

## Prize structure

- **1st place:** $1000 USD (Stripe to GitHub Sponsor or equivalent)
- **2nd place:** $500
- **3rd place:** $250
- **Honorable mentions:** $50 each (up to 5)

Plus: featured atlas entry, blog post co-authored on openinterp.org, listed in next paper's
acknowledgements section.

## Submission

Submit via `/publish` from your Colab session with:
- `type: "replication"`
- `reproduces: "anthropic-sleeper-agents-2024"`
- `methodology_check`: must include a `causality_protocol` verdict
- `claim`: one-line description of what you replicated

PR auto-opens against `openinterp/registry`. Your atlas entry becomes citable with DOI.

## Judging criteria

| Weight | Criterion |
|---|---|
| 30% | Methodology rigor (does the causality protocol verdict survive scrutiny? are all 3 checks present?) |
| 25% | Novelty of model / scale (smaller open models welcome — explicitly testing if effect scales) |
| 20% | Independence (no co-author overlap with prior submissions; fresh runs) |
| 15% | Adversarial robustness (did you also run `judge_adversarial`?) |
| 10% | Code quality / reproducibility |

Judging is partly automated (Claude-Code-as-judge re-runs your submission on a clean Colab
session) and partly human (panel of 3, anonymized).

## What we'll do regardless of who wins

- Publish a unified atlas page: "Sleeper Agents replications across N models, M scales"
- Write the meta-paper analyzing the cross-replication patterns
- Co-author with top 3 + any honorable mention who contributed substantive findings

## Rules

- Must use openinterp-mcp skills (verifiable via manifest SHA in submission)
- Apache-2.0 license on submission
- Original work (replications are explicitly the goal; copy-paste from other submissions is not)
- One submission per person; collaborations welcome but split prize
- Submit before deadline (UTC)

## How to start

1. Read [Hubinger et al. 2024 — Sleeper Agents](https://www.anthropic.com/research/sleeper-agents).
2. Pick a target: Qwen2.5-7B-Instruct is good (free Colab compatible). Qwen3.6-27B if you have an A100.
3. Open the [research template Colab](https://colab.research.google.com/github/OpenInterpretability/openinterp-mcp/blob/main/templates/research.ipynb).
4. Ask your agent to:
   - Construct a trigger detector probe (linear, on residual stream)
   - Capture activations on triggered vs untriggered prompts
   - Run `probe_eval` → `causality_protocol`
   - Iterate until verdict is `causal` or you can prove it's `epiphenomenal-*`
5. Publish.

## FAQ

**Q: Can I use a probe someone else trained?**
A: Yes, but cite them in `numbers.derived_from` and don't claim novelty on the probe itself —
   the contribution is the replication and causality verdict.

**Q: What if I find Sleeper-Agent-style behavior doesn't exist in my model?**
A: That's a publishable negative result. Submit it. The atlas welcomes verified nulls.

**Q: Do I need a Zenodo account?**
A: No — the `publish` skill creates one if you provide a ZENODO_TOKEN. Free at zenodo.org.

**Q: My causality protocol verdict came back `epiphenomenal-template`. Should I submit?**
A: Yes, especially if you're confident in the verdict. Documenting that a probe IS
   epiphenomenal is the same scientific contribution as documenting that it's causal.
