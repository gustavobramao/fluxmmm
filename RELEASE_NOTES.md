# FluxMMM 3.0.0 — RegretSet-MMM V9

Released 5 September 2026.

This release publishes the frozen RegretSet-MMM V9 research selector alongside
the local-first FluxMMM product. It is a scientific-method release, not a claim
that simulation makes observational MMM automatically causal.

## Main changes

- A permutation-invariant DeepSets selector learns from complete 48-candidate
  advertiser sets rather than scoring candidates in isolation.
- Each candidate is represented by 232 truth-blind tokens spanning validation,
  posterior decision geometry, evidence attribution, and model specification.
- Multi-head prediction covers mean, median, P90, P95, CVaR90, four decision
  scenarios, and dangerous-false-champion probability.
- The selection target is downstream economic regret under a declared 65% mean
  / 35% P90 business-risk preference.
- A finite-candidate theorem connects selected-model capped excess regret to
  regret-weighted oracle misranking.
- Evidence influence is continuously attributed to observational data,
  experiments, benchmarks, and regularization using local curvature and
  conditional information for correlated media.
- The frozen method passed all 9 predeclared checks on a new 140-business
  internal sealed audit with zero observed theorem violations.
- Without retraining, it was challenged on 100 businesses from Google AMSS
  1.0.1: 4,800 SVI fits and 19,200 frozen budget actions.
- Mean capped AMSS decision risk was 0.230 for RegretSet-MMM versus 0.400 for
  uniform random valid-candidate selection, a 42.5% reduction. The in-pool
  oracle lower bound was 0.012.

## Product changes

- A Score Research surface explains the V9 target, scientific firewall,
  development design, evidence attribution, internal audit, and AMSS external
  validation without relabeling the fast Agentic runtime as V9.
- Evidence-attribution and leave-source-out decision-dependence diagnostics are
  available in the shared MMM library.
- The PyMC sampler preserves channel-specific response contracts and remains
  contract-matched to promoted specifications.

## Reproducibility

```bash
pnpm install --frozen-lockfile
pnpm test:svi-score-v9
pnpm test:svi-score-v10-amss
```

The compact frozen selector, token registry, protocols, result summaries,
candidate-level AMSS losses, acceptance hashes, and publication-format paper
are versioned. Multi-gigabyte posterior checkpoints and generated cohorts
remain excluded from Git and are reproducible from the declared contracts.

## Claim boundary

The result supports external synthetic transport under the declared AMSS
adapter, candidate pool, FullRankADVI contract, and utility. It does not
establish real-advertiser effectiveness, universal SOTA performance, or causal
validity for an observational dataset. V2 remains available from tag `v2.0.0`.
