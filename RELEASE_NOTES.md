# FluxMMM V2.0.0

Released 18 August 2026.

FluxMMM V2 keeps the same local-first workflow and adds a versioned research
contract for model selection. It is a scientific-method release, not a claim
that synthetic validation makes observational MMM automatically causal.

## Main changes

- Learned validation ranker: twenty visible diagnostics are weighted by a
  constrained monotonic pairwise model trained on hidden synthetic budget
  decision loss. Generator families define train, validation, and sealed audit
  partitions. Immutable eligibility gates remain outside the learner.
- Same-period experiment estimand: campaign-window spend is contrasted with a
  zero-spend counterfactual through the same adstock and saturation response.
  Outcomes may extend through a declared post-test carryover window while the
  denominator remains tested spend in the campaign window.
- Consistent evidence contract: static MAP, advanced prior and likelihood
  calibration, causal anchor recovery, ROI coherence, and PyMC NUTS sampling
  use the same experiment-window definition.
- Channel-specific response contracts: media channels may use separate
  geometric or Weibull carryover and saturation settings during search and
  production sampling.
- Stronger numerical and evidence transparency: QR/SVD solving, rank and
  conditioning receipts, prior-influence labels, non-negative-boundary
  disclosure, evidence-rescue comparisons, and local workspace checkpoints.
- Agentic search uses the active learned score within the highest immutable
  eligibility tier and retains bounded multi-start, family coverage, local
  challenges, and explicit forced-promotion audit trails.

## Reproducibility

```bash
pnpm install --frozen-lockfile
pnpm test
pnpm research:score-v3:audit
pnpm research:score-v6:pilot

# optional full posterior checks
pnpm mcmc:setup
pnpm mcmc:test
```

The learned-score artifact records its simulator version, cohort split,
diagnostic weights, activation gates, held-out performance, and limitations.
Cached V1 artifacts are intentionally invalidated when the model, validation,
or sampling contract changes.

## Compatibility

- Existing daily, weekly, and monthly CSV contracts remain supported.
- Existing experiment files remain valid; `outcomeEndDate` is optional and
  defaults to the campaign end date.
- V1 remains available from the `v1.0.0` Git tag.
