# Evidence-Adaptive RegretSet development

This isolated development track tests whether RegretSet should learn different
reliance on predictive, causal-identification, posterior-decision, and
structural-specification signals according to the advertiser's evidence
environment. It does not place a prediction-only filter before RegretSet.

The selector retains one joint all-token encoder so interactions across
validation pillars cannot be lost. Four disjoint residual expert views are
then modulated by a learned business-level gate. The gate maps explicit
evidence availability, coverage, quality, compatibility, dependence, and
candidate-set summaries to four softmax weights. The joint and adaptive
representations are processed together by the same set-wise, multi-head
economic-loss architecture used by V9.

The reported softmax values are **adaptive residual routing weights**, not
total feature importance. The joint path also contains every candidate token.
Total reliance on a pillar must later be estimated with held-out token-group
ablation or another attribution method.

## Paired development design

The 20,160 already-fitted V9 development candidates contain 24 specifications
under each of two evidence contracts. For every one of the 420 businesses this
track constructs three set-wise training examples:

1. all 48 candidates;
2. the 24 experiments-only candidates;
3. the same 24 specifications with benchmark gap-fill.

All three sets retain the same observed outcome, spend paths, hidden causal
truth, advertiser fold, and simulator family. The economic oracle and excess
loss are recalculated within each candidate regime. This isolates the effect of
adding benchmark evidence without generating new posterior fits.

This cached design does **not** identify the counterfactual effect of removing
an experiment from a business that had one. Experiments affect the posterior;
masking experiment tokens after fitting would leak their influence and is
therefore forbidden. A future experiment-presence pairing requires new SVI
fits and a fresh independent audit.

## Governance

- Development inputs only: 420 businesses and cached FullRankADVI posteriors.
- No NUTS and no posterior refits.
- During V11 development and training, the completed AMSS cohort was not read,
  rescored, or used for model selection.
- Results are post-AMSS development evidence, not a new confirmatory claim.
- Any eventual claim requires a newly generated independent cohort.

```bash
npm run research:svi-score-v11-ea:prepare
npm run research:svi-score-v11-ea:train
npm run research:svi-score-v11-ea:verify
npm run test:svi-score-v11-ea
```

See `DEVELOPMENT_RESULT.md` for the grouped cross-validation result and its
interpretation boundary.

## Previously opened AMSS screening test

After the V11 development artifact was frozen, the already-opened 100-business
AMSS cohort may be reused as a zero-cost retrospective screen. Selection and
evaluation are deliberately separate: the first two commands read only the
cached truth-blind 232-token candidate artifact and the frozen V11 selector;
only the final command reads the accepted AMSS candidate-loss file.

```bash
npm run research:svi-score-v11-ea:amss-prepare
npm run research:svi-score-v11-ea:amss-select
npm run research:svi-score-v11-ea:amss-evaluate
npm run test:svi-score-v11-ea-amss
```

This reuse never becomes confirmatory. V11 was designed after the earlier AMSS
result had been inspected, so any result is labelled retrospective external
benchmark evidence. A newly generated untouched cohort remains necessary for
a V11 confirmatory claim.
