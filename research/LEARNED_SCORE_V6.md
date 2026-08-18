# Learned Score V6 contract

V6 learns how strongly twenty visible validation diagnostics should influence
candidate ranking. It is an offline research artifact, not a model trained on
an advertiser's uploaded data.

## Label

Every synthetic business has a hidden response surface. Each fitted candidate
recommends budgets for reduction, fixed-budget reallocation, growth, and an
economic-ceiling decision. The uncapped economic opportunity loss across those
decisions is the training label. Hidden ROI and allocation truth never enter the
MMM fit or its validation diagnostics.

## Model

Candidate pairs from the same business are ordered by lower hidden decision
loss. A constrained monotonic pairwise logistic ranker learns non-negative
diagnostic weights:

```text
100 × exp(Σ diagnostic_weight × log(diagnostic_score / 100))
```

No diagnostic can be rewarded for becoming worse. Each readable validation
layer retains a declared minimum and maximum total weight, and no individual
diagnostic may dominate the score.

## Split and evidence contract

Generator families, not candidate rows, define train, validation, and sealed
audit splits. Experiments are generated as uncertain measurements of the causal
ROI attributable to spend in a declared campaign window. The outcome window may
extend beyond campaign end to capture carryover. Candidate fitting sees only the
reported experiment and its uncertainty; the local causal ROI remains hidden.

V6 ranks only within the highest available immutable eligibility tier. It
cannot turn a failed temporal, structural, external-anchor, placebo, or ROI
coherence gate into a decision-grade candidate.

## Reproducibility

```bash
pnpm research:score-v3:audit
pnpm research:score-v6:pilot
```

The small runtime JSON artifact contains the learned weights, split metrics,
activation gates, versioned simulator contract, and limitations. The full
candidate cohort is retained locally for research reproducibility but is not
loaded by the product UI.
