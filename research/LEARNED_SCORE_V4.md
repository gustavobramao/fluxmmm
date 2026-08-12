# Learned Score V4 contract

V4 replaces the fixed ranking weights with a small, versioned model learned
offline from synthetic businesses whose causal response and budget-decision
truth are known. It does not train inside an advertiser session and does not
send data to an external service.

## Target and model

For candidate `m` in business `b`, the label is mean contribution-profit regret
across budget reduction, fixed-budget reallocation, budget growth, and economic
ceiling decisions. The learned score preserves the interpretable geometric form:

```text
100 × G^wG × S^wS × C^wC × D^wD
```

The weights are non-negative, sum to one, have a 0.05 floor, and are selected
from a deterministic 0.025 grid. Candidate ranking is evaluated only inside the
zero-failed-gate pool. When no candidate clears all gates, the model can order
review candidates but cannot change their status.

## Split contract

- Train: 120 businesses from balanced DTC, demand-harvesting search, and social
  frequency-pressure generators.
- Validation: 40 businesses from unseen delayed-TV and correlated-planning
  generator families.
- Untouched audit: 40 businesses with wrong external evidence and swapped
  channel mechanics.
- Each business fits eight model specifications under two evidence arms, for
  3,200 candidate rows.

Generator families, not random candidate rows, define the splits. This prevents
near-duplicate candidates from the same synthetic business leaking across train
and validation.

## Risk-aware activation

An artifact is active only if it achieves all of the following:

1. At least 2% lower mean regret on family-held-out validation.
2. No increase in held-out P90 regret.
3. No validation-family mean-regret increase above 1%.
4. No material mean or P90 regression on the untouched audit.

Otherwise the fixed 20/15/25/40 heuristic remains the runtime fallback.

## Immutable safeguards

Learning changes ranking, not evidence. Temporal integrity, structural
identification, external-anchor prediction, placebo behavior, evidence
coherence, and two-sided ROI plausibility checks remain hard gates. Parameter
bounds and the declared experiment/benchmark evidence are also immutable.

## Reproducibility

```bash
pnpm research:score-v4:learn   # rebuild cohort and artifact
pnpm research:score-v4:refit   # refit from the stored cohort
```

The browser imports only `research/score_v4/artifacts/learned-score-v4.json`.
The 3,200-row cohort is retained for reproducibility but is not loaded at
runtime.
