# FluxMMM V5D-SVI

**Release status:** public research release, frozen exactly as evaluated.

V5D-SVI closes the inference mismatch in V5D development. Current V5D learns
SVI posterior economic loss from predominantly MAP/Laplace diagnostics.
V5D-SVI instead adds information that was available from the stored FullRank
ADVI posterior for every candidate and defines candidate safety using that same
posterior.

## No posterior refit for the development comparison

The 24,000 SVI fits already exist. The 420-business development cohort uses
20,160 of them; the audit split remains excluded. Each stored record contains
posterior channel ROI centers, predictive coverage, ROI width, implausibility,
ELBO stability, and seed agreement. The first like-for-like experiment therefore
reuses immutable posterior artifacts and retrains only the lightweight V5D
selector.

## SVI safety contract

A candidate is posterior-safe only when:

- the posterior is finite;
- ELBO drift is within the predeclared SVI tolerance;
- independent SVI seeds agree;
- at least 80% of observations are covered by the posterior predictive interval;
- at most 20% of any material channel's ROI posterior is beyond its evidence-informed bound;
- maximum relative ROI interval width is at most 3.

If no candidate passes, Flux selects only for review and does not describe the
result as decision-grade.

## Controlled 2x2 ablation

The experiment reports four cells using identical business folds, candidate
regions, learner, loss target, search algorithms, and budgets:

1. MAP features + MAP safety: frozen V5D baseline.
2. SVI features + MAP safety: posterior-feature effect.
3. MAP features + SVI safety: posterior-safety effect.
4. SVI features + SVI safety: full V5D-SVI.

The evaluated method and its development results are published as **V5D-SVI**.
The release does not claim fresh-cohort generalization, state-of-the-art status,
or production authorization. Those limitations are disclosures, not changes to
the fitted model, score, gates, or reported results.

The public coefficient-importance table is generated with:

```bash
npm run research:svi-score-v5d-svi:importance
```

It reports the top 50 standardized cross-fitted risk coefficients with plain-
English descriptions. Importance is predictive rather than causal: correlated
features can share or exchange coefficient weight.
