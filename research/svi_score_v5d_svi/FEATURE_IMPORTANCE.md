# V5D-SVI learned feature importance

This is the frozen public V5D-SVI selector. Importance is the mean absolute standardized risk coefficient across 20 cross-fitted models and is normalized across all features. The signed coefficient is conditional on every other transformed feature in the model: positive predicts more economic loss; negative predicts less.

> These are predictive associations, not causal effects. Correlated features may share or exchange weight, so direction should not be read as a standalone intervention claim.

## Feature-family importance

| Rank | Feature family | Importance |
|---:|---|---:|
| 1 | model-specification-and-interactions | 23.92% |
| 2 | posterior-roi | 21.88% |
| 3 | temporal-evidence | 21.40% |
| 4 | structure | 7.89% |
| 5 | decision-coherence | 6.72% |
| 6 | posterior-decision-safety | 5.13% |
| 7 | generalization | 4.67% |
| 8 | causal-robustness | 4.14% |
| 9 | posterior-reliability | 3.64% |
| 10 | posterior-predictive | 0.61% |

## Top 50 learned features

| Rank | Feature | Family | Importance | Coefficient | Stability | Description |
|---:|---|---|---:|---:|---:|---|
| 1 | `svi:roi:log-center:paid-social` | posterior-roi | 7.38% | -1.232 | 100% | Log posterior ROI center for paid social. |
| 2 | `svi:roi:log-center:mean` | posterior-roi | 3.42% | -0.571 | 100% | Mean log posterior ROI center across modeled channels. |
| 3 | `svi:roi:log-center:minimum` | posterior-roi | 3.21% | -0.536 | 100% | Lowest channel log posterior ROI center. |
| 4 | `svi:roi:log-center:maximum` | posterior-roi | 3.14% | -0.523 | 100% | Highest channel log posterior ROI center. |
| 5 | `interaction:saturation:spend-regimes` | model-specification-and-interactions | 2.99% | -0.499 | 100% | Interaction between saturation and spend regimes. |
| 6 | `svi:convergence:seed-log-roi-difference` | posterior-reliability | 2.73% | +0.455 | 100% | Largest cross-seed difference in log ROI. |
| 7 | `diagnostic:functional-form` | structure | 2.31% | +0.386 | 100% | Adequacy of the response-curve and baseline functional form. |
| 8 | `diagnostic:spend-regimes` | generalization | 2.14% | -0.356 | 100% | Generalization across low- and high-spend periods. |
| 9 | `diagnostic:kernel-distinguishability` | temporal-evidence | 1.89% | +0.315 | 100% | Ability to distinguish competing carryover kernels. |
| 10 | `context:temporal:kernel-distinguishability` | temporal-evidence | 1.72% | +0.287 | 100% | Temporal-identification interaction for kernel distinguishability. |
| 11 | `svi:decision:log-relative-roi-width` | posterior-decision-safety | 1.51% | +0.253 | 100% | Log-scaled largest ROI interval width relative to its center. |
| 12 | `diagnostic:residual-independence` | structure | 1.47% | -0.245 | 100% | Absence of systematic time or regressor patterns in residuals. |
| 13 | `svi:roi:log-center:range` | posterior-roi | 1.47% | +0.245 | 100% | Range between highest and lowest channel log posterior ROI centers. |
| 14 | `svi:roi:log-center:standard-deviation` | posterior-roi | 1.42% | +0.237 | 100% | Dispersion of log posterior ROI centers across channels. |
| 15 | `svi:margin:roi-precision` | posterior-decision-safety | 1.40% | -0.234 | 100% | Distance inside or outside the posterior ROI precision gate. |
| 16 | `spec:likelihood:log-normal` | model-specification-and-interactions | 1.26% | +0.211 | 100% | Indicator for the likelihood / log normal model specification. |
| 17 | `diagnostic:roi-economic-consistency` | decision-coherence | 1.26% | -0.211 | 100% | Consistency between ROI estimates and marginal-return economics. |
| 18 | `spec:likelihood:student-t` | model-specification-and-interactions | 1.23% | -0.206 | 100% | Indicator for the likelihood / student t model specification. |
| 19 | `context:temporal:spec:likelihood:log-normal` | temporal-evidence | 1.18% | +0.197 | 100% | Interaction between temporal identification and the likelihood / log normal specification. |
| 20 | `deficit:carryover-support:50` | temporal-evidence | 1.15% | +0.192 | 100% | Shortfall below 50/100 for data support for the estimated persistence of media effects. |
| 21 | `svi:roi:log-center:nonbrand-search` | posterior-roi | 1.15% | +0.191 | 100% | Log posterior ROI center for non-brand search. |
| 22 | `interaction:planning:confounder` | model-specification-and-interactions | 1.09% | -0.168 | 90% | Interaction between planning and confounder. |
| 23 | `spec:planning:off` | model-specification-and-interactions | 1.05% | -0.176 | 100% | Indicator for the planning / off model specification. |
| 24 | `spec:planning:on` | model-specification-and-interactions | 1.05% | +0.176 | 100% | Indicator for the planning / on model specification. |
| 25 | `interaction:benchmark:compatibility` | model-specification-and-interactions | 1.00% | +0.167 | 100% | Interaction between benchmark and compatibility. |
| 26 | `spec:calibration:prior` | model-specification-and-interactions | 0.99% | -0.165 | 100% | Indicator for the calibration / prior model specification. |
| 27 | `spec:calibration:likelihood` | model-specification-and-interactions | 0.99% | +0.165 | 75% | Indicator for the calibration / likelihood model specification. |
| 28 | `context:temporal:spec:family:advanced` | temporal-evidence | 0.97% | +0.162 | 100% | Interaction between temporal identification and the family / advanced specification. |
| 29 | `interaction:log-normal:shape` | model-specification-and-interactions | 0.95% | -0.155 | 95% | Interaction between log normal and shape. |
| 30 | `diagnostic:likelihood-shape` | structure | 0.95% | +0.156 | 95% | Agreement between residual behavior and the selected likelihood. |
| 31 | `diagnostic:post-flight-residual-stability` | temporal-evidence | 0.95% | +0.157 | 70% | Residual behavior after campaigns stop. |
| 32 | `context:temporal:spec:planning:on` | temporal-evidence | 0.90% | +0.150 | 100% | Interaction between temporal identification and the planning / on specification. |
| 33 | `context:temporal:spec:family:bayesian` | temporal-evidence | 0.87% | -0.146 | 100% | Interaction between temporal identification and the family / bayesian specification. |
| 34 | `deficit:carryover-support:70` | temporal-evidence | 0.87% | +0.144 | 90% | Shortfall below 70/100 for data support for the estimated persistence of media effects. |
| 35 | `svi:convergence:log-elbo-drift` | posterior-reliability | 0.83% | -0.119 | 80% | Log-scaled remaining drift in the variational objective. |
| 36 | `svi:decision:implausible-probability` | posterior-decision-safety | 0.80% | +0.134 | 100% | Largest posterior probability of an implausible channel ROI. |
| 37 | `svi:margin:roi-plausibility` | posterior-decision-safety | 0.80% | -0.134 | 100% | Distance inside or outside the posterior ROI plausibility gate. |
| 38 | `interaction:student-t:shape` | model-specification-and-interactions | 0.80% | +0.126 | 90% | Interaction between student t and shape. |
| 39 | `context:temporal:spec:calibration:likelihood` | temporal-evidence | 0.79% | +0.131 | 75% | Interaction between temporal identification and the calibration / likelihood specification. |
| 40 | `interaction:benchmark:dependence` | model-specification-and-interactions | 0.76% | +0.127 | 100% | Interaction between benchmark and dependence. |
| 41 | `diagnostic:predictive-coverage` | generalization | 0.73% | +0.118 | 90% | How often predictive intervals cover observed outcomes. |
| 42 | `interaction:ctv-long:carryover` | model-specification-and-interactions | 0.72% | +0.113 | 90% | Interaction between ctv long and carryover. |
| 43 | `context:temporal:spec:likelihood:student-t` | temporal-evidence | 0.72% | -0.120 | 100% | Interaction between temporal identification and the likelihood / student t specification. |
| 44 | `svi:roi:log-center:ctv` | posterior-roi | 0.69% | -0.115 | 95% | Log posterior ROI center for connected TV. |
| 45 | `context:temporal:spec:planning:off` | temporal-evidence | 0.68% | -0.114 | 100% | Interaction between temporal identification and the planning / off specification. |
| 46 | `context:temporal:spec:ctv-adstock:weibull` | temporal-evidence | 0.65% | +0.109 | 100% | Interaction between temporal identification and the ctv adstock / weibull specification. |
| 47 | `deficit:future-media-placebo:70` | causal-robustness | 0.63% | +0.105 | 100% | Shortfall below 70/100 for whether future media spuriously explains current outcomes. |
| 48 | `diagnostic:confounder-sensitivity` | causal-robustness | 0.63% | -0.101 | 90% | Sensitivity to plausible unobserved demand confounding. |
| 49 | `diagnostic:evidence-decision-dependence` | decision-coherence | 0.62% | -0.104 | 100% | How much decisions depend on a particular evidence source. |
| 50 | `diagnostic:carryover-support` | temporal-evidence | 0.62% | +0.101 | 95% | Data support for the estimated persistence of media effects. |

## Interpretation contract

- Importance ranks how strongly a feature contributes to predictions after standardization; it is not a business-controlled weight.
- Coefficient sign is conditional on the other features. A counterintuitive sign can arise from interactions or correlated features.
- Stability is the share of the 20 cross-fitted models that agree with the majority sign.
- V5D-SVI combines predicted mean and tail loss as `0.65 × mean loss + 0.35 × P90 loss`.
- The model and results are frozen as evaluated; this report changes no fit, score, gate, or selection behavior.
