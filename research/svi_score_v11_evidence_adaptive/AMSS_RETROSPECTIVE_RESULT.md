# V11 retrospective AMSS screening result

## Scientific status

This is a zero-cost retrospective application of the frozen V11
evidence-adaptive selector to the previously opened 100-business AMSS cohort.
It reused 4,800 cached FullRankADVI candidate posteriors and the already
accepted AMSS candidate outcomes. V11 was not retrained, no posterior was
refitted, and no cloud compute was used.

The candidate choices were produced from the truth-blind AMSS observable
artifact and frozen before the accepted candidate-loss artifact was read by a
separate evaluation program. Nevertheless, the result is not confirmatory:
V11 was designed after researchers had inspected the earlier AMSS result.

## Primary result

The screening endpoint applies the frozen business preference to the cohort:

\[
0.65\,\operatorname{mean}(L_b)+0.35\,P_{90}(L_b).
\]

| Selector | Mean risk | P90 | P95 | 65/35 objective |
| --- | ---: | ---: | ---: | ---: |
| Evidence-adaptive V11 | 0.119 | 0.322 | 0.387 | **0.190** |
| Frozen V9 | 0.230 | 0.707 | 0.757 | 0.397 |
| Prediction only | 0.246 | 0.763 | 0.951 | 0.427 |
| In-pool oracle | 0.012 | 0.030 | 0.039 | 0.018 |

Relative to frozen V9, V11 reduced the screening objective by 52.1%. Relative
to prediction-only selection, it reduced the objective by 55.4%.

The paired V11-minus-V9 mean-risk difference was -0.111, with a stratified
bootstrap 95% interval of [-0.150, -0.074]. The paired 65/35-objective
difference was -0.207, with a bootstrap interval of [-0.253, -0.099]. V11 had
lower risk in 44 businesses, selected the same-risk candidate in 43, and had
higher risk in 13.

Against prediction-only selection, the paired mean-risk difference was -0.127
[-0.182, -0.075], and the objective difference was -0.237
[-0.331, -0.089]. V11 had lower risk in 57 businesses, tied in 10, and had
higher risk in 33.

## Descriptive evidence groups

Subgroup sample sizes are 25 and are descriptive only.

| Evidence group | V11 objective | V9 objective | Prediction-only objective |
| --- | ---: | ---: | ---: |
| No experiment | **0.226** | 0.539 | 0.359 |
| Paid-social experiment | **0.142** | 0.153 | 0.218 |
| Search experiment | **0.045** | 0.182 | 0.486 |
| TV experiment | **0.264** | 0.479 | 0.532 |

The intended no-experiment screen is encouraging: V11 outperformed both frozen
V9 and prediction-only selection. It also improved the point estimate in each
experiment subgroup, but those comparisons were not powered or sealed as
confirmatory subgroup tests.

## Interpretation and cautions

- V11 selected the same candidate as V9 for 43 businesses. Its gains therefore
  combine retained V9 decisions with targeted changes, rather than replacing
  every decision.
- The selected specifications are concentrated: 66 selections use base
  specification `V6-C01`, and 81 use the experiments-only evidence arm. This is
  a transport diagnostic to monitor in a fresh cohort, not evidence that those
  choices are universally optimal.
- Nine V11 selections recovered the business-specific in-pool oracle, versus
  eleven for V9 and three for prediction only. V11's advantage comes from
  avoiding costly errors across the full distribution, not from maximizing
  exact oracle recall.
- A small share of AMSS standardized inputs reaches V11's frozen clipping
  boundary. Most of the candidate-token shift is a nearly constant
  post-flight residual-stability diagnostic, but this remains an
  out-of-distribution diagnostic for prospective confirmation.
- The result supports paying for a fresh V11 AMSS audit. It does not itself
  authorize a confirmatory claim, production activation, or post-AMSS
  retraining.

The machine-readable result is in
`artifacts/v11-amss-retrospective-result.json`.
