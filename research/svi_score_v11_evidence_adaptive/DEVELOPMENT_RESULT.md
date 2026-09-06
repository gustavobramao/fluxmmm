# Evidence-Adaptive RegretSet development result

The implementation passed five-fold nested advertiser-grouped cross-validation
on 420 development businesses. The 20,160 candidate MMM fits and FullRankADVI
posteriors were reused unchanged. No posterior was refitted, and no AMSS or
earlier sealed-audit artifact was read.

## Full candidate-set result

| Endpoint | Frozen V9 | Evidence-adaptive | Relative change |
| --- | ---: | ---: | ---: |
| Mean excess economic loss | 1.473 | 1.262 | −14.3% |
| P90 excess economic loss | 3.631 | 3.332 | −8.2% |
| P95 excess economic loss | 6.728 | 5.270 | −21.7% |
| 65% mean + 35% P90 objective | 2.228 | 1.987 | −10.8% |
| Dangerous false champions | 33.3% | 31.4% | −1.9 pp |
| Oracle recall | 13.8% | 13.8% | unchanged |

Prediction-only produced a 3.492 selection objective on the same development
businesses. Evidence-adaptive RegretSet was 43.1% lower. These are development
comparisons, not new confirmatory results.

## Evidence adaptation

The average residual routing weights across cross-fitted full candidate sets
were:

| Residual expert | Average routing weight |
| --- | ---: |
| Predictive generalization | 22.3% |
| Causal identification | 21.1% |
| Posterior decision quality | 28.8% |
| Structural specification | 27.8% |

These values are not total feature importance. A joint all-token path preserves
interactions across pillars; the softmax weights describe only how the adaptive
residual representation is routed.

Only 24 of the 420 development businesses had no experiment. In this small
subgroup, the selection objective was 3.226 for evidence-adaptive RegretSet,
6.036 for frozen V9, and 5.707 for prediction-only. The adaptive residual gate
did not merely increase predictive weight: it routed more weight to structural
specification. This suggests that the economic value of prediction depends on
model-specification support, but the subgroup is too small for a general claim.

## What the paired design identifies

Each advertiser contributes a full 48-candidate set and two matched 24-candidate
sets: experiments-only and benchmark gap-fill. The same observed outcome,
spend, hidden truth, simulator family, and cross-validation fold are preserved.
The oracle is recalculated inside each evidence arm. This teaches the selector
about adding benchmark evidence without confounding the comparison with a
different advertiser.

It does not identify what would happen if an existing experiment were removed.
Experiments change the fitted posterior, so a valid experiment-presence pairing
requires new SVI fits. Token masking after fitting would be leakage and was not
used.

## Governance conclusion

The architecture is a successful development candidate, not a replacement for
the frozen public selector. The already-opened AMSS cohort cannot be used to
tune or confirm it. A future claim requires a new independent cohort generated
only after the method, token registry, endpoints, and comparator are frozen.
