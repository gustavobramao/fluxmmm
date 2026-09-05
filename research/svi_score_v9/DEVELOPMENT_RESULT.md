# V9 development result

V9 completed nested advertiser-grouped development on 420 businesses and
20,160 candidate MMMs. All labels came from the existing FullRankADVI research
contract. The V8 sealed audit was not read, and NUTS was not used.

| Endpoint | V8 | V9 | Relative change |
| --- | ---: | ---: | ---: |
| Mean excess loss, 100% coverage | 2.168 | 1.473 | −32.1% |
| P90 excess loss, 100% coverage | 6.349 | 3.631 | −42.8% |
| P95 excess loss, 100% coverage | 9.625 | 6.728 | −30.1% |
| Selection objective, 100% coverage | 3.631 | 2.228 | −38.6% |
| Dangerous false champions, 100% coverage | 42.4% | 33.3% | −9.0 pp |
| Oracle recall, 100% coverage | 10.2% | 13.8% | +3.6 pp |
| 60%–100% selective-risk area | 3.265 | 1.873 | −42.6% |

At V8's native 68.6% promotion coverage, V9's selection objective was 1.712
versus 2.979 for V8. P90 excess loss was 2.614 versus 5.175, and dangerous
false champions were 29.2% versus 37.8%. Each selector ordered advertisers by
its own truth-blind confidence score; equal coverage means equal numbers of
promotions, not necessarily the same promoted advertisers.

All predeclared development checks passed, including the full-coverage tail
checks and zero violations of the capped-regret selection bound. This is strong
development evidence, not confirmatory evidence.

## Unseen-mechanism challenge

V9 was also evaluated with leave-one-simulator-family-out fitting. For each of
the five families, model architecture and promotion policy were selected using
only the other four families. The held-out family was then evaluated once.

| Held-out simulator family | Businesses | Mean excess loss | P90 | P95 | Dangerous false champions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced DTC | 100 | 1.379 | 3.645 | 5.926 | 30.0% |
| Correlated planning | 60 | 1.264 | 3.309 | 5.190 | 33.3% |
| Delayed TV | 60 | 1.012 | 2.412 | 4.021 | 25.0% |
| Search demand harvesting | 100 | 0.992 | 2.483 | 4.281 | 31.0% |
| Social frequency pressure | 100 | 1.926 | 5.807 | 7.440 | 40.0% |
| **Aggregated V9** | **420** | **1.348** | **3.971** | **5.926** | **32.4%** |
| Descriptive V8 reference | 420 | 2.168 | 6.349 | 9.625 | 42.4% |

The V8 row is a conservative descriptive reference, not a like-for-like
mechanism-holdout baseline: V8 held out advertisers but did not exclude entire
simulator families during training. The V9 mechanism result therefore supports
developmental transfer across the five declared mechanisms; it does not prove
transfer to a sixth unknown mechanism or to real advertisers. Social frequency
pressure remains the clearest stress area.

The V8 audit remains unopened by V9. It is not valid to tune V9 on that already
used audit. The next legitimate step is to freeze the V9 code, token registry,
model, decision endpoints, and comparator, then generate a new untouched audit
cohort.

The result does not establish real-world generalization, production validity,
or a state-of-the-art claim. The current cohort contains five simulator
families and three channels per business; broader channel counts, additional
data-generating mechanisms, and an independent simulator remain necessary.
