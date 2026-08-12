# Score V2 pilot matrix

This report contains 84 candidate evaluations across 6 deterministic synthetic businesses.

- Correlation of current Flux score with lower profit regret: **-0.299**
- Correlation of current Flux score with lower ROI error: **0.394**
- Mean profit regret of the current Flux winner: **79.8%**
- Mean profit regret of the best evaluated candidate: **13.1%**
- Current score selected a lowest-profit-regret candidate in **33.3%** of scenarios.

| Scenario | Current Flux winner | Winner profit regret | Lowest-regret candidate | Lowest profit regret |
|---|---|---:|---|---:|
| clean-identification | channel-specific-long · benchmark-gap-fill | 100.0% | short-memory · experiments-only | 0.0% |
| demand-confounded-search | channel-specific-long · benchmark-gap-fill | 75.9% | short-memory · experiments-only | 75.9% |
| saturated-paid-social | short-memory · benchmark-gap-fill | 100.0% | weibull · experiments-only | 0.0% |
| delayed-tv | strong-saturation · benchmark-gap-fill | 100.0% | short-memory · experiments-only | 0.0% |
| correlated-media | weibull · benchmark-gap-fill | 2.7% | weibull · experiments-only | 2.7% |
| wrong-industry-benchmark | channel-specific-long · benchmark-gap-fill | 100.0% | weibull · experiments-only | 0.0% |

This is a feasibility diagnostic over a deliberately small candidate set. It is not sufficient to estimate new production weights.
