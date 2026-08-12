# Score V2 pilot matrix

This report contains 84 candidate evaluations across 6 deterministic synthetic businesses.

- Correlation of current Flux score with lower profit regret: **-0.092**
- Correlation of current Flux score with lower ROI error: **0.394**
- Mean profit regret of the current Flux winner: **67.3%**
- Mean profit regret of the best evaluated candidate: **30.2%**
- Current score selected a lowest-profit-regret candidate in **16.7%** of scenarios.

| Scenario | Current Flux winner | Winner profit regret | Lowest-regret candidate | Lowest profit regret |
|---|---|---:|---|---:|
| clean-identification | channel-specific-long · benchmark-gap-fill | 89.6% | weibull · experiments-only | 28.9% |
| demand-confounded-search | channel-specific-long · benchmark-gap-fill | 50.6% | channel-specific · experiments-only | 50.1% |
| saturated-paid-social | short-memory · benchmark-gap-fill | 100.0% | weibull · experiments-only | 25.0% |
| delayed-tv | strong-saturation · benchmark-gap-fill | 99.8% | weibull · benchmark-gap-fill | 46.2% |
| correlated-media | weibull · benchmark-gap-fill | 1.0% | weibull · benchmark-gap-fill | 1.0% |
| wrong-industry-benchmark | channel-specific-long · benchmark-gap-fill | 62.5% | short-memory · benchmark-gap-fill | 30.3% |

This is a feasibility diagnostic over a deliberately small candidate set. It is not sufficient to estimate new production weights.
