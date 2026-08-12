# Score V2 pilot: Demand-confounded search

Search auction volume rises with hidden demand, which also raises baseline revenue.

Seed: `20260814`. Truth was held out of model fitting. The declared benchmark has 52.8% spend-weighted log error versus causal truth and is recorded as fallible evidence, not an answer key.

## Injected truth

| Channel | Target ROI | Realized ROI |
|---|---:|---:|
| paid_social | 1.79x | 1.787137x |
| search | 3.90x | 3.900000x |
| tv | 1.64x | 1.644838x |

## Candidate results

| Candidate | Current Flux score | ROI truth error | Benchmark agreement error | Contribution error | Profit regret |
|---|---:|---:|---:|---:|---:|
| channel-specific-long · benchmark-gap-fill | 86.2 | 24.2% | 68.6% | 36.3% | 75.9% |
| channel-specific · benchmark-gap-fill | 83.9 | 28.7% | 68.1% | 40.3% | 75.9% |
| channel-specific-long · experiments-only | 80.8 | 24.4% | 69.9% | 37.4% | 75.9% |
| channel-specific · experiments-only | 80.1 | 28.9% | 69.2% | 41.3% | 75.9% |
| strong-saturation · benchmark-gap-fill | 55.0 | 25.3% | 74.7% | 14.9% | 75.9% |
| strong-saturation · experiments-only | 53.5 | 25.3% | 75.3% | 15.3% | 75.9% |
| balanced · experiments-only | 50.7 | 48.3% | 91.3% | 65.7% | 75.9% |
| short-memory · experiments-only | 46.6 | 65.2% | 101.8% | 92.1% | 75.9% |
| balanced · benchmark-gap-fill | 43.3 | 47.8% | 89.3% | 62.6% | 75.9% |
| long-memory · experiments-only | 41.5 | 37.8% | 89.6% | 65.8% | 75.9% |
| short-memory · benchmark-gap-fill | 39.3 | 64.8% | 99.9% | 88.3% | 75.9% |
| weibull · benchmark-gap-fill | 35.9 | 38.2% | 20.3% | 36.8% | 100.0% |
| long-memory · benchmark-gap-fill | 35.5 | 32.1% | 81.7% | 49.9% | 75.9% |
| weibull · experiments-only | 31.3 | 39.6% | 18.8% | 37.7% | 100.0% |

Profit regret is the share of the simulator-known incremental profit opportunity lost by following the candidate instead of the hidden profit-optimal spend and mix. Benchmark agreement is reported separately from causal truth and is never treated as the answer key. This single scenario is diagnostic evidence, not a learned score or a production claim.
