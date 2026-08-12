# Score V2 pilot: Delayed TV

Long, flighted TV carryover makes contemporaneous attribution misleading.

Seed: `20260816`. Truth was held out of model fitting. The declared benchmark has 49.1% spend-weighted log error versus causal truth and is recorded as fallible evidence, not an answer key.

## Injected truth

| Channel | Target ROI | Realized ROI |
|---|---:|---:|
| paid_social | 1.32x | 1.320701x |
| search | 1.09x | 1.094140x |
| tv | 2.95x | 2.948515x |

## Candidate results

| Candidate | Current Flux score | ROI truth error | Benchmark agreement error | Contribution error | Profit regret |
|---|---:|---:|---:|---:|---:|
| strong-saturation · benchmark-gap-fill | 71.7 | 49.9% | 44.5% | 62.5% | 100.0% |
| strong-saturation · experiments-only | 70.0 | 49.9% | 44.5% | 62.5% | 100.0% |
| long-memory · benchmark-gap-fill | 60.6 | 87.2% | 81.8% | 138.4% | 100.0% |
| long-memory · experiments-only | 56.5 | 88.5% | 83.1% | 141.8% | 100.0% |
| channel-specific-long · benchmark-gap-fill | 55.8 | 63.9% | 56.4% | 97.9% | 100.0% |
| channel-specific-long · experiments-only | 54.4 | 63.9% | 56.5% | 98.1% | 100.0% |
| weibull · benchmark-gap-fill | 50.8 | 38.6% | 35.5% | 42.2% | 0.0% |
| weibull · experiments-only | 47.2 | 20.4% | 53.5% | 22.1% | 0.0% |
| channel-specific · benchmark-gap-fill | 45.1 | 64.4% | 54.1% | 97.3% | 100.0% |
| channel-specific · experiments-only | 43.8 | 64.4% | 54.2% | 97.5% | 100.0% |
| balanced · benchmark-gap-fill | 42.6 | 74.4% | 68.9% | 118.7% | 100.0% |
| short-memory · benchmark-gap-fill | 42.4 | 85.9% | 73.8% | 149.0% | 0.0% |
| balanced · experiments-only | 41.3 | 74.9% | 69.4% | 119.7% | 100.0% |
| short-memory · experiments-only | 41.1 | 86.3% | 74.6% | 150.6% | 0.0% |

Profit regret is the share of the simulator-known incremental profit opportunity lost by following the candidate instead of the hidden profit-optimal spend and mix. Benchmark agreement is reported separately from causal truth and is never treated as the answer key. This single scenario is diagnostic evidence, not a learned score or a production claim.
