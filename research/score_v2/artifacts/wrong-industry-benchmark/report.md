# Score V2 pilot: Wrong industry benchmark

External benchmarks intentionally disagree with randomized truth to test non-circular validation.

Seed: `20260818`. Truth was held out of model fitting. The declared benchmark has 71.4% spend-weighted log error versus causal truth and is recorded as fallible evidence, not an answer key.

## Injected truth

| Channel | Target ROI | Realized ROI |
|---|---:|---:|
| paid_social | 1.15x | 1.150000x |
| search | 3.50x | 3.500000x |
| tv | 1.05x | 1.050000x |

## Candidate results

| Candidate | Current Flux score | ROI truth error | Benchmark agreement error | Contribution error | Profit regret |
|---|---:|---:|---:|---:|---:|
| channel-specific-long · benchmark-gap-fill | 85.2 | 31.7% | 81.7% | 48.8% | 100.0% |
| strong-saturation · benchmark-gap-fill | 79.6 | 21.3% | 84.9% | 22.3% | 100.0% |
| channel-specific · benchmark-gap-fill | 79.4 | 33.6% | 82.2% | 52.4% | 100.0% |
| channel-specific-long · experiments-only | 79.0 | 31.7% | 82.6% | 49.8% | 100.0% |
| channel-specific · experiments-only | 76.4 | 33.7% | 83.1% | 53.4% | 100.0% |
| strong-saturation · experiments-only | 76.3 | 21.3% | 85.4% | 22.7% | 100.0% |
| long-memory · experiments-only | 59.4 | 39.3% | 94.7% | 76.8% | 100.0% |
| weibull · benchmark-gap-fill | 56.1 | 62.5% | 61.5% | 54.7% | 0.0% |
| balanced · experiments-only | 51.2 | 39.1% | 99.3% | 73.8% | 100.0% |
| short-memory · experiments-only | 42.2 | 57.6% | 112.4% | 102.2% | 100.0% |
| balanced · benchmark-gap-fill | 40.7 | 38.7% | 97.9% | 71.0% | 100.0% |
| short-memory · benchmark-gap-fill | 37.7 | 57.2% | 110.9% | 98.8% | 100.0% |
| long-memory · benchmark-gap-fill | 35.5 | 35.2% | 89.5% | 63.1% | 100.0% |
| weibull · experiments-only | 33.7 | 81.4% | 80.1% | 61.9% | 0.0% |

Profit regret is the share of the simulator-known incremental profit opportunity lost by following the candidate instead of the hidden profit-optimal spend and mix. Benchmark agreement is reported separately from causal truth and is never treated as the answer key. This single scenario is diagnostic evidence, not a learned score or a production claim.
