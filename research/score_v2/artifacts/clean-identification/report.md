# Score V2 pilot: Clean identification

Low confounding, observable demand, and moderate noise.

Seed: `20260813`. Truth was held out of model fitting. The declared benchmark has 35.8% spend-weighted log error versus causal truth and is recorded as fallible evidence, not an answer key.

## Injected truth

| Channel | Target ROI | Realized ROI |
|---|---:|---:|
| paid_social | 1.49x | 1.488852x |
| search | 1.31x | 1.313477x |
| tv | 1.82x | 1.824276x |

## Candidate results

| Candidate | Current Flux score | ROI truth error | Benchmark agreement error | Contribution error | Profit regret |
|---|---:|---:|---:|---:|---:|
| channel-specific-long · benchmark-gap-fill | 82.2 | 41.2% | 29.4% | 51.9% | 89.6% |
| channel-specific · benchmark-gap-fill | 81.5 | 45.0% | 35.6% | 59.2% | 86.8% |
| channel-specific-long · experiments-only | 76.7 | 40.0% | 28.2% | 49.8% | 89.4% |
| channel-specific · experiments-only | 75.9 | 44.2% | 34.8% | 57.7% | 86.8% |
| long-memory · benchmark-gap-fill | 60.5 | 34.0% | 26.9% | 37.5% | 50.7% |
| long-memory · experiments-only | 58.2 | 29.5% | 22.1% | 31.6% | 40.1% |
| strong-saturation · benchmark-gap-fill | 54.2 | 51.7% | 46.9% | 54.1% | 89.2% |
| strong-saturation · experiments-only | 52.4 | 50.9% | 46.0% | 52.7% | 89.2% |
| balanced · benchmark-gap-fill | 50.8 | 67.3% | 56.6% | 85.8% | 89.2% |
| short-memory · benchmark-gap-fill | 50.5 | 66.2% | 56.1% | 81.2% | 59.9% |
| balanced · experiments-only | 49.2 | 67.6% | 56.9% | 86.4% | 89.2% |
| short-memory · experiments-only | 48.8 | 66.2% | 56.0% | 81.1% | 59.9% |
| weibull · experiments-only | 31.8 | 659.6% | 685.4% | 50.4% | 28.9% |
| weibull · benchmark-gap-fill | 30.9 | 659.1% | 686.2% | 49.8% | 28.9% |

Profit regret is the mean normalized opportunity loss across four hidden decision contracts: reduce budget, reallocate the current budget, grow budget, and find the economic ceiling. Component regrets remain in the JSON artifact. Benchmark agreement is reported separately from causal truth and is never treated as the answer key. This single scenario is diagnostic evidence, not a learned score or a production claim.
