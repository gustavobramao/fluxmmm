# Score V2 pilot: Saturated paid social

Paid social frequency rises quickly and creates strong diminishing returns.

Seed: `20260815`. Truth was held out of model fitting. The declared benchmark has 27.3% spend-weighted log error versus causal truth and is recorded as fallible evidence, not an answer key.

## Injected truth

| Channel | Target ROI | Realized ROI |
|---|---:|---:|
| paid_social | 2.29x | 2.294373x |
| search | 1.00x | 0.995640x |
| tv | 2.19x | 2.187374x |

## Candidate results

| Candidate | Current Flux score | ROI truth error | Benchmark agreement error | Contribution error | Profit regret |
|---|---:|---:|---:|---:|---:|
| short-memory · benchmark-gap-fill | 76.9 | 83.3% | 61.3% | 103.7% | 100.0% |
| channel-specific · benchmark-gap-fill | 76.7 | 59.8% | 37.9% | 67.7% | 100.0% |
| channel-specific · experiments-only | 75.0 | 60.0% | 38.1% | 68.1% | 100.0% |
| short-memory · experiments-only | 74.3 | 83.7% | 61.7% | 104.9% | 100.0% |
| channel-specific-long · benchmark-gap-fill | 60.1 | 62.9% | 41.1% | 73.2% | 100.0% |
| channel-specific-long · experiments-only | 58.5 | 63.2% | 41.4% | 73.7% | 100.0% |
| balanced · benchmark-gap-fill | 54.5 | 73.5% | 51.5% | 82.6% | 100.0% |
| strong-saturation · benchmark-gap-fill | 54.2 | 64.2% | 42.4% | 59.0% | 100.0% |
| balanced · experiments-only | 52.8 | 73.9% | 52.0% | 83.3% | 100.0% |
| strong-saturation · experiments-only | 52.6 | 64.2% | 42.4% | 59.0% | 100.0% |
| long-memory · benchmark-gap-fill | 44.1 | 74.0% | 52.2% | 86.5% | 100.0% |
| long-memory · experiments-only | 43.1 | 76.8% | 54.9% | 91.7% | 100.0% |
| weibull · experiments-only | 34.4 | 582.0% | 609.2% | 33.7% | 0.0% |
| weibull · benchmark-gap-fill | 33.1 | 583.9% | 611.2% | 35.7% | 0.0% |

Profit regret is the share of the simulator-known incremental profit opportunity lost by following the candidate instead of the hidden profit-optimal spend and mix. Benchmark agreement is reported separately from causal truth and is never treated as the answer key. This single scenario is diagnostic evidence, not a learned score or a production claim.
