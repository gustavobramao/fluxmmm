# Score V2 pilot: Correlated media

Channels share an unobserved commercial planning process and become difficult to separate.

Seed: `20260817`. Truth was held out of model fitting. The declared benchmark has 68.0% spend-weighted log error versus causal truth and is recorded as fallible evidence, not an answer key.

## Injected truth

| Channel | Target ROI | Realized ROI |
|---|---:|---:|
| paid_social | 5.60x | 5.600000x |
| search | 1.14x | 1.141749x |
| tv | 4.86x | 4.862371x |

## Candidate results

| Candidate | Current Flux score | ROI truth error | Benchmark agreement error | Contribution error | Profit regret |
|---|---:|---:|---:|---:|---:|
| weibull · benchmark-gap-fill | 66.1 | 10.3% | 63.1% | 8.9% | 2.7% |
| channel-specific · benchmark-gap-fill | 56.5 | 85.5% | 61.1% | 71.3% | 100.0% |
| channel-specific-long · benchmark-gap-fill | 56.3 | 93.6% | 54.9% | 77.6% | 100.0% |
| channel-specific · experiments-only | 55.2 | 88.6% | 60.8% | 74.4% | 100.0% |
| channel-specific-long · experiments-only | 54.8 | 97.5% | 54.4% | 81.1% | 100.0% |
| long-memory · experiments-only | 52.3 | 83.8% | 72.3% | 71.4% | 100.0% |
| long-memory · benchmark-gap-fill | 51.6 | 71.3% | 70.8% | 55.3% | 100.0% |
| balanced · benchmark-gap-fill | 50.6 | 101.2% | 80.7% | 82.1% | 100.0% |
| balanced · experiments-only | 49.3 | 110.0% | 80.4% | 93.0% | 100.0% |
| short-memory · experiments-only | 49.0 | 105.3% | 96.4% | 91.8% | 100.0% |
| strong-saturation · benchmark-gap-fill | 43.3 | 115.0% | 54.3% | 85.5% | 100.0% |
| short-memory · benchmark-gap-fill | 42.6 | 96.3% | 96.3% | 78.0% | 100.0% |
| strong-saturation · experiments-only | 42.2 | 119.0% | 53.7% | 88.8% | 100.0% |
| weibull · experiments-only | 40.8 | 9.9% | 75.2% | 7.6% | 2.7% |

Profit regret is the share of the simulator-known incremental profit opportunity lost by following the candidate instead of the hidden profit-optimal spend and mix. Benchmark agreement is reported separately from causal truth and is never treated as the answer key. This single scenario is diagnostic evidence, not a learned score or a production claim.
