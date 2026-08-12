# Simulator Audit V3

V3 separates two questions that the six-business V2 pilot could not answer:

1. Does the generator cover a credible range of DTC businesses?
2. Do validation diagnostics predict decision quality on generator families
   that were not used to tune the score?

## Population contract

`pnpm research:score-v3:audit` deterministically creates 500 businesses:

- 300 train businesses across balanced DTC, demand-harvesting search, and
  social-frequency-pressure families.
- 120 validation businesses across delayed-TV and correlated-planning
  families.
- 80 untouched audit businesses across wrong-evidence and swapped-channel-
  mechanics families.

The audit split is defined by generator family, not merely by random row. This
prevents the learned score from succeeding by memorizing a familiar simulator.

## Decision label

Each business exposes four separate hidden optimization problems:

- Reduce spend.
- Reallocate the current budget without changing its total.
- Increase spend.
- Find the unconstrained economic ceiling inside the declared support.

Candidate profit regret is averaged across the four decisions, while the
component regrets remain available for diagnosis. Allocations are explored on
a dense deterministic low-discrepancy surface rather than the V2 5% mix grid.

## Current boundary

V3 audits the simulator population and upgrades the decision label. It does not
yet train or display a replacement Flux score. The next gated step is to fit a
predeclared sentinel model set on a subset of businesses and demonstrate
calibration and lower selection regret on the untouched generator families.
