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

## Downstream score learning

The audited population now supports the V6 diagnostic ranker documented in
`research/LEARNED_SCORE_V6.md`. V6 uses a smaller predeclared cohort from these
families, learns only from the training families, and must improve decision loss
on family-held-out validation without regressing on the sealed adversarial
audit. The simulator audit and learned-score activation remain separate
artifacts so changing a score cannot silently redefine the data-generating
process.
