# Score V2 feasibility findings

## Outcome

The simulator and evaluation method work end to end, and the small pilot finds
a ranking problem worth investigating. Across 84 candidate fits on six
deterministic synthetic businesses:

- The current Flux score selected a lowest-profit-regret candidate in 2 of 6
  cases.
- Its selected candidates had 79.8% mean simulator-known profit regret.
- The best evaluated candidate averaged 13.1% profit regret.
- The correlation between current Flux score and lower profit regret was
  -0.299.
- The correlation between current Flux score and lower ROI error was 0.394.

The results are intentionally uncomfortable: a high validation score can still
recommend an economically poor next-dollar decision. In delayed TV, a Weibull
candidate reached 0% profit regret while the heuristic selected a globally
geometric strong-saturation candidate at 100% regret. In correlated media, the
Weibull family recovered the decision far better than most other candidates.
These are simulator-conditional diagnostics, not claims about real advertisers.

The pilot therefore establishes that known-truth simulation can distinguish
historical fit and validation performance from causal decision quality. It does
not yet establish what a learned score or its weights should be.

## What is verified

- Every declared channel ROI is injected exactly by construction.
- Observed CSVs exclude latent demand, planning intensity, true contributions,
  coefficients, and optimal allocations.
- Business-level ROI draws remain inside their versioned evidence envelopes
  unless an explicit stress scenario overrides them.
- Simulation and committed answer keys are reproducible from deterministic
  seeds and artifact identifiers.
- The profit oracle tests several incremental budget levels, can choose zero,
  enforces channel concentration constraints, and evaluates candidate decisions
  against the hidden nonlinear response curves.
- Misleading benchmarks remain fallible evidence and never define truth.
- Paid social, search, and TV use separate delivery, carryover, saturation, and
  experiment contracts.
- The same channel-specific response contract flows through MAP fitting,
  validation refits, budgeting, and production MCMC compilation.

## Important limitations

This is a feasibility pilot, not statistical evidence for production weights:

1. It uses six businesses, one seed per family, three channels, seven response
   specifications, and two evidence arms—not enough to learn score weights.
2. The profit oracle evaluates a small predeclared budget grid. Several models
   make the same corner decision, so regret can fail to distinguish otherwise
   different ROI errors.
3. Counterfactual allocation scales each channel's observed weekly delivery
   pattern. It does not yet optimize timing, flights, minimum commitments, or
   channel interactions.
4. Business-level ROI is drawn from only three versioned evidence distributions.
   Future simulations must add market, objective, brand-scale, execution, and
   evidence-quality strata.
5. Paid social and TV receive synthetic independent experiments; search is
   intentionally unanchored. Evidence availability and quality need more
   randomized arms.
6. The matched benchmark-gap-fill arm is useful for anti-circularity testing,
   but this pilot is too small to estimate when benchmark information helps.
7. No weight learning, confidence calibration, MCMC comparison, or held-out
   generator evaluation has been performed.

## Decision

Proceed to a larger research pilot, but do not replace the V1 Flux score yet.
The next stage should create hundreds of independently seeded businesses,
broaden the candidate space, add richer constraints and budget levels, and
predeclare train/validation/test splits at the business-generator level. Only
then should constrained geometric weights and a monotonic model be trained and
compared on held-out generator families.
