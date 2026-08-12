# Score V2 feasibility findings

## Outcome

The simulator and evaluation method work end to end, and the small pilot finds
a ranking problem worth investigating. Across 84 candidate fits on six
deterministic synthetic businesses:

- The current Flux score selected a lowest-profit-regret candidate in 1 of 6
  cases.
- Its selected candidates had 67.3% mean simulator-known profit regret.
- The best evaluated candidate averaged 30.2% profit regret.
- The correlation between current Flux score and lower profit regret was
  -0.092.
- The correlation between current Flux score and lower ROI error was 0.394.

The results are intentionally uncomfortable: a high validation score can still
recommend economically poor decisions. In delayed TV, the best evaluated
Weibull candidate averaged 46.2% regret across four decision contracts while
the heuristic selected a globally geometric strong-saturation candidate at
99.8% regret. In correlated media, the Weibull family recovered the decision
far better than most other candidates.
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
- The profit oracle evaluates budget reduction, fixed-budget reallocation,
  budget growth, and an economic ceiling against hidden nonlinear response
  curves, with concentration constraints and a no-change option.
- Misleading benchmarks remain fallible evidence and never define truth.
- Paid social, search, and TV use separate delivery, carryover, saturation, and
  experiment contracts.
- The same channel-specific response contract flows through MAP fitting,
  validation refits, budgeting, and production MCMC compilation.

## Important limitations

This is a feasibility pilot, not statistical evidence for production weights:

1. It uses six businesses, one seed per family, three channels, seven response
   specifications, and two evidence arms—not enough to learn score weights.
2. The profit oracle uses a dense deterministic budget and allocation surface,
   not a continuous mathematical optimizer. Extremely narrow optima may still
   be approximated rather than located exactly.
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

Proceed to the predeclared V3 simulator audit, but do not replace the V1 Flux
score yet. V3 creates hundreds of independently seeded businesses, broadens the
decision label, and freezes train/validation/audit splits at the generator-family
level. A replacement score is only justified after a sentinel model cohort is
fit and constrained alternatives outperform the heuristic on held-out families.
