# Score V2 feasibility findings

## Outcome

The simulator and evaluation method work end to end, and the small pilot finds
a real ranking problem worth investigating. Across 30 candidate fits on six
deterministic synthetic businesses:

- The current Flux score selected a lowest-regret candidate in 1 of 6 cases.
- Its selected candidates had 36.7% mean true budget regret.
- A lowest-regret candidate was present in the evaluated set in every case.
- The correlation between current Flux score and lower budget regret was
  -0.297.
- The correlation between current Flux score and lower ROI error was 0.265.

In the demand-confounded search scenario, the current winner scored 84.4 but
lost 61.2% of the hidden incremental allocation opportunity. A stronger-
saturation candidate reduced regret to 2.3%, while a Weibull candidate reached
0% regret but scored only 27.2. In the delayed-TV scenario, the data-generating
Weibull family recovered ROI with 2.8% weighted log error and 0% regret, while
the current winner produced 62.3% regret.

These results show that known-truth simulation can distinguish historical fit
and validation performance from actual causal decision quality. They do not
yet show what the new score or weights should be.

## What is verified

- Every declared channel ROI is injected exactly by construction.
- Observed CSVs exclude latent demand, planning intensity, true contributions,
  coefficients, and optimal allocations.
- Simulation is reproducible from deterministic seeds.
- The allocation oracle conserves budget and evaluates candidate decisions
  against the true nonlinear response curves.
- Misleading benchmarks remain fallible evidence and never define truth.
- The V1 estimator, validation code, production build, and regression suite
  remain unchanged and pass alongside the research harness.

## Important limitations

This is a feasibility pilot, not statistical evidence for production weights:

1. It uses six businesses, one seed per family, three channels, and five static
   MAP/Laplace candidate specifications.
2. The 10% incremental budget is allocated on a 5% grid. Several models make
   the same corner decision, so regret is sometimes unable to distinguish
   otherwise different ROI errors.
3. Counterfactual allocation scales each channel's observed weekly spend path.
   It does not yet optimize timing, flights, minimum commitments, or channel
   interaction effects.
4. Aggregate ROI truth is common across families. Future simulations must vary
   ROI levels, relative channel rankings, noise, spend support, and response
   parameters independently.
5. Paid social and TV receive synthetic independent experiments; search is
   intentionally unanchored. Evidence availability and quality need their own
   randomized experimental arms.
6. The wrong-benchmark family records benchmark agreement but does not yet fit
   matched calibrated and uncalibrated arms.
7. No weight learning, confidence calibration, MCMC comparison, or held-out
   generator evaluation has been performed.

## Decision

Proceed to a larger research pilot, but do not replace the V1 Flux score yet.
The next stage should create hundreds of independently seeded businesses,
broaden the candidate space, add realistic constraints and multiple budget
levels, and predeclare train/validation/test splits at the business-generator
level. Only then should constrained geometric weights and a monotonic model be
trained and compared on held-out generator families.
