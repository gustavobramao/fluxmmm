# FluxMMM V5C: search-robust selector

V5C responds to the optimizer-pressure failure detected by V5B. It remains a
development-only experiment and does not open fresh validation or the sealed
audit.

## Corrections

1. **Proposal invariance.** Search phase, proposal method, restart, and
   candidate identity are excluded from selector features. A mathematical
   specification receives the same representation regardless of how it was
   proposed.
2. **Two-axis cross-fitting.** Outer folds hold out businesses. Inside each
   outer training set, four balanced parameter-space regions are held out from
   their own selector fits. Evidence arms for one specification remain paired.
3. **Uncapped economic target.** The structured top-one loss uses the full
   candidate-versus-oracle economic gap. Catastrophic errors no longer collapse
   to the same target as a one-unit mistake.
4. **Selector uncertainty.** Each candidate is scored by three models trained
   on different business subfolds, all excluding that candidate's parameter
   region.
5. **Conservative promotion.** Search acquisition uses a one-standard-error
   lower confidence score. A challenger replaces the incumbent only when its
   mean advantage exceeds the joint uncertainty margin.
6. **End-to-end outer assessment.** The complete adaptive search runs inside
   each unseen-business fold. SVI economic loss remains hidden until the final
   assessment.

## Acceptance rule

The strongest adaptive policy at 16 evaluations must be within 2% of the
ordered seed-first baseline on the predeclared economic objective. Its
48-candidate result must also remain within 2% of its best earlier checkpoint.
Failure on either condition blocks expanded candidate-space search.

V5C still uses the historical 48-candidate universe. Passing V5C would justify
a separately predeclared neutral expanded-universe experiment; it would not
establish a global optimum.
