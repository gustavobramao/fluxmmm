# FluxMMM V5B: equal-compute search benchmark

V5B changes only how candidate specifications are ordered for evaluation. It
does not change V5A weights, the V4.2 ROI safety pool, stored SVI labels, the
420-business development cohort, or the 48-candidate universe.

## Search question

Given only 16 model evaluations for a new business, which search policy finds
the strongest candidate under the frozen V5A selector?

The tested policies are:

1. fixed parameter coverage;
2. deterministic uniform random search;
3. static maximin space filling;
4. Gaussian-process upper-confidence-bound search;
5. a hybrid that alternates global GP proposals with local challenges around
   the current safe champion.

Every policy receives exactly the same candidate-evaluation budget. No early
stopping or duplicate evaluations are allowed.

## Information firewall

Before a candidate is evaluated, the search policy sees only declared model
parameters and the evidence arm. After evaluation, it may see the frozen V5A
score and whether the candidate passes the V4.2 ROI safety policy. SVI economic
loss, injected truth, ROI truth error, contribution truth error, and generator
family remain hidden until assessment.

V5A is refit out of fold: each business is searched with a selector trained on
the other four business-grouped folds.

## Interpretation

This is a finite-universe replay benchmark. It measures sample efficiency over
the existing 48 candidates and cannot prove that any policy found the global
optimum of FluxMMM's much larger mixed continuous/discrete parameter space.
Expanded-space search is a separate, later experiment.

If an adaptive policy recovers the V5A champion more often but produces higher
economic loss, V5B records optimizer pressure and blocks expanded search. That
pattern means the optimizer is exploiting residual selector error; it is not
evidence that the adaptive algorithm is economically superior.
