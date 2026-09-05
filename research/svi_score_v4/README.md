# FluxMMM SVI score V4

V4 is a new research method created after V3 failed its one-time validation
screen, especially in the delayed-TV family. It does not alter the V3 training
release, failed validation receipt, or sealed audit.

## What changes

### 1. The loss matches the deployed top-one decision

For business `b`, let `c*` be the feasible candidate with the lowest
posterior-expected economic loss and let `ĉ` be the highest-scoring candidate.
With normalized excess loss `Delta`,

```text
Delta(ĉ) <= max[c != c*] { Delta(c) + score(c) - score(c*) }_+.
```

The inequality holds because the selected candidate is one of the terms and
`score(ĉ) >= score(c*)`. V4 trains the right-hand cost-augmented top-one loss.
This is tighter and more deployment-aligned than summing the regret of every
candidate that outranks the oracle.

### 2. Tail and generator-family risk are first-class objectives

Training minimizes a structured surrogate for:

```text
mean excess loss
+ lambda * CVaR90 excess loss
+ gamma * worst-family mean excess loss.
```

Mean performance therefore cannot hide a small number of economically severe
failures, and strong performance in common generator families cannot hide a
failure in delayed response or correlated planning.

Safety-threshold and risk-hyperparameter configurations are compared out of
fold using **absolute** decision loss (mean + 0.5 CVaR90 + 0.5 worst-family
mean). This prevents a strict gate from deleting the global oracle and then
claiming zero excess loss relative to the weaker candidates left behind.

### 3. Temporal identification is measured directly

Three truth-blind candidate diagnostics are added:

- `carryover-support`: whether history and observed blackout gaps support the
  candidate's declared lag-95 horizon;
- `post-flight-residual-stability`: whether residuals remain centered during a
  fixed post-flight window, so a short-memory model cannot choose its own easy
  diagnostic horizon;
- `kernel-distinguishability`: whether the observed spend path distinguishes
  the declared transformed response from contemporaneous spend.

Forward `whole-flight-generalization` is measured alongside these diagnostics.
All temporal measurements remain score features and descriptive evidence; none
is a non-compensatory eligibility gate or accepted as proof that a response
kernel is correct. Insufficient temporal evidence means that carryover is
unresolved, not that the candidate response is false.

Carryover support and whole-flight generalization form the recorded temporal-
identification summary. The business context used to mix the two scoring
experts is computed only from observed spend flighting, gaps, runs, and serial
dependence. Injected synthetic truth is not available to the feature
calculation.

### 4. ROI safety is non-compensatory

When at least one candidate passes ROI plausibility and ROI decision-stability
thresholds, candidates that fail either threshold are outside the selectable
set. High residual, confounder, predictive, or temporal scores cannot
compensate for a material ROI safety failure.

If no candidate passes, Flux returns a review-only selection rather than
claiming that an unsafe candidate is decision-grade.

### 5. The ranker remains interpretable and monotonic

V4 uses two non-negative linear experts over log diagnostic scores:

- ordinary-response expert;
- temporally-complex expert.

An observed, candidate-invariant temporal-context value mixes the experts.
Every feature weight remains non-negative, so improving a diagnostic cannot
lower the score while all else is fixed. This tests context adaptation without
the opacity and sample requirements of a large neural network.

## Research governance

1. The 300 former training businesses and 120 retired V3 validation businesses
   form a 420-business development set.
2. Configuration selection uses five-fold grouped cross-validation; every
   candidate for one business remains in one fold.
3. V3's 80-business audit remains sealed and is never used by V4.
4. After the method is frozen, a fresh independent validation cohort must be
   generated and evaluated once.
5. The old audit is not opened merely because a new method exists.
