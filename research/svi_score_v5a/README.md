# FluxMMM V5A: common-pool nonlinear selector experiment

V5A changes only the candidate selector. It does not expand the search space,
refit an MMM, open the sealed audit, or alter V6/V4 artifacts.

## Fair comparison contract

- 420 development businesses;
- the same 48 candidates per business;
- the same stored SVI decision-loss labels;
- the same five business-grouped, generator-stratified folds;
- the V4.2 ROI plausibility and decision-stability candidate pool;
- temporal identification remains a diagnostic rather than a gate.

V4.2 is the primary selector comparison. Historical V4.0 and V4.1 receipts are
reported with their original temporal policies, while active V6 remains a
locked descriptive comparator.

## Selector

V5A retains the exact capped top-one regret target, but expands the monotonic
feature basis with:

1. diagnostic deficit hinges at predeclared 50 and 70 thresholds;
2. interactions with advertiser-visible temporal context;
3. declared model-setting interactions, such as long CTV memory with flight
   evidence, planning intensity with confounder diagnostics, and benchmark use
   with evidence quality.

All weights are nonnegative. Within a fixed specification, improving a
diagnostic therefore cannot lower its score. Generator family, business id,
split, synthetic truth, decision loss, ROI truth error, and contribution truth
error are forbidden feature inputs.

## Governance

Configuration selection uses only grouped development cross-validation and the
predeclared absolute-loss objective:

```text
mean loss + 0.5 * CVaR90 loss + 0.5 * worst-family mean loss
```

Expanded search is permitted only after the selector result is inspected and
frozen. Fresh validation is not generated at this stage, and the existing
sealed audit remains closed.
