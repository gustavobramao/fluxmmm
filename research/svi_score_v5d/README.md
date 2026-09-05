# FluxMMM SVI score V5D

V5D addresses the optimizer's-curse failure exposed by V5B and V5C. It does
not learn another weighted 0–100 quality score. It predicts each candidate's
uncapped excess economic decision loss and P90 loss directly from information
available to Flux before hidden synthetic truth is opened.

## Frozen development contract

- 420 businesses and 48 candidates per business.
- The cohort combines 300 original training businesses and 120 validation
  businesses that were opened in earlier research. All 420 are now development;
  none may be described as fresh validation.
- The sealed audit remains inaccessible.
- Five outer business folds assess new-business generalization.
- Four candidate regions assess new-parameter-region generalization.
- Candidate rows are scored only by a model that excluded their region.
- Targets remain uncapped.

## Learning target

For business `b` and candidate `c`:

```text
excess loss = max(0, candidate economic loss - safe-oracle economic loss)
```

The mean model uses Huber regression plus an economically weighted hard-negative
ranking term. A second model estimates P90 loss using quantile loss. The deployed
research risk estimate is:

```text
predicted risk = predicted mean + 0.35 * (predicted P90 - predicted mean)
```

The 0–100 presentation score is deliberately absent from model selection.

## Order-independent selection

At every budget, V5D pools all candidates evaluated so far and chooses the
lowest predicted risk inside the unchanged ROI-safety pool. Arrival order is
discarded. Consequently every exhaustive search must select the same candidate
after all 48 candidates have been evaluated.

The exhaustive checkpoint is also compared with the best earlier checkpoint.
Agreement across algorithms fixes the order-dependence bug; deterioration versus
an earlier checkpoint still reveals optimizer's curse in the loss proxy.

## Governance

V5D is research-only. Its development receipt cannot authorize a global-optimum
claim or production activation. Passing development would authorize only a
newly generated, predeclared validation cohort.
