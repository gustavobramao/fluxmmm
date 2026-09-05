# FluxMMM V7 nonlinear multi-task economic surrogate

V7 is a research-only test of whether a nonlinear neural surrogate improves
economic-loss prediction over V5D. It uses the exact same 420-business,
48-candidate, 115-feature development cohort, the same five outer business
folds, the same four excluded candidate regions, and the same order-independent
selection and search budgets.

## Model

The deterministic NumPy MLP has two hidden layers (48 and 24 SiLU units) and
four outputs:

- mean excess economic loss;
- P90 excess economic loss;
- ROI recovery error;
- contribution recovery error.

Training combines Huber mean loss, P90 pinball loss, two auxiliary Huber losses,
and an economically weighted hard-negative ranking loss. Final research risk
retains the subjective business preference:

```text
risk = 65% predicted mean loss + 35% predicted P90 loss
```

## Result

V7 failed every predeclared improvement check except exhaustive algorithm
agreement. Relative to V5D:

- candidate-level correlation fell from 0.154 to 0.104;
- economically weighted pairwise accuracy fell from 0.661 to 0.610;
- mean absolute error increased from 2.667 to 2.907;
- P90 coverage fell from 0.900 to 0.829;
- dangerous false champions increased from 27.9% to 30.0%;
- the best 16-evaluation objective was 3.8% worse;
- the full-pool objective was 2.4% worse.

The gap between training and assessment indicates overfitting. Candidate rows
within a business are correlated, so 20,160 rows do not create 20,160
independent examples; the effective business-level sample is 420.

## Governance

V7 is frozen as a negative development result. It is not publication-ready,
does not establish fresh-data generalization, and cannot support a state-of-the-
art claim. Hyperparameters must not be changed using the opened outer-fold
results. Any further neural experiment requires a new declared version and
nested tuning or a materially larger business cohort.
