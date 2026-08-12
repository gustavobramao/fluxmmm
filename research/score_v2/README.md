# Flux Decision Score V2 research lab

This module is an isolated, reproducible research harness for testing whether
Flux validation evidence predicts causal recovery and budget decision quality.
It does not change the V1 product, model, UI, or published demo.

## Scientific contract

Each simulated business produces two logically separate objects:

- **Observed data**: dates, revenue, media spend, and observable controls. This
  is the only information available to Flux during fitting.
- **Hidden truth**: latent demand, channel contributions, response parameters,
  true aggregate ROI, and the true counterfactual allocation frontier. This is
  used only after fitting to evaluate the candidate.

The generator injects a target aggregate ROI `rho[j]` by setting

```text
beta[j] = rho[j] * sum(spend[j]) / sum(response[j])
contribution[j,t] = beta[j] * response[j,t]
```

Therefore `sum(contribution[j]) / sum(spend[j])` equals the declared ROI up to
floating-point precision. Confounding changes how difficult that truth is to
recover; it never changes the answer key.

## Initial scenario families

1. `clean-identification`
2. `demand-confounded-search`
3. `saturated-paid-social`
4. `delayed-tv`
5. `correlated-media`
6. `wrong-industry-benchmark`

The last family deliberately declares benchmark evidence that disagrees with
the injected truth. The pilot records both benchmark-to-truth error and fitted
candidate-to-benchmark agreement. The benchmark does not enter the answer key,
and this baseline arm does not calibrate the V1 fit to it; a later experiment
can compare calibrated and uncalibrated arms without mislabeling a benchmark as
experimental ground truth. This prevents future score learning from becoming a
circular benchmark-agreement test.

## Run the pilot

```bash
pnpm test:score-v2
pnpm research:score-v2:pilot
pnpm research:score-v2:matrix
```

The default command fits a small, predeclared candidate set to one
demand-confounded synthetic business. `--all` repeats it for all six scenario
families and writes an aggregate score-versus-regret selection report. Reports
and observed/truth artifacts are written beneath
`research/score_v2/artifacts/`, which is intentionally ignored by Git.

## What the pilot measures

- Spend-weighted absolute log ROI error.
- Channel contribution recovery error.
- Lost incremental budget opportunity (decision regret).
- Existing Generalization, Structure, Causal, Decision, and final Flux scores.

This pilot tests whether the current evidence is directionally informative. It
does not train a production meta-model. Learned weights, monotonic calibration,
held-out generator evaluation, and large-scale simulation only follow if the
pilot demonstrates signal without blindly rewarding external priors.

The first frozen feasibility result and its limitations are documented in
[`PILOT_FINDINGS.md`](./PILOT_FINDINGS.md). Regenerate its underlying ignored
artifacts with `pnpm research:score-v2:matrix`.
