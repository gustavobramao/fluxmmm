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

The generator draws business-level ROI from a versioned evidence registry,
bounded by each entry's declared interval. The explicit wrong-benchmark stress
test is the only family allowed to override that envelope. It then injects a
target aggregate ROI `rho[j]` by setting

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
candidate-to-benchmark agreement. Every scenario fits a matched
experiments-only arm and a benchmark-gap-fill arm; the benchmark fills only a
channel lacking simulated experimental evidence. Neither arm can see the
answer key. This prevents future score learning from becoming a circular
benchmark-agreement test.

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
`research/score_v2/artifacts/`. Compact reports and summaries are versioned
because the Score Lab imports them and a clean clone should be able to audit
the answer keys shown in the UI. Full observed CSV and latent truth arrays are
deterministically regenerated and ignored to keep the repository lean.

## What the pilot measures

- Spend-weighted absolute log ROI error.
- Channel contribution recovery error.
- Lost incremental profit opportunity (profit regret), averaged across four
  predeclared decisions: budget reduction, fixed-budget reallocation, budget
  growth, and the economic ceiling. A dense deterministic budget/allocation
  surface enforces concentration constraints and always includes no change.
- Existing Generalization, Structure, Causal, Decision, and final Flux scores.

This pilot tests whether the current evidence is directionally informative. It
does not train a production meta-model. Learned weights, monotonic calibration,
held-out generator evaluation, and large-scale simulation only follow if the
pilot demonstrates signal without blindly rewarding external priors.

The first frozen feasibility result and its limitations are documented in
[`PILOT_FINDINGS.md`](./PILOT_FINDINGS.md). Regenerate its underlying ignored
artifacts with `pnpm research:score-v2:matrix`.
