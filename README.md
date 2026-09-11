# FluxMMM

[![CI](https://github.com/gustavobramao/fluxmmm/actions/workflows/ci.yml/badge.svg)](https://github.com/gustavobramao/fluxmmm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-111827.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/Live_demo-open-5b5ce2.svg)](https://fluxmmm-demo.web.app/)
[![Research paper](https://img.shields.io/badge/Research_paper-read-0f766e.svg)](https://fluxmmm-web.web.app/paper/)

**Open measurement. Grounded ROI.**

FluxMMM is a local-first, open-source Bayesian marketing mix modeling workspace.
It connects data validation, explicit experiment or benchmark calibration,
model diagnostics, model search, production posterior sampling, and budget
planning in one auditable workflow.

This release adds **evidence-adaptive RegretSet-MMM V11**, a frozen research
selector that learns across synthetic advertisers which complete MMM
specification is least likely to produce costly downstream budget decisions.
V11 keeps the full joint candidate representation from V9 and adds learned
residual experts whose reliance changes with the advertiser's observable
evidence environment. Its model, token registries, frozen selection receipts,
and independent AMSS synthetic audit are published with the product.

[Explore the cached public demo](https://fluxmmm-demo.web.app/) ·
[Read the paper](https://fluxmmm-web.web.app/paper/) ·
[Read the V11 release notes](RELEASE_NOTES.md) ·
[Visit FluxMMM](https://fluxmmm-web.web.app/)

![FluxMMM — open measurement and grounded ROI](public/og.png)

## What makes FluxMMM different

MMM is often selected for predictive fit and then used for causal and budget
decisions that require stronger evidence. FluxMMM keeps those claims separate:

1. **Predictive generalization** — rolling-origin and spend-regime holdouts.
2. **Structural integrity** — residual behavior, identification, functional
   form, multicollinearity, influence, and assumption-aware diagnostics.
3. **Causal credibility** — experiment-anchor recovery, new-data stability,
   latent-confounder stress, and future-media placebos.
4. **ROI decision coherence** — business plausibility, refit stability,
   interval resolution, evidence dependence, and economic consistency.
5. **Continuous evidence attribution** — local ROI information is decomposed
   into observational data, experiments, benchmarks, and regularization rather
   than reduced to arbitrary “data-led” or “prior-led” point thresholds.

## Evidence-adaptive RegretSet-MMM V11

RegretSet-MMM changes the model-selection target. Instead of asking only which
candidate predicts best, it asks which candidate is least likely to recommend
an economically costly budget decision.

- **420 development businesses** generated across declared mechanism families,
  reusing **20,160 FullRankADVI candidate fits** under three matched evidence
  regimes per business.
- **48 truth-blind candidates per complete set** and **232 candidate tokens**
  spanning diagnostics, posterior geometry, evidence, and the model contract.
- **63 business-context tokens** encoding evidence availability, quality,
  compatibility, missingness, and candidate-set summaries.
- A permutation-invariant **DeepSets multi-head selector** with one full joint
  pathway plus four learned residual experts: predictive generalization,
  causal identification, posterior decision quality, and structural
  specification.
- Ten jointly trained loss and tail-risk heads covering four predeclared budget
  decisions; promotion minimizes the declared **65% mean / 35% P90** risk.
- A finite-candidate regret bound connecting oracle misranking to capped
  selected-model excess regret.
- A fresh external audit on **100 businesses from Google AMSS 1.0.1**: 4,800
  SVI fits and 19,200 budget actions, with no retraining or truth access before
  candidate actions and selections were frozen.

On that untouched AMSS cohort, V11 achieved a declared selection objective of
**0.203**, versus **0.353** for prediction-only selection, **0.401** for frozen
V9, **0.458** for uniform random valid-candidate selection, and **0.701** for a
conventional prediction/decomposition/experiment-calibration Pareto rule. The
unattainable in-pool oracle was **0.029**. V11 reduced the objective by **42.6%**
relative to prediction-only and passed all three predeclared confirmatory
checks. The Pareto comparison is secondary and used no new posterior fits.

These findings establish transport to a second synthetic universe under the
declared adapter, candidate pool, FullRankADVI contract, and utility. They do
**not** establish real-advertiser effectiveness or universal SOTA performance.

The research release lives in:

```text
research/svi_score_v9/                     Frozen joint selector lineage
research/svi_score_v11_evidence_adaptive/  V11 selector and development contract
research/svi_score_v11_amss_confirmatory/  Fresh AMSS protocol and audit receipts
docs/regretset-mmm-paper.pdf                Publication-format working paper
```

The interactive Agentic workspace now runs the frozen V11 selection contract
directly. It screens 24 predeclared specifications under two evidence regimes,
fits the resulting complete 48-candidate pool with the declared FullRankADVI
contract, builds the exact 232 candidate and 63 business-context tokens, and
ranks candidates by adjusted predicted economic risk. MAP/Laplace remains a
fast diagnostic screen inside each candidate workflow; it is not substituted
for the posterior inputs expected by V11. After promotion, NUTS independently
samples the selected specification before budget planning.

Inference-equivalent evidence arms share one posterior artifact when their
compiled data, likelihood, priors, and inference contract are identical. This
can reduce a fully experiment-anchored search from 48 to 24 unique posterior
fits without changing the V11 candidate set or its selection semantics.

## Product capabilities

- Daily, weekly, and monthly CSV ingestion with deterministic repairs and
  visible data-contract receipts.
- Progressive EDA for coverage, sparsity, variation, spend regimes, and
  cross-channel correlation.
- Channel-specific geometric or Weibull-PDF adstock, Hill saturation, trend,
  Fourier seasonality, cycle effects, and observed controls.
- Non-negative frequentist and Bayesian MAP/Laplace screening estimators.
- Experiment-informed prior or likelihood calibration, plus optional
  channel-level plausibility priors when experiments are absent.
- Optional KTR-style time-varying coefficients and a latent
  media-planning-intensity sensitivity factor.
- Half-normal or log-normal coefficient priors and Gaussian, Student-t, or
  log-normal outcome likelihoods.
- Pivoted QR/SVD fallback, rank and condition diagnostics, evidence attribution,
  and non-negative-boundary disclosure.
- Frozen V11 selection across 48 paired FullRankADVI candidates, with immutable
  evidence, complete-set scoring, validation gates, and reproducible receipts.
- PyMC NUTS sampling of the frozen winner with modern convergence diagnostics.
- Fixed-budget, outcome-target, and economic-ceiling allocation with posterior
  uncertainty, support warnings, and channel constraints.
- Deterministic model fingerprints and local artifact caching.

## Quick start

### Requirements

- macOS, Linux, or Windows through WSL2
- Node.js 22.13 or newer
- pnpm 11.9 through Corepack
- Python 3.12 for full posterior inference; optional for the analytic workspace

```bash
git clone https://github.com/gustavobramao/fluxmmm.git
cd fluxmmm
corepack enable
pnpm install --frozen-lockfile
pnpm dev
```

Open [http://localhost:3003](http://localhost:3003). No login, API key, hosted
account, or cloud project is required.

To enable the local PyMC environment:

```bash
pnpm mcmc:setup
pnpm dev
```

## Data and evidence contract

Each uploaded dataset needs one date column, one numeric outcome, and one or
more non-negative paid-media spend columns. Exposure, event, numeric-control,
and categorical-control columns are optional. FluxMMM normalizes safe cadence
issues with a visible receipt and stops on ambiguous dates, duplicate periods,
unresolved gaps, or unsafe reinterpretations.

Independent experiments are the preferred ROI anchor. Anchor recovery compares
the experiment with the MMM contribution attributable to spend in the same
declared campaign and outcome window, including an optional post-test carryover
window. When no compatible experiment exists, users may activate a
channel-specific industry prior as a visible plausibility guardrail.

Uploaded files and generated artifacts stay on the local machine under ignored
`.wrangler` and `.flux-artifacts` directories.

## Evidence attribution

FluxMMM decomposes local posterior curvature as:

```text
H_post = H_data + H_experiment + H_benchmark + H_regularization
```

Source sensitivity is computed from `H_post⁻¹ H_source`; correlated media use
conditional information rather than raw column magnitude. Influence is
reported separately from source quality, conflict, and decision dependence.

## Architecture

```text
app/                 Product UI and progressive client orchestration
app/components/      Interactive visualizations and research surface
lib/mmm/             Schema, EDA, transformations, estimators, search, budget
lib/mmm/validation/  Leakage-aware refits, validation, gates, attribution
scripts/             Local orchestration, PyMC sampler, and SVI workers
worker/              Local Worker API for uploads and cached artifacts
db/ + drizzle/       Local D1 schema and migrations
tests/               Product, inference, numerical, and research contracts
research/            Simulators, frozen selectors, audits, and results
```

## Verification

```bash
pnpm install --frozen-lockfile
pnpm lint
pnpm exec tsc --noEmit
pnpm test
pnpm test:svi-score-v9
pnpm test:svi-score-v10-amss
pnpm test:svi-score-v11-ea
pnpm test:svi-score-v11-amss
pnpm test:experiment-csv
pnpm test:regretset-v11-runtime
```

After `pnpm mcmc:setup`, the Python selector-equivalence checks are available as
`pnpm test:svi-score-v11-ea-python`.

To regenerate the publication-format PDF from the compact research artifacts:

```bash
pnpm research:paper:setup
pnpm research:paper:build
```

Large posterior checkpoints and generated cohorts are excluded from Git. The
repository includes frozen V9 and V11 selectors, compact audit results,
pre-truth selections, hashes, candidate-level external losses, and the code
needed to inspect the research.

## Scientific limits

- Observational MMM can remain confounded by demand, targeting, price,
  promotion, inventory, competition, and missing controls.
- Validation can falsify weak candidates; it cannot manufacture randomization.
- RegretSet-MMM was developed and audited in simulation. Independent AMSS
  transport is stronger than testing only in the Flux generator, but it is not
  a substitute for prospective validation on real advertiser experiments.
- V11's learned expert routing is not feature importance, evidence quality, or
  causal validity. The joint pathway still sees every candidate token.
- Results are conditional on the candidate pool, inference engine, token schema,
  channel adapter, and declared 65% mean / 35% P90 risk preference.
- Neither specification search nor budget optimization guarantees a global
  optimum. Budget outputs are scenario analysis, not financial advice.

## Project trust

- **Maintainer:** [Gustavo Bramao](https://github.com/gustavobramao)
- **License:** [MIT](LICENSE)
- **Security:** [responsible disclosure and local-data guidance](SECURITY.md)
- **Contributing:** [development and review expectations](CONTRIBUTING.md)
- **Citation:** [machine-readable citation metadata](CITATION.cff)

## License

Copyright © 2026 Gustavo Bramao and FluxMMM contributors. Released under the
[MIT License](LICENSE).
