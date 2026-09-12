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

This release ships the frozen **RegretSet-MMM relative-log-regret selector**.
It learns across synthetic advertisers which complete MMM specification is
least likely to produce costly downstream budget decisions. The final model
combines raw posterior and validation evidence with each candidate's relative
position inside the advertiser's candidate set. Its model, token registries,
cross-validation receipts, tests, and paper are published with the product.

[Explore the cached public demo](https://fluxmmm-demo.web.app/) ·
[Read the paper](https://fluxmmm-web.web.app/paper/) ·
[Read the v5 release notes](RELEASE_NOTES.md) ·
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

## RegretSet-MMM: relative log-regret

RegretSet-MMM changes the model-selection target. Instead of asking only which
candidate predicts best, it asks which candidate is least likely to recommend
an economically costly budget decision.

- **420 development businesses** generated across declared mechanism families,
  reusing **20,160 FullRankADVI candidate fits** under three matched evidence
  regimes per business.
- **48 truth-blind candidates per complete set**, with **232 raw candidate
  tokens** spanning diagnostics, posterior geometry, evidence, and the model
  contract.
- **464 candidate-relative tokens**: within-advertiser percentile ranks and
  robust median/IQR distances for each raw token.
- **63 business-context tokens** encoding evidence availability, quality,
  compatibility, missingness, and candidate-set summaries.
- A permutation-equivariant **DeepSets multi-head selector** that learns from
  696 candidate tokens and 63 context tokens.
- Ten jointly trained loss and tail-risk heads covering four predeclared budget
  decisions. Training uses `log(1 + uncapped excess economic loss)` so severe
  candidates remain ordered instead of collapsing at a cap.
- Promotion minimizes the declared **65% mean / 35% P90** predicted risk. No
  post-prediction danger or ensemble-disagreement penalty changes the ranking.

In five-fold advertiser-grouped cross-validation, the final selector achieved
a primary mean-tail objective of **0.805**, versus **1.031** for a
posterior-geometry-only DeepSets ablation, **1.188** for prediction-only,
**1.243** for Classic Pareto, and **1.377** for uniform random valid selection.
It reduced the objective by **32.3%** relative to prediction-only.

These findings are out-of-sample for advertisers held out from selector
training within the declared synthetic population. They do **not** establish
real-advertiser effectiveness or universal SOTA performance.

The research release lives in:

```text
research/regretset_relative_features/      Final selector, grouped-CV evidence, and interpretation
lib/mmm/artifacts/                          Frozen runtime selector and candidate registry
docs/regretset-mmm-paper.pdf                Publication-format working paper
```

The interactive Agentic workspace runs the frozen RegretSet-MMM selection contract
directly. It screens 24 predeclared specifications under two evidence regimes,
fits the resulting complete 48-candidate pool with the declared FullRankADVI
contract, builds the exact 696 candidate and 63 business-context tokens, and
ranks candidates by predicted economic risk. MAP/Laplace remains a
fast diagnostic screen inside each candidate workflow; it is not substituted
for the posterior inputs expected by RegretSet-MMM. After promotion, NUTS independently
samples the selected specification before budget planning.

Inference-equivalent evidence arms share one posterior artifact when their
compiled data, likelihood, priors, and inference contract are identical. This
can reduce a fully experiment-anchored search from 48 to 24 unique posterior
fits without changing the RegretSet candidate set or its selection semantics.

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
- Frozen RegretSet-MMM selection across 48 paired FullRankADVI candidates, with immutable
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
repository includes the frozen relative-log-regret selector, grouped
cross-validation results, interpretation artifacts, hashes, and the code needed
to inspect the research.

## Scientific limits

- Observational MMM can remain confounded by demand, targeting, price,
  promotion, inventory, competition, and missing controls.
- Validation can falsify weak candidates; it cannot manufacture randomization.
- RegretSet-MMM was developed and evaluated in simulation. Advertiser-grouped
  cross-validation is not a substitute for prospective validation on real
  advertiser experiments.
- Drop-column importance is predictive and conditional on correlated token
  groups; it is not evidence quality or causal validity.
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
