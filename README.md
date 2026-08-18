# FluxMMM

[![CI](https://github.com/gustavobramao/fluxmmm/actions/workflows/ci.yml/badge.svg)](https://github.com/gustavobramao/fluxmmm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-111827.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/Live_demo-open-5b5ce2.svg)](https://fluxmmm-demo.web.app/)
[![Working paper](https://img.shields.io/badge/Working_paper-read-0f766e.svg)](https://fluxmmm-web.web.app/paper/)

**Open measurement. Grounded ROI.**

FluxMMM is a local-first, open-source marketing mix modeling workspace that
makes the full path from data to budget decision inspectable. It combines fast
analytic model search, explicit experimental or benchmark calibration,
four-layer validation, full NUTS sampling of the promoted specification, and
posterior-aware budget planning in one auditable workflow.

[Explore the cached public demo](https://fluxmmm-demo.web.app/) ·
[Read the methodology](https://fluxmmm-web.web.app/paper/) ·
[Read the V2 release notes](RELEASE_NOTES.md) ·
[Visit FluxMMM](https://fluxmmm-web.web.app/)

![FluxMMM — open measurement and grounded ROI](public/og.png)

## Why FluxMMM

MMM is often selected for predictive fit and then used for causal and budget
decisions that require stronger evidence. FluxMMM keeps those claims separate.
Each candidate is challenged on:

1. **Predictive generalization** — rolling-origin and spend-regime holdouts.
2. **Structural integrity** — residual behavior, identification, functional
   form, multicollinearity, influence, and assumption-aware diagnostics.
3. **Causal credibility** — experiment-anchor recovery, new-data stability,
   latent-confounder stress, and future-media placebos.
4. **ROI decision coherence** — business plausibility, refit stability,
   interval resolution, prior dependence, and economic consistency by channel.

The V2 Flux Decision Score is a monotonic weighted geometric mean learned
offline from synthetic businesses with hidden causal and budget-decision truth:

```text
100 × exp(Σ diagnostic_weight × log(diagnostic_score / 100))
```

Twenty granular diagnostics remain organized into the same four readable
layers. Weights are non-negative and constrained by layer and feature caps.
Scores rank candidates only inside the highest available immutable eligibility
tier after applicability-aware gates are evaluated. A high score is not
presented as proof of causal identification.

## What ships in v2

- Daily, weekly, and monthly CSV ingestion with deterministic repairs,
  semantic column mapping, and visible data-contract receipts.
- Progressive EDA for coverage, sparsity, variation, spend regimes, and
  cross-channel correlation.
- Geometric or Weibull-PDF adstock, Hill saturation, trend, cadence-aware
  Fourier seasonality, cycle effects, and observed controls.
- A non-negative regularized frequentist baseline and fast Bayesian
  MAP/Laplace screening estimator.
- Experiment-informed prior or likelihood calibration, plus optional
  channel-level D2C ecommerce plausibility priors when experiments are absent.
- Experiment evidence is compared with the MMM response attributable to spend
  in the same declared test window. An optional outcome end date includes
  measured carryover without adding post-test spend to the denominator.
- Optional smooth KTR-style time-varying coefficients and a latent
  media-planning-intensity sensitivity factor.
- Half-normal or log-normal coefficient priors and Gaussian, Student-t, or
  log-normal outcome likelihoods.
- Numerical reliability receipts: pivoted QR/SVD fallback, rank and condition
  diagnostics, prior-influence labels, and non-negative-boundary disclosure.
- Bounded, family-aware agentic specification search with immutable evidence,
  parameter bounds, validation gates, multi-start coverage, and local
  challenges. It searches intelligently; it does **not** claim a global optimum.
- Full PyMC NUTS sampling of the frozen winner with R-hat, effective sample
  size, divergence, BFMI, tree-depth, MCSE, HDI, and posterior predictive gates.
- Fixed-budget, outcome-target, and economic-ceiling allocation with channel
  constraints, posterior uncertainty, support warnings, and interactive guides.
- Deterministic model fingerprints and local artifact caching.
- A versioned simulator audit and learned validation ranker trained only on
  hidden synthetic decision loss, with family-held-out validation and a sealed
  adversarial audit.

## Public fixture, not a Robyn wrapper

The bundled demonstration uses a public simulated weekly fixture distributed
with Meta's Robyn materials so the product can be inspected reproducibly. Robyn
supplies the example input data. FluxMMM supplies its own ingestion, model
contracts, estimators, calibration, validation, specification search, sampling,
and budget-decision workflow.

The fixture demonstrates software behavior. It is not an accuracy benchmark and
does not establish external validity for another advertiser.

## Quick start

### Requirements

- macOS, Linux, or Windows through WSL2
- Node.js 22.13 or newer
- pnpm 11.9 (Corepack is recommended)
- Python 3.12 for full production sampling; optional for the analytic workspace

### Run the analytic workspace

```bash
git clone https://github.com/gustavobramao/fluxmmm.git
cd fluxmmm
corepack enable
pnpm install --frozen-lockfile
pnpm dev
```

Open [http://localhost:3003](http://localhost:3003). No login, API key, OpenAI
account, or cloud project is required.

### Enable full NUTS sampling

```bash
pnpm mcmc:setup
pnpm dev
```

`mcmc:setup` creates an ignored `.venv` and installs the pinned PyMC sampling
environment. `pnpm dev` then starts both the product and the local sampling
service. If Python is not installed, the rest of FluxMMM remains usable and the
Production tab explains that sampling is unavailable.

## Data contract

Each dataset needs:

- one date column with consecutive daily, weekly, or monthly observations;
- one numeric outcome such as revenue or conversions;
- one or more non-negative paid-media spend columns; and
- no empty or non-numeric cells in the outcome, active media, or active controls.

Exposure, event, numeric control, and categorical control columns are optional.
Unambiguous date formats are normalized with a repair receipt. Consecutive daily
data is aggregated into complete seven-day periods; weekly and monthly data keep
their native cadence. Ambiguous dates, duplicate periods, gaps, and unsafe
reinterpretations stop with a specific diagnostic rather than being silently
changed.

Uploaded files and run artifacts stay on the local machine under ignored
`.wrangler` and `.flux-artifacts` directories. They are not sent to FluxMMM,
OpenAI, GitHub, or another hosted service by this codebase.

## Evidence and calibration

Independent experiments are the preferred ROI anchor. Compatible experiments
are pooled with uncertainty retained and can enter as an informative coefficient
prior or a noisy ROI measurement. Likelihood calibration requires an explicit
overlap review because reusing outcomes or controls can double-count evidence.
Anchor recovery always compares like with like: spend in the experiment window
is removed from the fitted response path, its incremental contribution is
measured through the declared outcome window, and the resulting period-specific
ROI is compared with the reported lift estimate.

When a channel has no compatible experiment, a user may activate an optional
channel-specific industry prior. These priors are pragmatic plausibility
guardrails for large D2C ecommerce businesses—not universal ground truth—and
their influence remains visible in the result.

## Search and production inference

Fast MAP/Laplace approximations make bounded specification search interactive.
After promotion, FluxMMM freezes the complete winning contract—data,
transformations, priors, likelihood, calibration route, advanced structure, and
ROI mapping—before PyMC samples it with NUTS. MAP and MCMC are compared on the
same contract, and approximation shifts are displayed rather than hidden.

Forced promotion is available as a recorded governance override when no
eligible model exists. It does not convert weak evidence into strong evidence.

## Architecture

```text
app/                 Product UI and progressive client orchestration
app/components/      Interactive ECharts visualizations
lib/mmm/             Schema, EDA, transformations, estimators, search, budget
lib/mmm/validation/  Leakage-aware refits, four validation layers, gates
scripts/             Local development orchestration and PyMC sampling service
worker/              Local Worker API for uploads and cached artifacts
db/ + drizzle/       Local D1 schema and migrations
public/data/         Versioned public demonstration fixtures
tests/               Data, numerical, model, MCMC, and rendered-product contracts
research/            Simulator audit, hidden decision labels, and learned-score artifacts
```

The modeling modules do not depend on React or persistence. The sampler consumes
the same compiled specification as the analytic fit, allowing the UI and
inference engines to evolve independently.

## Verification

```bash
pnpm lint
pnpm exec tsc --noEmit
pnpm test

# after pnpm mcmc:setup
pnpm mcmc:test
```

Permanent regression fixtures cover correlated and duplicated media, zero
variance, sparsity, extreme numerical scales, underdetermined systems, and
daily, weekly, and monthly cadence behavior. Tests require either finite,
reproducible estimates or a specific diagnostic.

## Scientific limits

FluxMMM is an evidence and governance system, not an automatic truth machine.

- Observational MMM can remain confounded by demand, targeting, price,
  promotion, inventory, competition, and missing controls.
- Validation can falsify weak candidates; it cannot manufacture randomization
  or variation absent from the data.
- Time-varying effects and planning intensity add flexibility and can worsen
  identification as well as improve fit.
- The V2 diagnostic weights are learned from a declared synthetic population.
  They reduce held-out simulator decision loss but are not universal empirical
  weights for every advertiser, category, or measurement design.
- Neither specification search nor nonlinear budget optimization guarantees a
  global optimum.
- Budget outputs are scenario analysis, not financial advice.

For the full equations, evidence hierarchy, diagnostics, references, and research
agenda, read the [FluxMMM working paper](https://fluxmmm-web.web.app/paper/).

## Project trust

- **Maintainer:** [Gustavo Bramao](https://github.com/gustavobramao)
- **License:** [MIT](LICENSE)
- **Security:** [responsible disclosure and local-data guidance](SECURITY.md)
- **Contributing:** [development and review expectations](CONTRIBUTING.md)
- **Citation:** [machine-readable citation metadata](CITATION.cff)

## Attribution

FluxMMM builds on established work in Bayesian MMM, experimental calibration,
time-varying coefficients, sequential optimization, sensitivity analysis, and
Hamiltonian Monte Carlo. The working paper distinguishes established components
from FluxMMM's software and governance contribution and links the primary
literature.

## License

Copyright © 2026 Gustavo Bramao and FluxMMM contributors. Released under the
[MIT License](LICENSE).
