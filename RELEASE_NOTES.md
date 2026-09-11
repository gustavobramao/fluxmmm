# FluxMMM 4.1.0 — Frozen V11 Runtime Selection

Released 11 September 2026.

This release activates the frozen evidence-adaptive RegretSet-MMM V11 contract
inside the local product workflow. The Agentic screen no longer presents an
analytic MAP/Laplace rank as though it were V11.

## Runtime alignment

- The Agentic workspace evaluates the complete 24-specification × 2-evidence-
  regime candidate set required by V11.
- Each candidate receives the frozen two-seed FullRankADVI posterior contract,
  232 observable candidate tokens, and 63 business-context tokens before the
  five-member DeepSets ensemble ranks adjusted economic risk.
- Standalone validation remains diagnostic evidence; a V11 score exists only
  after the complete 48-candidate set has been evaluated.
- Promotion preserves validation and ROI gates. NUTS then samples the frozen
  winner and cannot retroactively change which candidate V11 selected.

## Faster and safer local execution

- Inference-equivalent evidence arms reuse a single posterior artifact. When
  every media channel is already experiment-anchored, this reduces 48 declared
  candidates to at most 24 unique posterior fits without removing candidates.
- Existing artifacts created before the new canonical fingerprint are safely
  discovered and reused only when the statistical model and inference contract
  match exactly.
- Local SVI concurrency now adapts to installed memory, reducing memory
  contention on smaller machines.
- Oversized PyTensor graphs automatically retry through the deterministic C/VM
  backend after a compilation-limit failure.
- Experiment CSV upload, schema guidance, dynamic column mapping, and explicit
  missing-value repair are available in the calibration and data workflows.

## Verification

```bash
pnpm test
pnpm test:experiment-csv
pnpm test:regretset-v11-runtime
pnpm mcmc:test
```

The V11 weights, candidate registry, posterior-token contract, and selection
objective remain frozen. This is a product-runtime and reliability release, not
a retraining or a new research result.

---

# FluxMMM 4.0.0 — Evidence-Adaptive RegretSet-MMM

Released 6 September 2026.

This release publishes the frozen evidence-adaptive RegretSet-MMM V11 selector,
its fresh external synthetic audit, and a clearer product boundary between
research-grade SVI selection and responsive analytic model search.

## Model and evidence changes

- V11 retains V9's joint 232-token DeepSets pathway and adds four disjoint
  residual experts for predictive generalization, causal identification,
  posterior decision quality, and structural specification.
- A 63-token business context captures evidence availability, quality,
  compatibility, explicit missingness, and candidate-set summaries.
- The learned gate adapts residual expert routing to the advertiser's evidence
  environment. Routing values are not presented as source validity or total
  feature importance because the joint pathway always sees all tokens.
- Development reuses the frozen 20,160 FullRankADVI candidates across 420
  businesses and constructs three matched evidence regimes per business; it
  does not fabricate counterfactual posteriors or mask experiment effects after
  fitting.
- The ten-head selector and theorem-aligned loss target remain decision-focused:
  the declared promotion objective is 65% cohort mean plus 35% cohort P90
  capped economic loss across four budget decisions.

## Fresh AMSS audit

The frozen V11 selector was tested without retraining on 100 new businesses
from Google AMSS 1.0.1. The audit contains 4,800 SVI candidates, 19,200 frozen
actions, four balanced evidence environments, pre-truth candidate selections,
and 10,000 stratified paired-bootstrap resamples.

| Selector | Mean risk | P90 risk | 65/35 objective |
|---|---:|---:|---:|
| Evidence-adaptive V11 | 0.135 | 0.329 | **0.203** |
| Prediction-only | 0.217 | 0.605 | 0.353 |
| Frozen V9 | 0.247 | 0.688 | 0.401 |
| Uniform random valid expectation | 0.395 | 0.576 | 0.458 |
| Conventional Pareto | 0.539 | 1.000 | 0.701 |
| In-pool oracle | 0.017 | 0.052 | 0.029 |

V11 reduced the declared objective by 42.6% relative to prediction-only, 49.5%
relative to V9, 55.8% relative to random valid-candidate selection, and 71.1%
relative to the conventional Pareto selector. It passed all three predeclared
confirmatory checks. The Pareto comparison is a post-truth secondary benchmark;
it used the already accepted posterior actions and changed no V11 selection.

## Product and documentation

- The Score Research surface explains the V9-to-V11 lineage, adaptive expert
  architecture, exact benchmark hierarchy, truth firewall, and inference
  boundary without information overload.
- The public paper, website, demo, README, citation metadata, and research
  receipts now share the same frozen V11 numbers and terminology.
- The interactive Agentic modeler remains fast MAP/Laplace screening. Running
  the frozen V11 method requires its complete 48-candidate SVI pool and exact
  deployment-observable token/context contract.
- Continuous evidence attribution still separates observational data,
  experiments, benchmarks, and regularization; influence remains distinct from
  source quality, conflict, and decision dependence.

## Reproducibility

```bash
pnpm install --frozen-lockfile
pnpm test:svi-score-v11-ea
pnpm test:svi-score-v11-amss
```

The optional Python selector-equivalence test is available after
`pnpm mcmc:setup` as `pnpm test:svi-score-v11-ea-python`.

The paper can be rebuilt from the versioned compact artifacts with
`pnpm research:paper:setup` followed by `pnpm research:paper:build`.

Versioned artifacts include the selectors, contracts, manifests, compact audit
result, frozen pre-truth selections, candidate-level losses, secondary benchmark
receipt, hashes, and publication-format paper. Multi-gigabyte posterior
checkpoints and generated panels are excluded from Git.

## Claim boundary

This release supports external synthetic transport under the declared AMSS
adapter, candidate pool, FullRankADVI contract, and economic utility. It does
not prove causal validity on an observational dataset, prospective performance
for real advertisers, universal superiority, or a global optimum. V3 remains
available from tag `v3.0.0`.
