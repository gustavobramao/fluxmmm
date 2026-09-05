# Step 2 — AMSS reproduction and interface audit

## Result

**Pass, with one pre-cohort channel-transport decision still required.** The
official four-year weekly example runs locally from the pinned upstream source,
produces the documented modeler-facing table, regenerates outcomes under changed
budgets, and exposes simulator-derived ROAS and marginal ROAS.

This step used no GCP resources and produced no V10 candidate fits or research
labels.

## Reproducibility contract

| Item | Pinned value |
| --- | --- |
| Upstream repository | `https://github.com/google/amss.git` |
| Git revision | `cbf5e7f6c668de493077ce1d08a2ab963891f0cc` |
| AMSS package | `1.0.1` |
| License | Apache-2.0 |
| Verified R runtime | `4.6.1` |
| `assertthat` | `0.2.1` |
| `data.table` | `1.18.6.1` |
| Scenario seed | `1` |

The upstream code is fetched into temporary storage rather than copied into the
FluxMMM repository. Exact provenance is recorded in `amss-upstream.lock.json`.

## Reproduced official scenario

- Four years of weekly simulation: 208 time points.
- One-year burn-in: 156 modeler-facing observations remain.
- Two native media modules: television and paid search.
- 198 latent consumer states per week across market, satiation, activity,
  favorability, loyalty, and availability dimensions.
- Television changes brand awareness/favorability through a traditional media
  exposure module with flighting and a Hill frequency-response curve.
- Search changes activity through query volume, auction eligibility, bids, CPC,
  CTR, and keyword-list expansion.
- Sales depend on purchase state, brand state, competitor pressure, and price.

The generated table contains 18 columns after the vignette adds its observed
seasonal market-rate covariate. The variables needed for a two-channel MMM are
available directly: weekly revenue, TV spend/volume, search spend/clicks/
impressions/query volume, total spend, budgets, and the seasonal covariate.

The vignette explicitly describes `market.rate` as perfect knowledge that would
not usually be available. Reproducing it is correct for this smoke test, but the
recommended V10 cohort must not pass this latent truth directly to Flux. It
should use calendar/Fourier terms or a separately generated noisy observable
proxy under a frozen measurement-error contract.

## Truth and counterfactual interfaces

`CalculateROAS` does not read a coefficient labelled “true ROI.” It regenerates
the system under an original and a counterfactual budget and calculates

\[
\frac{\sum_t Y_t^{\mathrm{counterfactual}}-\sum_t Y_t^{\mathrm{original}}}
{\sum_t S_t^{\mathrm{counterfactual}}-\sum_t S_t^{\mathrm{original}}}.
\]

Setting a channel to zero estimates average ROAS. A small proportional budget
change estimates marginal ROAS. The evaluation window can extend beyond the
changed budget period when carryover is present.

`GenerateDataUnderNewBudget` exposes the lower-level counterfactual path. The
smoke test changed the fourth-year TV budget by -20% and search budget by +20%
and successfully regenerated paired weekly outcomes. `OptimizeSpend` is also
available, although V10 can calculate its own feasible action grid and use the
counterfactual generator as the value oracle.

The four ROAS numbers in `truth-interface-smoke.csv` use only two replicates and
a deliberately reduced population scale. They verify plumbing only and are
marked `research.label = FALSE`; they are not admissible V10 outcomes.

## Leakage boundary for V10

Before a candidate is selected, the adapter may expose only the modeler-facing
weekly table and declared business constraints. It must not expose:

- `data.full`, which contains latent consumer-state tables;
- simulator transition matrices or response parameters;
- `CalculateROAS` outputs;
- counterfactual outcomes under candidate allocations; or
- `OptimizeSpend` results.

After all 48 candidate actions are immutable, the truth process may evaluate
those actions and the common feasible oracle.

## Channel-transport finding

The default AMSS vignette has **TV and paid search only**. The frozen V9 selector
contains three archetype-specific ROI-center tokens for paid social, nonbrand
search, and CTV, in addition to channel-order-invariant posterior summaries.
Silently filling the absent paid-social token with zero, a prior mean, or a TV
value would create an artificial selector signal and is not acceptable.

The clean Step 3 design is therefore to keep the AMSS engine untouched but
instantiate a third channel through AMSS's public media-module interface before
the cohort is generated. The recommended mapping is:

| AMSS mechanism | Frozen V9 role | Interpretation |
| --- | --- | --- |
| Search module | Nonbrand search | Native query/auction demand-harvesting channel |
| Traditional module A | CTV | Long-memory reach-media role; reported as an archetype mapping, not literal CTV |
| Traditional module B with distinct targeting, flighting, and response | Paid social | Exposure-media stress proxy; explicitly not a platform-auction replica |

This preserves the frozen 232-token input contract and uses only independently
implemented AMSS mechanisms, but the paid-social mapping remains a limitation.
It must be predeclared before any cohort or truth label is generated.

AMSS search is also a generic paid-search auction rather than an explicit
brand/nonbrand split. V10 should call it a nonbrand-search *archetype mapping*
and predeclare keyword-list and targeting settings that make demand harvesting
visible; it must not claim that AMSS natively distinguishes search tactics.

## Decisions required before Step 3

1. Freeze the three-role channel mapping above, including its limitations.
2. Exclude the perfect `market.rate` series from candidate inputs or replace it
   with a predeclared noisy proxy.
3. Freeze AMSS scenario ranges, seeds, burn-in, and sample size before cohort
   generation.
4. Freeze ROAS/mROAS perturbations, counterfactual evaluation windows, carryover
   tails, replicate counts, and precision targets.
5. Freeze the mapping from AMSS budgets/actions to the same four decision
   contracts expected by V9.

## Step 2 artifacts

- `default-observed.csv`: the 156-row modeler-facing table.
- `default-budget.csv`: four budget periods by two native channels.
- `counterfactual-interface-smoke.csv`: paired original and changed-budget
  summaries.
- `truth-interface-smoke.csv`: non-research ROAS API checks.

The deterministic script includes assertions for dimensions, required columns,
finite values, the accounting identity `profit = revenue - total.spend`, media
names, and artifact completeness.
