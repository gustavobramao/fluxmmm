# V10 AMSS external-transport validation

V10 evaluates the frozen V9 selector in Google's independently developed
Aggregate Marketing System Simulator (AMSS). It is an external-transport
validation, not a V9 retraining exercise.

## Step 1 — frozen V9 selector

The snapshot in `frozen-v9/` was created before AMSS was installed or any AMSS
business was generated. It preserves:

- the exact V9 DeepSets ensemble;
- the ordered 232-token registry;
- the FullRankADVI inference contract;
- the four hard completeness gates;
- the 65% mean / 35% P90 economic-risk preference;
- the selected uncertainty and danger policy; and
- the source files that define these scientific contracts.

The AMSS adapter must consume the frozen selector and registry. It must not
retrain V9, add or remove tokens, tune thresholds, or expose AMSS ground truth
before model selection. Advertiser-visible AMSS inputs are permitted because
they are required to fit candidates and calculate truth-blind selector tokens.

```bash
npm run research:svi-score-v10-amss:verify-v9-freeze
```

The freeze command intentionally refuses to overwrite an existing snapshot.
Any future selector constitutes a new method and requires a separately named
external-validation protocol.

## Step 2 — reproduce and audit AMSS

The adapter is pinned to the upstream AMSS v1.0.1 source revision recorded in
`amss-upstream.lock.json`. The deterministic smoke script reproduces the
official four-year weekly vignette scenario and writes only small, inspectable
artifacts:

```bash
R_LIBS_USER=/private/tmp/fluxmmm-r-library \
  Rscript research/svi_score_v10_amss/reproduce-default-amss.R
```

The ROAS values produced by this script use only two simulator replicates and a
reduced population scale. They verify that the truth interface works; they are
explicitly marked `research.label = FALSE` and are not V10 audit outcomes.

## Step 3 — three-channel AMSS adapter

The paid-social stress proxy uses AMSS's unchanged public
`DefaultTraditionalMediaModule`. Its raw name and limitation are retained in
all provenance, while the modeler-facing adapter maps the three AMSS roles to
the frozen V9 schema. The smoke test exposes no simulator truth and creates no
research labels.

```bash
R_LIBS_USER=/private/tmp/fluxmmm-r-library \
  Rscript research/svi_score_v10_amss/smoke-paid-social-proxy.R

npm run research:svi-score-v10-amss:verify-step3
```

See `STEP_3_CHANNEL_PROTOCOL.md` for the scientific boundary.

## Steps 4–6 — sealed cohort, posterior fits and frozen decisions

Before generating any business, V10 froze the AMSS parameter ranges, 100
business seeds, four balanced evidence groups, all six media-module orders, 24
candidate specifications, two evidence arms, four budget decisions and the
economic-loss contract. The resulting workload contains 4,800 FullRankADVI
candidate fits. It completed with 4,779 accepted labels, 21 predeclared review
labels and no technical errors.

Candidate posteriors then generated 19,200 actions (100 businesses × 48
candidates × four decisions). The V9 and original-heuristic choices were frozen
for every business before AMSS truth was opened. Neither selector was retrained
or rerun after truth became available.

## Step 7 — confirmatory external-transport result

AMSS evaluated the frozen actions with four common-random-number replicates per
business and decision. All 100 Cloud Run tasks completed successfully, yielding
19,200 verified outcomes. The primary scenario loss is capped at the
predeclared value of one; uncapped loss is retained as a separate tail-risk
endpoint. Business risk is 65% mean loss plus 35% P90 loss across the four
decisions.

| Frozen selector | Mean risk | P90 | P95 |
| --- | ---: | ---: | ---: |
| V9 | 0.230 | 0.707 | 0.757 |
| Original Flux heuristic | 0.560 | 0.991 | 1.000 |
| Uniform random valid candidate, expected | 0.400 | 0.591 | 0.614 |
| In-pool oracle | 0.012 | 0.030 | 0.039 |

The paired V9-minus-heuristic mean difference is −0.330, with a 95% paired
bootstrap interval of [−0.402, −0.258]. V9 has lower loss for 80 businesses,
ties for two and has higher loss for 18. This is evidence of transport to a
second, independently developed simulator; it is not validation on real
advertiser outcomes.

Evidence-group summaries are descriptive only. In particular, the
paid-social-experiment group is close and slightly favors the heuristic on mean
risk (0.049 versus 0.094 for V9). The frozen protocol forbids subgroup inference
from 25 businesses per group, so this is a documented target for future stress
testing rather than a post-hoc tuning opportunity.

During final contract verification, the first audit assembly was found to have
reported uncapped loss as the primary endpoint despite the predeclared cap.
Technical amendment 004 preserves that preliminary output as invalid, applies
the declared cap without changing models, actions, selections or AMSS truth,
and reports uncapped results separately. The accepted result and its hashes are
recorded in `external-audit/acceptance-receipt.json`.

```bash
npm run research:svi-score-v10-amss:verify-cloud-truth
npm run research:svi-score-v10-amss:build-audit
npm run test:svi-score-v10-amss
```
