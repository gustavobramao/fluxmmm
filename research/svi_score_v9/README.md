# FluxMMM V9 set-wise distributional selector

V9 is an isolated development track. It does not modify V8, reopen the V8
sealed audit, or activate a production score.

The core change is the unit of learning. V8 predicts each candidate largely
in isolation and then ranks the predictions. V9 receives all 48 candidates
for an advertiser as one unordered set. A shared candidate encoder is pooled
into an advertiser-specific context, and the decoder compares every candidate
with that context. Reordering candidates cannot change a prediction.

V9 also adds truth-blind posterior decision tokens from the already-cached SVI
draws. The tokens summarize channel-level ROI shape, uncertainty, response
parameters, evidence source, and posterior trade-offs. Channel values are
reduced with symmetric mean, standard-deviation, minimum, and maximum
operations, so channel ordering cannot leak identity.

## Development protocol

1. Build the 232-token development dataset from the frozen 420 × 48 V8
   development cohort and cached FullRankADVI results.
2. Train a small DeepSets multi-head selector with five outer advertiser
   folds and four inner advertiser folds.
3. Learn mean, median, P90, P95, CVaR90, four absolute scenario losses, and
   dangerous false-champion probability. Candidate-independent loss offsets
   do not affect selection; exact excess regret remains the ranking target.
4. Optimize the expected normalized economic regret of selecting from the
   full 48-candidate set. Retain the exact theorem-aligned oracle-versus-
   candidate loss as a regularizer.
5. Compare V9 and V8 at 100% coverage, at identical promotion coverage, and
   across the full 60%–100% selective-risk curve.
6. Repeat fitting while holding out each complete simulator family. Model and
   policy selection use only the remaining families, so this challenges
   transfer to an unseen business mechanism rather than another random split.
7. Freeze V9 only if development checks pass. A new, untouched audit cohort is
   required for confirmatory evaluation; the V8 audit cannot be reused.

```bash
npm run research:svi-score-v9:prepare
npm run research:svi-score-v9:train
npm run research:svi-score-v9:mechanism-holdout
npm run research:svi-score-v9:verify
npm run test:svi-score-v9
```

No new MMM fit is required for this stage. It reuses cached SVI posteriors and
hidden economic labels from development only.

## One-time confirmatory audit

V9's confirmatory synthetic audit is a new 140-business cohort: 20 businesses
from each of seven declared simulator families and 48 candidates per business.
Its seed window is disjoint from all 500 earlier research businesses. The V9
and V8 selectors, source code, SVI contract, endpoint definitions, bootstrap
procedure, and audit scorer are hashed before the cohort is generated.

```bash
npm run research:svi-score-v9:audit-freeze
npm run research:svi-score-v9:audit-prepare
# Run the 6,720 immutable SVI payloads locally or on GCP, then place results
# in .flux-artifacts/svi-score-v9/audit/results.
npm run research:svi-score-v9:audit-build
npm run research:svi-score-v9:audit-score
npm run research:svi-score-v9:audit-verify
```

Scoring is one-time. Technical inference failures may be rerun only with the
identical frozen payload and inference contract. No model, token, threshold,
endpoint, or source change may be repaired against this cohort.

Audit 001 ultimately produced 171 posterior envelopes before its
quota-constrained cloud orchestration was stopped. Audit 002 was stopped during
truth-blind preparation after a zero-based shard-index defect was observed; it
produced no posterior envelopes. Neither audit's hidden economic truth,
selector score, or endpoint was opened. Both are permanently retired rather
than repaired against observed outcomes.

Audit 003 is the sole confirmatory cohort. It uses a new disjoint seed window,
a regression-tested zero-based parser, one immutable candidate per Cloud Run
task, and a frozen 180-way execution contract. The complete outcome-blind
execution and cost declaration is in `AUDIT_003_PROTOCOL.md`.

## Current development outcome

The nested run passed all development checks. Against V8 at 100% coverage,
mean excess loss decreased 32.1%, P90 decreased 42.8%, P95 decreased 30.1%,
and the selection objective decreased 38.6%. The 60%–100% selective-risk area
decreased 42.6%.

In the stricter leave-one-simulator-family-out assessment, V9's aggregated
mean excess loss was 1.348 and P90 was 3.971. The descriptive V8 reference was
2.168 and 6.349, respectively. V8 did not itself hold out complete mechanism
families, so this comparison is intentionally labelled descriptive rather than
like-for-like. See `DEVELOPMENT_RESULT.md` for the full comparison and the
interpretation boundary.
