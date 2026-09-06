import assert from "node:assert/strict";
import test from "node:test";
import {
  SVI_SCORE_V11_EA_CONTRACT,
  V11_EA_CONTEXT_NAMES,
  V11_EA_EXPERTS,
  V11_EA_PAIRED_METRICS,
  V11_EA_REGIMES,
  V11_EA_SUMMARY_METRICS,
  v11EaExpertForToken,
} from "../research/svi_score_v11_evidence_adaptive/contract";

test("evidence-adaptive development is isolated from AMSS and posterior refits", () => {
  assert.equal(SVI_SCORE_V11_EA_CONTRACT.governance.activation, "development-only");
  assert.equal(
    SVI_SCORE_V11_EA_CONTRACT.governance.amssConfirmatoryCohortAccessPermitted,
    false,
  );
  assert.equal(SVI_SCORE_V11_EA_CONTRACT.source.posteriorRefitsRequired, 0);
  assert.equal(
    SVI_SCORE_V11_EA_CONTRACT.pairedRegimes.experimentPresenceCounterfactualGenerated,
    false,
  );
  assert.equal(SVI_SCORE_V11_EA_CONTRACT.governance.freshIndependentAuditRequiredBeforeClaim, true);
});

test("context registry contains explicit masks and paired within-business regimes", () => {
  assert.deepEqual(V11_EA_REGIMES, [
    "full-candidate-set",
    "experiments-only",
    "benchmark-gap-fill",
  ]);
  assert.equal(new Set(V11_EA_CONTEXT_NAMES).size, V11_EA_CONTEXT_NAMES.length);
  assert.equal(V11_EA_CONTEXT_NAMES.length, 63);
  assert.ok(V11_EA_CONTEXT_NAMES.includes("evidence:experiment:available-mask"));
  assert.ok(V11_EA_CONTEXT_NAMES.includes("evidence:benchmark:available-mask"));
  assert.ok(V11_EA_CONTEXT_NAMES.includes("paired-evidence-arm:available-mask"));
  assert.equal(V11_EA_SUMMARY_METRICS.length, 11);
  assert.equal(V11_EA_PAIRED_METRICS.length, 8);
});

test("candidate tokens route to interpretable experts", () => {
  assert.equal(
    v11EaExpertForToken("diagnostic:rolling-oos"),
    "predictive-generalization",
  );
  assert.equal(
    v11EaExpertForToken("diagnostic:anchor-recovery"),
    "causal-identification",
  );
  assert.equal(
    v11EaExpertForToken("posterior:channel-set:log-roi-median:mean"),
    "posterior-decision",
  );
  assert.equal(
    v11EaExpertForToken("spec:global-adstock:weibull"),
    "structural-specification",
  );
  assert.deepEqual(V11_EA_EXPERTS, [
    "predictive-generalization",
    "causal-identification",
    "posterior-decision",
    "structural-specification",
  ]);
});
