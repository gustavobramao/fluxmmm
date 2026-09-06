import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { resolve } from "node:path";
import {
  V11_AMSS_CONFIRMATORY_CONTRACT,
  V11_AMSS_CONFIRMATORY_VERSION,
} from "../research/svi_score_v11_amss_confirmatory/contract";
import { calculateV11ConfirmatoryEconomicLoss } from "../research/svi_score_v11_amss_confirmatory/economic-loss";

test("fresh AMSS cohort is disjoint, balanced, and truth sealed", () => {
  assert.equal(V11_AMSS_CONFIRMATORY_VERSION, "flux-svi-score-v11-amss.1.0.0-fresh-confirmatory");
  assert.equal(V11_AMSS_CONFIRMATORY_CONTRACT.cohort.businesses, 100);
  assert.equal(V11_AMSS_CONFIRMATORY_CONTRACT.cohort.candidateFits, 4_800);
  assert.ok(V11_AMSS_CONFIRMATORY_CONTRACT.cohort.businessSeedStart > 1_000_000);
  assert.ok(V11_AMSS_CONFIRMATORY_CONTRACT.cohort.truthSeedStart > 1_000_000);
  assert.notEqual(V11_AMSS_CONFIRMATORY_CONTRACT.cohort.businessSeedStart, 710_001);
  assert.notEqual(V11_AMSS_CONFIRMATORY_CONTRACT.cohort.truthSeedStart, 910_001);
  assert.equal(V11_AMSS_CONFIRMATORY_CONTRACT.governance.postOpenTuningPermitted, false);
});

test("confirmatory decision rule is fixed before truth", () => {
  const rule = V11_AMSS_CONFIRMATORY_CONTRACT.evaluation.confirmatoryDecisionRule;
  assert.equal(rule.primarySuccess.length, 2);
  assert.match(rule.tailSafety, /0\.02/);
  assert.match(rule.uncertainty, /10,000/);
});

test("economic loss is nonnegative, capped at one, and retains uncapped safety", () => {
  const loss = calculateV11ConfirmatoryEconomicLoss({
    oracleProfit: 200,
    realizedProfit: 0,
    baselineProfit: 100,
    baselineBudget: 1_000,
    normalizedLossCap: 1,
  });
  assert.equal(loss.economicScale, 100);
  assert.equal(loss.uncapped, 2);
  assert.equal(loss.capped, 1);
});

test("known V10 defects are repaired before the fresh freeze", async () => {
  const [scenario, generator, truth, task] = await Promise.all([
    readFile(resolve("research/svi_score_v11_amss_confirmatory/amss-cohort-scenario.R"), "utf8"),
    readFile(resolve("research/svi_score_v11_amss_confirmatory/generate-cohort.R"), "utf8"),
    readFile(resolve("research/svi_score_v11_amss_confirmatory/open-truth.R"), "utf8"),
    readFile(resolve("research/svi_score_v11_amss_confirmatory/cloud/truth_task.py"), "utf8"),
  ]);
  assert.match(scenario, /min\(1 - 1e-9/);
  assert.match(generator, /utilization <- setNames/);
  assert.match(truth, /utilization <- setNames/);
  assert.doesNotMatch(task, /technical-amendment|amendment_00/);
});

test("Cloud Run inference is immutable one-candidate SVI", async () => {
  const yaml = await readFile(
    resolve("research/svi_score_v11_amss_confirmatory/cloud/inference-cloud-run.yaml"),
    "utf8",
  );
  assert.match(yaml, /taskCount: 4800/);
  assert.match(yaml, /parallelism: 180/);
  assert.match(yaml, /maxRetries: 0/);
  assert.match(yaml, /v11-amss-confirmatory-001\/execution-shards\/manifest\.json/);
  assert.match(yaml, /sha256:3276faca285998efdf40db81ec8b881d4102ab5b3eb528e4dba529e53c9e893e/);
});

test("truth job preserves task zero and new business identity", async () => {
  const [task, yaml] = await Promise.all([
    readFile(resolve("research/svi_score_v11_amss_confirmatory/cloud/truth_task.py"), "utf8"),
    readFile(resolve("research/svi_score_v11_amss_confirmatory/cloud/truth-cloud-run.yaml"), "utf8"),
  ]);
  assert.match(task, /task_index < 0/);
  assert.match(task, /amss-v11c-/);
  assert.match(task, /if_generation_match=0/);
  assert.match(yaml, /amss-truth-v11-confirmatory@sha256:ab78317233dd42d0391acd6b1611f451ef2616a1ed5b33fa08cf865bdcdec8a7/);
  assert.doesNotMatch(yaml, /amss-truth-v10/);
  assert.equal(V11_AMSS_CONFIRMATORY_CONTRACT.governance.discardedPreCohortBuildUsedForAudit, false);
});

test("published audit receipt preserves the frozen V11 result", async () => {
  const result = JSON.parse(await readFile(resolve(
    "research/svi_score_v11_amss_confirmatory/artifacts/result.json",
  ), "utf8"));
  assert.equal(result.status, "fresh-confirmatory-audit-complete-no-post-truth-tuning");
  assert.equal(result.cohort.businesses, 100);
  assert.equal(result.cohort.candidateModels, 4_800);
  assert.equal(result.cohort.decisionActions, 19_200);
  assert.equal(result.provenance.candidateActionsFrozenBeforeTruth, true);
  assert.equal(result.provenance.selectorRetrainedAfterTruth, false);
  assert.equal(result.confirmatoryDecisionRule.passed, true);
  assert.ok(Math.abs(
    result.endpoints.evidenceAdaptiveV11.selectionObjective - 0.20260835892779494,
  ) < 1e-12);
  assert.ok(
    result.endpoints.evidenceAdaptiveV11.selectionObjective
      < result.endpoints.predictionOnly.selectionObjective,
  );
  assert.ok(
    result.endpoints.evidenceAdaptiveV11.selectionObjective
      < result.endpoints.frozenV9.selectionObjective,
  );
});
