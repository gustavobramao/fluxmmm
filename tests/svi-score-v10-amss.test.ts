import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { SamplingDecisionDraw } from "../lib/mmm/sampling";
import { SCORE_V6_CANDIDATES } from "../research/score_v6/cohort";
import { recommendV10CandidateActions } from "../research/svi_score_v10_amss/actions";
import { calculateV10EconomicLoss } from "../research/svi_score_v10_amss/economic-loss";
import {
  V10_AMSS_CHANNELS,
  V10_AMSS_CONTRACT,
  V10_AMSS_DECISION_SCENARIOS,
  V10_AMSS_EVIDENCE_GROUPS,
  V10_AMSS_MODULE_ORDERS,
} from "../research/svi_score_v10_amss/contract";

test("V10 freezes a 100-business, 4800-fit external AMSS audit", () => {
  assert.equal(V10_AMSS_CONTRACT.cohort.businesses, 100);
  assert.equal(V10_AMSS_CONTRACT.cohort.candidateFits, 4_800);
  assert.equal(SCORE_V6_CANDIDATES.length, 24);
  assert.equal(V10_AMSS_CHANNELS.length, 3);
  assert.equal(V10_AMSS_EVIDENCE_GROUPS.length, 4);
  assert.equal(V10_AMSS_MODULE_ORDERS.length, 6);
  assert.equal(V10_AMSS_CONTRACT.selection.retrainingPermitted, false);
  assert.equal(V10_AMSS_CONTRACT.inference.nutsPermitted, false);
  assert.equal(V10_AMSS_CONTRACT.governance.openTruthExactlyOnce, true);
  assert.match(V10_AMSS_CONTRACT.decisions.oracle, /48 truth-blind candidate actions/);
});

test("V10 posterior actions obey the four frozen budget regimes", () => {
  const observed = Object.fromEntries(V10_AMSS_CHANNELS.map((channel, channelIndex) => [
    channel,
    Array.from({ length: 156 }, (_, index) => 80 + 10 * channelIndex + index % 13),
  ]));
  const baseline = {
    meta_acquisition_spend: 5_000,
    google_search_nonbrand_spend: 4_000,
    ctv_spend: 6_000,
  };
  const response = {
    adstockType: "geometric" as const,
    adstock: 0.4,
    weibullShape: 1.5,
    weibullScale: 3,
    saturation: 1.2,
    halfSaturationQuantile: 0.5,
    kernelNormalization: "sum" as const,
  };
  const draws: SamplingDecisionDraw[] = Array.from({ length: 4 }, (_, drawIndex) => ({
    kernelBandwidth: 0.2,
    channels: V10_AMSS_CHANNELS.map((channel, channelIndex) => ({
      channel,
      roi: 1 + channelIndex,
      contribution: (1 + channelIndex) * 20_000 * (1 + drawIndex * 0.02),
      response,
    })),
  }));
  const actions = recommendV10CandidateActions({
    grossMargin: 0.5,
    baselineAnnualSpend: baseline,
    observedSpend: observed,
  }, draws);
  assert.deepEqual(actions.map((action) => action.scenario), V10_AMSS_DECISION_SCENARIOS.map((item) => item.id));
  const baselineTotal = Object.values(baseline).reduce((total, value) => total + value, 0);
  const roundingTolerance = baselineTotal * 0.005 * 3;
  actions.forEach((action, index) => {
    const scenario = V10_AMSS_DECISION_SCENARIOS[index];
    assert.ok(Number.isFinite(action.predictedIncrementalProfit));
    assert.ok(action.totalAnnualSpend >= baselineTotal * scenario.minimumTotalShare - roundingTolerance);
    assert.ok(action.totalAnnualSpend <= baselineTotal * scenario.maximumTotalShare + roundingTolerance);
    V10_AMSS_CHANNELS.forEach((channel) => {
      assert.ok(action.annualSpend[channel] >= 0);
      assert.ok(action.annualSpend[channel] <= 2.5 * baseline[channel] + roundingTolerance);
    });
  });
});

test("V10 cloud truth execution pins AMSS and assigns one business per task", async () => {
  const [dockerfile, task, cloudRun] = await Promise.all([
    readFile(resolve("research/svi_score_v10_amss/cloud/Dockerfile.truth"), "utf8"),
    readFile(resolve("research/svi_score_v10_amss/cloud/truth_task.py"), "utf8"),
    readFile(resolve("research/svi_score_v10_amss/cloud/truth-cloud-run.yaml"), "utf8"),
  ]);
  assert.match(dockerfile, /rocker\/r-ver:4\.6\.1/);
  assert.match(dockerfile, /a440105d5089fc2142aca9b139492507475166dc950354069f771110e1d95505/);
  assert.match(task, /TASKS = 100/);
  assert.match(task, /if_generation_match=0/);
  assert.match(cloudRun, /taskCount: 100/);
  assert.match(cloudRun, /parallelism: 100/);
  assert.match(cloudRun, /maxRetries: 0/);
});

test("V10 caps the primary normalized loss and retains uncapped safety loss", () => {
  const loss = calculateV10EconomicLoss({
    oracleProfit: 1_100,
    realizedProfit: 700,
    baselineProfit: 1_000,
    baselineBudget: 1_000,
    normalizedLossCap: V10_AMSS_CONTRACT.decisions.normalizedLossCap,
  });
  assert.equal(loss.economicScale, 100);
  assert.equal(loss.uncapped, 4);
  assert.equal(loss.capped, 1);
});
