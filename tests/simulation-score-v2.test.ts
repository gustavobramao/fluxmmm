import assert from "node:assert/strict";
import test from "node:test";
import { parseCsv } from "../lib/mmm/csv";
import { DEFAULT_CONFIG, runModel } from "../lib/mmm/models";
import { createDataset, validateDataset } from "../lib/mmm/schema";
import { evaluateCandidateTruth } from "../research/score_v2/evaluate";
import { roiEvidence } from "../research/score_v2/evidence";
import { PILOT_SCENARIOS } from "../research/score_v2/scenarios";
import {
  generateSyntheticBusiness,
  trueContributionForAllocation,
  verifyInjectedTruth,
} from "../research/score_v2/simulator";

for (const scenario of PILOT_SCENARIOS) {
  test(`${scenario.id} injects exact ROI and produces model-ready observed data`, async () => {
    const first = generateSyntheticBusiness(scenario);
    const second = generateSyntheticBusiness(scenario);
    verifyInjectedTruth(first);

    assert.equal(first.observedCsv, second.observedCsv);
    assert.deepEqual(first.truth, second.truth);
    assert.equal(first.observedCsv.includes("latentDemand"), false);
    assert.equal(first.observedCsv.includes("planningIntensity"), false);
    first.truth.channels.forEach((channel) => {
      assert.ok(Math.abs(channel.realizedRoi - channel.targetRoi) < 1e-10);
      assert.ok(Number.isFinite(channel.marginalRoiAtObserved));
      assert.equal(channel.deliveryUnits.length, scenario.periods);
      const declaredOverride = scenario.channels.find(
        (configured) => configured.channel === channel.channel,
      )?.targetRoi;
      if (declaredOverride === undefined) {
        const evidence = roiEvidence(channel.evidenceId);
        assert.ok(channel.targetRoi >= evidence.low);
        assert.ok(channel.targetRoi <= evidence.high);
      }
    });
    assert.ok(first.truth.experiments.length > 0);
    first.truth.experiments.forEach((experiment) => {
      assert.notEqual(experiment.observedRoi, experiment.trueRoi);
    });

    const parsed = parseCsv(first.observedCsv);
    const dataset = await createDataset(
      `${scenario.id}.csv`,
      first.observedCsv,
      parsed.columns,
      parsed.rows,
    );
    assert.notEqual(validateDataset(dataset).status, "blocked");
  });
}

test("the hidden profit oracle conserves its chosen spend and can choose zero", () => {
  const business = generateSyntheticBusiness(PILOT_SCENARIOS[1]);
  const optimizedBudget = Object.values(
    business.truth.allocation.optimalAdditionalBudget,
  ).reduce((total, value) => total + value, 0);

  assert.ok(
    Math.abs(optimizedBudget - business.truth.allocation.optimalSpend) < 1e-6,
  );
  assert.equal(business.truth.decisionScenarios.length, 4);
  business.truth.decisionScenarios.forEach((decision) => {
    const allocated = Object.values(decision.optimalAdditionalBudget).reduce(
      (total, value) => total + value,
      0,
    );
    assert.ok(Math.abs(allocated - decision.optimalSpend) < 1e-5);
    assert.ok(decision.evaluations > 1);
  });
  assert.ok(business.truth.allocation.optimalSpend >= 0);
  assert.ok(business.truth.allocation.optimalIncrementalProfit >= -1e-6);
  assert.ok(
    business.truth.allocation.optimalSpend <=
      business.truth.allocation.maximumAdditionalBudget +
        1e-6,
  );
  assert.ok(
    business.truth.allocation.optimalContribution >=
      trueContributionForAllocation(
        business,
        Object.fromEntries(
          business.truth.channels.map((channel) => [channel.channel, 0]),
        ),
      ) -
        1e-6,
  );
});

test("a V1 model can fit observed data without receiving the hidden answer key", async () => {
  const business = generateSyntheticBusiness(PILOT_SCENARIOS[1]);
  const parsed = parseCsv(business.observedCsv);
  const dataset = await createDataset(
    "score-v2-smoke.csv",
    business.observedCsv,
    parsed.columns,
    parsed.rows,
  );
  const model = await runModel(
    dataset,
    DEFAULT_CONFIG,
    [],
    "frequentist",
    "score-v2-smoke",
  );
  const evaluation = evaluateCandidateTruth(
    business,
    model,
    DEFAULT_CONFIG,
  );

  assert.equal(model.actual.length, business.observedRows.length);
  assert.ok(model.predicted.every(Number.isFinite));
  assert.ok(Number.isFinite(evaluation.weightedLogRoiError));
  assert.ok(Number.isFinite(evaluation.weightedLogBenchmarkAgreement));
  assert.ok(Number.isFinite(evaluation.contributionError));
  assert.ok(Number.isFinite(evaluation.budgetRegret));
  assert.ok(Number.isFinite(evaluation.profitRegret));
  assert.ok(Number.isFinite(evaluation.revenueRegret));
  assert.ok(evaluation.budgetRegret >= 0);
  assert.deepEqual(
    Object.keys(evaluation.scenarioProfitRegret).sort(),
    [
      "budget-growth",
      "budget-reduction",
      "economic-ceiling",
      "fixed-budget-mix",
    ],
  );
  Object.values(evaluation.scenarioProfitRegret).forEach((regret) => {
    assert.ok(regret >= 0 && regret <= 1);
  });
});
