import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const root = "research/svi_score_v11_amss_confirmatory/artifacts";
const result = JSON.parse(readFileSync(
  `${root}/secondary-baselines.json`,
  "utf8",
));
const frozen = JSON.parse(readFileSync(
  `${root}/frozen-selections.json`,
  "utf8",
));

test("V11 AMSS secondary baselines preserve the accepted posterior actions", () => {
  assert.equal(result.artifactId, "flux-svi-score-v11-amss-secondary-baselines-v1");
  assert.equal(result.status, "complete-post-truth-secondary-analysis-not-confirmatory");
  assert.equal(result.governance.confirmatory, false);
  assert.equal(result.governance.newPosteriorFits, 0);
  assert.equal(result.governance.cloudComputeUsed, false);
  assert.equal(result.governance.selectorTrainingPerformed, false);
  assert.equal(result.governance.candidateActionsChanged, false);
  assert.equal(result.cohort.businesses, 100);
  assert.equal(result.cohort.candidates, 4_800);
  assert.equal(result.cohort.predictionOnlyTies, 0);
  assert.ok(result.cohort.maximumScreeningReconstructionRelativeRmse < 1e-10);
});

test("V11 and prediction-only selections reproduce the pre-truth freeze", () => {
  const frozenByBusiness = new Map(frozen.selections.map((row: {
    businessId: string;
    v11CandidateId: string;
    predictionOnlyCandidateId: string;
  }) => [row.businessId, row]));
  assert.equal(result.selections.length, 100);
  for (const row of result.selections) {
    const expected = frozenByBusiness.get(row.businessId) as {
      v11CandidateId: string;
      predictionOnlyCandidateId: string;
    } | undefined;
    assert.ok(expected);
    assert.equal(row.regretSetCandidateId, expected.v11CandidateId);
    assert.equal(row.predictionOnlyCandidateId, expected.predictionOnlyCandidateId);
  }
});

test("Classic Pareto covers all businesses under the declared objective contract", () => {
  assert.equal(new Set(result.selections.map(
    (row: { businessId: string }) => row.businessId,
  )).size, 100);
  assert.equal(result.selections.filter(
    (row: { conventionalParetoActiveObjectives: number }) =>
      row.conventionalParetoActiveObjectives === 3,
  ).length, 75);
  assert.equal(result.selections.filter(
    (row: { conventionalParetoActiveObjectives: number }) =>
      row.conventionalParetoActiveObjectives === 2,
  ).length, 25);
  assert.ok(result.selections.every(
    (row: { conventionalParetoFrontSize: number }) =>
      row.conventionalParetoFrontSize >= 1 && row.conventionalParetoFrontSize <= 48,
  ));
});

test("secondary endpoint and paired bootstrap arithmetic is internally consistent", () => {
  Object.values(result.endpoints).forEach((value) => {
    const row = value as { mean: number; p90: number; selectionObjective: number };
    assert.ok(Number.isFinite(row.mean));
    assert.ok(Number.isFinite(row.p90));
    assert.ok(Number.isFinite(row.selectionObjective));
    assert.ok(Math.abs(row.selectionObjective - (0.65 * row.mean + 0.35 * row.p90)) < 1e-12);
  });
  Object.values(result.pairedComparisons).forEach((comparison) => {
    const row = comparison as {
      interval95: [number, number];
      selectionObjectiveInterval95: [number, number];
      selectedMethodLowerRisk: number;
      tied: number;
      comparatorLowerRisk: number;
      resamples: number;
    };
    assert.equal(row.selectedMethodLowerRisk + row.tied + row.comparatorLowerRisk, 100);
    assert.equal(row.resamples, 10_000);
    assert.ok(row.interval95[0] <= row.interval95[1]);
    assert.ok(row.selectionObjectiveInterval95[0] <= row.selectionObjectiveInterval95[1]);
  });
});
