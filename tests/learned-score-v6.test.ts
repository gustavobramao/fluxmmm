import assert from "node:assert/strict";
import test from "node:test";
import { inferIndustryPrior } from "../lib/mmm/benchmarks";
import {
  assessExternalAnchorEligibility,
  runModelValidation,
  validationFingerprint,
} from "../lib/mmm/validation";
import { evaluateCandidateTruth } from "../research/score_v2/evaluate";
import { PILOT_SCENARIOS } from "../research/score_v2/scenarios";
import { generateSyntheticBusiness } from "../research/score_v2/simulator";
import { syntheticIndustryPriorOverrides } from "../research/score_v6/evidence";
import { scoreV6 } from "../research/score_v6/learn";
import {
  SCORE_V6_DIAGNOSTIC_GROUPS,
  type ScoreV6CandidateRow,
  type ScoreV6FeatureValues,
  type ScoreV6FeatureWeights,
} from "../research/score_v6/types";
import { DEFAULT_CONFIG, runModel } from "../lib/mmm/models";
import { DEFAULT_ADVANCED_CONFIG } from "../lib/mmm/advanced";
import { parseCsv } from "../lib/mmm/csv";
import { createDataset } from "../lib/mmm/schema";

test("scenario evidence overrides are model-visible but remain separate from truth", () => {
  const business = generateSyntheticBusiness(PILOT_SCENARIOS[5]);
  const overrides = syntheticIndustryPriorOverrides(business);
  for (const channel of business.truth.channels) {
    const benchmark = inferIndustryPrior(channel.spendColumn, overrides);
    assert.equal(
      benchmark.median,
      business.scenario.industryBenchmarks[channel.channel],
    );
    assert.notEqual(benchmark.median, channel.targetRoi);
  }
});

test("replicated non-overlapping synthetic experiments qualify for honest anchor holdout", () => {
  const scenario = structuredClone(PILOT_SCENARIOS[0]);
  scenario.channels.forEach((channel) => {
    if (channel.experiment.design !== "none") channel.experiment.replicates = 2;
  });
  const business = generateSyntheticBusiness(scenario);
  const assessment = assessExternalAnchorEligibility(
    business.truth.experiments.map((study) => study.experiment),
    true,
  );
  assert.equal(assessment.status, "qualified");
  assert.ok(assessment.qualifiedCount >= 2);
});

test("anchor recovery compares held-out experiments with the same fitted ROI window", async () => {
  const scenario = structuredClone(PILOT_SCENARIOS[0]);
  scenario.channels.forEach((channel) => {
    if (channel.experiment.design !== "none") channel.experiment.replicates = 2;
  });
  const business = generateSyntheticBusiness(scenario);
  const experiments = business.truth.experiments.map((study) => study.experiment);
  const parsed = parseCsv(business.observedCsv);
  const dataset = await createDataset(
    "window-anchor.csv",
    business.observedCsv,
    parsed.columns,
    parsed.rows,
  );
  const model = await runModel(
    dataset,
    DEFAULT_CONFIG,
    experiments,
    "bayesian",
    "window-anchor",
  );
  const options = {
    anchorIndependenceConfirmed: true,
    industryPriorChannels: [],
    industryBenchmarkScreeningEnabled: true,
  };
  const fingerprint = await validationFingerprint(
    dataset,
    model,
    DEFAULT_CONFIG,
    DEFAULT_ADVANCED_CONFIG,
    experiments,
    options,
  );
  const validation = await runModelValidation(
    dataset,
    model,
    DEFAULT_CONFIG,
    DEFAULT_ADVANCED_CONFIG,
    experiments,
    fingerprint,
    undefined,
    options,
  );
  const anchors = validation.layers.causal.evidence.kind === "causal"
    ? validation.layers.causal.evidence.anchors
    : [];
  assert.ok(anchors.length >= 2);
  assert.ok(
    anchors.every((anchor) => anchor.comparisonBasis === "experiment-window"),
  );
  assert.ok(
    anchors.every((anchor) => anchor.comparisonLabel?.includes("outcomes through")),
  );
});

test("V6 economic loss remains uncapped when a candidate is catastrophically wrong", async () => {
  const business = generateSyntheticBusiness(PILOT_SCENARIOS[1]);
  const parsed = parseCsv(business.observedCsv);
  const dataset = await createDataset(
    "v6-loss.csv",
    business.observedCsv,
    parsed.columns,
    parsed.rows,
  );
  const model = await runModel(dataset, DEFAULT_CONFIG, [], "frequentist", "v6-loss");
  const evaluation = evaluateCandidateTruth(business, model, DEFAULT_CONFIG);
  assert.ok(Number.isFinite(evaluation.decisionLoss));
  assert.ok(Object.values(evaluation.scenarioDecisionLoss).every((loss) => loss >= 0));
});

test("V6 monotonic ranker never punishes an improved diagnostic", () => {
  const features = Object.values(SCORE_V6_DIAGNOSTIC_GROUPS).flat();
  const weights = Object.fromEntries(
    features.map((feature) => [feature, 1 / features.length]),
  ) as ScoreV6FeatureWeights;
  const diagnostics = Object.fromEntries(
    features.map((feature) => [feature, 70]),
  ) as ScoreV6FeatureValues;
  const base = {
    businessId: "b",
    family: "f",
    split: "train",
    candidateId: "c",
    evidenceArm: "experiments-only",
    modelFamily: "bayesian",
    eligible: true,
    reviewEligible: true,
    eligibilityTier: "decision-grade",
    failedGateCount: 0,
    layerScores: { generalization: 70, structure: 70, causal: 70, decision: 70 },
    diagnostics,
    heuristicScore: 70,
    decisionLoss: 0.2,
    cappedRegret: 0.2,
    roiError: 0.2,
    contributionError: 0.2,
  } satisfies ScoreV6CandidateRow;
  for (const feature of features) {
    const improved: ScoreV6CandidateRow = {
      ...base,
      diagnostics: {
        ...base.diagnostics,
        [feature]: 90,
      } as ScoreV6FeatureValues,
    };
    assert.ok(scoreV6(improved, weights) >= scoreV6(base, weights));
  }
});
