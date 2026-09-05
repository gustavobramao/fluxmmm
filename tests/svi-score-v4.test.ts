import assert from "node:assert/strict";
import test from "node:test";
import type { ScoreV6CandidateRow } from "../research/score_v6/types";
import type { SviScoreV4TrainingConfiguration } from "../research/svi_score_v4/contract";
import { crossValidateScoreV4 } from "../research/svi_score_v4/cross-validation";
import {
  evaluateScoreV4,
  evaluateTopOneRegretBound,
  latentV4Score,
  learnScoreV4,
  selectScoreV4Candidates,
} from "../research/svi_score_v4/ranker";
import { kernelReceipt } from "../research/svi_score_v4/temporal";
import type {
  SviScoreV4CandidateRow,
  SviScoreV4Model,
} from "../research/svi_score_v4/types";

const CONFIGURATION: SviScoreV4TrainingConfiguration = {
  id: "test",
  regularization: 0.01,
  epochs: 80,
  cvarWeight: 0.5,
  familyDroWeight: 0.5,
  safety: {
    roiPlausibility: 40,
    decisionStability: 40,
  },
};

function row(
  businessId: string,
  family: string,
  candidateId: string,
  loss: number,
  diagnostic: number,
  temporalIdentification = 70,
): SviScoreV4CandidateRow {
  return {
    businessId,
    family,
    split: "train",
    candidateId,
    evidenceArm: "experiments-only",
    modelFamily: "bayesian",
    eligible: true,
    reviewEligible: true,
    eligibilityTier: "decision-grade",
    failedGateCount: 0,
    layerScores: {
      generalization: diagnostic,
      structure: diagnostic,
      causal: diagnostic,
      decision: diagnostic,
    },
    diagnostics: {
      "rolling-oos": diagnostic,
      "roi-posterior-plausibility": diagnostic,
      "roi-decision-stability": diagnostic,
    },
    heuristicScore: diagnostic,
    economicScale: 1,
    decisionLoss: loss,
    cappedRegret: Math.min(loss, 1),
    roiError: loss,
    contributionError: loss,
    temporal: {
      context: family === "delayed" ? 0.9 : 0.1,
      wholeFlightGeneralization: diagnostic,
      carryoverSupport: diagnostic,
      postFlightResidualStability: diagnostic,
      kernelDistinguishability: diagnostic,
      temporalIdentification,
      maximumEffectiveLag: 4,
      maximumLag95: 10,
      materialFlightCount: 5,
      detail: "test",
    },
  } satisfies ScoreV6CandidateRow & SviScoreV4CandidateRow;
}

function fixedModel(rows: SviScoreV4CandidateRow[]): SviScoreV4Model {
  const learned = learnScoreV4(rows, { ...CONFIGURATION, epochs: 1 });
  return {
    ...learned,
    ordinaryWeights: { "rolling-oos": 1 },
    temporalWeights: { "rolling-oos": 1 },
  };
}

test("the top-one theorem bounds the regret of the selected candidate", () => {
  const rows = [
    row("b1", "balanced", "oracle", 0, 60),
    row("b1", "balanced", "selected", 0.7, 90),
    row("b1", "balanced", "other", 0.3, 50),
  ];
  const model = fixedModel(rows);
  assert.equal(selectScoreV4Candidates(rows, model)[0].candidateId, "selected");
  const receipt = evaluateTopOneRegretBound(rows, model);
  assert.equal(receipt.violations, 0);
  assert.ok(receipt.meanTopOneHingeBound >= receipt.meanSelectedNormalizedRegret);
  assert.ok(receipt.meanSmoothTopOneBound >= receipt.meanTopOneHingeBound);
});

test("temporal identification remains diagnostic rather than an eligibility gate", () => {
  const rows = [
    row("b1", "delayed", "low-temporal-high-score", 0.1, 95, 20),
    row("b1", "delayed", "high-temporal", 0.2, 65, 70),
  ];
  const selection = selectScoreV4Candidates(rows, fixedModel(rows))[0];
  assert.equal(selection.candidateId, "low-temporal-high-score");
  assert.equal(selection.safetyAccepted, true);
  assert.equal(selection.reviewOnlyFallback, false);
});

test("a high compensatory score cannot override an ROI safety failure", () => {
  const unsafe = row("b1", "delayed", "unsafe-high-score", 0.1, 95, 70);
  unsafe.diagnostics["roi-posterior-plausibility"] = 20;
  const rows = [
    unsafe,
    row("b1", "delayed", "safe", 0.2, 65, 20),
  ];
  const selection = selectScoreV4Candidates(rows, fixedModel(rows))[0];
  assert.equal(selection.candidateId, "safe");
  assert.equal(selection.safetyAccepted, true);
  assert.equal(selection.reviewOnlyFallback, false);
});

test("V4 learns two monotonic experts and reports tail and family risk", () => {
  const rows = [
    row("b1", "balanced", "good", 0.1, 90),
    row("b1", "balanced", "bad", 0.8, 45),
    row("b2", "delayed", "good", 0.2, 88),
    row("b2", "delayed", "bad", 1.2, 48),
  ];
  const model = learnScoreV4(rows, CONFIGURATION);
  const ordinaryTotal = Object.values(model.ordinaryWeights).reduce(
    (sum, value) => sum + (value ?? 0),
    0,
  );
  const temporalTotal = Object.values(model.temporalWeights).reduce(
    (sum, value) => sum + (value ?? 0),
    0,
  );
  assert.ok(Math.abs(ordinaryTotal - 1) < 1e-9);
  assert.ok(Math.abs(temporalTotal - 1) < 1e-9);
  assert.ok(rows.every((candidate) => Number.isFinite(latentV4Score(candidate, model))));
  const metrics = evaluateScoreV4(rows, model);
  assert.equal(metrics.businesses, 2);
  assert.ok(metrics.cvar90Loss >= metrics.meanLoss);
  assert.ok(metrics.cvar90ExcessLoss >= metrics.meanExcessLoss);
  assert.ok(metrics.worstFamilyMeanExcessLoss >= 0);
});

test("kernel receipts expose effective and 95% carryover", () => {
  const short = kernelReceipt({
    adstockType: "geometric",
    adstock: 0.1,
    weibullShape: 2,
    weibullScale: 4,
    saturation: 1.2,
    halfSaturationQuantile: 0.5,
    kernelNormalization: "sum",
  });
  const long = kernelReceipt({
    adstockType: "geometric",
    adstock: 0.8,
    weibullShape: 2,
    weibullScale: 4,
    saturation: 1.2,
    halfSaturationQuantile: 0.5,
    kernelNormalization: "sum",
  });
  assert.ok(long.effectiveLag > short.effectiveLag);
  assert.ok(long.lag95 > short.lag95);
});

test("grouped V4 cross-validation never admits audit rows", () => {
  const rows = Array.from({ length: 10 }, (_, index) => {
    const business = `b${index + 1}`;
    const family = index % 2 ? "delayed" : "balanced";
    return [
      row(business, family, "good", 0.1, 90),
      row(business, family, "middle", 0.4, 70),
      row(business, family, "bad", 0.9, 45),
    ];
  }).flat();
  const receipt = crossValidateScoreV4(
    rows,
    [{ ...CONFIGURATION, epochs: 10 }],
    2,
  );
  assert.equal(receipt.data.businesses, 10);
  assert.equal(receipt.data.rows, 30);
  assert.equal(receipt.configurations[0].folds.length, 2);
  assert.equal(receipt.finalFit.theorem.violations, 0);
  assert.throws(
    () => crossValidateScoreV4(
      [{ ...rows[0], split: "audit" }],
      [{ ...CONFIGURATION, epochs: 1 }],
      2,
    ),
    /sealed V3 audit/,
  );
});
