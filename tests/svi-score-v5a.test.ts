import assert from "node:assert/strict";
import test from "node:test";
import type { SviScoreV4CandidateRow } from "../research/svi_score_v4/types";
import { crossValidateScoreV5A } from "../research/svi_score_v5a/cross-validation";
import { v5aFeatureNames, v5aFeatureVector } from "../research/svi_score_v5a/features";
import {
  evaluateV5ATopOneRegretBound,
  latentV5AScore,
  selectScoreV5ACandidates,
} from "../research/svi_score_v5a/ranker";
import type { SviScoreV5AModel } from "../research/svi_score_v5a/types";

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
    evidenceArm: candidateId.includes("benchmark-gap-fill")
      ? "benchmark-gap-fill"
      : "experiments-only",
    modelFamily: candidateId.startsWith("V6-C02") ? "advanced" : "bayesian",
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
      context: family === "delayed-tv" ? 0.9 : 0.1,
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
  };
}

function uniformModel(sample: SviScoreV4CandidateRow): SviScoreV5AModel {
  const featureNames = v5aFeatureNames(sample);
  return {
    featureNames,
    weights: featureNames.map(() => 1 / featureNames.length),
    configuration: {
      id: "test",
      regularization: 0.02,
      epochs: 4,
      cvarWeight: 0.4,
      familyDroWeight: 0.85,
      featureCap: 0.2,
      learningRate: 0.1,
    },
  };
}

test("V5A features cannot read generator identity or hidden loss", () => {
  const first = row(
    "business-a",
    "balanced-dtc",
    "V6-C01 · experiments-only",
    0.1,
    70,
  );
  const second = {
    ...first,
    businessId: "different-business",
    family: "delayed-tv",
    split: "validation" as const,
    decisionLoss: 99,
    roiError: 77,
    contributionError: 55,
  };
  assert.deepEqual(v5aFeatureNames(first), v5aFeatureNames(second));
  assert.deepEqual(v5aFeatureVector(first), v5aFeatureVector(second));
});

test("nonnegative V5A weights preserve diagnostic monotonicity", () => {
  const lower = row(
    "b1",
    "balanced-dtc",
    "V6-C01 · experiments-only",
    0.2,
    45,
  );
  const higher = row(
    "b1",
    "balanced-dtc",
    "V6-C01 · experiments-only",
    0.2,
    85,
  );
  const model = uniformModel(lower);
  assert.ok(latentV5AScore(higher, model) >= latentV5AScore(lower, model));
});

test("temporal identification is not a V5A eligibility gate", () => {
  const rows = [
    row("b1", "delayed-tv", "V6-C01 · experiments-only", 0.1, 95, 10),
    row("b1", "delayed-tv", "V6-C02 · experiments-only", 0.2, 60, 90),
  ];
  const selection = selectScoreV5ACandidates(rows, uniformModel(rows[0]))[0];
  assert.equal(selection.candidateId, "V6-C01 · experiments-only");
  assert.equal(selection.safetyAccepted, true);
});

test("ROI plausibility remains a non-compensatory V5A gate", () => {
  const unsafe = row(
    "b1",
    "balanced-dtc",
    "V6-C01 · experiments-only",
    0.1,
    95,
  );
  unsafe.diagnostics["roi-posterior-plausibility"] = 20;
  const safe = row(
    "b1",
    "balanced-dtc",
    "V6-C02 · experiments-only",
    0.2,
    60,
  );
  const selection = selectScoreV5ACandidates(
    [unsafe, safe],
    uniformModel(unsafe),
  )[0];
  assert.equal(selection.candidateId, safe.candidateId);
});

test("grouped V5A cross-validation is deterministic and keeps audit sealed", () => {
  const rows = Array.from({ length: 10 }, (_, index) => {
    const business = `b${index + 1}`;
    const family = index % 2 ? "delayed-tv" : "balanced-dtc";
    return [
      row(business, family, "V6-C01 · experiments-only", 0.1, 90),
      row(business, family, "V6-C02 · experiments-only", 0.4, 70),
      row(business, family, "V6-C03 · benchmark-gap-fill", 0.8, 55),
      row(business, family, "V6-C04 · benchmark-gap-fill", 1.1, 45),
    ];
  }).flat();
  const configuration = [{
    id: "test",
    regularization: 0.02,
    epochs: 6,
    cvarWeight: 0.4,
    familyDroWeight: 0.85,
    featureCap: 0.2,
    learningRate: 0.1,
  }];
  const first = crossValidateScoreV5A(rows, configuration, 2);
  const second = crossValidateScoreV5A(rows, configuration, 2);
  assert.deepEqual(first, second);
  assert.equal(first.data.businesses, 10);
  assert.equal(first.data.candidatesPerBusiness, 4);
  assert.equal(first.finalFit.theorem.violations, 0);
  assert.equal(evaluateV5ATopOneRegretBound(
    rows,
    first.finalFit.model,
  ).violations, 0);
  assert.throws(
    () => crossValidateScoreV5A(
      [{ ...rows[0], split: "audit" }],
      configuration,
      2,
    ),
    /sealed V3 audit/,
  );
});
