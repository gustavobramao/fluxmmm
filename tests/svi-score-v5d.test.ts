import assert from "node:assert/strict";
import test from "node:test";
import type { SviScoreV4CandidateRow } from "../research/svi_score_v4/types";
import { V5D_TRAINING_CONFIGURATION } from "../research/svi_score_v5d/contract";
import { evaluateV5D, selectV5DOrderIndependent } from "../research/svi_score_v5d/evaluate";
import { v5dFeatureNames, v5dFeatureVector } from "../research/svi_score_v5d/features";
import { learnV5DModel, predictV5DLoss } from "../research/svi_score_v5d/model";
import type { V5DLossPrediction } from "../research/svi_score_v5d/types";

function row(
  businessId: string,
  family: string,
  candidateId: string,
  loss: number,
  diagnostic = 70,
): SviScoreV4CandidateRow {
  return {
    businessId,
    family,
    split: "train",
    candidateId,
    evidenceArm: candidateId.includes("benchmark-gap-fill")
      ? "benchmark-gap-fill"
      : "experiments-only",
    modelFamily: Number(candidateId.match(/C(\d+)/)?.[1] ?? 1) % 2
      ? "bayesian"
      : "advanced",
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
      temporalIdentification: diagnostic,
      maximumEffectiveLag: 4,
      maximumLag95: 10,
      materialFlightCount: 5,
      detail: "test",
    },
  };
}

test("V5D features cannot read hidden loss, truth, business, family, or split", () => {
  const first = row("b1", "balanced-dtc", "V6-C01 · experiments-only", 0.1);
  const changed = {
    ...first,
    businessId: "different",
    family: "delayed-tv",
    split: "validation" as const,
    decisionLoss: 999,
    roiError: 888,
    contributionError: 777,
  };
  assert.deepEqual(v5dFeatureNames(first), v5dFeatureNames(changed));
  assert.deepEqual(v5dFeatureVector(first), v5dFeatureVector(changed));
});

test("V5D learns finite uncapped mean and P90 loss predictions", () => {
  const ids = Array.from({ length: 8 }, (_, index) => {
    const id = `V6-C${String(index + 1).padStart(2, "0")}`;
    return [`${id} · experiments-only`, `${id} · benchmark-gap-fill`];
  }).flat();
  const rows = Array.from({ length: 10 }, (_, businessIndex) =>
    ids.map((candidateId, candidateIndex) => row(
      `b${businessIndex}`,
      businessIndex % 2 ? "delayed-tv" : "balanced-dtc",
      candidateId,
      candidateIndex === ids.length - 1 ? 12 : candidateIndex / 5,
      85 - candidateIndex,
    ))
  ).flat();
  const model = learnV5DModel(rows, {
    ...V5D_TRAINING_CONFIGURATION,
    epochs: 4,
  });
  const prediction = predictV5DLoss(rows[0], model);
  assert.ok(Number.isFinite(prediction.mean));
  assert.ok(Number.isFinite(prediction.p90));
  assert.ok(prediction.p90 >= prediction.mean);
  assert.ok(model.receipt.p90Target > 1, "the target must remain uncapped");
  assert.ok(model.receipt.hardNegativeUpdates > 0);
});

test("V5D final selection is invariant to candidate arrival order", () => {
  const rows = [
    row("b", "balanced-dtc", "V6-C01 · experiments-only", 2),
    row("b", "balanced-dtc", "V6-C02 · experiments-only", 1),
    row("b", "balanced-dtc", "V6-C03 · experiments-only", 3),
  ];
  const predictions = new Map<string, V5DLossPrediction>([
    [rows[0].candidateId, { mean: 2, p90: 3, tailWidth: 1, risk: 2.35 }],
    [rows[1].candidateId, { mean: 1, p90: 1.5, tailWidth: 0.5, risk: 1.175 }],
    [rows[2].candidateId, { mean: 3, p90: 4, tailWidth: 1, risk: 3.35 }],
  ]);
  const forward = selectV5DOrderIndependent(rows, predictions).row.candidateId;
  const reverse = selectV5DOrderIndependent([...rows].reverse(), predictions).row.candidateId;
  assert.equal(forward, rows[1].candidateId);
  assert.equal(reverse, forward);
});

test("V5D rejects sealed audit rows before fitting", () => {
  const audit = {
    ...row("audit", "balanced-dtc", "V6-C01 · experiments-only", 1),
    split: "audit" as const,
  };
  assert.throws(
    () => evaluateV5D(
      [audit],
      [{ businessId: audit.businessId, family: audit.family, fold: 0 }],
    ),
    /sealed audit/,
  );
});
