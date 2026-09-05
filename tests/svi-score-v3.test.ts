import assert from "node:assert/strict";
import test from "node:test";
import type { ScoreV6CandidateRow } from "../research/score_v6/types";
import { selectSviGoldAudit } from "../research/svi_score_v3/audit-selection";
import { SVI_SCORE_V3_CONTRACT } from "../research/svi_score_v3/contract";
import { recoverCompleteJsonObjects } from "../research/svi_score_v3/checkpoint";
import {
  crossValidateScoreV3,
  groupedBusinessFolds,
} from "../research/svi_score_v3/cross-validation";
import { verifyTrainingFreeze } from "../research/svi_score_v3/verify-training-freeze";
import { verifyValidationProtocol } from "../research/svi_score_v3/verify-validation-protocol";
import { sviValidationRows } from "../research/svi_score_v3/validation-rows";
import { verifyValidationResult } from "../research/svi_score_v3/verify-validation-result";
import type { SviScoreV3Record } from "../research/svi_score_v3/types";
import {
  evaluateSviScoreV3Validation,
  SVI_SCORE_V3_VALIDATION_PROTOCOL,
} from "../research/svi_score_v3/validation-protocol";
import {
  evaluateRegretBound,
  regretWeightedOraclePairs,
} from "../research/score_v6/learn";
import { sviDesignBusinesses } from "../research/svi_score_v3/design";
import {
  SCORE_V6_DIAGNOSTIC_GROUPS,
  type ScoreV6FeatureValues,
} from "../research/score_v6/types";

function rows(): ScoreV6CandidateRow[] {
  return sviDesignBusinesses().flatMap(({ family, scenario }) =>
    Array.from({ length: 48 }, (_, index) => ({
      businessId: scenario.id,
      family: family.id,
      split: family.split,
      candidateId: `candidate-${index}`,
      evidenceArm:
        index % 2 === 0 ? "experiments-only" : "benchmark-gap-fill",
      modelFamily: index % 4 < 2 ? "bayesian" : "advanced",
      eligible: index % 3 !== 0,
      reviewEligible: true,
      eligibilityTier: index % 3 !== 0 ? "decision-grade" : "review",
      failedGateCount: index % 3 === 0 ? 1 : 0,
      layerScores: {
        generalization: 40 + index,
        structure: 50,
        causal: 60,
        decision: 70,
      },
      diagnostics: {} as ScoreV6CandidateRow["diagnostics"],
      heuristicScore: index,
      decisionLoss: 10_000 - index,
      cappedRegret: index / 100,
      roiError: index / 50,
      contributionError: index / 40,
    } satisfies ScoreV6CandidateRow)),
  );
}

function groupedCvRows(): ScoreV6CandidateRow[] {
  const features = Object.values(SCORE_V6_DIAGNOSTIC_GROUPS).flat();
  return ["family-a", "family-b", "family-c"].flatMap((family, familyIndex) =>
    Array.from({ length: 10 }, (_, businessIndex) =>
      Array.from({ length: 3 }, (_, candidateIndex) => {
        const quality = 45 + candidateIndex * 20 + ((businessIndex + familyIndex) % 3);
        return {
          businessId: `${family}-business-${businessIndex}`,
          family,
          split: "train",
          candidateId: `candidate-${candidateIndex}`,
          evidenceArm: candidateIndex % 2 === 0
            ? "experiments-only"
            : "benchmark-gap-fill",
          modelFamily: candidateIndex === 2 ? "advanced" : "bayesian",
          eligible: true,
          reviewEligible: true,
          eligibilityTier: "decision-grade",
          failedGateCount: 0,
          layerScores: {
            generalization: quality,
            structure: quality,
            causal: quality,
            decision: quality,
          },
          diagnostics: Object.fromEntries(
            features.map((feature, featureIndex) => [
              feature,
              Math.min(99, quality + (featureIndex % 4)),
            ]),
          ) as ScoreV6FeatureValues,
          heuristicScore: quality,
          decisionLoss: 0.8 - candidateIndex * 0.25 + businessIndex * 0.001,
          cappedRegret: 0.5 - candidateIndex * 0.1,
          roiError: 0.5 - candidateIndex * 0.1,
          contributionError: 0.5 - candidateIndex * 0.1,
        } satisfies ScoreV6CandidateRow;
      })
    ).flat()
  );
}

test("500-business SVI design preserves predeclared split and fit counts", () => {
  const design = sviDesignBusinesses();
  assert.equal(design.length, 500);
  for (const split of ["train", "validation", "audit"] as const) {
    assert.equal(
      design.filter((business) => business.family.split === split).length,
      SVI_SCORE_V3_CONTRACT.splits[split],
    );
  }
  assert.equal(
    design.length * SVI_SCORE_V3_CONTRACT.candidatesPerBusiness,
    24_000,
  );
});

test("gold MCMC audit is stratified and independent of scores and hidden loss", () => {
  const original = rows();
  const altered = original.map((row, index) => ({
    ...row,
    heuristicScore: 100 - row.heuristicScore,
    decisionLoss: index * 1_001,
    eligible: !row.eligible,
    failedGateCount: 9 - row.failedGateCount,
  }));
  const first = selectSviGoldAudit(original);
  const second = selectSviGoldAudit(altered);
  assert.equal(first.length, 96);
  assert.deepEqual(
    first.map((item) => `${item.businessId}:${item.candidateId}:${item.stratum}`),
    second.map((item) => `${item.businessId}:${item.candidateId}:${item.stratum}`),
  );
  assert.ok(first.every((item) => item.jointInclusionProbability > 0));
  assert.equal(new Set(first.map((item) => item.businessId)).size, 24);
  assert.deepEqual(
    Object.fromEntries(
      ["train", "validation", "audit"].map((split) => [
        split,
        new Set(
          first.filter((item) => item.split === split).map((item) => item.businessId),
        ).size,
      ]),
    ),
    { train: 12, validation: 6, audit: 6 },
  );
});

test("interrupted SVI checkpoints recover only complete records", () => {
  const source = '[\n  {"id":1,"nested":{"text":"} inside"}},\n  {"id":2},\n  {"id":';
  assert.deepEqual(recoverCompleteJsonObjects(source), [
    { id: 1, nested: { text: "} inside" } },
    { id: 2 },
  ]);
});

test("grouped folds keep every business intact and balance generator families", () => {
  const trainingRows = groupedCvRows();
  const assignments = groupedBusinessFolds(trainingRows, 5);
  assert.equal(assignments.length, 30);
  assert.equal(new Set(assignments.map((item) => item.businessId)).size, 30);
  for (let fold = 0; fold < 5; fold += 1) {
    const assessment = assignments.filter((item) => item.fold === fold);
    assert.equal(assessment.length, 6);
    assert.deepEqual(
      Object.fromEntries(
        ["family-a", "family-b", "family-c"].map((family) => [
          family,
          assessment.filter((item) => item.family === family).length,
        ]),
      ),
      { "family-a": 2, "family-b": 2, "family-c": 2 },
    );
  }
  const foldByBusiness = new Map(
    assignments.map((item) => [item.businessId, item.fold]),
  );
  assert.ok(trainingRows.every((row) => foldByBusiness.has(row.businessId)));
});

test("grouped score tuning is deterministic and produces one out-of-fold assessment per business", () => {
  const trainingRows = groupedCvRows();
  const configurations = [
    { id: "light", regularization: 0.005, epochs: 3 },
    { id: "strong", regularization: 0.05, epochs: 3 },
  ];
  const first = crossValidateScoreV3(trainingRows, configurations, 5);
  const second = crossValidateScoreV3(trainingRows, configurations, 5);
  assert.equal(first.data.businesses, 30);
  assert.deepEqual(first.data.foldBusinesses, {
    "fold-0": 6,
    "fold-1": 6,
    "fold-2": 6,
    "fold-3": 6,
    "fold-4": 6,
  });
  assert.equal(
    first.configurations.reduce(
      (total, configuration) =>
        total + configuration.folds.reduce(
          (foldTotal, fold) => foldTotal + fold.assessmentBusinesses,
          0,
        ),
      0,
    ),
    configurations.length * 30,
  );
  assert.deepEqual(first, second);
});

test("grouped score tuning rejects validation or audit rows", () => {
  const heldOut = groupedCvRows().map((row, index) =>
    index === 0 ? { ...row, split: "validation" as const } : row
  );
  assert.throws(
    () => groupedBusinessFolds(heldOut, 5),
    /Held-out split 'validation' is not permitted/,
  );
});

test("the theorem target uses every oracle comparison with exact capped excess regret", () => {
  const candidates = groupedCvRows().filter(
    (row) => row.businessId === "family-a-business-0",
  ).map((row, index) => ({
    ...row,
    economicScale: 1,
    decisionLoss: [0.1, 0.4, 1.6][index],
  }));
  const pairs = regretWeightedOraclePairs(candidates);
  assert.equal(pairs.length, 2);
  assert.ok(Math.abs(pairs[0].weight - 0.3) < 1e-12);
  assert.equal(pairs[1].weight, 1);
  const scoreByCandidate = new Map([
    ["candidate-0", 10],
    ["candidate-1", 5],
    ["candidate-2", 100],
  ]);
  const receipt = evaluateRegretBound(
    candidates,
    (row) => scoreByCandidate.get(row.candidateId)!,
  );
  assert.equal(receipt.violations, 0);
  assert.equal(receipt.meanSelectedNormalizedRegret, 1);
  assert.equal(receipt.meanMisrankingUpperBound, 1);
  assert.ok(
    receipt.meanLogisticSurrogateUpperBound >=
      receipt.meanMisrankingUpperBound,
  );
});

test("the theorem target rejects candidate-dependent economic scales", () => {
  const candidates = groupedCvRows().filter(
    (row) => row.businessId === "family-a-business-0",
  ).map((row, index) => ({
    ...row,
    economicScale: index === 0 ? 1 : 2,
  }));
  assert.throws(
    () => regretWeightedOraclePairs(candidates),
    /candidate-independent/,
  );
});

test("the frozen training release verifies without reading local held-out checkpoints", async () => {
  const receipt = await verifyTrainingFreeze(process.cwd(), {
    verifyLocalSources: false,
  });
  assert.equal(receipt.checkedFiles, 8);
  assert.equal(receipt.artifactSha256.length, 64);
  assert.equal(receipt.weightsSha256.length, 64);
});

test("the predeclared validation protocol uses paired family-stratified economic loss", () => {
  const validationRows = ["delayed-tv", "correlated-planning"].flatMap(
    (family) => Array.from({ length: 60 }, (_, businessIndex) =>
      Array.from({ length: 48 }, (_, candidateIndex) => ({
        businessId: `${family}-${businessIndex}`,
        family,
        split: "validation" as const,
        candidateId: `candidate-${candidateIndex}`,
        evidenceArm: candidateIndex % 2 === 0
          ? "experiments-only" as const
          : "benchmark-gap-fill" as const,
        modelFamily: candidateIndex % 3 === 0
          ? "advanced" as const
          : "bayesian" as const,
        eligible: true,
        reviewEligible: true,
        eligibilityTier: "decision-grade" as const,
        failedGateCount: 0,
        layerScores: {
          generalization: 70,
          structure: 70,
          causal: 70,
          decision: 70,
        },
        diagnostics: {
          "rolling-oos": candidateIndex === 0 ? 100 : candidateIndex === 1 ? 20 : 10,
          "spend-regimes": candidateIndex === 1 ? 100 : candidateIndex === 0 ? 20 : 10,
        },
        heuristicScore: 70,
        economicScale: 1,
        decisionLoss: candidateIndex,
        cappedRegret: Math.min(candidateIndex, 1),
        roiError: candidateIndex / 10,
        contributionError: candidateIndex / 10,
      } satisfies ScoreV6CandidateRow)).flat(),
    ).flat(),
  );
  const receipt = evaluateSviScoreV3Validation(
    validationRows,
    { "rolling-oos": 1 },
    { "spend-regimes": 1 },
  );
  assert.equal(receipt.businesses, 120);
  assert.equal(receipt.challenger.meanExcessLoss, 0);
  assert.equal(receipt.comparator.meanExcessLoss, 1);
  assert.ok(receipt.pairedPrimary.oneSided95UpperBound < 0);
  assert.equal(receipt.pairedPrimary.statisticallySuperior, true);
  assert.equal(receipt.pairedPrimary.statisticallyNonInferior, true);
  assert.ok(
    receipt.challenger.meanRegretWeightedMisrankingBound <
      receipt.comparator.meanRegretWeightedMisrankingBound,
  );
  assert.equal(receipt.proceedToSealedAudit, true);
});

test("validation refuses incomplete cohorts and every non-validation split", () => {
  assert.equal(SVI_SCORE_V3_VALIDATION_PROTOCOL.rows, 5_760);
  assert.throws(
    () => evaluateSviScoreV3Validation([], {}, {}),
    /exactly 5760 candidate rows/,
  );
  const trainingRows = groupedCvRows();
  assert.throws(
    () => evaluateSviScoreV3Validation(trainingRows, {}, {}),
    /exactly 5760 candidate rows/,
  );
});

test("the validation protocol is frozen before any held-out outcome is opened", async () => {
  const receipt = await verifyValidationProtocol();
  assert.equal(receipt.checkedFiles, 6);
  assert.equal(
    receipt.protocolId,
    "flux-svi-score-v3-validation-v2-powered-screen",
  );
});

test("validation assembly filters the split before reading sealed-audit labels", () => {
  const base = groupedCvRows()[0];
  const validation = {
    ...base,
    split: "validation",
    status: "labelled",
    sviDecisionLoss: 0.42,
    sviProfitRegret: 0.1,
    sviRoiError: 0.2,
    sviContributionError: 0.3,
  } as unknown as SviScoreV3Record;
  const audit = {
    ...base,
    businessId: "sealed-audit-business",
    split: "audit",
    status: "labelled",
    get sviDecisionLoss(): number {
      throw new Error("sealed audit label was accessed");
    },
  } as unknown as SviScoreV3Record;
  const rows = sviValidationRows([audit, validation], [{
    businessId: validation.businessId,
    candidateId: validation.candidateId,
    eligible: validation.eligible,
    reviewEligible: validation.reviewEligible,
    eligibilityTier: validation.eligibilityTier,
    failedGateCount: validation.failedGateCount,
    layerScores: validation.layerScores,
    diagnostics: validation.diagnostics,
    heuristicScore: validation.heuristicScore,
  }]);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].decisionLoss, 0.42);
  assert.equal(rows[0].split, "validation");
});

test("the one-time validation result is frozen and keeps audit sealed", async () => {
  const receipt = await verifyValidationResult();
  assert.equal(receipt.status, "did-not-pass");
  assert.equal(receipt.auditMayOpen, false);
  assert.equal(receipt.auditAccessed, false);
  assert.equal(receipt.checkedSources, 5);
});
