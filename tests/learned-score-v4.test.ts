import assert from "node:assert/strict";
import test from "node:test";
import {
  ACTIVE_SCORE_CONTRACT,
  HEURISTIC_SCORE_WEIGHTS,
  LEARNED_SCORE_ARTIFACT,
  scoreValidationLayers,
} from "../lib/mmm/score-contract";

test("learned score artifact clears its predeclared activation contract", () => {
  assert.equal(LEARNED_SCORE_ARTIFACT.activation, "active");
  assert.equal(ACTIVE_SCORE_CONTRACT.kind, "learned");
  assert.equal(LEARNED_SCORE_ARTIFACT.cohort.businesses, 200);
  assert.equal(LEARNED_SCORE_ARTIFACT.cohort.candidates, 3_200);
  assert.ok(
    LEARNED_SCORE_ARTIFACT.performance.validation.relativeMeanRegretReduction >=
      0.02,
  );
  assert.ok(
    LEARNED_SCORE_ARTIFACT.performance.audit.learned.meanRegret <=
      LEARNED_SCORE_ARTIFACT.performance.audit.heuristic.meanRegret * 1.02,
  );
  assert.ok(
    LEARNED_SCORE_ARTIFACT.performance.validation.learned.p90Regret <=
      LEARNED_SCORE_ARTIFACT.performance.validation.heuristic.p90Regret *
        1.000001,
  );
  for (const [family, regret] of Object.entries(
    LEARNED_SCORE_ARTIFACT.performance.validation.heuristic.familyMeanRegret,
  )) {
    assert.ok(
      (LEARNED_SCORE_ARTIFACT.performance.validation.learned.familyMeanRegret[
        family
      ] ?? Infinity) <= regret * 1.01,
    );
  }
});

test("learned weights are normalized, bounded, and can change a ranking", () => {
  const weights = ACTIVE_SCORE_CONTRACT.weights;
  assert.ok(
    Math.abs(Object.values(weights).reduce((total, value) => total + value, 0) - 1) <
      1e-12,
  );
  Object.values(weights).forEach((weight) => {
    assert.ok(weight >= 0.05);
    assert.ok(weight <= 1);
  });

  const structurallyStrong = {
    generalization: 80,
    structure: 98,
    causal: 80,
    decision: 50,
  };
  const decisionStrong = {
    generalization: 80,
    structure: 55,
    causal: 80,
    decision: 95,
  };
  assert.ok(
    scoreValidationLayers(structurallyStrong, HEURISTIC_SCORE_WEIGHTS) <
      scoreValidationLayers(decisionStrong, HEURISTIC_SCORE_WEIGHTS),
  );
  assert.ok(
    scoreValidationLayers(structurallyStrong) >
      scoreValidationLayers(decisionStrong),
  );
});

test("score remains finite at a failed layer boundary for review-mode display", () => {
  const score = scoreValidationLayers({
    generalization: 82,
    structure: 91,
    causal: 0,
    decision: 88,
  });
  assert.ok(Number.isFinite(score));
  assert.ok(score >= 0 && score <= 100);
});
