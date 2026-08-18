import assert from "node:assert/strict";
import test from "node:test";
import {
  ACTIVE_SCORE_CONTRACT,
  HEURISTIC_SCORE_WEIGHTS,
  LEARNED_SCORE_ARTIFACT,
  LEARNED_SCORE_V6_ARTIFACT,
  scoreValidationDiagnostics,
  scoreValidationLayers,
} from "../lib/mmm/score-contract";
import {
  SCORE_DIAGNOSTIC_GROUPS,
  type ValidationDiagnosticValues,
} from "../lib/mmm/score-diagnostics";

const diagnosticIds = Object.values(SCORE_DIAGNOSTIC_GROUPS).flat();

test("V6 is the active diagnostic ranker while V4 remains auditable", () => {
  assert.equal(LEARNED_SCORE_ARTIFACT.activation, "active");
  assert.equal(LEARNED_SCORE_V6_ARTIFACT.activation, "active");
  assert.equal(ACTIVE_SCORE_CONTRACT.kind, "learned");
  assert.equal(ACTIVE_SCORE_CONTRACT.resolution, "diagnostic");
  assert.equal(
    ACTIVE_SCORE_CONTRACT.version,
    "flux-score-learner-v6.0.0-pilot",
  );
  assert.equal(
    Object.keys(ACTIVE_SCORE_CONTRACT.diagnosticWeights ?? {}).length,
    20,
  );
});

test("V6 diagnostic and readable group weights are normalized and monotonic", () => {
  const diagnosticWeights = ACTIVE_SCORE_CONTRACT.diagnosticWeights;
  assert.ok(diagnosticWeights);
  assert.ok(
    Math.abs(
      Object.values(diagnosticWeights).reduce(
        (total, value) => total + value,
        0,
      ) - 1,
    ) < 1e-9,
  );
  Object.values(diagnosticWeights).forEach((weight) => {
    assert.ok(weight >= 0);
    assert.ok(weight <= 0.16 + 1e-9);
  });
  assert.ok(
    Math.abs(
      Object.values(ACTIVE_SCORE_CONTRACT.weights).reduce(
        (total, value) => total + value,
        0,
      ) - 1,
    ) < 1e-9,
  );

  const baseline = Object.fromEntries(
    diagnosticIds.map((id) => [id, 60]),
  ) as ValidationDiagnosticValues;
  const improved = { ...baseline, "spend-regimes": 90 };
  assert.ok(
    scoreValidationDiagnostics(improved) >
      scoreValidationDiagnostics(baseline),
  );
});

test("legacy four-layer heuristic remains available as a transparent reference", () => {
  const score = scoreValidationLayers(
    {
      generalization: 82,
      structure: 91,
      causal: 0,
      decision: 88,
    },
    HEURISTIC_SCORE_WEIGHTS,
  );
  assert.ok(Number.isFinite(score));
  assert.ok(score >= 0 && score <= 100);
});
