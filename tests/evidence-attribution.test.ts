import assert from "node:assert/strict";
import test from "node:test";
import {
  channelEvidenceAttribution,
  EVIDENCE_SOURCES,
} from "../lib/mmm/evidence-attribution";
import {
  diagonalPenalty,
  solveLeastSquares,
} from "../lib/mmm/math";

function gradient(columns: number, index: number): number[] {
  return Array.from({ length: columns }, (_, column) =>
    column === index ? 1 : 0
  );
}

test("ROI uncertainty attribution is nonnegative and exhaustive", () => {
  const matrix = [
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
  ];
  const regularization = [
    diagonalPenalty(3, 1, 0.2),
    diagonalPenalty(3, 2, 0.2),
  ];
  const experiment = [diagonalPenalty(3, 1, 2, 1)];
  const solution = solveLeastSquares(matrix, [0, 1, 1, 2], {
    penalties: [...regularization, ...experiment],
  });
  const attribution = channelEvidenceAttribution({
    matrix,
    posteriorPrecisionInverse: solution.precisionInverse,
    penalties: {
      experiment,
      benchmark: [],
      regularization,
    },
    channels: [
      { channel: "a", coefficientIndexes: [1], roiGradient: gradient(3, 1) },
      { channel: "b", coefficientIndexes: [2], roiGradient: gradient(3, 2) },
    ],
  });
  attribution.forEach((channel) => {
    const total = EVIDENCE_SOURCES.reduce(
      (sum, source) => sum + channel.uncertaintyShare[source],
      0,
    );
    assert.ok(Math.abs(total - 1) < 1e-10);
    assert.ok(
      EVIDENCE_SOURCES.every(
        (source) => channel.uncertaintyShare[source] >= 0,
      ),
    );
    assert.ok(channel.attributionResidual < 1e-9);
  });
  assert.ok(attribution[0].uncertaintyShare.experiment > 0);
  assert.equal(attribution[1].uncertaintyShare.experiment, 0);
});

test("duplicated media have negligible conditional separating information", () => {
  const matrix = [
    [1, 0, 0],
    [1, 1, 1],
    [1, 2, 2],
    [1, 3, 3],
  ];
  const regularization = [
    diagonalPenalty(3, 1, 0.1),
    diagonalPenalty(3, 2, 0.1),
  ];
  const solution = solveLeastSquares(matrix, [0, 1, 2, 3], {
    penalties: regularization,
  });
  const attribution = channelEvidenceAttribution({
    matrix,
    posteriorPrecisionInverse: solution.precisionInverse,
    penalties: { experiment: [], benchmark: [], regularization },
    channels: [
      { channel: "a", coefficientIndexes: [1], roiGradient: gradient(3, 1) },
      { channel: "b", coefficientIndexes: [2], roiGradient: gradient(3, 2) },
    ],
  });
  assert.ok(attribution[0].conditionalDataShare < 1e-10);
  assert.ok(attribution[1].conditionalDataShare < 1e-10);
});
