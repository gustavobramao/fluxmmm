import {
  matrixVector,
  solveLeastSquares,
  type LeastSquaresPenalty,
} from "./math";

export const EVIDENCE_SOURCES = [
  "observational",
  "experiment",
  "benchmark",
  "regularization",
] as const;

export type EvidenceSource = (typeof EVIDENCE_SOURCES)[number];

export type EvidenceSourceShares = Record<EvidenceSource, number>;

export interface ChannelEvidenceAttribution {
  channel: string;
  /** Variance attribution for the channel ROI under the local Laplace model. */
  uncertaintyShare: EvidenceSourceShares;
  /** Schur-complement information after conditioning on the other regressors. */
  conditionalDataInformation: number;
  conditionalDataShare: number;
  attributionResidual: number;
}

export interface EvidenceCurvatureContract {
  matrix: number[][];
  weights?: number[];
  posteriorPrecisionInverse: number[][];
  penalties: Record<Exclude<EvidenceSource, "observational">, LeastSquaresPenalty[]>;
  channels: {
    channel: string;
    coefficientIndexes: number[];
    roiGradient: number[];
  }[];
}

function zeros(size: number): number[][] {
  return Array.from({ length: size }, () => Array(size).fill(0));
}

function quadratic(vector: number[], matrix: number[][]): number {
  const projected = matrixVector(matrix, vector);
  return vector.reduce(
    (total, value, index) => total + value * (projected[index] ?? 0),
    0,
  );
}

export function weightedGram(matrix: number[][], weights?: number[]): number[][] {
  const columns = matrix[0]?.length ?? 0;
  const gram = zeros(columns);
  matrix.forEach((row, rowIndex) => {
    const weight = Math.max(weights?.[rowIndex] ?? 1, 0);
    for (let left = 0; left < columns; left += 1) {
      const leftValue = row[left] ?? 0;
      for (let right = left; right < columns; right += 1) {
        const value = weight * leftValue * (row[right] ?? 0);
        gram[left][right] += value;
        if (left !== right) gram[right][left] += value;
      }
    }
  });
  return gram;
}

export function penaltyCurvature(
  penalties: readonly LeastSquaresPenalty[],
  columns: number,
): number[][] {
  const curvature = zeros(columns);
  penalties.forEach((penalty) => {
    if (
      penalty.coefficients.length !== columns ||
      !Number.isFinite(penalty.precision) ||
      penalty.precision <= 0
    ) {
      return;
    }
    for (let left = 0; left < columns; left += 1) {
      const leftValue = penalty.coefficients[left] ?? 0;
      for (let right = left; right < columns; right += 1) {
        const value =
          penalty.precision * leftValue * (penalty.coefficients[right] ?? 0);
        curvature[left][right] += value;
        if (left !== right) curvature[right][left] += value;
      }
    }
  });
  return curvature;
}

function conditionalInformation(
  matrix: number[][],
  weights: number[] | undefined,
  coefficientIndexes: number[],
  roiGradient: number[],
): { information: number; share: number } {
  if (!matrix.length || !coefficientIndexes.length) {
    return { information: 0, share: 0 };
  }
  const gradientNorm = Math.sqrt(
    coefficientIndexes.reduce(
      (total, index) => total + (roiGradient[index] ?? 0) ** 2,
      0,
    ),
  );
  if (gradientNorm <= 1e-12) return { information: 0, share: 0 };
  const indexSet = new Set(coefficientIndexes);
  const controls = Array.from(
    { length: matrix[0]?.length ?? 0 },
    (_, index) => index,
  ).filter((index) => !indexSet.has(index));
  const direction = matrix.map((row) =>
    coefficientIndexes.reduce(
      (total, index) =>
        total + (row[index] ?? 0) * (roiGradient[index] ?? 0) / gradientNorm,
      0,
    ),
  );
  const totalInformation = direction.reduce(
    (total, value, rowIndex) =>
      total + Math.max(weights?.[rowIndex] ?? 1, 0) * value ** 2,
    0,
  );
  if (!controls.length || totalInformation <= 1e-12) {
    return {
      information: totalInformation,
      share: totalInformation <= 1e-12 ? 0 : 1,
    };
  }
  const controlMatrix = matrix.map((row) =>
    controls.map((index) => row[index] ?? 0),
  );
  const projection = solveLeastSquares(controlMatrix, direction, { weights });
  const fitted = matrixVector(controlMatrix, projection.coefficients);
  const information = direction.reduce(
    (total, value, rowIndex) =>
      total +
        Math.max(weights?.[rowIndex] ?? 1, 0) *
          (value - fitted[rowIndex]) ** 2,
    0,
  );
  return {
    information,
    share: Math.min(1, Math.max(0, information / totalInformation)),
  };
}

/**
 * Decomposes local posterior ROI variance as
 * g' H^-1 H_s H^-1 g / (g' H^-1 g). Curvatures are PSD outer products and
 * are all evaluated at the same fit, making the displayed shares nonnegative
 * and exhaustive up to floating-point error.
 */
export function channelEvidenceAttribution(
  contract: EvidenceCurvatureContract,
): ChannelEvidenceAttribution[] {
  const columns = contract.matrix[0]?.length ?? 0;
  const curvatures: Record<EvidenceSource, number[][]> = {
    observational: weightedGram(contract.matrix, contract.weights),
    experiment: penaltyCurvature(contract.penalties.experiment, columns),
    benchmark: penaltyCurvature(contract.penalties.benchmark, columns),
    regularization: penaltyCurvature(
      contract.penalties.regularization,
      columns,
    ),
  };
  return contract.channels.map((channel) => {
    const posteriorGradient = matrixVector(
      contract.posteriorPrecisionInverse,
      channel.roiGradient,
    );
    const posteriorVariance = Math.max(
      channel.roiGradient.reduce(
        (total, value, index) =>
          total + value * (posteriorGradient[index] ?? 0),
        0,
      ),
      1e-18,
    );
    const raw = Object.fromEntries(
      EVIDENCE_SOURCES.map((source) => [
        source,
        Math.max(0, quadratic(posteriorGradient, curvatures[source])) /
          posteriorVariance,
      ]),
    ) as EvidenceSourceShares;
    const rawTotal = EVIDENCE_SOURCES.reduce(
      (total, source) => total + raw[source],
      0,
    );
    const uncertaintyShare = Object.fromEntries(
      EVIDENCE_SOURCES.map((source) => [
        source,
        raw[source] / Math.max(rawTotal, 1e-18),
      ]),
    ) as EvidenceSourceShares;
    const conditional = conditionalInformation(
      contract.matrix,
      contract.weights,
      channel.coefficientIndexes,
      channel.roiGradient,
    );
    return {
      channel: channel.channel,
      uncertaintyShare,
      conditionalDataInformation: conditional.information,
      conditionalDataShare: conditional.share,
      attributionResidual: Math.abs(1 - rawTotal),
    };
  });
}
