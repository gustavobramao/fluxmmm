import { diagonalPenalty, mean, solveLeastSquares, std } from "../math";

export function clampScore(value: number): number {
  return Math.max(0, Math.min(100, value));
}

export function correlation(a: number[], b: number[]): number {
  const aMean = mean(a);
  const bMean = mean(b);
  const numerator = a.reduce(
    (total, value, index) =>
      total + (value - aMean) * ((b[index] ?? bMean) - bMean),
    0,
  );
  const denominator =
    Math.sqrt(
      a.reduce((total, value) => total + (value - aMean) ** 2, 0) *
        b.reduce((total, value) => total + (value - bMean) ** 2, 0),
    ) || 1;
  return numerator / denominator;
}

export function weightedScore(
  items: { score: number; weight: number }[],
): number {
  const totalWeight = items.reduce((total, item) => total + item.weight, 0);
  return clampScore(
    items.reduce((total, item) => total + item.score * item.weight, 0) /
      Math.max(totalWeight, 1),
  );
}

export function percentile(values: number[], probability: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * Math.max(0, Math.min(1, probability));
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

export function skewness(values: number[]): number {
  const average = mean(values);
  const scale = std(values);
  return mean(values.map((value) => ((value - average) / scale) ** 3));
}

export function excessKurtosis(values: number[]): number {
  const average = mean(values);
  const scale = std(values);
  return mean(values.map((value) => ((value - average) / scale) ** 4)) - 3;
}

export function varianceInflationFactors(
  columns: number[][],
): number[] {
  if (columns.length < 2 || !columns[0]?.length) {
    return columns.map(() => 1);
  }
  return columns.map((target, targetIndex) => {
    const predictors = columns.filter((_, index) => index !== targetIndex);
    const rowCount = target.length;
    const matrix = Array.from({ length: rowCount }, (_, rowIndex) => [
      1,
      ...predictors.map((column) => column[rowIndex]),
    ]);
    const columnCount = matrix[0].length;
    const coefficients = solveLeastSquares(matrix, target, {
      penalties: Array.from(
        { length: Math.max(columnCount - 1, 0) },
        (_, offset) =>
          diagonalPenalty(columnCount, offset + 1, 1e-4),
      ),
    }).coefficients;
    const predicted = matrix.map((row) =>
      row.reduce(
        (total, value, index) => total + value * coefficients[index],
        0,
      ),
    );
    const total = target.reduce(
      (sum, value) => sum + (value - mean(target)) ** 2,
      0,
    );
    const residual = target.reduce(
      (sum, value, index) => sum + (value - predicted[index]) ** 2,
      0,
    );
    const r2 = 1 - residual / Math.max(total, 1e-9);
    return 1 / Math.max(1 - Math.min(r2, 0.999), 1e-3);
  });
}
