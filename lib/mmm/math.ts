export function mean(values: number[]): number {
  return values.length
    ? values.reduce((total, value) => total + value, 0) / values.length
    : 0;
}

export function variance(values: number[]): number {
  const average = mean(values);
  return mean(values.map((value) => (value - average) ** 2));
}

export function std(values: number[]): number {
  return Math.sqrt(Math.max(variance(values), 1e-12));
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

export function transpose(matrix: number[][]): number[][] {
  return matrix[0].map((_, column) => matrix.map((row) => row[column]));
}

export function multiply(a: number[][], b: number[][]): number[][] {
  const bT = transpose(b);
  return a.map((row) =>
    bT.map((column) =>
      row.reduce((total, value, index) => total + value * column[index], 0),
    ),
  );
}

export function matrixVector(matrix: number[][], vector: number[]): number[] {
  return matrix.map((row) =>
    row.reduce(
      (total, value, index) => total + value * (vector[index] ?? 0),
      0,
    ),
  );
}

export interface LeastSquaresPenalty {
  coefficients: number[];
  target: number;
  precision: number;
}

export interface LinearSolverDiagnostics {
  method: "pivoted-qr" | "svd";
  rank: number;
  columns: number;
  conditionNumber: number;
  status: "stable" | "review" | "rank-deficient";
}

export interface LeastSquaresSolution {
  coefficients: number[];
  precisionInverse: number[][];
  diagnostics: LinearSolverDiagnostics;
}

export function diagnoseDesignMatrix(
  matrix: number[][],
  weights?: number[],
): LinearSolverDiagnostics {
  const weightedMatrix = matrix.map((row, rowIndex) => {
    const scale = Math.sqrt(Math.max(weights?.[rowIndex] ?? 1, 1e-12));
    return row.map((value) => value * scale);
  });
  const zeroOutcome = Array(weightedMatrix.length).fill(0);
  return (
    pivotedQr(weightedMatrix, zeroOutcome) ??
    jacobiSvdSolve(weightedMatrix, zeroOutcome)
  ).diagnostics;
}

const QR_FALLBACK_CONDITION = 1e8;
const PRACTICAL_RANK_TOLERANCE = 1e-10;

function identity(size: number): number[][] {
  return Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, column) => (row === column ? 1 : 0)),
  );
}

function hypot(values: number[]): number {
  return values.reduce((magnitude, value) => Math.hypot(magnitude, value), 0);
}

function solverStatus(
  rank: number,
  columns: number,
  conditionNumber: number,
): LinearSolverDiagnostics["status"] {
  if (rank < columns || !Number.isFinite(conditionNumber)) {
    return "rank-deficient";
  }
  return conditionNumber > 1e6 ? "review" : "stable";
}

function pivotedQr(
  matrix: number[][],
  outcome: number[],
): LeastSquaresSolution | undefined {
  const rowCount = matrix.length;
  const columnCount = matrix[0]?.length ?? 0;
  if (!rowCount || !columnCount || rowCount < columnCount) return undefined;

  const qr = matrix.map((row) => [...row]);
  const transformedOutcome = [...outcome];
  const permutation = Array.from({ length: columnCount }, (_, index) => index);

  for (let column = 0; column < columnCount; column += 1) {
    let pivot = column;
    let pivotNorm = -1;
    for (let candidate = column; candidate < columnCount; candidate += 1) {
      const norm = hypot(
        qr.slice(column).map((row) => row[candidate] ?? 0),
      );
      if (norm > pivotNorm) {
        pivot = candidate;
        pivotNorm = norm;
      }
    }
    if (pivot !== column) {
      qr.forEach((row) => {
        [row[column], row[pivot]] = [row[pivot], row[column]];
      });
      [permutation[column], permutation[pivot]] = [
        permutation[pivot],
        permutation[column],
      ];
    }

    const columnValues = qr.slice(column).map((row) => row[column] ?? 0);
    const norm = hypot(columnValues);
    if (norm === 0) continue;
    const reflectedDiagonal = (qr[column][column] ?? 0) >= 0 ? -norm : norm;
    const reflector = columnValues;
    reflector[0] -= reflectedDiagonal;
    const reflectorNormSquared = reflector.reduce(
      (total, value) => total + value * value,
      0,
    );
    if (reflectorNormSquared === 0) continue;
    const scale = 2 / reflectorNormSquared;

    for (let targetColumn = column; targetColumn < columnCount; targetColumn += 1) {
      let projection = 0;
      for (let row = column; row < rowCount; row += 1) {
        projection += reflector[row - column] * qr[row][targetColumn];
      }
      projection *= scale;
      for (let row = column; row < rowCount; row += 1) {
        qr[row][targetColumn] -= projection * reflector[row - column];
      }
    }
    let outcomeProjection = 0;
    for (let row = column; row < rowCount; row += 1) {
      outcomeProjection +=
        reflector[row - column] * transformedOutcome[row];
    }
    outcomeProjection *= scale;
    for (let row = column; row < rowCount; row += 1) {
      transformedOutcome[row] -=
        outcomeProjection * reflector[row - column];
    }
  }

  const diagonal = Array.from(
    { length: columnCount },
    (_, index) => Math.abs(qr[index]?.[index] ?? 0),
  );
  const maximumDiagonal = Math.max(...diagonal, 0);
  const rankTolerance = Math.max(
    Number.EPSILON * Math.max(rowCount, columnCount),
    PRACTICAL_RANK_TOLERANCE,
  ) * maximumDiagonal;
  const rank = diagonal.filter((value) => value > rankTolerance).length;
  const minimumDiagonal = Math.min(
    ...diagonal.filter((value) => value > rankTolerance),
  );
  const conditionNumber =
    rank === columnCount && minimumDiagonal > 0
      ? maximumDiagonal / minimumDiagonal
      : Number.POSITIVE_INFINITY;

  if (rank < columnCount || conditionNumber > QR_FALLBACK_CONDITION) {
    return undefined;
  }

  const solveUpper = (rightHandSide: number[]): number[] => {
    const solution = Array(columnCount).fill(0);
    for (let row = columnCount - 1; row >= 0; row -= 1) {
      let value = rightHandSide[row] ?? 0;
      for (let column = row + 1; column < columnCount; column += 1) {
        value -= qr[row][column] * solution[column];
      }
      solution[row] = value / qr[row][row];
    }
    return solution;
  };
  const solveUpperTranspose = (rightHandSide: number[]): number[] => {
    const solution = Array(columnCount).fill(0);
    for (let row = 0; row < columnCount; row += 1) {
      let value = rightHandSide[row] ?? 0;
      for (let column = 0; column < row; column += 1) {
        value -= qr[column][row] * solution[column];
      }
      solution[row] = value / qr[row][row];
    }
    return solution;
  };

  const permutedCoefficients = solveUpper(
    transformedOutcome.slice(0, columnCount),
  );
  const coefficients = Array(columnCount).fill(0);
  permutation.forEach((originalColumn, permutedColumn) => {
    coefficients[originalColumn] = permutedCoefficients[permutedColumn];
  });

  const permutedInverse = Array.from({ length: columnCount }, () =>
    Array(columnCount).fill(0),
  );
  for (let column = 0; column < columnCount; column += 1) {
    const basis = Array(columnCount).fill(0);
    basis[column] = 1;
    const solved = solveUpper(solveUpperTranspose(basis));
    solved.forEach((value, row) => {
      permutedInverse[row][column] = value;
    });
  }
  const precisionInverse = Array.from({ length: columnCount }, () =>
    Array(columnCount).fill(0),
  );
  permutation.forEach((originalRow, permutedRow) => {
    permutation.forEach((originalColumn, permutedColumn) => {
      precisionInverse[originalRow][originalColumn] =
        (permutedInverse[permutedRow][permutedColumn] +
          permutedInverse[permutedColumn][permutedRow]) /
        2;
    });
  });

  return {
    coefficients,
    precisionInverse,
    diagnostics: {
      method: "pivoted-qr",
      rank,
      columns: columnCount,
      conditionNumber,
      status: solverStatus(rank, columnCount, conditionNumber),
    },
  };
}

function jacobiSvdSolve(
  matrix: number[][],
  outcome: number[],
): LeastSquaresSolution {
  const rowCount = matrix.length;
  const columnCount = matrix[0]?.length ?? 0;
  const rotated = matrix.map((row) => [...row]);
  const rightVectors = identity(columnCount);
  const maximumSweeps = Math.max(30, columnCount * 4);

  for (let sweep = 0; sweep < maximumSweeps; sweep += 1) {
    let changed = false;
    for (let left = 0; left < columnCount - 1; left += 1) {
      for (let right = left + 1; right < columnCount; right += 1) {
        let alpha = 0;
        let beta = 0;
        let gamma = 0;
        for (let row = 0; row < rowCount; row += 1) {
          const leftValue = rotated[row][left];
          const rightValue = rotated[row][right];
          alpha += leftValue * leftValue;
          beta += rightValue * rightValue;
          gamma += leftValue * rightValue;
        }
        if (
          alpha === 0 ||
          beta === 0 ||
          Math.abs(gamma) <= 1e-12 * Math.sqrt(alpha * beta)
        ) {
          continue;
        }
        changed = true;
        const zeta = (beta - alpha) / (2 * gamma);
        const tangent =
          (zeta >= 0 ? 1 : -1) /
          (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const cosine = 1 / Math.sqrt(1 + tangent * tangent);
        const sine = cosine * tangent;
        for (let row = 0; row < rowCount; row += 1) {
          const leftValue = rotated[row][left];
          const rightValue = rotated[row][right];
          rotated[row][left] = cosine * leftValue - sine * rightValue;
          rotated[row][right] = sine * leftValue + cosine * rightValue;
        }
        for (let row = 0; row < columnCount; row += 1) {
          const leftValue = rightVectors[row][left];
          const rightValue = rightVectors[row][right];
          rightVectors[row][left] = cosine * leftValue - sine * rightValue;
          rightVectors[row][right] = sine * leftValue + cosine * rightValue;
        }
      }
    }
    if (!changed) break;
  }

  const singularValues = Array.from({ length: columnCount }, (_, column) =>
    hypot(rotated.map((row) => row[column] ?? 0)),
  );
  const order = singularValues
    .map((value, index) => ({ value, index }))
    .sort((left, right) => right.value - left.value);
  const maximumSingularValue = order[0]?.value ?? 0;
  const rankTolerance = Math.max(
    Number.EPSILON * Math.max(rowCount, columnCount),
    PRACTICAL_RANK_TOLERANCE,
  ) * maximumSingularValue;
  const retained = order.filter(({ value }) => value > rankTolerance);
  const rank = retained.length;
  const minimumSingularValue = retained.at(-1)?.value ?? 0;
  const conditionNumber =
    rank === columnCount && minimumSingularValue > 0
      ? maximumSingularValue / minimumSingularValue
      : Number.POSITIVE_INFINITY;
  const coefficients = Array(columnCount).fill(0);
  const precisionInverse = Array.from({ length: columnCount }, () =>
    Array(columnCount).fill(0),
  );

  retained.forEach(({ value: singularValue, index: singularIndex }) => {
    let projectedOutcome = 0;
    for (let row = 0; row < rowCount; row += 1) {
      projectedOutcome += rotated[row][singularIndex] * outcome[row];
    }
    const coefficientScale = projectedOutcome / (singularValue ** 2);
    const inverseScale = 1 / (singularValue ** 2);
    for (let row = 0; row < columnCount; row += 1) {
      const rightRow = rightVectors[row][singularIndex];
      coefficients[row] += rightRow * coefficientScale;
      for (let column = 0; column < columnCount; column += 1) {
        precisionInverse[row][column] +=
          rightRow * rightVectors[column][singularIndex] * inverseScale;
      }
    }
  });

  return {
    coefficients,
    precisionInverse,
    diagnostics: {
      method: "svd",
      rank,
      columns: columnCount,
      conditionNumber,
      status: solverStatus(rank, columnCount, conditionNumber),
    },
  };
}

export function solveLeastSquares(
  matrix: number[][],
  outcome: number[],
  options: {
    weights?: number[];
    penalties?: LeastSquaresPenalty[];
  } = {},
): LeastSquaresSolution {
  const columnCount = matrix[0]?.length ?? 0;
  if (!matrix.length || !columnCount || outcome.length !== matrix.length) {
    throw new Error("Least-squares inputs must contain aligned, non-empty rows.");
  }
  if (matrix.some((row) => row.length !== columnCount)) {
    throw new Error("Least-squares design rows must have a consistent width.");
  }

  const augmentedMatrix: number[][] = [];
  const augmentedOutcome: number[] = [];
  matrix.forEach((row, rowIndex) => {
    const weight = Math.max(options.weights?.[rowIndex] ?? 1, 1e-12);
    const scale = Math.sqrt(weight);
    augmentedMatrix.push(row.map((value) => value * scale));
    augmentedOutcome.push(outcome[rowIndex] * scale);
  });
  (options.penalties ?? []).forEach((penalty) => {
    if (
      penalty.coefficients.length !== columnCount ||
      !Number.isFinite(penalty.precision) ||
      penalty.precision <= 0
    ) {
      return;
    }
    const scale = Math.sqrt(penalty.precision);
    augmentedMatrix.push(
      penalty.coefficients.map((value) => value * scale),
    );
    augmentedOutcome.push(penalty.target * scale);
  });

  return (
    pivotedQr(augmentedMatrix, augmentedOutcome) ??
    jacobiSvdSolve(augmentedMatrix, augmentedOutcome)
  );
}

export function diagonalPenalty(
  columns: number,
  coefficient: number,
  precision: number,
  target = 0,
): LeastSquaresPenalty {
  return {
    coefficients: Array.from(
      { length: columns },
      (_, index) => (index === coefficient ? 1 : 0),
    ),
    target,
    precision,
  };
}

export function adstock(values: number[], decay: number): number[] {
  return values.reduce<number[]>((output, value, index) => {
    output.push(value + (output[index - 1] ?? 0) * decay);
    return output;
  }, []);
}

export function weibullAdstock(
  values: number[],
  shape: number,
  scale: number,
  maxLag = 52,
): number[] {
  const safeShape = Math.max(shape, 0.1);
  const safeScale = Math.max(scale, 0.5);
  const lagCount = Math.min(maxLag, Math.max(values.length - 1, 0));
  const weights = Array.from({ length: lagCount + 1 }, (_, lag) => {
    const time = lag + 0.5;
    const ratio = time / safeScale;
    return (
      (safeShape / safeScale) *
      ratio ** (safeShape - 1) *
      Math.exp(-(ratio ** safeShape))
    );
  });
  const peak = Math.max(...weights, 1e-12);
  const scaledWeights = weights.map((weight) => weight / peak);

  return values.map((_, index) =>
    scaledWeights
      .slice(0, Math.min(index, lagCount) + 1)
      .reduce(
        (total, weight, lag) =>
          total + Math.max(values[index - lag] ?? 0, 0) * weight,
        0,
      ),
  );
}

export function hill(values: number[], shape: number): number[] {
  const positive = values.filter((value) => value > 0).sort((a, b) => a - b);
  const half = positive[Math.floor(positive.length / 2)] || 1;
  return values.map((value) => {
    const powered = Math.max(value, 0) ** shape;
    return powered / (powered + half ** shape || 1);
  });
}

export function normalise(values: number[]): number[] {
  const average = mean(values);
  const scale = std(values);
  return values.map((value) => (value - average) / scale);
}

export function rmse(actual: number[], predicted: number[]): number {
  return Math.sqrt(
    mean(actual.map((value, index) => (value - predicted[index]) ** 2)),
  );
}

export function mape(actual: number[], predicted: number[]): number {
  return (
    mean(
      actual.map(
        (value, index) =>
          Math.abs(value - predicted[index]) / Math.max(Math.abs(value), 1),
      ),
    ) * 100
  );
}

export function rSquared(actual: number[], predicted: number[]): number {
  const average = mean(actual);
  const total = actual.reduce((sum, value) => sum + (value - average) ** 2, 0);
  const residual = actual.reduce(
    (sum, value, index) => sum + (value - predicted[index]) ** 2,
    0,
  );
  return 1 - residual / Math.max(total, 1);
}
