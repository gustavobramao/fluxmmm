import { toNumber } from "../csv";
import { activeIndustryPrior } from "../benchmarks";
import {
  adstock,
  diagonalPenalty,
  matrixVector,
  mean,
  solveLeastSquares,
  std,
  weibullAdstock,
} from "../math";
import type { LeastSquaresPenalty } from "../math";
import type {
  AdvancedModelConfig,
  Dataset,
  Experiment,
  ModelConfig,
} from "../types";
import type { ValidationModelKind } from "./types";
import type { ValidationOptions } from "./types";

export interface ValidationModelSpec {
  kind: ValidationModelKind;
  config: ModelConfig;
  advancedConfig: AdvancedModelConfig;
  experiments: Experiment[];
  validationOptions: ValidationOptions;
}

export interface FoldPrediction {
  actual: number[];
  predicted: number[];
  lower: number[];
  upper: number[];
  trainActual: number[];
  trainPredicted: number[];
}

function scaleFromTraining(values: number[], trainIndexes: number[]): number[] {
  const training = trainIndexes.map((index) => values[index]);
  const average = mean(training);
  const scale = std(training);
  return values.map((value) => (value - average) / scale);
}

function hillFromTraining(
  values: number[],
  shape: number,
  trainIndexes: number[],
): number[] {
  const positive = trainIndexes
    .map((index) => values[index])
    .filter((value) => value > 0)
    .sort((a, b) => a - b);
  const half = positive[Math.floor(positive.length / 2)] || 1;
  return values.map((value) => {
    const powered = Math.max(value, 0) ** shape;
    return powered / Math.max(powered + half ** shape, 1e-9);
  });
}

function kernelWeights(
  rowCount: number,
  trainIndexes: number[],
  knotCount: number,
  bandwidth: number,
): number[][] {
  const first = Math.min(...trainIndexes);
  const last = Math.max(...trainIndexes);
  const span = Math.max(last - first, 1);
  const knots = Array.from(
    { length: Math.max(2, knotCount) },
    (_, index) => index / Math.max(knotCount - 1, 1),
  );
  return Array.from({ length: rowCount }, (_, rowIndex) => {
    const time = (rowIndex - first) / span;
    const raw = knots.map((knot) =>
      Math.exp(-((time - knot) ** 2) / (2 * bandwidth ** 2)),
    );
    const total = raw.reduce((sum, value) => sum + value, 0) || 1;
    return raw.map((value) => value / total);
  });
}

function buildValidationDesign(
  dataset: Dataset,
  spec: ValidationModelSpec,
  trainIndexes: number[],
): {
  matrix: number[][];
  outcome: number[];
  mediaIndexes: number[][];
  mediaVectors: number[][];
  spendVectors: number[][];
  kernel: number[][];
} {
  const { config, advancedConfig } = spec;
  const rowCount = dataset.rows.length;
  const outcome = dataset.rows.map((row) =>
    toNumber(row[dataset.outcomeColumn]),
  );
  const time = Array.from({ length: rowCount }, (_, index) => index);
  const columns: number[][] = [
    Array(rowCount).fill(1),
    scaleFromTraining(time, trainIndexes),
    time.map((index) =>
      Math.sin((2 * Math.PI * index) / config.cyclePeriod),
    ),
    time.map((index) =>
      Math.cos((2 * Math.PI * index) / config.cyclePeriod),
    ),
  ];
  for (let order = 1; order <= config.fourierOrder; order += 1) {
    columns.push(
      time.map((index) =>
        Math.sin((2 * Math.PI * order * index) / dataset.periodsPerYear),
      ),
      time.map((index) =>
        Math.cos((2 * Math.PI * order * index) / dataset.periodsPerYear),
      ),
    );
  }
  dataset.controlColumns.slice(0, 4).forEach((column) => {
    const values = dataset.rows.map((row) => toNumber(row[column]));
    if (std(trainIndexes.map((index) => values[index])) > 1e-9) {
      columns.push(scaleFromTraining(values, trainIndexes));
    }
  });

  const spendVectors = dataset.mediaColumns.map((column) =>
    dataset.rows.map((row) => Math.max(0, toNumber(row[column]))),
  );
  const mediaVectors = spendVectors.map((values) => {
    const carried =
      config.adstockType === "weibull"
        ? weibullAdstock(values, config.weibullShape, config.weibullScale)
        : adstock(values, config.adstock);
    return hillFromTraining(carried, config.saturation, trainIndexes);
  });
  if (spec.kind === "advanced" && advancedConfig.planningIntensity) {
    const shared = dataset.rows.map((_, rowIndex) =>
      mean(
        spendVectors.map((values) =>
          Math.log1p(Math.max(values[rowIndex], 0)),
        ),
      ),
    );
    columns.push(scaleFromTraining(shared, trainIndexes));
  }
  const kernel =
    spec.kind === "advanced" && advancedConfig.timeVarying
      ? kernelWeights(
          rowCount,
          trainIndexes,
          advancedConfig.kernelKnots,
          advancedConfig.kernelBandwidth,
        )
      : Array.from({ length: rowCount }, () => [1]);
  const mediaIndexes: number[][] = [];
  mediaVectors.forEach((media) => {
    const indexes: number[] = [];
    kernel[0].forEach((_, knotIndex) => {
      indexes.push(columns.length);
      columns.push(
        media.map(
          (value, rowIndex) => value * kernel[rowIndex][knotIndex],
        ),
      );
    });
    mediaIndexes.push(indexes);
  });
  return {
    matrix: Array.from({ length: rowCount }, (_, rowIndex) =>
      columns.map((column) => column[rowIndex]),
    ),
    outcome,
    mediaIndexes,
    mediaVectors,
    spendVectors,
    kernel,
  };
}

function dateIndexesForExperiment(
  dataset: Dataset,
  experiment: Experiment,
  allowed: Set<number>,
): number[] {
  const start = Date.parse(experiment.startDate);
  const end = Date.parse(experiment.endDate);
  return dataset.rows
    .map((row, index) => ({
      index,
      date: Date.parse(String(row[dataset.dateColumn])),
    }))
    .filter(
      ({ index, date }) =>
        allowed.has(index) &&
        Number.isFinite(date) &&
        (!Number.isFinite(start) || date >= start) &&
        (!Number.isFinite(end) || date <= end),
    )
    .map(({ index }) => index);
}

function applyExperimentEvidence(
  dataset: Dataset,
  spec: ValidationModelSpec,
  design: ReturnType<typeof buildValidationDesign>,
  trainIndexes: number[],
  variance: number,
  penalties: LeastSquaresPenalty[],
): void {
  if (spec.kind === "frequentist") return;
  const allowed = new Set(trainIndexes);
  const logOutcome =
    spec.kind === "advanced" &&
    spec.advancedConfig.likelihoodDistribution === "log-normal";
  const effectMultiplier = logOutcome ? Math.max(mean(design.outcome), 1) : 1;

  dataset.mediaColumns.forEach((channel, mediaIndex) => {
    const evidence = spec.experiments.filter(
      (experiment) =>
        experiment.channel.toLowerCase() === channel.toLowerCase(),
    );
    const benchmark = activeIndustryPrior(
      channel,
      spec.experiments,
      spec.validationOptions.industryPriorChannels ?? [],
    );
    if (!evidence.length && !benchmark) return;
    const indexes = design.mediaIndexes[mediaIndex];
    if (
      evidence.length &&
      spec.kind === "advanced" &&
      spec.advancedConfig.calibrationMode === "likelihood"
    ) {
      evidence.forEach((experiment) => {
        const rows = dateIndexesForExperiment(dataset, experiment, allowed);
        if (!rows.length) return;
        const spend = rows.reduce(
          (total, rowIndex) =>
            total + design.spendVectors[mediaIndex][rowIndex],
          0,
        );
        if (spend <= 0) return;
        const h = indexes.map((_, knotIndex) =>
          rows.reduce(
            (total, rowIndex) =>
              total +
              design.mediaVectors[mediaIndex][rowIndex] *
                design.kernel[rowIndex][knotIndex] *
                effectMultiplier,
            0,
          ) / spend,
        );
        const roi =
          experiment.incrementalOutcome /
          Math.max(experiment.incrementalSpend, 1);
        const precision =
          variance / Math.max(experiment.standardError ** 2, 1e-6);
        const coefficients = Array(design.matrix[0].length).fill(0);
        indexes.forEach((coefficientIndex, index) => {
          coefficients[coefficientIndex] = h[index];
        });
        penalties.push({ coefficients, target: roi, precision });
      });
      return;
    }

    const weights = evidence.map(
      (experiment) => 1 / Math.max(experiment.standardError ** 2, 1e-6),
    );
    const totalWeight = weights.reduce((total, value) => total + value, 0);
    const pooledRoi = evidence.length
      ? evidence.reduce(
          (total, experiment, index) =>
            total +
            (experiment.incrementalOutcome /
              Math.max(experiment.incrementalSpend, 1)) *
              weights[index],
          0,
        ) / totalWeight
      : benchmark?.median ?? 0;
    const pooledSe = evidence.length
      ? Math.sqrt(1 / totalWeight)
      : benchmark?.standardDeviation ?? pooledRoi * 2;
    const spend = trainIndexes.reduce(
      (total, rowIndex) =>
        total + design.spendVectors[mediaIndex][rowIndex],
      0,
    );
    const transformed = trainIndexes.reduce(
      (total, rowIndex) =>
        total + design.mediaVectors[mediaIndex][rowIndex],
      0,
    );
    const priorMean =
      (pooledRoi * spend) /
      Math.max(transformed * effectMultiplier, 1e-9);
    const priorSd =
      (Math.max(pooledSe, Math.abs(pooledRoi) * 0.12) * spend) /
      Math.max(transformed * effectMultiplier, 1e-9);
    const precision =
      variance / Math.max(priorSd ** 2 * indexes.length, 1e-9);
    indexes.forEach((coefficientIndex) => {
      penalties.push(
        diagonalPenalty(
          design.matrix[0].length,
          coefficientIndex,
          precision,
          priorMean,
        ),
      );
    });
  });
}

export function fitValidationFold(
  dataset: Dataset,
  spec: ValidationModelSpec,
  trainIndexes: number[],
  testIndexes: number[],
): FoldPrediction {
  const design = buildValidationDesign(dataset, spec, trainIndexes);
  const logOutcome =
    spec.kind === "advanced" &&
    spec.advancedConfig.likelihoodDistribution === "log-normal";
  const robust =
    spec.kind === "advanced" &&
    spec.advancedConfig.likelihoodDistribution === "student-t";
  const target = logOutcome
    ? design.outcome.map((value) => Math.log(Math.max(value, 1e-6)))
    : design.outcome;
  let weights = Array(trainIndexes.length).fill(1);
  let coefficients: number[] = [];
  for (let iteration = 0; iteration < (robust ? 4 : 1); iteration += 1) {
    const trainingMatrix = trainIndexes.map((index) => design.matrix[index]);
    const trainingTarget = trainIndexes.map((index) => target[index]);
    const columnCount = design.matrix[0]?.length ?? 0;
    const penalties = Array.from(
      { length: Math.max(columnCount - 1, 0) },
      (_, offset) =>
        diagonalPenalty(columnCount, offset + 1, spec.config.ridge),
    );
    const preliminary = solveLeastSquares(trainingMatrix, trainingTarget, {
      weights,
      penalties,
    }).coefficients;
    design.mediaIndexes.flat().forEach((index) => {
      preliminary[index] = Math.max(0, preliminary[index]);
    });
    const preliminaryPrediction = matrixVector(design.matrix, preliminary);
    const variance =
      trainIndexes.reduce(
        (total, rowIndex) =>
          total + (target[rowIndex] - preliminaryPrediction[rowIndex]) ** 2,
        0,
      ) / Math.max(trainIndexes.length - preliminary.length, 1);
    applyExperimentEvidence(
      dataset,
      spec,
      design,
      trainIndexes,
      variance,
      penalties,
    );
    coefficients = solveLeastSquares(trainingMatrix, trainingTarget, {
      weights,
      penalties,
    }).coefficients;
    design.mediaIndexes.flat().forEach((index) => {
      coefficients[index] = Math.max(0, coefficients[index]);
    });
    if (robust) {
      const predicted = matrixVector(design.matrix, coefficients);
      const residuals = trainIndexes.map(
        (rowIndex) => target[rowIndex] - predicted[rowIndex],
      );
      const scale = std(residuals);
      const nu = Math.max(spec.advancedConfig.studentTDegreesFreedom, 2.1);
      weights = residuals.map((residual) => {
        const standardized = residual / scale;
        return (nu + 1) / (nu + standardized ** 2);
      });
    }
  }
  const linear = matrixVector(design.matrix, coefficients);
  let allPredicted = linear;
  if (logOutcome) {
    const smearing = mean(
      trainIndexes.map((rowIndex) =>
        Math.exp(target[rowIndex] - linear[rowIndex]),
      ),
    );
    allPredicted = linear.map((value) => Math.exp(value) * smearing);
  }
  const trainResiduals = trainIndexes.map(
    (rowIndex) => design.outcome[rowIndex] - allPredicted[rowIndex],
  );
  const residualScale = std(trainResiduals);
  return {
    actual: testIndexes.map((index) => design.outcome[index]),
    predicted: testIndexes.map((index) => allPredicted[index]),
    lower: testIndexes.map((index) =>
      Math.max(0, allPredicted[index] - 1.96 * residualScale),
    ),
    upper: testIndexes.map((index) =>
      allPredicted[index] + 1.96 * residualScale,
    ),
    trainActual: trainIndexes.map((index) => design.outcome[index]),
    trainPredicted: trainIndexes.map((index) => allPredicted[index]),
  };
}
