import { toNumber } from "../csv";
import { activeIndustryPrior } from "../benchmarks";
import { experimentResponseContrast } from "../experiment-window";
import {
  diagonalPenalty,
  matrixVector,
  mean,
  solveLeastSquares,
  std,
} from "../math";
import {
  carryoverForResponse,
  positiveQuantile,
  responseForChannel,
} from "../response";
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
  fixedHalf?: number,
): number[] {
  const half =
    fixedHalf ??
    positiveQuantile(
      trainIndexes.map((index) => values[index]),
      0.5,
    );
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
  const mediaVectors = spendVectors.map((values, index) => {
    const response = responseForChannel(
      config,
      dataset.mediaColumns[index],
    );
    const carried = carryoverForResponse(values, response);
    const half = positiveQuantile(
      trainIndexes.map((rowIndex) => carried[rowIndex]),
      response.halfSaturationQuantile,
    );
    return hillFromTraining(
      carried,
      response.saturation,
      trainIndexes,
      half,
    );
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
      spec.validationOptions.industryPriorOverrides,
    );
    if (!evidence.length && !benchmark) return;
    const indexes = design.mediaIndexes[mediaIndex];
    const usableEvidence = evidence.flatMap((experiment) => {
      const contrast = experimentResponseContrast(
        dataset,
        spec.config,
        channel,
        experiment,
      );
      if (
        !contrast ||
        !contrast.spendRows.every((rowIndex) => allowed.has(rowIndex)) ||
        !contrast.outcomeRows.every((rowIndex) => allowed.has(rowIndex))
      ) {
        return [];
      }
      return [{ experiment, contrast }];
    });
    if (
      usableEvidence.length &&
      spec.kind === "advanced" &&
      spec.advancedConfig.calibrationMode === "likelihood"
    ) {
      usableEvidence.forEach(({ experiment, contrast }) => {
        const spend = contrast.spend;
        if (spend <= 0) return;
        const h = indexes.map((_, knotIndex) =>
          contrast.outcomeRows.reduce(
            (total, rowIndex) =>
              total +
              (contrast.deltaTransformed[rowIndex] ?? 0) *
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
    let priorMean: number;
    let priorSd: number;
    if (usableEvidence.length) {
      const mapped = usableEvidence.flatMap(({ experiment, contrast }) => {
        const transformedContrast = contrast.outcomeRows.reduce(
          (total, rowIndex) =>
            total + (contrast.deltaTransformed[rowIndex] ?? 0),
          0,
        );
        const roiScale =
          (transformedContrast * effectMultiplier) /
          Math.max(contrast.spend, 1);
        if (roiScale <= 1e-12) return [];
        const roi =
          experiment.incrementalOutcome /
          Math.max(experiment.incrementalSpend, 1);
        return [{
          mean: roi / roiScale,
          standardDeviation:
            Math.max(experiment.standardError, Math.abs(roi) * 0.12) /
            roiScale,
        }];
      });
      if (!mapped.length) return;
      const mappedWeights = mapped.map(
        (item) => 1 / Math.max(item.standardDeviation ** 2, 1e-9),
      );
      const mappedWeight = mappedWeights.reduce(
        (total, value) => total + value,
        0,
      );
      priorMean = mapped.reduce(
        (total, item, index) => total + item.mean * mappedWeights[index],
        0,
      ) / mappedWeight;
      priorSd = Math.sqrt(1 / mappedWeight);
    } else {
      if (evidence.length || !benchmark) return;
      const pooledRoi = benchmark.median;
      const pooledSe = benchmark.standardDeviation;
      priorMean =
        (pooledRoi * spend) /
        Math.max(transformed * effectMultiplier, 1e-9);
      priorSd =
        (Math.max(pooledSe, Math.abs(pooledRoi) * 0.12) * spend) /
        Math.max(transformed * effectMultiplier, 1e-9);
    }
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
