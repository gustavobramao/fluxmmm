import { sha256, toNumber } from "./csv";
import {
  diagonalPenalty,
  diagnoseDesignMatrix,
  mape,
  matrixVector,
  mean,
  multiply,
  normalise,
  rSquared,
  rmse,
  solveLeastSquares,
  std,
  transpose,
} from "./math";
import { responseForChannel, responseTransform } from "./response";
import type { LeastSquaresPenalty } from "./math";
import {
  activeIndustryPrior,
  INDUSTRY_BENCHMARK_VERSION,
} from "./benchmarks";
import type { IndustryPriorSelection } from "./benchmarks";
import type {
  AdvancedModelConfig,
  ChannelEstimate,
  Dataset,
  Experiment,
  ModelCalibrationReceipt,
  ModelConfig,
  ModelResult,
  NumericalStabilityDiagnostics,
} from "./types";

const ADVANCED_MODEL_VERSION =
  "flux-mmm-advanced-v2.0-channel-response-contracts";

interface AdvancedDesign {
  matrix: number[][];
  outcome: number[];
  baselineCount: number;
  mediaVectors: number[][];
  spendVectors: number[][];
  channelCoefficientIndexes: number[][];
  kernelWeights: number[][];
  planningIntensity: number[];
}

export interface AdvancedSamplingDesign {
  matrix: number[][];
  outcome: number[];
  target: number[];
  baselineCount: number;
  mediaVectors: number[][];
  spendVectors: number[][];
  channelCoefficientIndexes: number[][];
  kernelWeights: number[][];
  planningIntensity: number[];
  effectMultiplier: number;
}

export function logNormalLaplaceFromMedian(
  median: number,
  low: number,
  high: number,
): { median: number; mode: number; localSd: number; logSigma: number } {
  const safeMedian = Math.max(median, 1e-9);
  const logSigma = Math.max(
    (Math.log(Math.max(high, 1e-9)) -
      Math.log(Math.max(low, 1e-9))) /
      (2 * 1.281551565545),
    0.05,
  );
  const mode = safeMedian * Math.exp(-(logSigma ** 2));
  return {
    median: safeMedian,
    mode,
    localSd: Math.max(mode * logSigma, mode * 0.05),
    logSigma,
  };
}

function smooth(values: number[], radius = 2): number[] {
  return values.map((_, index) => {
    const start = Math.max(0, index - radius);
    const end = Math.min(values.length, index + radius + 1);
    return mean(values.slice(start, end));
  });
}

function planningFactor(spendVectors: number[][]): {
  modelValues: number[];
  displayValues: number[];
} {
  if (!spendVectors.length || !spendVectors[0]?.length) {
    return { modelValues: [], displayValues: [] };
  }

  const rowCount = spendVectors[0].length;
  const standardised = spendVectors.map((values) =>
    normalise(values.map((value) => Math.log1p(Math.max(value, 0)))),
  );
  const observations = Array.from({ length: rowCount }, (_, rowIndex) =>
    standardised.map((channel) => channel[rowIndex]),
  );
  const covariance = multiply(transpose(observations), observations).map(
    (row) => row.map((value) => value / Math.max(rowCount, 1)),
  );

  let loading = Array(standardised.length).fill(
    1 / Math.sqrt(standardised.length),
  );
  for (let iteration = 0; iteration < 40; iteration += 1) {
    const next = matrixVector(covariance, loading);
    const magnitude =
      Math.sqrt(next.reduce((total, value) => total + value ** 2, 0)) || 1;
    loading = next.map((value) => value / magnitude);
  }

  let scores = observations.map((row) =>
    row.reduce(
      (total, value, index) => total + value * loading[index],
      0,
    ),
  );
  const aggregate = observations.map((row) => mean(row));
  const orientation = scores.reduce(
    (total, value, index) => total + value * aggregate[index],
    0,
  );
  if (orientation < 0) scores = scores.map((value) => -value);

  const smoothed = smooth(scores);
  const minimum = Math.min(...smoothed);
  const maximum = Math.max(...smoothed);
  const displayValues = smoothed.map(
    (value) => ((value - minimum) / Math.max(maximum - minimum, 1e-9)) * 100,
  );
  return { modelValues: normalise(smoothed), displayValues };
}

function gaussianKernel(
  rowCount: number,
  knotCount: number,
  bandwidth: number,
): number[][] {
  const safeKnotCount = Math.max(2, Math.round(knotCount));
  const safeBandwidth = Math.max(0.04, bandwidth);
  const knots = Array.from(
    { length: safeKnotCount },
    (_, index) => index / Math.max(safeKnotCount - 1, 1),
  );

  return Array.from({ length: rowCount }, (_, rowIndex) => {
    const time = rowIndex / Math.max(rowCount - 1, 1);
    const raw = knots.map((knot) =>
      Math.exp(-((time - knot) ** 2) / (2 * safeBandwidth ** 2)),
    );
    const total = raw.reduce((sum, value) => sum + value, 0) || 1;
    return raw.map((value) => value / total);
  });
}

function transformedMedia(
  dataset: Dataset,
  config: ModelConfig,
): { mediaVectors: number[][]; spendVectors: number[][] } {
  const spendVectors = dataset.mediaColumns.map((column) =>
    dataset.rows.map((row) => Math.max(0, toNumber(row[column]))),
  );
  const mediaVectors = spendVectors.map((values, index) =>
    responseTransform(
      values,
      responseForChannel(config, dataset.mediaColumns[index]),
    ).transformed,
  );
  return { mediaVectors, spendVectors };
}

function buildAdvancedDesign(
  dataset: Dataset,
  config: ModelConfig,
  advancedConfig: AdvancedModelConfig,
): AdvancedDesign {
  const rowCount = dataset.rows.length;
  const outcome = dataset.rows.map((row) =>
    toNumber(row[dataset.outcomeColumn]),
  );
  const columns: number[][] = [
    Array(rowCount).fill(1),
    normalise(Array.from({ length: rowCount }, (_, index) => index)),
    Array.from({ length: rowCount }, (_, index) =>
      Math.sin((2 * Math.PI * index) / config.cyclePeriod),
    ),
    Array.from({ length: rowCount }, (_, index) =>
      Math.cos((2 * Math.PI * index) / config.cyclePeriod),
    ),
  ];

  for (let order = 1; order <= config.fourierOrder; order += 1) {
    columns.push(
      Array.from({ length: rowCount }, (_, index) =>
        Math.sin((2 * Math.PI * order * index) / dataset.periodsPerYear),
      ),
      Array.from({ length: rowCount }, (_, index) =>
        Math.cos((2 * Math.PI * order * index) / dataset.periodsPerYear),
      ),
    );
  }

  dataset.controlColumns.slice(0, 4).forEach((column) => {
    const values = dataset.rows.map((row) => toNumber(row[column]));
    if (std(values) > 1e-9) columns.push(normalise(values));
  });

  const { mediaVectors, spendVectors } = transformedMedia(dataset, config);
  const intensity = planningFactor(spendVectors);
  if (advancedConfig.planningIntensity && intensity.modelValues.length) {
    columns.push(intensity.modelValues);
  }
  const baselineCount = columns.length;
  const kernelWeights = advancedConfig.timeVarying
    ? gaussianKernel(
        rowCount,
        advancedConfig.kernelKnots,
        advancedConfig.kernelBandwidth,
      )
    : Array.from({ length: rowCount }, () => [1]);
  const channelCoefficientIndexes: number[][] = [];

  mediaVectors.forEach((media) => {
    const indexes: number[] = [];
    for (
      let knotIndex = 0;
      knotIndex < kernelWeights[0].length;
      knotIndex += 1
    ) {
      indexes.push(columns.length);
      columns.push(
        media.map(
          (value, rowIndex) => value * kernelWeights[rowIndex][knotIndex],
        ),
      );
    }
    channelCoefficientIndexes.push(indexes);
  });

  return {
    matrix: transpose(columns),
    outcome,
    baselineCount,
    mediaVectors,
    spendVectors,
    channelCoefficientIndexes,
    kernelWeights,
    planningIntensity: intensity.displayValues,
  };
}

function matchingExperiments(
  dataset: Dataset,
  experiments: Experiment[],
  channel: string,
): { experiment: Experiment; rows: number[] }[] {
  return experiments
    .filter(
      (experiment) =>
        experiment.channel.toLowerCase() === channel.toLowerCase(),
    )
    .map((experiment) => {
      const start = Date.parse(experiment.startDate);
      const end = Date.parse(experiment.endDate);
      const rows = dataset.rows
        .map((row, index) => ({
          index,
          date: Date.parse(String(row[dataset.dateColumn])),
        }))
        .filter(
          ({ date }) =>
            Number.isFinite(date) &&
            (!Number.isFinite(start) || date >= start) &&
            (!Number.isFinite(end) || date <= end),
        )
        .map(({ index }) => index);
      return {
        experiment,
        rows: rows.length
          ? rows
          : Array.from({ length: dataset.rows.length }, (_, index) => index),
      };
    });
}

function initialVariance(
  design: AdvancedDesign,
  target: number[],
  weights: number[],
  penalties: LeastSquaresPenalty[],
): number {
  const coefficients = solveLeastSquares(design.matrix, target, {
    weights,
    penalties,
  }).coefficients;
  design.channelCoefficientIndexes.flat().forEach((index) => {
    coefficients[index] = Math.max(0, coefficients[index]);
  });
  const predicted = matrixVector(design.matrix, coefficients);
  return (
    target.reduce(
      (total, value, index) =>
        total + (value - predicted[index]) ** 2,
      0,
    ) / Math.max(target.length - coefficients.length, 1)
  );
}

function applyPriorCalibration(
  dataset: Dataset,
  design: AdvancedDesign,
  experiments: Experiment[],
  likelihoodVariance: number,
  effectMultiplier: number,
  priorDistribution: AdvancedModelConfig["priorDistribution"],
  penalties: LeastSquaresPenalty[],
  priorPrecisions: Map<number, number>,
): number {
  let calibrated = 0;
  dataset.mediaColumns.forEach((channel, mediaIndex) => {
    const evidence = matchingExperiments(dataset, experiments, channel);
    if (!evidence.length) return;
    calibrated += evidence.length;

    const weights = evidence.map(
      ({ experiment }) =>
        1 / Math.max(experiment.standardError ** 2, 1e-6),
    );
    const rois = evidence.map(
      ({ experiment }) =>
        experiment.incrementalOutcome /
        Math.max(experiment.incrementalSpend, 1),
    );
    const totalWeight = weights.reduce((total, value) => total + value, 0);
    const pooledRoi = rois.reduce(
      (total, roi, index) => total + roi * weights[index],
      0,
    ) / totalWeight;
    const pooledSe = Math.sqrt(1 / totalWeight);
    const spend = design.spendVectors[mediaIndex].reduce(
      (total, value) => total + value,
      0,
    );
    const transformed = design.mediaVectors[mediaIndex].reduce(
      (total, value) => total + value,
      0,
    );
    let priorMean =
      (pooledRoi * spend) /
      Math.max(transformed * effectMultiplier, 1e-9);
    let mappedSd =
      (Math.max(pooledSe, Math.abs(pooledRoi) * 0.12) * spend) /
      Math.max(transformed * effectMultiplier, 1e-9);
    if (priorDistribution === "log-normal" && priorMean > 0) {
      const logVariance = Math.log(
        1 + mappedSd ** 2 / Math.max(priorMean ** 2, 1e-9),
      );
      priorMean *= Math.exp(-1.5 * logVariance);
      mappedSd = Math.max(
        priorMean * Math.sqrt(Math.max(logVariance, 1e-6)),
        priorMean * 0.05,
      );
    }
    const indexes = design.channelCoefficientIndexes[mediaIndex];
    const precision =
      likelihoodVariance /
      Math.max(mappedSd ** 2 * indexes.length, 1e-9);

    indexes.forEach((coefficientIndex) => {
      penalties.push(
        diagonalPenalty(
          design.matrix[0].length,
          coefficientIndex,
          precision,
          priorMean,
        ),
      );
      priorPrecisions.set(
        coefficientIndex,
        (priorPrecisions.get(coefficientIndex) ?? 0) + precision,
      );
    });
  });
  return calibrated;
}

function applyIndustryPriorCalibration(
  dataset: Dataset,
  design: AdvancedDesign,
  experiments: Experiment[],
  industryPriorSelection: IndustryPriorSelection,
  likelihoodVariance: number,
  effectMultiplier: number,
  priorDistribution: AdvancedModelConfig["priorDistribution"],
  penalties: LeastSquaresPenalty[],
  priorPrecisions: Map<number, number>,
): number {
  let calibrated = 0;
  dataset.mediaColumns.forEach((channel, mediaIndex) => {
    const benchmark = activeIndustryPrior(
      channel,
      experiments,
      industryPriorSelection,
    );
    if (!benchmark) return;
    calibrated += 1;

    const spend = design.spendVectors[mediaIndex].reduce(
      (total, value) => total + value,
      0,
    );
    const transformed = design.mediaVectors[mediaIndex].reduce(
      (total, value) => total + value,
      0,
    );
    const coefficientScale =
      spend / Math.max(transformed * effectMultiplier, 1e-9);
    let priorMean = benchmark.median * coefficientScale;
    let mappedSd =
      Math.max(benchmark.standardDeviation, benchmark.median * 0.25) *
      coefficientScale;
    if (priorDistribution === "log-normal" && priorMean > 0) {
      // Industry tables declare a median and an 80% range. Preserve that
      // contract in log space, then use the true log-normal mode and local
      // curvature for the fast Laplace penalty. The previous mean-based
      // conversion shifted a declared median down by exp(-1.5 sigma²).
      const approximation = logNormalLaplaceFromMedian(
        benchmark.median,
        benchmark.low,
        benchmark.high,
      );
      priorMean = approximation.mode * coefficientScale;
      mappedSd = approximation.localSd * coefficientScale;
    }
    const indexes = design.channelCoefficientIndexes[mediaIndex];
    const precision =
      likelihoodVariance /
      Math.max(mappedSd ** 2 * indexes.length, 1e-9);

    indexes.forEach((coefficientIndex) => {
      penalties.push(
        diagonalPenalty(
          design.matrix[0].length,
          coefficientIndex,
          precision,
          priorMean,
        ),
      );
      priorPrecisions.set(
        coefficientIndex,
        (priorPrecisions.get(coefficientIndex) ?? 0) + precision,
      );
    });
  });
  return calibrated;
}

function applyLikelihoodCalibration(
  dataset: Dataset,
  design: AdvancedDesign,
  experiments: Experiment[],
  likelihoodVariance: number,
  effectMultiplier: number,
  penalties: LeastSquaresPenalty[],
): number {
  let calibrated = 0;
  dataset.mediaColumns.forEach((channel, mediaIndex) => {
    const evidence = matchingExperiments(dataset, experiments, channel);
    evidence.forEach(({ experiment, rows }) => {
      const spend = rows.reduce(
        (total, rowIndex) =>
          total + design.spendVectors[mediaIndex][rowIndex],
        0,
      );
      if (spend <= 0) return;
      calibrated += 1;

      const indexes = design.channelCoefficientIndexes[mediaIndex];
      const h = indexes.map((_, knotIndex) =>
        rows.reduce(
          (total, rowIndex) =>
            total +
            design.mediaVectors[mediaIndex][rowIndex] *
              design.kernelWeights[rowIndex][knotIndex] *
              effectMultiplier,
          0,
        ) / spend,
      );
      const observedRoi =
        experiment.incrementalOutcome /
        Math.max(experiment.incrementalSpend, 1);
      const precision =
        likelihoodVariance /
        Math.max(experiment.standardError ** 2, 1e-6);

      const coefficients = Array(design.matrix[0].length).fill(0);
      indexes.forEach((coefficientIndex, index) => {
        coefficients[coefficientIndex] = h[index];
      });
      penalties.push({ coefficients, target: observedRoi, precision });
    });
  });
  return calibrated;
}

function coefficientPath(
  design: AdvancedDesign,
  coefficients: number[],
  mediaIndex: number,
): number[] {
  const indexes = design.channelCoefficientIndexes[mediaIndex];
  return design.kernelWeights.map((weights) =>
    indexes.reduce(
      (total, coefficientIndex, knotIndex) =>
        total + coefficients[coefficientIndex] * weights[knotIndex],
      0,
    ),
  );
}

function channelEstimates(
  dataset: Dataset,
  design: AdvancedDesign,
  coefficients: number[],
  covariance: number[][],
  experiments: Experiment[],
  industryPriorSelection: IndustryPriorSelection,
  contributionVectors?: number[][],
): ChannelEstimate[] {
  const estimates = dataset.mediaColumns.map((channel, mediaIndex) => {
    const indexes = design.channelCoefficientIndexes[mediaIndex];
    const path = coefficientPath(design, coefficients, mediaIndex);
    const linearContribution = design.mediaVectors[mediaIndex].reduce(
      (total, value, rowIndex) => total + value * path[rowIndex],
      0,
    );
    const contribution = contributionVectors
      ? contributionVectors[mediaIndex].reduce(
          (total, value) => total + value,
          0,
        )
      : linearContribution;
    const spend = design.spendVectors[mediaIndex].reduce(
      (total, value) => total + value,
      0,
    );
    const contributionGradient = indexes.map((_, knotIndex) =>
      design.mediaVectors[mediaIndex].reduce(
        (total, value, rowIndex) =>
          total + value * design.kernelWeights[rowIndex][knotIndex],
        0,
      ),
    );
    const contributionVariance = contributionGradient.reduce(
      (outerTotal, outerValue, outerIndex) =>
        outerTotal +
        contributionGradient.reduce(
          (innerTotal, innerValue, innerIndex) =>
            innerTotal +
            outerValue *
              (covariance[indexes[outerIndex]]?.[indexes[innerIndex]] ?? 0) *
              innerValue,
          0,
        ),
      0,
    );
    const roi = contribution / Math.max(spend, 1);
    const linearSe = Math.sqrt(Math.max(contributionVariance, 0));
    const roiSe = contributionVectors
      ? (Math.abs(roi) * linearSe) /
        Math.max(Math.abs(linearContribution), 1e-6)
      : linearSe / Math.max(spend, 1);
    const evidence = experiments.filter(
      (experiment) =>
        experiment.channel.toLowerCase() === channel.toLowerCase(),
    );
    const priorRoi = evidence.length
      ? mean(
          evidence.map(
            (experiment) =>
              experiment.incrementalOutcome /
              Math.max(experiment.incrementalSpend, 1),
          ),
        )
      : undefined;
    const benchmark = activeIndustryPrior(
      channel,
      experiments,
      industryPriorSelection,
    );

    return {
      channel,
      contribution,
      contributionShare: 0,
      roi,
      roiLow: Math.max(0, roi - 1.96 * roiSe),
      roiHigh: Math.max(0, roi + 1.96 * roiSe),
      coefficient: mean(path),
      priorRoi: priorRoi ?? benchmark?.median,
      priorSource: priorRoi === undefined && benchmark
        ? "industry" as const
        : priorRoi === undefined
          ? undefined
          : "experiment" as const,
      priorLabel:
        priorRoi === undefined
          ? benchmark?.label
          : evidence.length === 1
            ? evidence[0].source
            : `${evidence.length} pooled experiments`,
    };
  });
  const totalContribution = estimates.reduce(
    (total, estimate) => total + Math.max(estimate.contribution, 0),
    0,
  );
  return estimates
    .map((estimate) => ({
      ...estimate,
      contributionShare:
        estimate.contribution / Math.max(totalContribution, 1),
    }))
    .sort((a, b) => b.contribution - a.contribution);
}

function comparableRoiEstimate(
  design: AdvancedDesign,
  coefficients: number[],
  covariance: number[][],
  mediaIndex: number,
  rows: number[],
  contributionVectors?: number[][],
): { roi: number; low: number; high: number } {
  const indexes = design.channelCoefficientIndexes[mediaIndex];
  const path = coefficientPath(design, coefficients, mediaIndex);
  const spend = rows.reduce(
    (total, rowIndex) =>
      total + Math.max(0, design.spendVectors[mediaIndex][rowIndex] ?? 0),
    0,
  );
  const linearContribution = rows.reduce(
    (total, rowIndex) =>
      total +
      (design.mediaVectors[mediaIndex][rowIndex] ?? 0) *
        (path[rowIndex] ?? 0),
    0,
  );
  const contribution = contributionVectors
    ? rows.reduce(
        (total, rowIndex) =>
          total + (contributionVectors[mediaIndex][rowIndex] ?? 0),
        0,
      )
    : linearContribution;
  const gradient = indexes.map((_, knotIndex) =>
    rows.reduce(
      (total, rowIndex) =>
        total +
        (design.mediaVectors[mediaIndex][rowIndex] ?? 0) *
          (design.kernelWeights[rowIndex][knotIndex] ?? 0),
      0,
    ),
  );
  const contributionVariance = gradient.reduce(
    (outerTotal, outerValue, outerIndex) =>
      outerTotal +
      gradient.reduce(
        (innerTotal, innerValue, innerIndex) =>
          innerTotal +
          outerValue *
            (covariance[indexes[outerIndex]]?.[indexes[innerIndex]] ?? 0) *
            innerValue,
        0,
      ),
    0,
  );
  const roi = contribution / Math.max(spend, 1);
  const linearSe = Math.sqrt(Math.max(contributionVariance, 0));
  const roiSe = contributionVectors
    ? (Math.abs(roi) * linearSe) /
      Math.max(Math.abs(linearContribution), 1e-6)
    : linearSe / Math.max(spend, 1);
  return {
    roi,
    low: Math.max(0, roi - 1.96 * roiSe),
    high: Math.max(0, roi + 1.96 * roiSe),
  };
}

function calibrationReceipts(
  dataset: Dataset,
  design: AdvancedDesign,
  advancedConfig: AdvancedModelConfig,
  coefficients: number[],
  covariance: number[][],
  channels: ChannelEstimate[],
  experiments: Experiment[],
  industryPriorSelection: IndustryPriorSelection,
  contributionVectors?: number[][],
): ModelCalibrationReceipt[] {
  const channelEstimate = new Map(
    channels.map((channel) => [channel.channel.toLowerCase(), channel]),
  );
  const receipts: ModelCalibrationReceipt[] = [];
  dataset.mediaColumns.forEach((channel, mediaIndex) => {
    const evidence = matchingExperiments(dataset, experiments, channel);
    evidence.forEach(({ experiment, rows }) => {
      const comparable = comparableRoiEstimate(
        design,
        coefficients,
        covariance,
        mediaIndex,
        rows,
        contributionVectors,
      );
      const targetRoi =
        experiment.incrementalOutcome /
        Math.max(experiment.incrementalSpend, 1);
      receipts.push({
        channel,
        source: "experiment",
        route: advancedConfig.calibrationMode,
        comparisonBasis: "experiment-window",
        evidenceLabel: experiment.source,
        targetRoi,
        targetLow: Math.max(0, targetRoi - 1.96 * experiment.standardError),
        targetHigh: targetRoi + 1.96 * experiment.standardError,
        modelRoi: comparable.roi,
        modelLow: comparable.low,
        modelHigh: comparable.high,
        startDate: experiment.startDate,
        endDate: experiment.endDate,
        scope: experiment.scope,
      });
    });

    const benchmark = activeIndustryPrior(
      channel,
      experiments,
      industryPriorSelection,
    );
    const estimate = channelEstimate.get(channel.toLowerCase());
    if (!benchmark || !estimate) return;
    receipts.push({
      channel,
      source: "industry",
      route: "prior",
      comparisonBasis: "full-history",
      evidenceLabel: benchmark.label,
      targetRoi: benchmark.median,
      targetLow: benchmark.low,
      targetHigh: benchmark.high,
      modelRoi: estimate.roi,
      modelLow: estimate.roiLow,
      modelHigh: estimate.roiHigh,
    });
  });
  return receipts;
}

function coefficientPenalties(
  design: AdvancedDesign,
  target: number[],
  config: ModelConfig,
  advancedConfig: AdvancedModelConfig,
  likelihoodVariance: number,
  weights: number[],
  priorPrecisions: Map<number, number>,
  externallyPriorCalibratedMedia: ReadonlySet<number>,
): LeastSquaresPenalty[] {
  const columnCount = design.matrix[0]?.length ?? 0;
  const penalties = Array.from(
    { length: Math.max(columnCount - 1, 0) },
    (_, offset) => {
      const index = offset + 1;
    const isMedia = index >= design.baselineCount;
      return diagonalPenalty(
        columnCount,
        index,
        isMedia && advancedConfig.priorDistribution === "log-normal"
          ? config.ridge * 0.15
          : config.ridge,
      );
    },
  );

  if (advancedConfig.priorDistribution !== "log-normal") return penalties;
  const targetScale = Math.max(std(target), 1e-3);
  const priorMean = Math.max(targetScale * 0.12, 1e-4);
  const priorSd = Math.max(targetScale * 0.35, 1e-4);
  const logVariance = Math.log1p((priorSd / priorMean) ** 2);
  const logSigma = Math.sqrt(logVariance);
  const logMu = Math.log(priorMean) - logVariance / 2;
  const localMode = Math.exp(logMu - logVariance);
  const localSd = Math.max(localMode * logSigma, priorMean * 0.05);
  design.channelCoefficientIndexes.forEach((indexes, mediaIndex) => {
    if (externallyPriorCalibratedMedia.has(mediaIndex)) return;
    indexes.forEach((coefficientIndex) => {
      const precision =
        likelihoodVariance /
        Math.max(localSd ** 2 * indexes.length, 1e-9);
      penalties.push(
        diagonalPenalty(
          columnCount,
          coefficientIndex,
          precision,
          localMode,
        ),
      );
      priorPrecisions.set(
        coefficientIndex,
        (priorPrecisions.get(coefficientIndex) ?? 0) + precision,
      );
    });
  });
  return penalties;
}

function transformedOutcome(
  outcome: number[],
  distribution: AdvancedModelConfig["likelihoodDistribution"],
): number[] {
  return distribution === "log-normal"
    ? outcome.map((value) => Math.log(Math.max(value, 1e-6)))
    : outcome;
}

export function compileAdvancedSamplingDesign(
  dataset: Dataset,
  config: ModelConfig,
  advancedConfig: AdvancedModelConfig,
): AdvancedSamplingDesign {
  const design = buildAdvancedDesign(dataset, config, advancedConfig);
  return {
    ...design,
    target: transformedOutcome(
      design.outcome,
      advancedConfig.likelihoodDistribution,
    ),
    effectMultiplier:
      advancedConfig.likelihoodDistribution === "log-normal"
        ? Math.max(mean(design.outcome), 1)
        : 1,
  };
}

function studentTWeights(
  target: number[],
  predicted: number[],
  degreesFreedom: number,
): number[] {
  const residuals = target.map(
    (value, index) => value - predicted[index],
  );
  const scale = Math.max(rmse(target, predicted), 1e-6);
  const nu = Math.max(degreesFreedom, 2.1);
  return residuals.map((residual) => {
    const standardized = residual / scale;
    return (nu + 1) / (nu + standardized ** 2);
  });
}

export const DEFAULT_ADVANCED_CONFIG: AdvancedModelConfig = {
  timeVarying: true,
  kernelKnots: 6,
  kernelBandwidth: 0.18,
  planningIntensity: false,
  calibrationMode: "prior",
  priorDistribution: "log-normal",
  likelihoodDistribution: "gaussian",
  studentTDegreesFreedom: 4,
};

export async function advancedModelFingerprint(
  dataset: Dataset,
  config: ModelConfig,
  advancedConfig: AdvancedModelConfig,
  experiments: Experiment[],
  industryPriorSelection: IndustryPriorSelection = false,
): Promise<string> {
  return sha256(
    JSON.stringify({
      dataset: dataset.hash,
      config,
      advancedConfig,
      experiments,
      industryPriors: {
        selected:
          industryPriorSelection === true
            ? "all"
            : industryPriorSelection === false
              ? []
              : [...industryPriorSelection].sort(),
        version: INDUSTRY_BENCHMARK_VERSION,
      },
      version: ADVANCED_MODEL_VERSION,
    }),
  );
}

export async function runAdvancedModel(
  dataset: Dataset,
  config: ModelConfig,
  advancedConfig: AdvancedModelConfig,
  experiments: Experiment[],
  fingerprint: string,
  industryPriorSelection: IndustryPriorSelection = false,
): Promise<ModelResult> {
  const design = buildAdvancedDesign(dataset, config, advancedConfig);
  const target = transformedOutcome(
    design.outcome,
    advancedConfig.likelihoodDistribution,
  );
  const effectMultiplier =
    advancedConfig.likelihoodDistribution === "log-normal"
      ? Math.max(mean(design.outcome), 1)
      : 1;
  const iterations =
    advancedConfig.likelihoodDistribution === "student-t" ? 4 : 1;
  let observationWeights = Array(target.length).fill(1);
  let finalCoefficients: number[] = [];
  let finalUnconstrainedCoefficients: number[] = [];
  let finalSolution: ReturnType<typeof solveLeastSquares> | undefined;
  let finalPriorPrecisions = new Map<number, number>();
  let likelihoodVariance = 1;
  let calibratedExperiments = 0;
  let calibratedIndustryPriors = 0;
  const externallyPriorCalibratedMedia = new Set(
    dataset.mediaColumns.flatMap((channel, mediaIndex) => {
      const hasExperimentPrior =
        advancedConfig.calibrationMode === "prior" &&
        matchingExperiments(dataset, experiments, channel).length > 0;
      const hasIndustryPrior = Boolean(
        activeIndustryPrior(
          channel,
          experiments,
          industryPriorSelection,
        ),
      );
      return hasExperimentPrior || hasIndustryPrior ? [mediaIndex] : [];
    }),
  );

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const columnCount = design.matrix[0]?.length ?? 0;
    const preliminaryPenalties = Array.from(
      { length: Math.max(columnCount - 1, 0) },
      (_, offset) =>
        diagonalPenalty(columnCount, offset + 1, config.ridge),
    );
    likelihoodVariance = initialVariance(
      design,
      target,
      observationWeights,
      preliminaryPenalties,
    );
    const priorPrecisions = new Map<number, number>();
    const penalties = coefficientPenalties(
      design,
      target,
      config,
      advancedConfig,
      likelihoodVariance,
      observationWeights,
      priorPrecisions,
      externallyPriorCalibratedMedia,
    );
    calibratedExperiments =
      advancedConfig.calibrationMode === "likelihood"
        ? applyLikelihoodCalibration(
            dataset,
            design,
            experiments,
            likelihoodVariance,
            effectMultiplier,
            penalties,
          )
        : applyPriorCalibration(
            dataset,
            design,
            experiments,
            likelihoodVariance,
            effectMultiplier,
            advancedConfig.priorDistribution,
            penalties,
            priorPrecisions,
          );
    calibratedIndustryPriors = applyIndustryPriorCalibration(
      dataset,
      design,
      experiments,
      industryPriorSelection,
      likelihoodVariance,
      effectMultiplier,
      advancedConfig.priorDistribution,
      penalties,
      priorPrecisions,
    );

    finalSolution = solveLeastSquares(design.matrix, target, {
      weights: observationWeights,
      penalties,
    });
    finalUnconstrainedCoefficients = [...finalSolution.coefficients];
    finalCoefficients = [...finalUnconstrainedCoefficients];
    finalPriorPrecisions = priorPrecisions;
    design.channelCoefficientIndexes.flat().forEach((index) => {
      finalCoefficients[index] = Math.max(0, finalCoefficients[index]);
    });
    if (advancedConfig.likelihoodDistribution === "student-t") {
      observationWeights = studentTWeights(
        target,
        matrixVector(design.matrix, finalCoefficients),
        advancedConfig.studentTDegreesFreedom,
      );
    }
  }

  if (!finalSolution) {
    throw new Error("Advanced model did not produce a numerical solution.");
  }
  const covarianceBase = finalSolution.precisionInverse;
  const coefficients = finalCoefficients;
  const predictedTarget = matrixVector(design.matrix, coefficients);
  const coefficientPaths = dataset.mediaColumns.map((channel, mediaIndex) => ({
    channel,
    values: coefficientPath(design, coefficients, mediaIndex),
  }));
  const mediaByRow = design.outcome.map((_, rowIndex) =>
    coefficientPaths.reduce(
      (total, path, mediaIndex) =>
        total + design.mediaVectors[mediaIndex][rowIndex] * path.values[rowIndex],
      0,
    ),
  );
  let predicted = predictedTarget;
  let baseline = predictedTarget.map(
    (value, rowIndex) => value - mediaByRow[rowIndex],
  );
  let contributionVectors: number[][] | undefined;
  if (advancedConfig.likelihoodDistribution === "log-normal") {
    const smearing = mean(
      target.map((value, rowIndex) =>
        Math.exp(value - predictedTarget[rowIndex]),
      ),
    );
    predicted = predictedTarget.map((value) => Math.exp(value) * smearing);
    baseline = predictedTarget.map(
      (value, rowIndex) =>
        Math.exp(value - mediaByRow[rowIndex]) * smearing,
    );
    contributionVectors = coefficientPaths.map((path, mediaIndex) =>
      design.outcome.map((_, rowIndex) => {
        const totalIncrement = predicted[rowIndex] - baseline[rowIndex];
        const channelEffect =
          design.mediaVectors[mediaIndex][rowIndex] * path.values[rowIndex];
        return mediaByRow[rowIndex] > 1e-9
          ? totalIncrement * (channelEffect / mediaByRow[rowIndex])
          : 0;
      }),
    );
  }
  const residuals = design.outcome.map(
    (value, rowIndex) => value - predicted[rowIndex],
  );
  const targetResiduals = target.map(
    (value, rowIndex) => value - predictedTarget[rowIndex],
  );
  const sigmaSquared =
    targetResiduals.reduce((total, value) => total + value ** 2, 0) /
    Math.max(target.length - coefficients.length, 1);
  const covariance = covarianceBase.map((row) =>
    row.map((value) => value * sigmaSquared),
  );
  const channels = channelEstimates(
    dataset,
    design,
    coefficients,
    covariance,
    experiments,
    industryPriorSelection,
    contributionVectors,
  );
  const calibrationReceipt = calibrationReceipts(
    dataset,
    design,
    advancedConfig,
    coefficients,
    covariance,
    channels,
    experiments,
    industryPriorSelection,
    contributionVectors,
  );
  const channelEstimateByName = new Map(
    channels.map((channel) => [channel.channel, channel]),
  );
  const unconstrainedPrediction = matrixVector(
    design.matrix,
    finalUnconstrainedCoefficients,
  );
  const predictionShiftShare =
    rmse(unconstrainedPrediction, predictedTarget) /
    Math.max(std(target), 1e-9);
  const clippedChannels = dataset.mediaColumns
    .map((channel, mediaIndex) => {
      const indexes = design.channelCoefficientIndexes[mediaIndex];
      const rawValues = indexes.map(
        (index) => finalUnconstrainedCoefficients[index],
      );
      const minimumIndex = rawValues.reduce(
        (minimum, value, index) =>
          value < rawValues[minimum] ? index : minimum,
        0,
      );
      const rawPath = coefficientPath(
        design,
        finalUnconstrainedCoefficients,
        mediaIndex,
      );
      const constrainedPath = coefficientPath(
        design,
        coefficients,
        mediaIndex,
      );
      const rawContribution = design.mediaVectors[mediaIndex].reduce(
        (total, value, rowIndex) =>
          total + value * rawPath[rowIndex] * effectMultiplier,
        0,
      );
      const constrainedContribution = design.mediaVectors[mediaIndex].reduce(
        (total, value, rowIndex) =>
          total + value * constrainedPath[rowIndex] * effectMultiplier,
        0,
      );
      const contributionChange = constrainedContribution - rawContribution;
      const spend = design.spendVectors[mediaIndex].reduce(
        (total, value) => total + value,
        0,
      );
      const estimate = channelEstimateByName.get(channel);
      const constrainedRoi =
        estimate?.roi ?? constrainedContribution / Math.max(spend, 1);
      const unconstrainedRoi =
        constrainedRoi - contributionChange / Math.max(spend, 1);
      const intervalWidth = Math.max(
        (estimate?.roiHigh ?? constrainedRoi) -
          (estimate?.roiLow ?? constrainedRoi),
        0,
      );
      return {
        channel,
        unconstrainedCoefficient: Math.min(...rawValues),
        constrainedCoefficient: coefficients[indexes[minimumIndex]],
        contributionChange,
        contributionChangeShare:
          Math.abs(contributionChange) /
          Math.max(
            design.outcome.reduce(
              (total, value) => total + Math.abs(value),
              0,
            ),
            1,
          ),
        unconstrainedRoi,
        constrainedRoi,
        roiIntervalMaterial:
          Math.abs(constrainedRoi - unconstrainedRoi) >
          Math.max(0.1, intervalWidth * 0.1),
      };
    })
    .filter(({ unconstrainedCoefficient }) => unconstrainedCoefficient < 0);
  const clippingMaterial =
    predictionShiftShare >= 0.01 ||
    clippedChannels.length >= 2 ||
    clippedChannels.some((channel) => channel.roiIntervalMaterial);
  const dataDiagnostics = diagnoseDesignMatrix(
    design.matrix,
    observationWeights,
  );
  const priorInfluence = dataset.mediaColumns.map((channel, mediaIndex) => {
    const indexes = design.channelCoefficientIndexes[mediaIndex];
    const dataPrecision = indexes.reduce(
      (channelTotal, coefficientIndex) =>
        channelTotal +
        design.matrix.reduce(
          (total, row, rowIndex) =>
            total +
            (observationWeights[rowIndex] ?? 1) *
              (row[coefficientIndex] ?? 0) ** 2,
          0,
        ),
      0,
    );
    const priorPrecision = indexes.reduce(
      (total, index) => total + (finalPriorPrecisions.get(index) ?? 0),
      0,
    );
    const precisionShare =
      priorPrecision / Math.max(dataPrecision + priorPrecision, 1e-12);
    const experimentEvidence = experiments.filter(
      (experiment) =>
        experiment.channel.toLowerCase() === channel.toLowerCase(),
    );
    const benchmark = activeIndustryPrior(
      channel,
      experiments,
      industryPriorSelection,
    );
    const source =
      advancedConfig.calibrationMode === "prior" && experimentEvidence.length
        ? ("experiment" as const)
        : benchmark
          ? ("industry" as const)
          : advancedConfig.priorDistribution === "log-normal"
            ? ("regularizing" as const)
            : ("none" as const);
    return {
      channel,
      precisionShare,
      classification:
        precisionShare >= 0.7
          ? ("prior-led" as const)
          : precisionShare >= 0.3
            ? ("data-and-prior" as const)
            : ("data-led" as const),
      source,
      sourceLabel:
        source === "experiment"
          ? experimentEvidence.length === 1
            ? experimentEvidence[0].source
            : `${experimentEvidence.length} pooled experiments`
          : source === "industry"
            ? benchmark?.label
            : source === "regularizing"
              ? "Weak positive regularizing prior"
              : undefined,
    };
  });
  const numericalStatus =
    finalSolution.diagnostics.status === "rank-deficient"
      ? "rank-deficient"
      : dataDiagnostics.rank < dataDiagnostics.columns ||
          dataDiagnostics.status !== "stable" ||
          clippingMaterial
        ? "review"
        : finalSolution.diagnostics.status;
  const numerical: NumericalStabilityDiagnostics = {
    solver: finalSolution.diagnostics.method,
    rank: finalSolution.diagnostics.rank,
    parameterCount: finalSolution.diagnostics.columns,
    conditionNumber: Number.isFinite(finalSolution.diagnostics.conditionNumber)
      ? finalSolution.diagnostics.conditionNumber
      : null,
    dataRank: dataDiagnostics.rank,
    dataConditionNumber: Number.isFinite(dataDiagnostics.conditionNumber)
      ? dataDiagnostics.conditionNumber
      : null,
    status: numericalStatus,
    priorInfluence,
    clipping: {
      applied: clippedChannels.length > 0,
      material: clippingMaterial,
      severity: !clippedChannels.length
        ? "none"
        : clippingMaterial
          ? "warning"
          : "informational",
      predictionShiftShare,
      channels: clippedChannels,
    },
  };
  const totalOutcome = design.outcome.reduce(
    (total, value) => total + value,
    0,
  );
  const baselineTotal = baseline.reduce((total, value) => total + value, 0);
  const fitR2 = rSquared(design.outcome, predicted);

  return {
    kind: "advanced",
    fingerprint,
    cached: false,
    r2: fitR2,
    mape: mape(design.outcome, predicted),
    rmse: rmse(design.outcome, predicted),
    baselineShare: baselineTotal / Math.max(totalOutcome, 1),
    channels,
    actual: design.outcome,
    predicted,
    baseline,
    residuals,
    numerical,
    diagnostics: [
      {
        label: "Numerical stability",
        value: `${numerical.status === "stable" ? "Stable" : "Review"} · ${numerical.solver === "pivoted-qr" ? "QR" : "SVD"}`,
        state: numerical.status === "stable" ? "good" : "warn",
      },
      {
        label: "Coefficient structure",
        value: advancedConfig.timeVarying
          ? `${advancedConfig.kernelKnots} kernel knots`
          : "Static",
        state: "good",
      },
      {
        label: "Planning factor",
        value: advancedConfig.planningIntensity ? "Included" : "Off",
        state: "good",
      },
      {
        label: "Calibration",
        value: `${advancedConfig.calibrationMode} · ${calibratedExperiments} exp + ${calibratedIndustryPriors} benchmark`,
        state:
          calibratedExperiments || calibratedIndustryPriors
            ? "good"
            : "warn",
      },
      {
        label: "Distributions",
        value: `${advancedConfig.priorDistribution} · ${advancedConfig.likelihoodDistribution}`,
        state: "good",
      },
      {
        label: "Time-series fit",
        value: fitR2 > 0.7 ? "Strong" : "Review",
        state: fitR2 > 0.7 ? "good" : "warn",
      },
    ],
    advanced: {
      timeVarying: advancedConfig.timeVarying,
      calibrationMode: advancedConfig.calibrationMode,
      priorDistribution: advancedConfig.priorDistribution,
      likelihoodDistribution: advancedConfig.likelihoodDistribution,
      coefficientPaths,
      planningIntensity: advancedConfig.planningIntensity
        ? design.planningIntensity
        : [],
      calibratedExperiments,
      calibratedIndustryPriors,
      calibrationReceipts: calibrationReceipt,
    },
    runAt: new Date().toISOString(),
  };
}
