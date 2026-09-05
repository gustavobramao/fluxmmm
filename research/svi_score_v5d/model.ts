import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";
import { v4SelectionPool } from "../svi_score_v4/ranker";
import { V5A_COMMON_POOL_CONFIGURATION } from "../svi_score_v5a/contract";
import type { V5DTrainingConfiguration } from "./contract";
import {
  v5dFeatureNames,
  v5dFeatureVector,
} from "./features";
import type { V5DLossPrediction, V5DModel } from "./types";

const FEATURE_CACHE = new WeakMap<SviScoreV4CandidateRow, number[]>();

export interface V5DFeatureProvider {
  names: (row: SviScoreV4CandidateRow) => string[];
  vector: (row: SviScoreV4CandidateRow) => number[];
}

export interface V5DSelectionPolicy {
  passes: (row: SviScoreV4CandidateRow) => boolean;
  pool: (rows: readonly SviScoreV4CandidateRow[]) => {
    rows: SviScoreV4CandidateRow[];
    hasSafetyAcceptedCandidate: boolean;
  };
}

export const V5D_MAP_FEATURE_PROVIDER: V5DFeatureProvider = {
  names: v5dFeatureNames,
  vector: v5dFeatureVector,
};

export const V5D_MAP_SELECTION_POLICY: V5DSelectionPolicy = {
  passes: (row) =>
    v4SelectionPool([row], V5A_COMMON_POOL_CONFIGURATION)
      .hasSafetyAcceptedCandidate,
  pool: (rows) => v4SelectionPool(rows, V5A_COMMON_POOL_CONFIGURATION),
};

const FEATURE_PROVIDER_CACHE = new WeakMap<
  V5DFeatureProvider,
  WeakMap<SviScoreV4CandidateRow, number[]>
>();

function vector(
  row: SviScoreV4CandidateRow,
  provider: V5DFeatureProvider,
): number[] {
  if (provider === V5D_MAP_FEATURE_PROVIDER) {
    const cached = FEATURE_CACHE.get(row);
    if (cached) return cached;
    const output = provider.vector(row);
    FEATURE_CACHE.set(row, output);
    return output;
  }
  let cache = FEATURE_PROVIDER_CACHE.get(provider);
  if (!cache) {
    cache = new WeakMap<SviScoreV4CandidateRow, number[]>();
    FEATURE_PROVIDER_CACHE.set(provider, cache);
  }
  const cached = cache.get(row);
  if (cached) return cached;
  const output = provider.vector(row);
  cache.set(row, output);
  return output;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function average(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) /
    Math.max(values.length, 1);
}

function quantile(values: readonly number[], probability: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const position = clamp(probability, 0, 1) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function dot(left: readonly number[], right: readonly number[]): number {
  return left.reduce((sum, value, index) => sum + value * (right[index] ?? 0), 0);
}

function sigmoid(value: number): number {
  if (value >= 0) return 1 / (1 + Math.exp(-Math.min(value, 40)));
  const exponential = Math.exp(Math.max(value, -40));
  return exponential / (1 + exponential);
}

function grouped(
  rows: readonly SviScoreV4CandidateRow[],
): SviScoreV4CandidateRow[][] {
  const output = new Map<string, SviScoreV4CandidateRow[]>();
  rows.forEach((row) => {
    output.set(row.businessId, [...(output.get(row.businessId) ?? []), row]);
  });
  return [...output.values()].sort((left, right) =>
    (left[0]?.businessId ?? "").localeCompare(right[0]?.businessId ?? "")
  );
}

interface TrainingExample {
  row: SviScoreV4CandidateRow;
  features: number[];
  target: number;
  family: string;
}

function targets(
  rows: readonly SviScoreV4CandidateRow[],
  configuration: V5DTrainingConfiguration,
  selectionPolicy: V5DSelectionPolicy,
): Map<SviScoreV4CandidateRow, number> {
  const output = new Map<SviScoreV4CandidateRow, number>();
  grouped(rows).forEach((business) => {
    const pool = selectionPolicy.pool(business).rows;
    const oracle = Math.min(...pool.map((row) => row.decisionLoss));
    business.forEach((row) => {
      output.set(
        row,
        Math.max(0, row.decisionLoss - oracle) /
          Math.max(configuration.economicScale, 1e-12),
      );
    });
  });
  return output;
}

function standardization(
  rows: readonly SviScoreV4CandidateRow[],
  provider: V5DFeatureProvider,
): {
  means: number[];
  scales: number[];
} {
  const columns = vector(rows[0], provider).length;
  const means = Array(columns).fill(0);
  rows.forEach((row) => {
    vector(row, provider).forEach((value, index) => means[index] += value / rows.length);
  });
  const scales = Array(columns).fill(0);
  rows.forEach((row) => {
    vector(row, provider).forEach((value, index) => {
      scales[index] += (value - means[index]) ** 2 / Math.max(rows.length - 1, 1);
    });
  });
  return {
    means,
    scales: scales.map((value) => Math.max(Math.sqrt(value), 1e-6)),
  };
}

function standardized(
  row: SviScoreV4CandidateRow,
  means: readonly number[],
  scales: readonly number[],
  provider: V5DFeatureProvider,
): number[] {
  return [
    1,
    ...vector(row, provider).map((value, index) =>
      clamp((value - means[index]) / scales[index], -8, 8)
    ),
  ];
}

function prediction(
  row: SviScoreV4CandidateRow,
  model: Pick<V5DModel, "means" | "scales" | "meanWeights" | "p90Weights" | "configuration">,
  provider: V5DFeatureProvider,
): V5DLossPrediction {
  const features = standardized(row, model.means, model.scales, provider);
  const mean = Math.max(0, dot(features, model.meanWeights));
  const p90 = Math.max(mean, dot(features, model.p90Weights));
  const tailWidth = p90 - mean;
  return {
    mean,
    p90,
    tailWidth,
    risk: mean + model.configuration.riskAversion * tailWidth,
  };
}

export function predictV5DLoss(
  row: SviScoreV4CandidateRow,
  model: V5DModel,
  provider: V5DFeatureProvider = V5D_MAP_FEATURE_PROVIDER,
): V5DLossPrediction {
  return prediction(row, model, provider);
}

export function learnV5DModel(
  rows: readonly SviScoreV4CandidateRow[],
  configuration: V5DTrainingConfiguration,
  provider: V5DFeatureProvider = V5D_MAP_FEATURE_PROVIDER,
  selectionPolicy: V5DSelectionPolicy = V5D_MAP_SELECTION_POLICY,
): V5DModel {
  if (!rows.length) throw new Error("V5D requires training rows.");
  const featureNames = provider.names(rows[0]);
  rows.forEach((row) => {
    const names = provider.names(row);
    if (
      names.length !== featureNames.length ||
      names.some((name, index) => name !== featureNames[index]) ||
      vector(row, provider).some((value) => !Number.isFinite(value))
    ) {
      throw new Error(`V5D feature contract changed for ${row.candidateId}.`);
    }
  });
  const { means, scales } = standardization(rows, provider);
  const targetByRow = targets(rows, configuration, selectionPolicy);
  const examples: TrainingExample[] = rows.map((row) => ({
    row,
    features: standardized(row, means, scales, provider),
    target: targetByRow.get(row) ?? 0,
    family: row.family,
  }));
  const targetValues = examples.map((example) => example.target);
  const tailThreshold = quantile(targetValues, 0.9);
  const familyTargets = new Map<string, number[]>();
  examples.forEach((example) => {
    familyTargets.set(example.family, [
      ...(familyTargets.get(example.family) ?? []),
      example.target,
    ]);
  });
  const worstFamily = [...familyTargets.entries()].sort(
    (left, right) =>
      average(right[1]) - average(left[1]) || left[0].localeCompare(right[0]),
  )[0]?.[0];
  const width = examples[0].features.length;
  let meanWeights = Array(width).fill(0);
  let p90Weights = Array(width).fill(0);
  meanWeights[0] = average(targetValues);
  p90Weights[0] = quantile(targetValues, configuration.quantile);
  const businesses = grouped(rows);
  let hardNegativeUpdates = 0;

  for (let epoch = 0; epoch < configuration.epochs; epoch += 1) {
    const meanGradient = Array(width).fill(0);
    const quantileGradient = Array(width).fill(0);
    let totalRowWeight = 0;
    examples.forEach((example) => {
      const rowWeight = 1 +
        (example.target >= tailThreshold - 1e-12 ? configuration.tailRowWeight : 0) +
        (example.family === worstFamily ? configuration.worstFamilyRowWeight : 0);
      totalRowWeight += rowWeight;
      const meanError = dot(example.features, meanWeights) - example.target;
      const huberDerivative = clamp(
        meanError,
        -configuration.huberDelta,
        configuration.huberDelta,
      );
      const quantileError = dot(example.features, p90Weights) - example.target;
      const pinballDerivative = quantileError >= 0
        ? 1 - configuration.quantile
        : -configuration.quantile;
      example.features.forEach((value, index) => {
        meanGradient[index] += rowWeight * huberDerivative * value;
        quantileGradient[index] += rowWeight * pinballDerivative * value;
      });
    });
    meanGradient.forEach((_, index) => {
      meanGradient[index] /= Math.max(totalRowWeight, 1);
      quantileGradient[index] /= Math.max(totalRowWeight, 1);
    });

    businesses.forEach((business) => {
      const pool = selectionPolicy.pool(business).rows;
      const oracle = [...pool].sort(
        (left, right) =>
          (targetByRow.get(left) ?? Infinity) - (targetByRow.get(right) ?? Infinity) ||
          left.candidateId.localeCompare(right.candidateId),
      )[0];
      if (!oracle) return;
      const oracleFeatures = standardized(oracle, means, scales, provider);
      const oraclePrediction = dot(oracleFeatures, meanWeights);
      let hardest: SviScoreV4CandidateRow | undefined;
      let hardestViolation = -Infinity;
      pool.forEach((candidate) => {
        if (candidate === oracle) return;
        const gap = Math.max(0, (targetByRow.get(candidate) ?? 0) - (targetByRow.get(oracle) ?? 0));
        const violation = gap + oraclePrediction -
          dot(standardized(candidate, means, scales, provider), meanWeights);
        if (
          violation > hardestViolation + 1e-12 ||
          (Math.abs(violation - hardestViolation) <= 1e-12 &&
            candidate.candidateId.localeCompare(hardest?.candidateId ?? "\uffff") < 0)
        ) {
          hardest = candidate;
          hardestViolation = violation;
        }
      });
      if (!hardest) return;
      hardNegativeUpdates += 1;
      const gap = Math.max(0, (targetByRow.get(hardest) ?? 0) - (targetByRow.get(oracle) ?? 0));
      const derivative = configuration.pairwiseWeight *
        Math.max(gap, 0.05) * sigmoid(hardestViolation) /
        Math.max(businesses.length, 1);
      const challengerFeatures = standardized(hardest, means, scales, provider);
      oracleFeatures.forEach((value, index) => {
        meanGradient[index] += derivative * (value - challengerFeatures[index]);
      });
    });

    meanGradient.forEach((_, index) => {
      if (index > 0) {
        meanGradient[index] += configuration.regularization * meanWeights[index];
        quantileGradient[index] += configuration.regularization * p90Weights[index];
      }
    });
    const rate = configuration.learningRate / Math.sqrt(1 + epoch / 8);
    meanWeights = meanWeights.map((weight, index) =>
      clamp(weight - rate * meanGradient[index], -12, 12)
    );
    p90Weights = p90Weights.map((weight, index) =>
      clamp(weight - rate * quantileGradient[index], -12, 12)
    );
  }

  return {
    featureNames,
    means,
    scales,
    meanWeights,
    p90Weights,
    configuration: structuredClone(configuration),
    receipt: {
      trainingRows: rows.length,
      trainingBusinesses: new Set(rows.map((row) => row.businessId)).size,
      meanTarget: average(targetValues),
      p90Target: quantile(targetValues, 0.9),
      hardNegativeUpdates,
    },
  };
}
