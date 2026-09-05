import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";
import { V5A_COMMON_POOL_CONFIGURATION } from "../svi_score_v5a/contract";
import { v4SelectionPool } from "../svi_score_v4/ranker";
import type { V5CTrainingConfiguration } from "./contract";
import {
  v5cFeatureNames,
  v5cFeatureVector,
  v5cPriorWeights,
} from "./features";
import type { V5CModel } from "./types";

const FEATURE_CACHE = new WeakMap<SviScoreV4CandidateRow, number[]>();

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

function vector(row: SviScoreV4CandidateRow): number[] {
  const cached = FEATURE_CACHE.get(row);
  if (cached) return cached;
  const output = v5cFeatureVector(row);
  FEATURE_CACHE.set(row, output);
  return output;
}

function groups(rows: readonly SviScoreV4CandidateRow[]): SviScoreV4CandidateRow[][] {
  const output = new Map<string, SviScoreV4CandidateRow[]>();
  rows.forEach((row) => {
    output.set(row.businessId, [...(output.get(row.businessId) ?? []), row]);
  });
  return [...output.values()];
}

function normalizeCapped(source: readonly number[], cap: number): number[] {
  const raw = source.map((value) => Math.max(0, Number.isFinite(value) ? value : 0));
  if (!raw.length || cap * raw.length < 1 - 1e-12) {
    throw new Error("V5C cannot form its capped nonnegative simplex.");
  }
  if (raw.every((value) => value <= 1e-12)) return raw.map(() => 1 / raw.length);
  const output = raw.map(() => 0);
  let remaining = 1;
  let active = raw.map((_, index) => index);
  while (active.length && remaining > 1e-12) {
    const total = active.reduce((sum, index) => sum + raw[index], 0);
    const provisional = active.map((index) => ({
      index,
      value: total > 1e-12
        ? remaining * raw[index] / total
        : remaining / active.length,
    }));
    const capped = provisional.filter((item) => item.value > cap);
    if (!capped.length) {
      provisional.forEach((item) => output[item.index] = item.value);
      remaining = 0;
      break;
    }
    capped.forEach((item) => {
      output[item.index] = cap;
      remaining -= cap;
    });
    const indexes = new Set(capped.map((item) => item.index));
    active = active.filter((index) => !indexes.has(index));
  }
  const total = output.reduce((sum, value) => sum + value, 0);
  return output.map((value) => value / Math.max(total, 1e-12));
}

export function latentV5CScore(row: SviScoreV4CandidateRow, model: V5CModel): number {
  const values = vector(row);
  if (values.length !== model.weights.length) {
    throw new Error("V5C feature vector does not match its fitted selector.");
  }
  return values.reduce(
    (sum, value, index) => sum + value * model.weights[index],
    0,
  );
}

export function v5cUncappedGap(
  candidateLoss: number,
  oracleLoss: number,
  economicScale: number,
): number {
  return Math.max(0, candidateLoss - oracleLoss) / Math.max(economicScale, 1e-12);
}

interface HardNegative {
  family: string;
  loss: number;
  gradient: number[];
}

function hardNegative(
  business: readonly SviScoreV4CandidateRow[],
  model: V5CModel,
): HardNegative {
  const pool = v4SelectionPool(business, V5A_COMMON_POOL_CONFIGURATION).rows;
  const oracle = [...pool].sort(
    (left, right) =>
      left.decisionLoss - right.decisionLoss ||
      left.candidateId.localeCompare(right.candidateId),
  )[0];
  if (!oracle) throw new Error("V5C business has no candidate pool.");
  const oracleScore = latentV5CScore(oracle, model);
  let maximum = 0;
  let challenger: SviScoreV4CandidateRow | undefined;
  pool.filter((candidate) => candidate !== oracle).forEach((candidate) => {
    const uncappedGap = v5cUncappedGap(
      candidate.decisionLoss,
      oracle.decisionLoss,
      model.configuration.economicScale,
    );
    const violation = uncappedGap + latentV5CScore(candidate, model) - oracleScore;
    if (
      violation > maximum + 1e-12 ||
      (Math.abs(violation - maximum) <= 1e-12 &&
        candidate.candidateId.localeCompare(challenger?.candidateId ?? "\uffff") < 0)
    ) {
      maximum = violation;
      challenger = candidate;
    }
  });
  if (!challenger || maximum <= 0) {
    return {
      family: oracle.family,
      loss: 0,
      gradient: model.weights.map(() => 0),
    };
  }
  const oracleVector = vector(oracle);
  return {
    family: oracle.family,
    loss: maximum,
    gradient: vector(challenger).map(
      (value, index) => value - oracleVector[index],
    ),
  };
}

export function learnScoreV5C(
  rows: readonly SviScoreV4CandidateRow[],
  configuration: V5CTrainingConfiguration,
): V5CModel {
  if (!rows.length) throw new Error("V5C requires training rows.");
  const featureNames = v5cFeatureNames(rows[0]);
  rows.forEach((row) => {
    const names = v5cFeatureNames(row);
    if (
      names.length !== featureNames.length ||
      names.some((name, index) => name !== featureNames[index]) ||
      vector(row).some((value) => !Number.isFinite(value))
    ) {
      throw new Error(`V5C feature contract changed for ${row.candidateId}.`);
    }
  });
  const prior = normalizeCapped(v5cPriorWeights(rows[0]), configuration.featureCap);
  let model: V5CModel = {
    featureNames,
    weights: [...prior],
    configuration: structuredClone(configuration),
  };
  for (let epoch = 0; epoch < configuration.epochs; epoch += 1) {
    const negatives = groups(rows).map((business) => hardNegative(business, model));
    const threshold = quantile(negatives.map((item) => item.loss), 0.9);
    const tail = negatives.filter((item) => item.loss >= threshold - 1e-12);
    const byFamily = new Map<string, HardNegative[]>();
    negatives.forEach((item) => {
      byFamily.set(item.family, [...(byFamily.get(item.family) ?? []), item]);
    });
    const worstFamily = [...byFamily.entries()].sort(
      (left, right) =>
        average(right[1].map((item) => item.loss)) -
          average(left[1].map((item) => item.loss)) ||
        left[0].localeCompare(right[0]),
    )[0]?.[0];
    const worst = byFamily.get(worstFamily ?? "") ?? [];
    const gradient = model.weights.map(() => 0);
    negatives.forEach((item) => {
      const multiplier = 1 / negatives.length +
        (item.loss >= threshold - 1e-12
          ? configuration.cvarWeight / Math.max(tail.length, 1)
          : 0) +
        (item.family === worstFamily
          ? configuration.familyDroWeight / Math.max(worst.length, 1)
          : 0);
      item.gradient.forEach((value, index) => {
        gradient[index] += multiplier * value;
      });
    });
    gradient.forEach((_, index) => {
      gradient[index] += configuration.regularization *
        (model.weights[index] - prior[index]);
    });
    const rate = configuration.learningRate / Math.sqrt(epoch + 8);
    model = {
      ...model,
      weights: normalizeCapped(
        model.weights.map((weight, index) => weight - rate * gradient[index]),
        configuration.featureCap,
      ),
    };
  }
  return model;
}
