import { HEURISTIC_WEIGHTS, scoreFromWeights } from "../score_v4/learn";
import {
  SCORE_V6_DIAGNOSTIC_GROUPS,
  type ScoreV6CandidateRow,
  type ScoreV6Feature,
  type ScoreV6FeatureWeights,
  type ScoreV6Metrics,
} from "./types";

const FEATURES = Object.values(SCORE_V6_DIAGNOSTIC_GROUPS).flat() as ScoreV6Feature[];
const GROUP_FLOOR = 0.1;
const GROUP_CAP = 0.4;
const FEATURE_CAP = 0.16;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function average(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) /
    Math.max(values.length, 1);
}

function quantile(values: number[], probability: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  const position = clamp(probability, 0, 1) * Math.max(sorted.length - 1, 0);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return (sorted[lower] ?? 0) * (1 - weight) +
    (sorted[upper] ?? 0) * weight;
}

function groupRows(rows: ScoreV6CandidateRow[]): ScoreV6CandidateRow[][] {
  const grouped = new Map<string, ScoreV6CandidateRow[]>();
  rows.forEach((row) => {
    grouped.set(row.businessId, [...(grouped.get(row.businessId) ?? []), row]);
  });
  return [...grouped.values()];
}

function selectionPool(rows: ScoreV6CandidateRow[]): ScoreV6CandidateRow[] {
  const decisionGrade = rows.filter((row) => row.eligible);
  if (decisionGrade.length) return decisionGrade;
  const review = rows.filter((row) => row.reviewEligible);
  return review.length ? review : rows;
}

function transformedFeature(row: ScoreV6CandidateRow, feature: ScoreV6Feature): number {
  const value = row.diagnostics[feature];
  return Math.log(clamp((value ?? 50) / 100, 0.01, 1));
}

export function scoreV6(
  row: ScoreV6CandidateRow,
  weights: ScoreV6FeatureWeights,
): number {
  return 100 * Math.exp(
    FEATURES.reduce(
      (total, feature) => total + weights[feature] * transformedFeature(row, feature),
      0,
    ),
  );
}

function groupForFeature(feature: ScoreV6Feature) {
  return (Object.keys(SCORE_V6_DIAGNOSTIC_GROUPS) as Array<keyof typeof SCORE_V6_DIAGNOSTIC_GROUPS>)
    .find((group) => SCORE_V6_DIAGNOSTIC_GROUPS[group].includes(feature as never))!;
}

function normalizeConstrained(raw: Record<ScoreV6Feature, number>): ScoreV6FeatureWeights {
  let weights = Object.fromEntries(
    FEATURES.map((feature) => [feature, Math.max(0, raw[feature])]),
  ) as ScoreV6FeatureWeights;
  if (Object.values(weights).every((value) => value <= 1e-12)) {
    weights = Object.fromEntries(FEATURES.map((feature) => [feature, 1])) as ScoreV6FeatureWeights;
  }
  for (let iteration = 0; iteration < 30; iteration += 1) {
    const total = Object.values(weights).reduce((sum, value) => sum + value, 0);
    FEATURES.forEach((feature) => {
      weights[feature] = Math.min(FEATURE_CAP, weights[feature] / Math.max(total, 1e-12));
    });
    for (const [group, features] of Object.entries(SCORE_V6_DIAGNOSTIC_GROUPS)) {
      const groupTotal = features.reduce(
        (sum, feature) => sum + weights[feature],
        0,
      );
      const target = clamp(groupTotal, GROUP_FLOOR, GROUP_CAP);
      const scale = target / Math.max(groupTotal, 1e-12);
      features.forEach((feature) => {
        weights[feature] = Math.min(FEATURE_CAP, weights[feature] * scale);
      });
      void group;
    }
  }
  const total = Object.values(weights).reduce((sum, value) => sum + value, 0);
  FEATURES.forEach((feature) => {
    weights[feature] /= Math.max(total, 1e-12);
  });
  return weights;
}

interface Pair {
  difference: number[];
  weight: number;
}

function trainingPairs(rows: ScoreV6CandidateRow[]): Pair[] {
  const pairs: Pair[] = [];
  groupRows(rows).forEach((business) => {
    const pool = selectionPool(business).sort(
      (left, right) => left.decisionLoss - right.decisionLoss,
    );
    const candidates: Pair[] = [];
    for (let leftIndex = 0; leftIndex < pool.length; leftIndex += 1) {
      for (let rightIndex = leftIndex + 1; rightIndex < pool.length; rightIndex += 1) {
        const left = pool[leftIndex];
        const right = pool[rightIndex];
        const gap = Math.abs(left.decisionLoss - right.decisionLoss);
        if (gap < 0.01) continue;
        const better = left.decisionLoss < right.decisionLoss ? left : right;
        const worse = better === left ? right : left;
        candidates.push({
          difference: FEATURES.map(
            (feature) => transformedFeature(better, feature) - transformedFeature(worse, feature),
          ),
          weight: Math.min(3, 0.25 + Math.log1p(gap * 4)),
        });
      }
    }
    // Preserve the widest regret contrasts plus a deterministic spread of
    // the remaining comparisons. Labels and eligibility tiers are unchanged.
    candidates.sort((left, right) => right.weight - left.weight);
    const limit = Math.min(192, candidates.length);
    if (candidates.length <= limit) {
      pairs.push(...candidates);
      return;
    }
    const head = Math.min(64, limit);
    pairs.push(...candidates.slice(0, head));
    for (let index = head; index < limit; index += 1) {
      const position = Math.floor(
        ((index - head + 0.5) / Math.max(limit - head, 1)) *
          candidates.length,
      );
      pairs.push(candidates[Math.min(candidates.length - 1, position)]);
    }
  });
  return pairs;
}

export function learnScoreV6(
  trainRows: ScoreV6CandidateRow[],
  regularization = 0.015,
  epochs = 180,
): { weights: ScoreV6FeatureWeights; pairCount: number } {
  const pairs = trainingPairs(trainRows);
  let weights = normalizeConstrained(
    Object.fromEntries(FEATURES.map((feature) => [feature, 1])) as ScoreV6FeatureWeights,
  );
  const baseline = { ...weights };
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const gradient = FEATURES.map(() => 0);
    pairs.forEach((pair) => {
      const margin = pair.difference.reduce(
        (total, value, index) =>
          total + value * weights[FEATURES[index]],
        0,
      );
      const factor =
        -pair.weight / (1 + Math.exp(clamp(margin, -30, 30)));
      pair.difference.forEach((value, index) => {
        gradient[index] += factor * value;
      });
    });
    FEATURES.forEach((feature, index) => {
      gradient[index] =
        gradient[index] / Math.max(pairs.length, 1) +
        regularization * (weights[feature] - baseline[feature]);
    });
    const learningRate = 0.35 / Math.sqrt(epoch + 8);
    const raw = Object.fromEntries(
      FEATURES.map((feature, index) => [
        feature,
        weights[feature] - learningRate * gradient[index],
      ]),
    ) as ScoreV6FeatureWeights;
    weights = normalizeConstrained(raw);
  }
  return { weights, pairCount: pairs.length };
}

export function evaluateScoreV6(
  rows: ScoreV6CandidateRow[],
  score: (row: ScoreV6CandidateRow) => number,
): ScoreV6Metrics {
  const selections = groupRows(rows).map((business) => {
    const pool = selectionPool(business);
    const selected = [...pool].sort(
      (left, right) => score(right) - score(left) || left.candidateId.localeCompare(right.candidateId),
    )[0];
    const best = [...pool].sort((left, right) => left.decisionLoss - right.decisionLoss)[0];
    return {
      family: selected.family,
      decisionGrade: business.some((row) => row.eligible),
      loss: selected.decisionLoss,
      excess: Math.max(0, selected.decisionLoss - best.decisionLoss),
      lowest: Math.abs(selected.decisionLoss - best.decisionLoss) < 1e-10,
    };
  });
  const familyMeanLoss = Object.fromEntries(
    [...new Set(selections.map((selection) => selection.family))].map((family) => [
      family,
      average(selections.filter((selection) => selection.family === family).map((selection) => selection.loss)),
    ]),
  );
  const losses = selections.map((selection) => selection.loss);
  return {
    businesses: selections.length,
    decisionGradeBusinessShare: average(selections.map((selection) => selection.decisionGrade ? 1 : 0)),
    meanLoss: average(losses),
    medianLoss: quantile(losses, 0.5),
    p90Loss: quantile(losses, 0.9),
    meanExcessLoss: average(selections.map((selection) => selection.excess)),
    lowestLossSelectionRate: average(selections.map((selection) => selection.lowest ? 1 : 0)),
    familyMeanLoss,
  };
}

export function evaluateV4Heuristic(rows: ScoreV6CandidateRow[]): ScoreV6Metrics {
  return evaluateScoreV6(rows, (row) => scoreFromWeights(row.layerScores, HEURISTIC_WEIGHTS));
}

export function groupWeights(weights: ScoreV6FeatureWeights) {
  return Object.fromEntries(
    Object.entries(SCORE_V6_DIAGNOSTIC_GROUPS).map(([group, features]) => [
      group,
      features.reduce((total, feature) => total + weights[feature], 0),
    ]),
  ) as Record<keyof typeof SCORE_V6_DIAGNOSTIC_GROUPS, number>;
}

export function featureGroup(feature: ScoreV6Feature) {
  return groupForFeature(feature);
}
