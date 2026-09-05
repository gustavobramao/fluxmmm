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
export const PREDECLARED_NORMALIZED_ECONOMIC_SCALE = 1;

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
      (total, feature) =>
        total + (weights[feature] ?? 0) * transformedFeature(row, feature),
      0,
    ),
  );
}

function groupForFeature(feature: ScoreV6Feature) {
  return (Object.keys(SCORE_V6_DIAGNOSTIC_GROUPS) as Array<keyof typeof SCORE_V6_DIAGNOSTIC_GROUPS>)
    .find((group) => SCORE_V6_DIAGNOSTIC_GROUPS[group].includes(feature as never))!;
}

function normalizeConstrained(raw: ScoreV6FeatureWeights): ScoreV6FeatureWeights {
  let weights = Object.fromEntries(
    FEATURES.map((feature) => [feature, Math.max(0, raw[feature] ?? 0)]),
  ) as ScoreV6FeatureWeights;
  if (Object.values(weights).every((value) => value <= 1e-12)) {
    weights = Object.fromEntries(FEATURES.map((feature) => [feature, 1])) as ScoreV6FeatureWeights;
  }
  for (let iteration = 0; iteration < 30; iteration += 1) {
    const total = Object.values(weights).reduce((sum, value) => sum + value, 0);
    FEATURES.forEach((feature) => {
      weights[feature] = Math.min(
        FEATURE_CAP,
        (weights[feature] ?? 0) / Math.max(total, 1e-12),
      );
    });
    for (const [group, features] of Object.entries(SCORE_V6_DIAGNOSTIC_GROUPS)) {
      const groupTotal = features.reduce(
        (sum, feature) => sum + (weights[feature] ?? 0),
        0,
      );
      const target = clamp(groupTotal, GROUP_FLOOR, GROUP_CAP);
      const scale = target / Math.max(groupTotal, 1e-12);
      features.forEach((feature) => {
        weights[feature] = Math.min(
          FEATURE_CAP,
          (weights[feature] ?? 0) * scale,
        );
      });
      void group;
    }
  }
  const total = Object.values(weights).reduce((sum, value) => sum + value, 0);
  FEATURES.forEach((feature) => {
    weights[feature] = (weights[feature] ?? 0) / Math.max(total, 1e-12);
  });
  return weights;
}

export interface RegretWeightedOraclePair {
  businessId: string;
  oracleCandidateId: string;
  candidateId: string;
  difference: number[];
  weight: number;
}

function economicScaleForBusiness(rows: ScoreV6CandidateRow[]): number {
  const scales = new Set(
    rows.map((row) => row.economicScale ?? PREDECLARED_NORMALIZED_ECONOMIC_SCALE),
  );
  if (scales.size !== 1) {
    throw new Error(
      `Economic scale must be candidate-independent within ${rows[0]?.businessId ?? "a business"}.`,
    );
  }
  const scale = [...scales][0];
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error("The predeclared economic scale must be finite and positive.");
  }
  return scale;
}

export function normalizedExcessRegret(
  candidateLoss: number,
  oracleLoss: number,
  economicScale: number,
): number {
  return clamp(
    Math.max(0, candidateLoss - oracleLoss) / Math.max(economicScale, 1e-12),
    0,
    1,
  );
}

/**
 * Exact theorem target: compare the deterministic in-pool oracle with every
 * other candidate. No score, eligibility outcome, or regret gap is used to
 * subsample pairs.
 */
export function regretWeightedOraclePairs(
  rows: ScoreV6CandidateRow[],
): RegretWeightedOraclePair[] {
  const pairs: RegretWeightedOraclePair[] = [];
  groupRows(rows).forEach((business) => {
    const pool = [...selectionPool(business)].sort(
      (left, right) =>
        left.decisionLoss - right.decisionLoss ||
        left.candidateId.localeCompare(right.candidateId),
    );
    const oracle = pool[0];
    if (!oracle) return;
    const economicScale = economicScaleForBusiness(pool);
    pool.slice(1).forEach((candidate) => {
      pairs.push({
        businessId: oracle.businessId,
        oracleCandidateId: oracle.candidateId,
        candidateId: candidate.candidateId,
        difference: FEATURES.map(
          (feature) =>
            transformedFeature(oracle, feature) -
            transformedFeature(candidate, feature),
        ),
        weight: normalizedExcessRegret(
          candidate.decisionLoss,
          oracle.decisionLoss,
          economicScale,
        ),
      });
    });
  });
  return pairs;
}

function softplus(value: number): number {
  if (value > 30) return value;
  if (value < -30) return Math.exp(value);
  return Math.log1p(Math.exp(value));
}

export interface RegretBoundReceipt {
  businesses: number;
  violations: number;
  meanSelectedNormalizedRegret: number;
  meanMisrankingUpperBound: number;
  meanLogisticSurrogateUpperBound: number;
  maximumNumericalSlack: number;
}

/**
 * Audits the finite-candidate regret theorem and its logistic surrogate. The
 * supplied score may be on any positive monotone scale; it is converted to a
 * latent log score before the smooth bound is evaluated.
 */
export function evaluateRegretBound(
  rows: ScoreV6CandidateRow[],
  score: (row: ScoreV6CandidateRow) => number,
): RegretBoundReceipt {
  const receipts = groupRows(rows).map((business) => {
    const pool = selectionPool(business);
    const economicScale = economicScaleForBusiness(pool);
    const oracle = [...pool].sort(
      (left, right) =>
        left.decisionLoss - right.decisionLoss ||
        left.candidateId.localeCompare(right.candidateId),
    )[0];
    const selected = [...pool].sort(
      (left, right) =>
        score(right) - score(left) ||
        left.candidateId.localeCompare(right.candidateId),
    )[0];
    const oracleScore = score(oracle);
    const oracleLatentScore = Math.log(Math.max(oracleScore, 1e-12) / 100);
    const selectedRegret = normalizedExcessRegret(
      selected.decisionLoss,
      oracle.decisionLoss,
      economicScale,
    );
    let misrankingBound = 0;
    let logisticBound = 0;
    pool.forEach((candidate) => {
      if (candidate === oracle) return;
      const gap = normalizedExcessRegret(
        candidate.decisionLoss,
        oracle.decisionLoss,
        economicScale,
      );
      const candidateScore = score(candidate);
      if (candidateScore >= oracleScore) misrankingBound += gap;
      const candidateLatentScore = Math.log(
        Math.max(candidateScore, 1e-12) / 100,
      );
      logisticBound +=
        gap * softplus(candidateLatentScore - oracleLatentScore) / Math.log(2);
    });
    return { selectedRegret, misrankingBound, logisticBound };
  });
  const slack = receipts.map(
    (receipt) => receipt.selectedRegret - receipt.misrankingBound,
  );
  return {
    businesses: receipts.length,
    violations: slack.filter((value) => value > 1e-10).length,
    meanSelectedNormalizedRegret: average(
      receipts.map((receipt) => receipt.selectedRegret),
    ),
    meanMisrankingUpperBound: average(
      receipts.map((receipt) => receipt.misrankingBound),
    ),
    meanLogisticSurrogateUpperBound: average(
      receipts.map((receipt) => receipt.logisticBound),
    ),
    maximumNumericalSlack: Math.max(0, ...slack),
  };
}

export function learnScoreV6(
  trainRows: ScoreV6CandidateRow[],
  regularization = 0.015,
  epochs = 180,
): { weights: ScoreV6FeatureWeights; pairCount: number } {
  const pairs = regretWeightedOraclePairs(trainRows);
  let weights = normalizeConstrained(
    Object.fromEntries(FEATURES.map((feature) => [feature, 1])) as ScoreV6FeatureWeights,
  );
  const baseline = { ...weights };
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const gradient = FEATURES.map(() => 0);
    pairs.forEach((pair) => {
      const margin = pair.difference.reduce(
        (total, value, index) =>
          total + value * (weights[FEATURES[index]] ?? 0),
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
        regularization *
          ((weights[feature] ?? 0) - (baseline[feature] ?? 0));
    });
    const learningRate = 0.35 / Math.sqrt(epoch + 8);
    const raw = Object.fromEntries(
      FEATURES.map((feature, index) => [
        feature,
        (weights[feature] ?? 0) - learningRate * gradient[index],
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
  const selections = selectScoreV6Candidates(rows, score);
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
    p95Loss: quantile(losses, 0.95),
    meanExcessLoss: average(selections.map((selection) => selection.excess)),
    meanNormalizedExcessRegret: average(
      groupRows(rows).map((business) => {
        const pool = selectionPool(business);
        const selected = selections.find(
          (selection) => selection.businessId === business[0]?.businessId,
        )!;
        const oracleLoss = Math.min(...pool.map((row) => row.decisionLoss));
        return normalizedExcessRegret(
          selected.loss,
          oracleLoss,
          economicScaleForBusiness(pool),
        );
      }),
    ),
    lowestLossSelectionRate: average(selections.map((selection) => selection.lowest ? 1 : 0)),
    familyMeanLoss,
  };
}

export interface ScoreV6Selection {
  businessId: string;
  candidateId: string;
  family: string;
  decisionGrade: boolean;
  loss: number;
  excess: number;
  lowest: boolean;
}

export function selectScoreV6Candidates(
  rows: ScoreV6CandidateRow[],
  score: (row: ScoreV6CandidateRow) => number,
): ScoreV6Selection[] {
  return groupRows(rows).map((business) => {
    const pool = selectionPool(business);
    const selected = [...pool].sort(
      (left, right) => score(right) - score(left) || left.candidateId.localeCompare(right.candidateId),
    )[0];
    const best = [...pool].sort((left, right) => left.decisionLoss - right.decisionLoss)[0];
    return {
      businessId: selected.businessId,
      candidateId: selected.candidateId,
      family: selected.family,
      decisionGrade: business.some((row) => row.eligible),
      loss: selected.decisionLoss,
      excess: Math.max(0, selected.decisionLoss - best.decisionLoss),
      lowest: Math.abs(selected.decisionLoss - best.decisionLoss) < 1e-10,
    };
  });
}

export function evaluateV4Heuristic(rows: ScoreV6CandidateRow[]): ScoreV6Metrics {
  return evaluateScoreV6(rows, (row) => scoreFromWeights(row.layerScores, HEURISTIC_WEIGHTS));
}

export function groupWeights(weights: ScoreV6FeatureWeights) {
  return Object.fromEntries(
    Object.entries(SCORE_V6_DIAGNOSTIC_GROUPS).map(([group, features]) => [
      group,
      features.reduce(
        (total, feature) => total + (weights[feature] ?? 0),
        0,
      ),
    ]),
  ) as Record<keyof typeof SCORE_V6_DIAGNOSTIC_GROUPS, number>;
}

export function featureGroup(feature: ScoreV6Feature) {
  return groupForFeature(feature);
}
