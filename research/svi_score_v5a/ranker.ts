import type { SviScoreV4CandidateRow, SviScoreV4Metrics } from "../svi_score_v4/types";
import {
  metricsFromV4Selections,
  passesV4Safety,
  v4SelectionPool,
} from "../svi_score_v4/ranker";
import {
  V5A_COMMON_POOL_CONFIGURATION,
  type SviScoreV5ATrainingConfiguration,
} from "./contract";
import {
  v5aFeatureNames,
  v5aFeatureVector,
  v5aPriorWeights,
} from "./features";
import type {
  SviScoreV5AModel,
  SviScoreV5ASelection,
  SviScoreV5ATheoremReceipt,
} from "./types";

const ECONOMIC_SCALE = 1;
const FEATURE_CACHE = new WeakMap<SviScoreV4CandidateRow, number[]>();

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function average(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0) /
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
  const value = v5aFeatureVector(row);
  FEATURE_CACHE.set(row, value);
  return value;
}

function groupRows(
  rows: readonly SviScoreV4CandidateRow[],
): SviScoreV4CandidateRow[][] {
  const grouped = new Map<string, SviScoreV4CandidateRow[]>();
  rows.forEach((row) => {
    grouped.set(row.businessId, [...(grouped.get(row.businessId) ?? []), row]);
  });
  return [...grouped.values()].sort((left, right) =>
    (left[0]?.businessId ?? "").localeCompare(right[0]?.businessId ?? "")
  );
}

function normalizedGap(candidateLoss: number, oracleLoss: number): number {
  return clamp(Math.max(0, candidateLoss - oracleLoss) / ECONOMIC_SCALE, 0, 1);
}

function normalizeCapped(
  source: readonly number[],
  featureCap: number,
): number[] {
  const raw = source.map((value) => Math.max(0, Number.isFinite(value) ? value : 0));
  if (!raw.length) throw new Error("V5A cannot normalize an empty feature vector.");
  if (featureCap * raw.length < 1 - 1e-12) {
    throw new Error("V5A feature cap is too small to form a unit simplex.");
  }
  if (raw.every((value) => value <= 1e-12)) {
    return raw.map(() => 1 / raw.length);
  }
  const output = raw.map(() => 0);
  let remaining = 1;
  let active = raw.map((_, index) => index);
  while (active.length && remaining > 1e-12) {
    const activeTotal = active.reduce((sum, index) => sum + raw[index], 0);
    const provisional = active.map((index) => ({
      index,
      value: activeTotal > 1e-12
        ? remaining * raw[index] / activeTotal
        : remaining / active.length,
    }));
    const capped = provisional.filter((item) => item.value > featureCap);
    if (!capped.length) {
      provisional.forEach((item) => {
        output[item.index] = item.value;
      });
      remaining = 0;
      break;
    }
    capped.forEach((item) => {
      output[item.index] = featureCap;
      remaining -= featureCap;
    });
    const cappedIndexes = new Set(capped.map((item) => item.index));
    active = active.filter((index) => !cappedIndexes.has(index));
  }
  const total = output.reduce((sum, value) => sum + value, 0);
  return output.map((value) => value / Math.max(total, 1e-12));
}

export function latentV5AScore(
  row: SviScoreV4CandidateRow,
  model: Pick<SviScoreV5AModel, "featureNames" | "weights">,
): number {
  const values = vector(row);
  if (values.length !== model.weights.length) {
    throw new Error("V5A feature vector does not match the fitted selector.");
  }
  return values.reduce(
    (total, value, index) => total + value * model.weights[index],
    0,
  );
}

export function scoreV5A(
  row: SviScoreV4CandidateRow,
  model: Pick<SviScoreV5AModel, "featureNames" | "weights">,
): number {
  return 100 * Math.exp(latentV5AScore(row, model));
}

interface BusinessTopOneLoss {
  businessId: string;
  family: string;
  loss: number;
  gradient: number[];
}

function topOneLossForBusiness(
  business: readonly SviScoreV4CandidateRow[],
  model: SviScoreV5AModel,
): BusinessTopOneLoss {
  const pool = v4SelectionPool(business, V5A_COMMON_POOL_CONFIGURATION).rows;
  const ordered = [...pool].sort(
    (left, right) =>
      left.decisionLoss - right.decisionLoss ||
      left.candidateId.localeCompare(right.candidateId),
  );
  const oracle = ordered[0];
  if (!oracle) throw new Error("A V5A business has no candidate rows.");
  const oracleScore = latentV5AScore(oracle, model);
  let maximum = 0;
  let challenger: SviScoreV4CandidateRow | undefined;
  ordered.slice(1).forEach((candidate) => {
    const term = normalizedGap(candidate.decisionLoss, oracle.decisionLoss) +
      latentV5AScore(candidate, model) - oracleScore;
    if (
      term > maximum + 1e-12 ||
      (Math.abs(term - maximum) <= 1e-12 &&
        candidate.candidateId.localeCompare(challenger?.candidateId ?? "\uffff") < 0)
    ) {
      maximum = term;
      challenger = candidate;
    }
  });
  if (!challenger || maximum <= 0) {
    return {
      businessId: oracle.businessId,
      family: oracle.family,
      loss: 0,
      gradient: model.weights.map(() => 0),
    };
  }
  const oracleFeatures = vector(oracle);
  const challengerFeatures = vector(challenger);
  return {
    businessId: oracle.businessId,
    family: oracle.family,
    loss: maximum,
    gradient: challengerFeatures.map(
      (value, index) => value - oracleFeatures[index],
    ),
  };
}

function assertFeatureContract(rows: readonly SviScoreV4CandidateRow[]): string[] {
  const names = v5aFeatureNames(rows[0]);
  rows.forEach((row) => {
    const rowNames = v5aFeatureNames(row);
    if (
      rowNames.length !== names.length ||
      rowNames.some((name, index) => name !== names[index])
    ) {
      throw new Error(`V5A feature contract changed for ${row.businessId}/${row.candidateId}.`);
    }
    if (vector(row).some((value) => !Number.isFinite(value))) {
      throw new Error(`V5A feature is not finite for ${row.businessId}/${row.candidateId}.`);
    }
  });
  return names;
}

export function learnScoreV5A(
  rows: readonly SviScoreV4CandidateRow[],
  configuration: SviScoreV5ATrainingConfiguration,
): SviScoreV5AModel {
  if (!rows.length) throw new Error("V5A score learning requires candidate rows.");
  const featureNames = assertFeatureContract(rows);
  const prior = normalizeCapped(v5aPriorWeights(rows[0]), configuration.featureCap);
  let model: SviScoreV5AModel = {
    featureNames,
    weights: [...prior],
    configuration: structuredClone(configuration),
  };
  for (let epoch = 0; epoch < configuration.epochs; epoch += 1) {
    const losses = groupRows(rows).map((business) =>
      topOneLossForBusiness(business, model)
    );
    const tailThreshold = quantile(losses.map((item) => item.loss), 0.9);
    const tail = losses.filter((item) => item.loss >= tailThreshold - 1e-12);
    const familyLosses = new Map<string, number[]>();
    losses.forEach((item) => {
      familyLosses.set(item.family, [
        ...(familyLosses.get(item.family) ?? []),
        item.loss,
      ]);
    });
    const worstFamily = [...familyLosses.entries()].sort(
      (left, right) =>
        average(right[1]) - average(left[1]) || left[0].localeCompare(right[0]),
    )[0]?.[0];
    const worstFamilyRows = losses.filter((item) => item.family === worstFamily);
    const gradient = model.weights.map(() => 0);
    losses.forEach((item) => {
      const multiplier = 1 / losses.length +
        (item.loss >= tailThreshold - 1e-12
          ? configuration.cvarWeight / Math.max(tail.length, 1)
          : 0) +
        (item.family === worstFamily
          ? configuration.familyDroWeight / Math.max(worstFamilyRows.length, 1)
          : 0);
      item.gradient.forEach((value, index) => {
        gradient[index] += multiplier * value;
      });
    });
    gradient.forEach((_, index) => {
      gradient[index] += configuration.regularization *
        (model.weights[index] - prior[index]);
    });
    const learningRate = configuration.learningRate / Math.sqrt(epoch + 8);
    model = {
      ...model,
      weights: normalizeCapped(
        model.weights.map(
          (weight, index) => weight - learningRate * gradient[index],
        ),
        configuration.featureCap,
      ),
    };
  }
  return model;
}

export function selectScoreV5ACandidates(
  rows: readonly SviScoreV4CandidateRow[],
  model: SviScoreV5AModel,
): SviScoreV5ASelection[] {
  return groupRows(rows).map((business) => {
    const poolReceipt = v4SelectionPool(business, V5A_COMMON_POOL_CONFIGURATION);
    const selected = [...poolReceipt.rows].sort(
      (left, right) =>
        latentV5AScore(right, model) - latentV5AScore(left, model) ||
        left.candidateId.localeCompare(right.candidateId),
    )[0];
    const oracle = [...poolReceipt.rows].sort(
      (left, right) =>
        left.decisionLoss - right.decisionLoss ||
        left.candidateId.localeCompare(right.candidateId),
    )[0];
    if (!selected || !oracle) throw new Error("A V5A business has no selectable candidates.");
    const excessLoss = Math.max(0, selected.decisionLoss - oracle.decisionLoss);
    return {
      businessId: selected.businessId,
      candidateId: selected.candidateId,
      family: selected.family,
      loss: selected.decisionLoss,
      excessLoss,
      normalizedExcessRegret: normalizedGap(selected.decisionLoss, oracle.decisionLoss),
      oracle: excessLoss <= 1e-10,
      decisionGrade: selected.eligible && poolReceipt.hasSafetyAcceptedCandidate,
      safetyAccepted: passesV4Safety(selected, V5A_COMMON_POOL_CONFIGURATION),
      reviewOnlyFallback: !poolReceipt.hasSafetyAcceptedCandidate,
    };
  });
}

export function evaluateScoreV5A(
  rows: readonly SviScoreV4CandidateRow[],
  model: SviScoreV5AModel,
): SviScoreV4Metrics {
  return metricsFromV4Selections(selectScoreV5ACandidates(rows, model));
}

function logSumExp(values: readonly number[], temperature: number): number {
  const scaled = values.map((value) => value / temperature);
  const maximum = Math.max(...scaled);
  return temperature * (
    maximum + Math.log(scaled.reduce(
      (total, value) => total + Math.exp(value - maximum),
      0,
    ))
  );
}

export function evaluateV5ATopOneRegretBound(
  rows: readonly SviScoreV4CandidateRow[],
  model: SviScoreV5AModel,
  temperature = 0.05,
): SviScoreV5ATheoremReceipt {
  const receipts = groupRows(rows).map((business) => {
    const pool = v4SelectionPool(business, V5A_COMMON_POOL_CONFIGURATION).rows;
    const oracle = [...pool].sort(
      (left, right) =>
        left.decisionLoss - right.decisionLoss ||
        left.candidateId.localeCompare(right.candidateId),
    )[0];
    const selected = [...pool].sort(
      (left, right) =>
        latentV5AScore(right, model) - latentV5AScore(left, model) ||
        left.candidateId.localeCompare(right.candidateId),
    )[0];
    const oracleScore = latentV5AScore(oracle, model);
    const terms = pool
      .filter((candidate) => candidate !== oracle)
      .map((candidate) =>
        normalizedGap(candidate.decisionLoss, oracle.decisionLoss) +
          latentV5AScore(candidate, model) - oracleScore
      );
    const hinge = Math.max(0, ...terms);
    const smooth = logSumExp([0, ...terms], temperature);
    const regret = normalizedGap(selected.decisionLoss, oracle.decisionLoss);
    return { regret, hinge, smooth, slack: regret - hinge };
  });
  return {
    businesses: receipts.length,
    violations: receipts.filter((receipt) => receipt.slack > 1e-10).length,
    meanSelectedNormalizedRegret: average(receipts.map((receipt) => receipt.regret)),
    meanTopOneHingeBound: average(receipts.map((receipt) => receipt.hinge)),
    meanSmoothTopOneBound: average(receipts.map((receipt) => receipt.smooth)),
    maximumNumericalSlack: Math.max(0, ...receipts.map((receipt) => receipt.slack)),
  };
}

export function v5aSelectionObjective(metrics: SviScoreV4Metrics): number {
  return metrics.meanLoss +
    0.5 * metrics.cvar90Loss +
    0.5 * metrics.worstFamilyMeanLoss;
}
