import type { SviScoreV4TrainingConfiguration } from "./contract";
import {
  SVI_SCORE_V4_FEATURES,
  transformedV4Feature,
  v4FeatureValue,
} from "./features";
import type {
  SviScoreV4CandidateRow,
  SviScoreV4Feature,
  SviScoreV4FeatureWeights,
  SviScoreV4Metrics,
  SviScoreV4Model,
  SviScoreV4Selection,
  SviScoreV4TopOneBoundReceipt,
} from "./types";

const ECONOMIC_SCALE = 1;
const FEATURE_CAP = 0.18;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function average(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) /
    Math.max(values.length, 1);
}

function quantile(values: number[], probability: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const position = clamp(probability, 0, 1) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
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

function existingEligibilityPool(
  rows: readonly SviScoreV4CandidateRow[],
): SviScoreV4CandidateRow[] {
  const decisionGrade = rows.filter((row) => row.eligible);
  if (decisionGrade.length) return decisionGrade;
  const review = rows.filter((row) => row.reviewEligible);
  return review.length ? review : [...rows];
}

export function passesV4Safety(
  row: SviScoreV4CandidateRow,
  configuration: SviScoreV4TrainingConfiguration,
): boolean {
  return v4FeatureValue(row, "roi-posterior-plausibility") >=
      configuration.safety.roiPlausibility &&
    v4FeatureValue(row, "roi-decision-stability") >=
      configuration.safety.decisionStability;
}

export interface SviScoreV4Pool {
  rows: SviScoreV4CandidateRow[];
  hasSafetyAcceptedCandidate: boolean;
}

export function v4SelectionPool(
  rows: readonly SviScoreV4CandidateRow[],
  configuration: SviScoreV4TrainingConfiguration,
): SviScoreV4Pool {
  const eligible = existingEligibilityPool(rows);
  const safe = eligible.filter((row) => passesV4Safety(row, configuration));
  return {
    rows: safe.length ? safe : eligible,
    hasSafetyAcceptedCandidate: safe.length > 0,
  };
}

function uniformWeights(): SviScoreV4FeatureWeights {
  return Object.fromEntries(
    SVI_SCORE_V4_FEATURES.map((feature) => [
      feature,
      1 / SVI_SCORE_V4_FEATURES.length,
    ]),
  );
}

function normalizeCapped(
  source: SviScoreV4FeatureWeights,
): SviScoreV4FeatureWeights {
  const raw = SVI_SCORE_V4_FEATURES.map((feature) =>
    Math.max(0, source[feature] ?? 0)
  );
  if (raw.every((value) => value <= 1e-12)) return uniformWeights();
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
    const capped = provisional.filter((item) => item.value > FEATURE_CAP);
    if (!capped.length) {
      provisional.forEach((item) => {
        output[item.index] = item.value;
      });
      remaining = 0;
      break;
    }
    capped.forEach((item) => {
      output[item.index] = FEATURE_CAP;
      remaining -= FEATURE_CAP;
    });
    const cappedIndexes = new Set(capped.map((item) => item.index));
    active = active.filter((index) => !cappedIndexes.has(index));
  }
  const total = output.reduce((sum, value) => sum + value, 0);
  if (total <= 1e-12) return uniformWeights();
  return Object.fromEntries(
    SVI_SCORE_V4_FEATURES.map((feature, index) => [feature, output[index] / total]),
  );
}

function assertContextInvariant(rows: readonly SviScoreV4CandidateRow[]): number {
  const contexts = rows.map((row) => row.temporal.context);
  const minimum = Math.min(...contexts);
  const maximum = Math.max(...contexts);
  if (maximum - minimum > 1e-9) {
    throw new Error(
      `Temporal context must be candidate-invariant within ${rows[0]?.businessId}.`,
    );
  }
  return clamp(average(contexts), 0, 1);
}

export function latentV4Score(
  row: SviScoreV4CandidateRow,
  model: Pick<SviScoreV4Model, "ordinaryWeights" | "temporalWeights">,
): number {
  const context = clamp(row.temporal.context, 0, 1);
  return SVI_SCORE_V4_FEATURES.reduce((total, feature) => {
    const weight = (1 - context) * (model.ordinaryWeights[feature] ?? 0) +
      context * (model.temporalWeights[feature] ?? 0);
    return total + weight * transformedV4Feature(row, feature);
  }, 0);
}

export function scoreV4(
  row: SviScoreV4CandidateRow,
  model: Pick<SviScoreV4Model, "ordinaryWeights" | "temporalWeights">,
): number {
  return 100 * Math.exp(latentV4Score(row, model));
}

function normalizedGap(candidateLoss: number, oracleLoss: number): number {
  return clamp(
    Math.max(0, candidateLoss - oracleLoss) / ECONOMIC_SCALE,
    0,
    1,
  );
}

interface BusinessTopOneLoss {
  businessId: string;
  family: string;
  loss: number;
  shortGradient: number[];
  temporalGradient: number[];
}

function topOneLossForBusiness(
  business: readonly SviScoreV4CandidateRow[],
  model: SviScoreV4Model,
): BusinessTopOneLoss {
  const pool = v4SelectionPool(business, model.configuration).rows;
  const context = assertContextInvariant(pool);
  const ordered = [...pool].sort(
    (left, right) =>
      left.decisionLoss - right.decisionLoss ||
      left.candidateId.localeCompare(right.candidateId),
  );
  const oracle = ordered[0];
  if (!oracle) throw new Error("A V4 business has no candidate rows.");
  const oracleScore = latentV4Score(oracle, model);
  let maximum = 0;
  let challenger: SviScoreV4CandidateRow | undefined;
  ordered.slice(1).forEach((candidate) => {
    const term = normalizedGap(candidate.decisionLoss, oracle.decisionLoss) +
      latentV4Score(candidate, model) - oracleScore;
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
      shortGradient: SVI_SCORE_V4_FEATURES.map(() => 0),
      temporalGradient: SVI_SCORE_V4_FEATURES.map(() => 0),
    };
  }
  const difference = SVI_SCORE_V4_FEATURES.map(
    (feature) =>
      transformedV4Feature(challenger!, feature) -
      transformedV4Feature(oracle, feature),
  );
  return {
    businessId: oracle.businessId,
    family: oracle.family,
    loss: maximum,
    shortGradient: difference.map((value) => (1 - context) * value),
    temporalGradient: difference.map((value) => context * value),
  };
}

export function learnScoreV4(
  rows: readonly SviScoreV4CandidateRow[],
  configuration: SviScoreV4TrainingConfiguration,
): SviScoreV4Model {
  if (!rows.length) throw new Error("V4 score learning requires candidate rows.");
  let model: SviScoreV4Model = {
    ordinaryWeights: uniformWeights(),
    temporalWeights: uniformWeights(),
    configuration,
  };
  const baseline = uniformWeights();
  for (let epoch = 0; epoch < configuration.epochs; epoch += 1) {
    const losses = groupRows(rows).map((business) =>
      topOneLossForBusiness(business, model)
    );
    const tailThreshold = quantile(
      losses.map((item) => item.loss),
      0.9,
    );
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
    const ordinaryGradient = SVI_SCORE_V4_FEATURES.map(() => 0);
    const temporalGradient = SVI_SCORE_V4_FEATURES.map(() => 0);
    losses.forEach((item) => {
      const multiplier = 1 / losses.length +
        (item.loss >= tailThreshold - 1e-12
          ? configuration.cvarWeight / Math.max(tail.length, 1)
          : 0) +
        (item.family === worstFamily
          ? configuration.familyDroWeight / Math.max(worstFamilyRows.length, 1)
          : 0);
      item.shortGradient.forEach((value, index) => {
        ordinaryGradient[index] += multiplier * value;
      });
      item.temporalGradient.forEach((value, index) => {
        temporalGradient[index] += multiplier * value;
      });
    });
    SVI_SCORE_V4_FEATURES.forEach((feature, index) => {
      ordinaryGradient[index] += configuration.regularization *
        ((model.ordinaryWeights[feature] ?? 0) - (baseline[feature] ?? 0));
      temporalGradient[index] += configuration.regularization *
        ((model.temporalWeights[feature] ?? 0) - (baseline[feature] ?? 0));
    });
    const learningRate = 0.18 / Math.sqrt(epoch + 8);
    model = {
      ...model,
      ordinaryWeights: normalizeCapped(Object.fromEntries(
        SVI_SCORE_V4_FEATURES.map((feature, index) => [
          feature,
          (model.ordinaryWeights[feature] ?? 0) -
            learningRate * ordinaryGradient[index],
        ]),
      )),
      temporalWeights: normalizeCapped(Object.fromEntries(
        SVI_SCORE_V4_FEATURES.map((feature, index) => [
          feature,
          (model.temporalWeights[feature] ?? 0) -
            learningRate * temporalGradient[index],
        ]),
      )),
    };
  }
  return model;
}

export function selectScoreV4Candidates(
  rows: readonly SviScoreV4CandidateRow[],
  model: SviScoreV4Model,
): SviScoreV4Selection[] {
  return groupRows(rows).map((business) => {
    const poolReceipt = v4SelectionPool(business, model.configuration);
    const pool = poolReceipt.rows;
    const selected = [...pool].sort(
      (left, right) =>
        latentV4Score(right, model) - latentV4Score(left, model) ||
        left.candidateId.localeCompare(right.candidateId),
    )[0];
    const oracle = [...pool].sort(
      (left, right) =>
        left.decisionLoss - right.decisionLoss ||
        left.candidateId.localeCompare(right.candidateId),
    )[0];
    if (!selected || !oracle) throw new Error("A V4 business has no selectable candidates.");
    const excessLoss = Math.max(0, selected.decisionLoss - oracle.decisionLoss);
    return {
      businessId: selected.businessId,
      candidateId: selected.candidateId,
      family: selected.family,
      loss: selected.decisionLoss,
      excessLoss,
      normalizedExcessRegret: normalizedGap(
        selected.decisionLoss,
        oracle.decisionLoss,
      ),
      oracle: excessLoss <= 1e-10,
      decisionGrade: selected.eligible && poolReceipt.hasSafetyAcceptedCandidate,
      safetyAccepted: passesV4Safety(selected, model.configuration),
      reviewOnlyFallback: !poolReceipt.hasSafetyAcceptedCandidate,
    };
  });
}

export function evaluateScoreV4(
  rows: readonly SviScoreV4CandidateRow[],
  model: SviScoreV4Model,
): SviScoreV4Metrics {
  return metricsFromV4Selections(selectScoreV4Candidates(rows, model));
}

export function metricsFromV4Selections(
  selections: readonly SviScoreV4Selection[],
): SviScoreV4Metrics {
  const losses = selections.map((selection) => selection.loss);
  const excess = selections.map((selection) => selection.excessLoss);
  const lossTailThreshold = quantile(losses, 0.9);
  const lossTail = losses.filter((value) => value >= lossTailThreshold - 1e-12);
  const tailThreshold = quantile(excess, 0.9);
  const tail = excess.filter((value) => value >= tailThreshold - 1e-12);
  const families = [...new Set(selections.map((selection) => selection.family))];
  const familyMeanLoss = Object.fromEntries(
    families.sort().map((family) => [
      family,
      average(
        selections
          .filter((selection) => selection.family === family)
          .map((selection) => selection.loss),
      ),
    ]),
  );
  const familyMeanExcessLoss = Object.fromEntries(
    families.sort().map((family) => [
      family,
      average(
        selections
          .filter((selection) => selection.family === family)
          .map((selection) => selection.excessLoss),
      ),
    ]),
  );
  return {
    businesses: selections.length,
    meanLoss: average(losses),
    medianLoss: quantile(losses, 0.5),
    p90Loss: quantile(losses, 0.9),
    p95Loss: quantile(losses, 0.95),
    cvar90Loss: average(lossTail),
    worstFamilyMeanLoss: Math.max(0, ...Object.values(familyMeanLoss)),
    familyMeanLoss,
    meanExcessLoss: average(excess),
    medianExcessLoss: quantile(excess, 0.5),
    p90ExcessLoss: quantile(excess, 0.9),
    p95ExcessLoss: quantile(excess, 0.95),
    cvar90ExcessLoss: average(tail),
    worstFamilyMeanExcessLoss: Math.max(0, ...Object.values(familyMeanExcessLoss)),
    familyMeanExcessLoss,
    oracleSelectionRate: average(
      selections.map((selection) => selection.oracle ? 1 : 0),
    ),
    safetyAcceptedShare: average(
      selections.map((selection) => selection.safetyAccepted ? 1 : 0),
    ),
    reviewOnlyFallbackShare: average(
      selections.map((selection) => selection.reviewOnlyFallback ? 1 : 0),
    ),
  };
}

function logSumExp(values: number[], temperature: number): number {
  const scaled = values.map((value) => value / temperature);
  const maximum = Math.max(...scaled);
  return temperature * (
    maximum + Math.log(scaled.reduce(
      (total, value) => total + Math.exp(value - maximum),
      0,
    ))
  );
}

/**
 * For the in-pool oracle c* and selected candidate ĉ = argmax s(c):
 *
 *   Delta(ĉ) <= max_{c != c*} [Delta(c) + s(c) - s(c*)]_+.
 *
 * The selected candidate itself appears on the right and s(ĉ) >= s(c*), so
 * this finite-candidate inequality is exact. The log-sum-exp receipt is a
 * differentiable upper bound on the same maximum.
 */
export function evaluateTopOneRegretBound(
  rows: readonly SviScoreV4CandidateRow[],
  model: SviScoreV4Model,
  temperature = 0.05,
): SviScoreV4TopOneBoundReceipt {
  const receipts = groupRows(rows).map((business) => {
    const pool = v4SelectionPool(business, model.configuration).rows;
    const oracle = [...pool].sort(
      (left, right) =>
        left.decisionLoss - right.decisionLoss ||
        left.candidateId.localeCompare(right.candidateId),
    )[0];
    const selected = [...pool].sort(
      (left, right) =>
        latentV4Score(right, model) - latentV4Score(left, model) ||
        left.candidateId.localeCompare(right.candidateId),
    )[0];
    const oracleScore = latentV4Score(oracle, model);
    const terms = pool
      .filter((candidate) => candidate !== oracle)
      .map((candidate) =>
        normalizedGap(candidate.decisionLoss, oracle.decisionLoss) +
          latentV4Score(candidate, model) - oracleScore
      );
    const hinge = Math.max(0, ...terms);
    const smooth = logSumExp([0, ...terms], temperature);
    const regret = normalizedGap(selected.decisionLoss, oracle.decisionLoss);
    return { regret, hinge, smooth, slack: regret - hinge };
  });
  return {
    businesses: receipts.length,
    violations: receipts.filter((receipt) => receipt.slack > 1e-10).length,
    meanSelectedNormalizedRegret: average(
      receipts.map((receipt) => receipt.regret),
    ),
    meanTopOneHingeBound: average(receipts.map((receipt) => receipt.hinge)),
    meanSmoothTopOneBound: average(receipts.map((receipt) => receipt.smooth)),
    maximumNumericalSlack: Math.max(0, ...receipts.map((receipt) => receipt.slack)),
  };
}

export function riskObjective(metrics: SviScoreV4Metrics, configuration: SviScoreV4TrainingConfiguration): number {
  return metrics.meanLoss +
    configuration.cvarWeight * metrics.cvar90Loss +
    configuration.familyDroWeight * metrics.worstFamilyMeanLoss;
}
