import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";
import {
  metricsFromV4Selections,
  passesV4Safety,
  v4SelectionPool,
} from "../svi_score_v4/ranker";
import { V5A_COMMON_POOL_CONFIGURATION } from "../svi_score_v5a/contract";
import { v5aSelectionObjective } from "../svi_score_v5a/ranker";
import {
  V5B_SEARCH_ALGORITHMS,
  type V5BSearchAlgorithm,
} from "../svi_score_v5b/contract";
import { v5bCandidateUniverse } from "../svi_score_v5b/embedding";
import { runV5BSearch } from "../svi_score_v5b/search";
import type {
  V5BCheckpointMetrics,
  V5BObservedCandidate,
  V5BSearchCandidate,
} from "../svi_score_v5b/types";
import { selectV5DOrderIndependent } from "../svi_score_v5d/evaluate";
import type { V5DLossPrediction } from "../svi_score_v5d/types";
import { SVI_SCORE_V7_CONTRACT, SVI_SCORE_V7_VERSION } from "./contract";

export interface V7Prediction extends V5DLossPrediction {
  businessId: string;
  candidateId: string;
  roiError: number;
  contributionError: number;
  fold: number;
  region: number;
}

interface Selection {
  businessId: string;
  family: string;
  algorithm: V5BSearchAlgorithm;
  budget: number;
  candidateId: string;
  loss: number;
  excessLoss: number;
  safetyAccepted: boolean;
  reviewOnlyFallback: boolean;
  championRecovered: boolean;
  oracleRecovered: boolean;
  riskGap: number;
}

export type V7CheckpointMetrics = V5BCheckpointMetrics;

function average(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) /
    Math.max(values.length, 1);
}

function quantile(values: readonly number[], probability: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const position = Math.min(1, Math.max(0, probability)) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function correlation(left: readonly number[], right: readonly number[]): number {
  const leftMean = average(left);
  const rightMean = average(right);
  const numerator = left.reduce(
    (sum, value, index) => sum + (value - leftMean) * ((right[index] ?? rightMean) - rightMean),
    0,
  );
  const denominator = Math.sqrt(
    left.reduce((sum, value) => sum + (value - leftMean) ** 2, 0) *
      right.reduce((sum, value) => sum + (value - rightMean) ** 2, 0),
  );
  return denominator > 1e-12 ? numerator / denominator : 0;
}

function grouped(
  rows: readonly SviScoreV4CandidateRow[],
): Map<string, SviScoreV4CandidateRow[]> {
  const output = new Map<string, SviScoreV4CandidateRow[]>();
  rows.forEach((row) => {
    output.set(row.businessId, [...(output.get(row.businessId) ?? []), row]);
  });
  return output;
}

function oracle(rows: readonly SviScoreV4CandidateRow[]): SviScoreV4CandidateRow {
  return [...v4SelectionPool(rows, V5A_COMMON_POOL_CONFIGURATION).rows].sort(
    (left, right) =>
      left.decisionLoss - right.decisionLoss ||
      left.candidateId.localeCompare(right.candidateId),
  )[0];
}

function selectionAtBudget(
  businessRows: readonly SviScoreV4CandidateRow[],
  predictions: Map<string, V7Prediction>,
  algorithm: V5BSearchAlgorithm,
  order: readonly string[],
  budget: number,
): Selection {
  const byCandidate = new Map(businessRows.map((row) => [row.candidateId, row]));
  const evaluated = order.slice(0, budget).map((candidateId) => {
    const row = byCandidate.get(candidateId);
    if (!row) throw new Error(`V7 search revealed unknown ${candidateId}.`);
    return row;
  });
  const selected = selectV5DOrderIndependent(evaluated, predictions);
  const champion = selectV5DOrderIndependent(businessRows, predictions).row;
  const economicOracle = oracle(businessRows);
  const selectedRisk = predictions.get(selected.row.candidateId)?.risk;
  const championRisk = predictions.get(champion.candidateId)?.risk;
  if (selectedRisk === undefined || championRisk === undefined) {
    throw new Error("V7 prediction map is incomplete.");
  }
  const excessLoss = Math.max(0, selected.row.decisionLoss - economicOracle.decisionLoss);
  return {
    businessId: selected.row.businessId,
    family: selected.row.family,
    algorithm,
    budget,
    candidateId: selected.row.candidateId,
    loss: selected.row.decisionLoss,
    excessLoss,
    safetyAccepted: passesV4Safety(selected.row, V5A_COMMON_POOL_CONFIGURATION),
    reviewOnlyFallback: selected.reviewOnlyFallback,
    championRecovered: selected.row.candidateId === champion.candidateId,
    oracleRecovered: excessLoss <= 1e-10,
    riskGap: selectedRisk - championRisk,
  };
}

function metrics(selections: readonly Selection[], budget: number): V7CheckpointMetrics {
  const base = metricsFromV4Selections(selections.map((selection) => ({
    businessId: selection.businessId,
    candidateId: selection.candidateId,
    family: selection.family,
    loss: selection.loss,
    excessLoss: selection.excessLoss,
    normalizedExcessRegret: selection.excessLoss,
    oracle: selection.oracleRecovered,
    decisionGrade: selection.safetyAccepted && !selection.reviewOnlyFallback,
    safetyAccepted: selection.safetyAccepted,
    reviewOnlyFallback: selection.reviewOnlyFallback,
  })));
  return {
    ...base,
    budget,
    selectionObjective: v5aSelectionObjective(base),
    fullPoolChampionRecall: average(
      selections.map((selection) => selection.championRecovered ? 1 : 0),
    ),
    economicOracleRecall: average(
      selections.map((selection) => selection.oracleRecovered ? 1 : 0),
    ),
    meanSelectorScoreGap: average(selections.map((selection) => selection.riskGap)),
  };
}

export function evaluateV7(
  rows: readonly SviScoreV4CandidateRow[],
  predictions: readonly V7Prediction[],
): {
  version: typeof SVI_SCORE_V7_VERSION;
  contract: typeof SVI_SCORE_V7_CONTRACT;
  proxyAssessment: {
    candidateLevelCorrelation: number;
    meanAbsoluteError: number;
    p90Coverage: number;
    roiErrorMeanAbsoluteError: number;
    contributionErrorMeanAbsoluteError: number;
    pairwiseOrderingAccuracy: number;
    economicallyWeightedPairwiseAccuracy: number;
    riskChampionOracleRecall: number;
    riskChampionMeanExcessLoss: number;
    riskChampionP90ExcessLoss: number;
    dangerousFalseChampionShare: number;
  };
  algorithms: Array<{
    algorithm: V5BSearchAlgorithm;
    checkpoints: V7CheckpointMetrics[];
    primary: V7CheckpointMetrics;
    primaryObjective: number;
  }>;
  primaryWinner: V5BSearchAlgorithm;
  fullPoolAlgorithmAgreement: boolean;
} {
  if (rows.some((row) => row.split === "audit")) {
    throw new Error("The sealed audit is forbidden in V7.");
  }
  const groupedRows = grouped(rows);
  if (groupedRows.size !== 420 || rows.length !== 20_160) {
    throw new Error("V7 requires the frozen 420 x 48 development cohort.");
  }
  const predictionsByKey = new Map(
    predictions.map((prediction) => [
      `${prediction.businessId}\u0000${prediction.candidateId}`,
      prediction,
    ]),
  );
  if (predictionsByKey.size !== rows.length) {
    throw new Error("V7 predictions are incomplete or duplicated.");
  }
  const universe = v5bCandidateUniverse(rows.map((row) => row.candidateId));
  const selections = new Map<V5BSearchAlgorithm, Selection[]>();
  V5B_SEARCH_ALGORITHMS.forEach((algorithm) => selections.set(algorithm, []));
  const actualExcess: number[] = [];
  const predictedMean: number[] = [];
  const predictedP90: number[] = [];
  const predictedRoi: number[] = [];
  const actualRoi: number[] = [];
  const predictedContribution: number[] = [];
  const actualContribution: number[] = [];
  let correctPairs = 0;
  let totalPairs = 0;
  let correctWeight = 0;
  let totalWeight = 0;
  const championExcess: number[] = [];

  [...groupedRows.entries()].sort().forEach(([businessId, businessRows]) => {
    const businessPredictions = new Map<string, V7Prediction>();
    businessRows.forEach((row) => {
      const prediction = predictionsByKey.get(`${businessId}\u0000${row.candidateId}`);
      if (!prediction) throw new Error(`V7 prediction missing for ${businessId}/${row.candidateId}.`);
      businessPredictions.set(row.candidateId, prediction);
    });
    const economicOracle = oracle(businessRows);
    const champion = selectV5DOrderIndependent(businessRows, businessPredictions).row;
    businessRows.forEach((row) => {
      const prediction = businessPredictions.get(row.candidateId)!;
      actualExcess.push(Math.max(0, row.decisionLoss - economicOracle.decisionLoss));
      predictedMean.push(prediction.mean);
      predictedP90.push(prediction.p90);
      predictedRoi.push(prediction.roiError);
      actualRoi.push(row.roiError);
      predictedContribution.push(prediction.contributionError);
      actualContribution.push(row.contributionError);
    });
    championExcess.push(Math.max(0, champion.decisionLoss - economicOracle.decisionLoss));
    for (let left = 0; left < businessRows.length; left += 1) {
      for (let right = left + 1; right < businessRows.length; right += 1) {
        const actualDifference = businessRows[left].decisionLoss - businessRows[right].decisionLoss;
        if (Math.abs(actualDifference) <= 1e-10) continue;
        const predictedDifference =
          businessPredictions.get(businessRows[left].candidateId)!.risk -
          businessPredictions.get(businessRows[right].candidateId)!.risk;
        const correct = actualDifference * predictedDifference > 0;
        totalPairs += 1;
        totalWeight += Math.abs(actualDifference);
        if (correct) {
          correctPairs += 1;
          correctWeight += Math.abs(actualDifference);
        }
      }
    }
    V5B_SEARCH_ALGORITHMS.forEach((algorithm) => {
      const revealed = new Set<string>();
      const trace = runV5BSearch(
        businessId,
        algorithm,
        universe,
        (candidate: V5BSearchCandidate): V5BObservedCandidate => {
          if (revealed.has(candidate.candidateId)) {
            throw new Error(`V7 ${algorithm} duplicated ${candidate.candidateId}.`);
          }
          const row = businessRows.find((item) => item.candidateId === candidate.candidateId);
          const prediction = businessPredictions.get(candidate.candidateId);
          if (!row || !prediction) throw new Error("V7 reveal failed.");
          revealed.add(candidate.candidateId);
          return {
            ...candidate,
            selectorScore: -prediction.risk,
            safetyAccepted: passesV4Safety(row, V5A_COMMON_POOL_CONFIGURATION),
          };
        },
      );
      [4, 8, 12, 16, 24, 32, 48].forEach((budget) => {
        selections.get(algorithm)!.push(selectionAtBudget(
          businessRows,
          businessPredictions,
          algorithm,
          trace.orderedCandidateIds,
          budget,
        ));
      });
    });
  });
  const algorithms = V5B_SEARCH_ALGORITHMS.map((algorithm) => {
    const all = selections.get(algorithm)!;
    const checkpoints = [4, 8, 12, 16, 24, 32, 48].map((budget) =>
      metrics(all.filter((selection) => selection.budget === budget), budget)
    );
    const primary = checkpoints.find((checkpoint) => checkpoint.budget === 16)!;
    return { algorithm, checkpoints, primary, primaryObjective: primary.selectionObjective };
  });
  const fullPoolMaps = V5B_SEARCH_ALGORITHMS.map((algorithm) =>
    new Map(
      selections.get(algorithm)!
        .filter((selection) => selection.budget === 48)
        .map((selection) => [selection.businessId, selection.candidateId]),
    )
  );
  const reference = fullPoolMaps[0];
  const fullPoolAlgorithmAgreement = fullPoolMaps.slice(1).every((candidateMap) =>
    [...reference.entries()].every(([businessId, candidateId]) =>
      candidateMap.get(businessId) === candidateId
    )
  );
  return {
    version: SVI_SCORE_V7_VERSION,
    contract: SVI_SCORE_V7_CONTRACT,
    proxyAssessment: {
      candidateLevelCorrelation: correlation(predictedMean, actualExcess),
      meanAbsoluteError: average(
        predictedMean.map((value, index) => Math.abs(value - actualExcess[index])),
      ),
      p90Coverage: average(
        predictedP90.map((value, index) => actualExcess[index] <= value + 1e-12 ? 1 : 0),
      ),
      roiErrorMeanAbsoluteError: average(
        predictedRoi.map((value, index) => Math.abs(value - actualRoi[index])),
      ),
      contributionErrorMeanAbsoluteError: average(
        predictedContribution.map(
          (value, index) => Math.abs(value - actualContribution[index]),
        ),
      ),
      pairwiseOrderingAccuracy: correctPairs / Math.max(totalPairs, 1),
      economicallyWeightedPairwiseAccuracy: correctWeight / Math.max(totalWeight, 1e-12),
      riskChampionOracleRecall: average(championExcess.map((value) => value <= 1e-10 ? 1 : 0)),
      riskChampionMeanExcessLoss: average(championExcess),
      riskChampionP90ExcessLoss: quantile(championExcess, 0.9),
      dangerousFalseChampionShare: average(championExcess.map((value) => value >= 1 ? 1 : 0)),
    },
    algorithms,
    primaryWinner: [...algorithms].sort(
      (left, right) =>
        left.primaryObjective - right.primaryObjective ||
        left.algorithm.localeCompare(right.algorithm),
    )[0].algorithm,
    fullPoolAlgorithmAgreement,
  };
}
