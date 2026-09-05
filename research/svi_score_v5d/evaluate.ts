import { createHash } from "node:crypto";
import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";
import { metricsFromV4Selections } from "../svi_score_v4/ranker";
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
import { v5cCandidateRegions } from "../svi_score_v5c/crossfit";
import {
  fitV5DCrossFittedModel,
  predictV5DCrossFitted,
  v5dCrossFitHash,
} from "./crossfit";
import { SVI_SCORE_V5D_CONTRACT, SVI_SCORE_V5D_VERSION } from "./contract";
import type { V5DLossPrediction } from "./types";
import {
  V5D_MAP_FEATURE_PROVIDER,
  V5D_MAP_SELECTION_POLICY,
  type V5DFeatureProvider,
  type V5DSelectionPolicy,
} from "./model";

interface V5DSelection {
  businessId: string;
  family: string;
  algorithm: V5BSearchAlgorithm;
  budget: number;
  candidateId: string;
  loss: number;
  excessLoss: number;
  safetyAccepted: boolean;
  reviewOnlyFallback: boolean;
  riskChampionRecovered: boolean;
  economicOracleRecovered: boolean;
  predictedRiskGap: number;
  predictedMeanLoss: number;
  predictedP90Loss: number;
}

interface V5DPredictionAssessment {
  businessId: string;
  candidateId: string;
  actualExcessLoss: number;
  predictedMeanLoss: number;
  predictedP90Loss: number;
  predictedRisk: number;
  riskChampion: boolean;
}

export interface V5DCheckpointMetrics extends V5BCheckpointMetrics {
  meanPredictedLoss: number;
  meanPredictedP90Loss: number;
}

export interface V5DAlgorithmReceipt {
  algorithm: V5BSearchAlgorithm;
  checkpoints: V5DCheckpointMetrics[];
  primary: V5DCheckpointMetrics;
  primaryObjective: number;
  duplicateEvaluations: number;
  incompleteTraces: number;
}

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

function proxyAssessment(
  rows: readonly V5DPredictionAssessment[],
): V5DDevelopmentReceipt["proxyAssessment"] {
  const errors = rows.map((row) => row.predictedMeanLoss - row.actualExcessLoss);
  const byBusiness = new Map<string, V5DPredictionAssessment[]>();
  rows.forEach((row) => {
    byBusiness.set(row.businessId, [...(byBusiness.get(row.businessId) ?? []), row]);
  });
  let correctPairs = 0;
  let comparablePairs = 0;
  let correctEconomicWeight = 0;
  let totalEconomicWeight = 0;
  [...byBusiness.values()].forEach((business) => {
    for (let left = 0; left < business.length; left += 1) {
      for (let right = left + 1; right < business.length; right += 1) {
        const actualDifference = business[left].actualExcessLoss - business[right].actualExcessLoss;
        if (Math.abs(actualDifference) <= 1e-10) continue;
        const predictedDifference = business[left].predictedRisk - business[right].predictedRisk;
        const correct = actualDifference * predictedDifference > 0;
        comparablePairs += 1;
        totalEconomicWeight += Math.abs(actualDifference);
        if (correct) {
          correctPairs += 1;
          correctEconomicWeight += Math.abs(actualDifference);
        }
      }
    }
  });
  const champions = rows.filter((row) => row.riskChampion);
  const championExcess = champions.map((row) => row.actualExcessLoss);
  return {
    candidateRows: rows.length,
    businesses: byBusiness.size,
    meanPredictionBias: average(errors),
    meanAbsoluteError: average(errors.map(Math.abs)),
    rootMeanSquaredError: Math.sqrt(average(errors.map((error) => error ** 2))),
    candidateLevelCorrelation: correlation(
      rows.map((row) => row.predictedMeanLoss),
      rows.map((row) => row.actualExcessLoss),
    ),
    pairwiseOrderingAccuracy: correctPairs / Math.max(comparablePairs, 1),
    economicallyWeightedPairwiseAccuracy:
      correctEconomicWeight / Math.max(totalEconomicWeight, 1e-12),
    p90Coverage: average(
      rows.map((row) => row.actualExcessLoss <= row.predictedP90Loss + 1e-12 ? 1 : 0),
    ),
    riskChampionOracleRecall: average(
      champions.map((row) => row.actualExcessLoss <= 1e-10 ? 1 : 0),
    ),
    riskChampionMeanExcessLoss: average(championExcess),
    riskChampionP90ExcessLoss: quantile(championExcess, 0.9),
    dangerousFalseChampionShare: average(
      champions.map((row) => row.actualExcessLoss >= 1 ? 1 : 0),
    ),
  };
}

function groupRows(
  rows: readonly SviScoreV4CandidateRow[],
): Map<string, SviScoreV4CandidateRow[]> {
  const grouped = new Map<string, SviScoreV4CandidateRow[]>();
  rows.forEach((row) => {
    grouped.set(row.businessId, [...(grouped.get(row.businessId) ?? []), row]);
  });
  return grouped;
}

export function selectV5DOrderIndependent(
  rows: readonly SviScoreV4CandidateRow[],
  predictions: Map<string, V5DLossPrediction>,
  selectionPolicy: V5DSelectionPolicy = V5D_MAP_SELECTION_POLICY,
): { row: SviScoreV4CandidateRow; reviewOnlyFallback: boolean } {
  const pool = selectionPolicy.pool(rows);
  const row = [...pool.rows].sort((left, right) =>
    (predictions.get(left.candidateId)?.risk ?? Infinity) -
      (predictions.get(right.candidateId)?.risk ?? Infinity) ||
    left.candidateId.localeCompare(right.candidateId)
  )[0];
  if (!row) throw new Error("V5D cannot select from an empty candidate pool.");
  return { row, reviewOnlyFallback: !pool.hasSafetyAcceptedCandidate };
}

function economicOracle(
  rows: readonly SviScoreV4CandidateRow[],
  selectionPolicy: V5DSelectionPolicy,
): SviScoreV4CandidateRow {
  const pool = selectionPolicy.pool(rows).rows;
  return [...pool].sort(
    (left, right) =>
      left.decisionLoss - right.decisionLoss ||
      left.candidateId.localeCompare(right.candidateId),
  )[0];
}

function selectionAtBudget(
  businessRows: readonly SviScoreV4CandidateRow[],
  predictions: Map<string, V5DLossPrediction>,
  algorithm: V5BSearchAlgorithm,
  orderedCandidateIds: readonly string[],
  budget: number,
  selectionPolicy: V5DSelectionPolicy,
): V5DSelection {
  const byCandidate = new Map(businessRows.map((row) => [row.candidateId, row]));
  const evaluated = orderedCandidateIds.slice(0, budget).map((candidateId) => {
    const row = byCandidate.get(candidateId);
    if (!row) throw new Error(`V5D trace contains unknown ${candidateId}.`);
    return row;
  });
  const selected = selectV5DOrderIndependent(evaluated, predictions, selectionPolicy);
  const riskChampion = selectV5DOrderIndependent(
    businessRows,
    predictions,
    selectionPolicy,
  ).row;
  const oracle = economicOracle(businessRows, selectionPolicy);
  const selectedPrediction = predictions.get(selected.row.candidateId);
  const championPrediction = predictions.get(riskChampion.candidateId);
  if (!selectedPrediction || !championPrediction) {
    throw new Error("V5D prediction map is incomplete.");
  }
  const excessLoss = Math.max(0, selected.row.decisionLoss - oracle.decisionLoss);
  return {
    businessId: selected.row.businessId,
    family: selected.row.family,
    algorithm,
    budget,
    candidateId: selected.row.candidateId,
    loss: selected.row.decisionLoss,
    excessLoss,
    safetyAccepted: selectionPolicy.passes(selected.row),
    reviewOnlyFallback: selected.reviewOnlyFallback,
    riskChampionRecovered: selected.row.candidateId === riskChampion.candidateId,
    economicOracleRecovered: excessLoss <= 1e-10,
    predictedRiskGap: selectedPrediction.risk - championPrediction.risk,
    predictedMeanLoss: selectedPrediction.mean,
    predictedP90Loss: selectedPrediction.p90,
  };
}

function checkpointMetrics(
  selections: readonly V5DSelection[],
  budget: number,
): V5DCheckpointMetrics {
  const metrics = metricsFromV4Selections(selections.map((selection) => ({
    businessId: selection.businessId,
    candidateId: selection.candidateId,
    family: selection.family,
    loss: selection.loss,
    excessLoss: selection.excessLoss,
    normalizedExcessRegret: selection.excessLoss,
    oracle: selection.economicOracleRecovered,
    decisionGrade: selection.safetyAccepted && !selection.reviewOnlyFallback,
    safetyAccepted: selection.safetyAccepted,
    reviewOnlyFallback: selection.reviewOnlyFallback,
  })));
  return {
    ...metrics,
    budget,
    selectionObjective: v5aSelectionObjective(metrics),
    fullPoolChampionRecall: average(
      selections.map((selection) => selection.riskChampionRecovered ? 1 : 0),
    ),
    economicOracleRecall: average(
      selections.map((selection) => selection.economicOracleRecovered ? 1 : 0),
    ),
    meanSelectorScoreGap: average(
      selections.map((selection) => selection.predictedRiskGap),
    ),
    meanPredictedLoss: average(
      selections.map((selection) => selection.predictedMeanLoss),
    ),
    meanPredictedP90Loss: average(
      selections.map((selection) => selection.predictedP90Loss),
    ),
  };
}

export interface V5DDevelopmentReceipt {
  version: typeof SVI_SCORE_V5D_VERSION;
  contract: typeof SVI_SCORE_V5D_CONTRACT;
  data: {
    businesses: number;
    rows: number;
    candidatesPerBusiness: number;
    outerAssignmentHash: string;
    modelHashes: Record<string, string>;
  };
  algorithms: V5DAlgorithmReceipt[];
  proxyAssessment: {
    candidateRows: number;
    businesses: number;
    meanPredictionBias: number;
    meanAbsoluteError: number;
    rootMeanSquaredError: number;
    candidateLevelCorrelation: number;
    pairwiseOrderingAccuracy: number;
    economicallyWeightedPairwiseAccuracy: number;
    p90Coverage: number;
    riskChampionOracleRecall: number;
    riskChampionMeanExcessLoss: number;
    riskChampionP90ExcessLoss: number;
    dangerousFalseChampionShare: number;
  };
  primaryWinner: V5BSearchAlgorithm;
  acceptance: {
    adaptiveAlgorithm: V5BSearchAlgorithm;
    adaptivePrimaryNonInferior: boolean;
    adaptivePrimaryRelativeDifference: number;
    fullPoolAlgorithmAgreement: boolean;
    fullPoolRelativeDifferenceVersusBestEarlier: number;
    expandedSearchSafe: boolean;
  };
}

export function evaluateV5D(
  rows: readonly SviScoreV4CandidateRow[],
  assignments: readonly { businessId: string; family: string; fold: number }[],
  provider: V5DFeatureProvider = V5D_MAP_FEATURE_PROVIDER,
  selectionPolicy: V5DSelectionPolicy = V5D_MAP_SELECTION_POLICY,
): V5DDevelopmentReceipt {
  if (!rows.length) throw new Error("V5D requires development rows.");
  if (rows.some((row) => row.split === "audit")) {
    throw new Error("The sealed audit is forbidden in V5D development.");
  }
  const grouped = groupRows(rows);
  if (
    grouped.size !== SVI_SCORE_V5D_CONTRACT.development.businesses ||
    [...grouped.values()].some(
      (business) => business.length !== SVI_SCORE_V5D_CONTRACT.development.candidatesPerBusiness,
    )
  ) {
    throw new Error("V5D cohort changed from its frozen 420 x 48 contract.");
  }
  const foldByBusiness = new Map(
    assignments.map((assignment) => [assignment.businessId, assignment.fold]),
  );
  const universe = v5bCandidateUniverse(rows.map((row) => row.candidateId));
  const regionByCandidateId = v5cCandidateRegions(
    universe.map((candidate) => candidate.candidateId),
  );
  const selections = new Map<V5BSearchAlgorithm, V5DSelection[]>();
  const checks = new Map<V5BSearchAlgorithm, { duplicates: number; incomplete: number }>();
  V5B_SEARCH_ALGORITHMS.forEach((algorithm) => {
    selections.set(algorithm, []);
    checks.set(algorithm, { duplicates: 0, incomplete: 0 });
  });
  const modelHashes: Record<string, string> = {};
  const predictionAssessments: V5DPredictionAssessment[] = [];
  const folds = [...new Set(assignments.map((assignment) => assignment.fold))].sort(
    (left, right) => left - right,
  );
  folds.forEach((fold) => {
    const training = rows.filter((row) => foldByBusiness.get(row.businessId) !== fold);
    const crossFitted = fitV5DCrossFittedModel(
      training,
      regionByCandidateId,
      provider,
      selectionPolicy,
    );
    modelHashes[String(fold)] = v5dCrossFitHash(crossFitted);
    assignments
      .filter((assignment) => assignment.fold === fold)
      .map((assignment) => assignment.businessId)
      .sort()
      .forEach((businessId) => {
        const businessRows = grouped.get(businessId);
        if (!businessRows) throw new Error(`V5D is missing ${businessId}.`);
        const rowsByCandidate = new Map(
          businessRows.map((row) => [row.candidateId, row]),
        );
        const predictions = new Map(
          businessRows.map((row) => [
            row.candidateId,
            predictV5DCrossFitted(row, crossFitted, provider),
          ]),
        );
        const oracle = economicOracle(businessRows, selectionPolicy);
        const champion = selectV5DOrderIndependent(
          businessRows,
          predictions,
          selectionPolicy,
        ).row;
        businessRows.forEach((row) => {
          const predicted = predictions.get(row.candidateId)!;
          predictionAssessments.push({
            businessId,
            candidateId: row.candidateId,
            actualExcessLoss: Math.max(0, row.decisionLoss - oracle.decisionLoss),
            predictedMeanLoss: predicted.mean,
            predictedP90Loss: predicted.p90,
            predictedRisk: predicted.risk,
            riskChampion: row.candidateId === champion.candidateId,
          });
        });
        V5B_SEARCH_ALGORITHMS.forEach((algorithm) => {
          const revealed = new Set<string>();
          const reveal = (candidate: V5BSearchCandidate): V5BObservedCandidate => {
            if (revealed.has(candidate.candidateId)) {
              throw new Error(`${algorithm} duplicated ${candidate.candidateId}.`);
            }
            const row = rowsByCandidate.get(candidate.candidateId);
            const predicted = predictions.get(candidate.candidateId);
            if (!row || !predicted) throw new Error("V5D reveal contract failed.");
            revealed.add(candidate.candidateId);
            return {
              ...candidate,
              selectorScore: -predicted.risk,
              safetyAccepted: selectionPolicy.passes(row),
            };
          };
          const trace = runV5BSearch(businessId, algorithm, universe, reveal);
          const check = checks.get(algorithm)!;
          check.duplicates += trace.orderedCandidateIds.length -
            new Set(trace.orderedCandidateIds).size;
          check.incomplete += trace.orderedCandidateIds.length === universe.length ? 0 : 1;
          SVI_SCORE_V5D_CONTRACT.search.checkpoints.forEach((budget) => {
            selections.get(algorithm)!.push(selectionAtBudget(
              businessRows,
              predictions,
              algorithm,
              trace.orderedCandidateIds,
              budget,
              selectionPolicy,
            ));
          });
        });
      });
  });
  const algorithms = V5B_SEARCH_ALGORITHMS.map((algorithm) => {
    const all = selections.get(algorithm)!;
    const checkpoints = SVI_SCORE_V5D_CONTRACT.search.checkpoints.map((budget) =>
      checkpointMetrics(all.filter((selection) => selection.budget === budget), budget)
    );
    const primary = checkpoints.find(
      (checkpoint) => checkpoint.budget ===
        SVI_SCORE_V5D_CONTRACT.search.primaryEvaluationsPerBusiness,
    );
    if (!primary) throw new Error(`${algorithm} has no V5D primary result.`);
    const check = checks.get(algorithm)!;
    return {
      algorithm,
      checkpoints,
      primary,
      primaryObjective: primary.selectionObjective,
      duplicateEvaluations: check.duplicates,
      incompleteTraces: check.incomplete,
    } satisfies V5DAlgorithmReceipt;
  });
  const primaryWinner = [...algorithms].sort(
    (left, right) =>
      left.primaryObjective - right.primaryObjective ||
      left.algorithm.localeCompare(right.algorithm),
  )[0].algorithm;
  const fixed = algorithms.find((item) => item.algorithm === "fixed-coverage")!;
  const adaptive = algorithms
    .filter((item) => item.algorithm === "gp-ucb" || item.algorithm === "hybrid-global-local")
    .sort((left, right) => left.primaryObjective - right.primaryObjective)[0];
  const adaptivePrimaryRelativeDifference =
    (adaptive.primaryObjective - fixed.primaryObjective) /
    Math.max(Math.abs(fixed.primaryObjective), 1e-12);
  const fullPoolSelections = V5B_SEARCH_ALGORITHMS.map((algorithm) => {
    const items = selections.get(algorithm)!.filter((selection) => selection.budget === 48);
    return new Map(items.map((selection) => [selection.businessId, selection.candidateId]));
  });
  const reference = fullPoolSelections[0];
  const fullPoolAlgorithmAgreement = fullPoolSelections.slice(1).every((candidateMap) =>
    [...reference.entries()].every(([businessId, candidateId]) =>
      candidateMap.get(businessId) === candidateId
    )
  );
  const adaptiveFull = adaptive.checkpoints.find((checkpoint) => checkpoint.budget === 48)!;
  const adaptiveBestEarlier = [...adaptive.checkpoints]
    .filter((checkpoint) => checkpoint.budget < 48)
    .sort((left, right) => left.selectionObjective - right.selectionObjective)[0];
  const fullPoolRelativeDifferenceVersusBestEarlier =
    (adaptiveFull.selectionObjective - adaptiveBestEarlier.selectionObjective) /
    Math.max(Math.abs(adaptiveBestEarlier.selectionObjective), 1e-12);
  const adaptivePrimaryNonInferior = adaptivePrimaryRelativeDifference <=
    SVI_SCORE_V5D_CONTRACT.acceptance.adaptivePrimaryNonInferiorityMargin;
  const fullPoolStable = fullPoolRelativeDifferenceVersusBestEarlier <=
    SVI_SCORE_V5D_CONTRACT.acceptance.fullPoolVersusBestEarlierCheckpointMargin;
  return {
    version: SVI_SCORE_V5D_VERSION,
    contract: SVI_SCORE_V5D_CONTRACT,
    data: {
      businesses: grouped.size,
      rows: rows.length,
      candidatesPerBusiness: universe.length,
      outerAssignmentHash: createHash("sha256")
        .update(JSON.stringify(assignments))
        .digest("hex"),
      modelHashes,
    },
    algorithms,
    proxyAssessment: proxyAssessment(predictionAssessments),
    primaryWinner,
    acceptance: {
      adaptiveAlgorithm: adaptive.algorithm,
      adaptivePrimaryNonInferior,
      adaptivePrimaryRelativeDifference,
      fullPoolAlgorithmAgreement,
      fullPoolRelativeDifferenceVersusBestEarlier,
      expandedSearchSafe:
        adaptivePrimaryNonInferior && fullPoolAlgorithmAgreement && fullPoolStable,
    },
  };
}
