import { createHash } from "node:crypto";
import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";
import {
  metricsFromV4Selections,
  passesV4Safety,
  v4SelectionPool,
} from "../svi_score_v4/ranker";
import { V5A_COMMON_POOL_CONFIGURATION } from "../svi_score_v5a/contract";
import { v5aSelectionObjective } from "../svi_score_v5a/ranker";
import {
  SVI_SCORE_V5B_CONTRACT,
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
import {
  fitV5CSelectorEnsemble,
  scoreV5CDistribution,
  v5cCandidateRegions,
} from "./crossfit";
import { SVI_SCORE_V5C_CONTRACT, SVI_SCORE_V5C_VERSION } from "./contract";
import type {
  V5CScoreDistribution,
  V5CSelectorEnsemble,
} from "./types";

interface V5CSelection {
  businessId: string;
  family: string;
  algorithm: V5BSearchAlgorithm;
  budget: number;
  candidateId: string;
  loss: number;
  excessLoss: number;
  normalizedExcessRegret: number;
  safetyAccepted: boolean;
  reviewOnlyFallback: boolean;
  robustChampionRecovered: boolean;
  robustChampionScoreGap: number;
  economicOracleRecovered: boolean;
  promotionCount: number;
}

export interface V5CAlgorithmReceipt {
  algorithm: V5BSearchAlgorithm;
  checkpoints: Array<V5BCheckpointMetrics & { meanPromotions: number }>;
  primary: V5BCheckpointMetrics & { meanPromotions: number };
  primaryObjective: number;
  duplicateEvaluations: number;
  incompleteTraces: number;
}

function average(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) /
    Math.max(values.length, 1);
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

function scoreMap(
  rows: readonly SviScoreV4CandidateRow[],
  ensemble: V5CSelectorEnsemble,
): Map<string, V5CScoreDistribution> {
  return new Map(rows.map((row) => [
    row.candidateId,
    scoreV5CDistribution(row, ensemble),
  ]));
}

export function confidencePromoted(
  orderedRows: readonly SviScoreV4CandidateRow[],
  scores: Map<string, V5CScoreDistribution>,
): {
  row: SviScoreV4CandidateRow;
  reviewOnlyFallback: boolean;
  promotions: number;
} {
  const pool = v4SelectionPool(orderedRows, V5A_COMMON_POOL_CONFIGURATION);
  const permitted = new Set(pool.rows.map((row) => row.candidateId));
  const sequence = orderedRows.filter((row) => permitted.has(row.candidateId));
  let incumbent = sequence[0];
  let promotions = 0;
  if (!incumbent) throw new Error("V5C cannot promote from an empty pool.");
  sequence.slice(1).forEach((challenger) => {
    const current = scores.get(incumbent.candidateId);
    const proposed = scores.get(challenger.candidateId);
    if (!current || !proposed) throw new Error("V5C score distribution is missing.");
    const margin = SVI_SCORE_V5C_CONTRACT.uncertainty.promotionStandardErrors *
      Math.sqrt(
        current.standardDeviation ** 2 + proposed.standardDeviation ** 2,
      );
    if (
      proposed.mean - current.mean > margin + 1e-12 ||
      (Math.abs(proposed.mean - current.mean - margin) <= 1e-12 &&
        challenger.candidateId.localeCompare(incumbent.candidateId) < 0)
    ) {
      incumbent = challenger;
      promotions += 1;
    }
  });
  return {
    row: incumbent,
    reviewOnlyFallback: !pool.hasSafetyAcceptedCandidate,
    promotions,
  };
}

function robustChampion(
  rows: readonly SviScoreV4CandidateRow[],
  scores: Map<string, V5CScoreDistribution>,
): SviScoreV4CandidateRow {
  const pool = v4SelectionPool(rows, V5A_COMMON_POOL_CONFIGURATION);
  return [...pool.rows].sort((left, right) =>
    (scores.get(right.candidateId)?.robust ?? -Infinity) -
      (scores.get(left.candidateId)?.robust ?? -Infinity) ||
    left.candidateId.localeCompare(right.candidateId)
  )[0];
}

function economicOracle(
  rows: readonly SviScoreV4CandidateRow[],
): SviScoreV4CandidateRow {
  const pool = v4SelectionPool(rows, V5A_COMMON_POOL_CONFIGURATION);
  return [...pool.rows].sort(
    (left, right) =>
      left.decisionLoss - right.decisionLoss ||
      left.candidateId.localeCompare(right.candidateId),
  )[0];
}

function selectionAtBudget(
  businessRows: readonly SviScoreV4CandidateRow[],
  scores: Map<string, V5CScoreDistribution>,
  algorithm: V5BSearchAlgorithm,
  orderedCandidateIds: readonly string[],
  budget: number,
): V5CSelection {
  const byCandidate = new Map(businessRows.map((row) => [row.candidateId, row]));
  const evaluated = orderedCandidateIds.slice(0, budget).map((candidateId) => {
    const row = byCandidate.get(candidateId);
    if (!row) throw new Error(`V5C trace contains unknown ${candidateId}.`);
    return row;
  });
  const promoted = confidencePromoted(evaluated, scores);
  const champion = robustChampion(businessRows, scores);
  const oracle = economicOracle(businessRows);
  const promotedScore = scores.get(promoted.row.candidateId);
  const championScore = scores.get(champion.candidateId);
  if (!promotedScore || !championScore) {
    throw new Error("V5C champion score distribution is missing.");
  }
  const excessLoss = Math.max(0, promoted.row.decisionLoss - oracle.decisionLoss);
  return {
    businessId: promoted.row.businessId,
    family: promoted.row.family,
    algorithm,
    budget,
    candidateId: promoted.row.candidateId,
    loss: promoted.row.decisionLoss,
    excessLoss,
    normalizedExcessRegret: excessLoss,
    safetyAccepted: passesV4Safety(promoted.row, V5A_COMMON_POOL_CONFIGURATION),
    reviewOnlyFallback: promoted.reviewOnlyFallback,
    robustChampionRecovered: promoted.row.candidateId === champion.candidateId,
    robustChampionScoreGap: championScore.robust - promotedScore.robust,
    economicOracleRecovered: excessLoss <= 1e-10,
    promotionCount: promoted.promotions,
  };
}

function checkpointMetrics(
  selections: readonly V5CSelection[],
  budget: number,
): V5BCheckpointMetrics & { meanPromotions: number } {
  const metrics = metricsFromV4Selections(selections.map((selection) => ({
    businessId: selection.businessId,
    candidateId: selection.candidateId,
    family: selection.family,
    loss: selection.loss,
    excessLoss: selection.excessLoss,
    normalizedExcessRegret: selection.normalizedExcessRegret,
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
      selections.map((selection) => selection.robustChampionRecovered ? 1 : 0),
    ),
    economicOracleRecall: average(
      selections.map((selection) => selection.economicOracleRecovered ? 1 : 0),
    ),
    meanSelectorScoreGap: average(
      selections.map((selection) => selection.robustChampionScoreGap),
    ),
    meanPromotions: average(selections.map((selection) => selection.promotionCount)),
  };
}

export interface V5CDevelopmentReceipt {
  version: typeof SVI_SCORE_V5C_VERSION;
  contract: typeof SVI_SCORE_V5C_CONTRACT;
  data: {
    businesses: number;
    rows: number;
    candidatesPerBusiness: number;
    candidateRegionCounts: Record<string, number>;
    outerAssignmentHash: string;
    selectorEnsembleHashes: Record<string, string>;
  };
  algorithms: V5CAlgorithmReceipt[];
  primaryWinner: V5BSearchAlgorithm;
  acceptance: {
    adaptiveAlgorithm: V5BSearchAlgorithm;
    adaptivePrimaryNonInferior: boolean;
    adaptivePrimaryRelativeDifference: number;
    fullPoolStable: boolean;
    fullPoolRelativeDifferenceVersusBestEarlier: number;
    expandedSearchSafe: boolean;
  };
}

export function evaluateV5C(
  rows: readonly SviScoreV4CandidateRow[],
  assignments: readonly { businessId: string; family: string; fold: number }[],
): V5CDevelopmentReceipt {
  if (!rows.length) throw new Error("V5C requires development rows.");
  if (rows.some((row) => row.split === "audit")) {
    throw new Error("The sealed V3 audit is forbidden in V5C.");
  }
  const grouped = groupRows(rows);
  if (
    grouped.size !== SVI_SCORE_V5C_CONTRACT.development.businesses ||
    [...grouped.values()].some(
      (business) => business.length !== SVI_SCORE_V5C_CONTRACT.development.candidatesPerBusiness,
    )
  ) {
    throw new Error("V5C cohort changed from its frozen 420 x 48 contract.");
  }
  const foldByBusiness = new Map(
    assignments.map((assignment) => [assignment.businessId, assignment.fold]),
  );
  const universe = v5bCandidateUniverse(rows.map((row) => row.candidateId));
  const regionByCandidateId = v5cCandidateRegions(
    universe.map((candidate) => candidate.candidateId),
  );
  const candidateRegionCounts = Object.fromEntries(
    Array.from({ length: SVI_SCORE_V5C_CONTRACT.development.candidateRegions }, (_, region) => [
      String(region),
      Object.values(regionByCandidateId).filter((value) => value === region).length,
    ]),
  );
  const selections = new Map<V5BSearchAlgorithm, V5CSelection[]>();
  const checks = new Map<V5BSearchAlgorithm, { duplicates: number; incomplete: number }>();
  V5B_SEARCH_ALGORITHMS.forEach((algorithm) => {
    selections.set(algorithm, []);
    checks.set(algorithm, { duplicates: 0, incomplete: 0 });
  });
  const selectorEnsembleHashes: Record<string, string> = {};
  const folds = [...new Set(assignments.map((assignment) => assignment.fold))].sort(
    (left, right) => left - right,
  );
  folds.forEach((fold) => {
    const training = rows.filter((row) => foldByBusiness.get(row.businessId) !== fold);
    const ensemble = fitV5CSelectorEnsemble(training, regionByCandidateId);
    selectorEnsembleHashes[String(fold)] = createHash("sha256")
      .update(JSON.stringify(ensemble))
      .digest("hex");
    assignments
      .filter((assignment) => assignment.fold === fold)
      .map((assignment) => assignment.businessId)
      .sort()
      .forEach((businessId) => {
        const businessRows = grouped.get(businessId);
        if (!businessRows) throw new Error(`V5C is missing ${businessId}.`);
        const rowsByCandidate = new Map(
          businessRows.map((row) => [row.candidateId, row]),
        );
        const distributions = scoreMap(businessRows, ensemble);
        V5B_SEARCH_ALGORITHMS.forEach((algorithm) => {
          const revealed = new Set<string>();
          const reveal = (candidate: V5BSearchCandidate): V5BObservedCandidate => {
            if (revealed.has(candidate.candidateId)) {
              throw new Error(`${algorithm} duplicated ${candidate.candidateId}.`);
            }
            const row = rowsByCandidate.get(candidate.candidateId);
            const distribution = distributions.get(candidate.candidateId);
            if (!row || !distribution) throw new Error("V5C reveal contract failed.");
            revealed.add(candidate.candidateId);
            return {
              ...candidate,
              selectorScore: distribution.robust,
              safetyAccepted: passesV4Safety(row, V5A_COMMON_POOL_CONFIGURATION),
            };
          };
          const trace = runV5BSearch(businessId, algorithm, universe, reveal);
          const receipt = checks.get(algorithm)!;
          receipt.duplicates += trace.orderedCandidateIds.length -
            new Set(trace.orderedCandidateIds).size;
          receipt.incomplete += trace.orderedCandidateIds.length === universe.length ? 0 : 1;
          SVI_SCORE_V5C_CONTRACT.search.checkpoints.forEach((budget) => {
            selections.get(algorithm)!.push(selectionAtBudget(
              businessRows,
              distributions,
              algorithm,
              trace.orderedCandidateIds,
              budget,
            ));
          });
        });
      });
  });
  const algorithms = V5B_SEARCH_ALGORITHMS.map((algorithm) => {
    const all = selections.get(algorithm)!;
    const checkpoints = SVI_SCORE_V5C_CONTRACT.search.checkpoints.map((budget) =>
      checkpointMetrics(all.filter((selection) => selection.budget === budget), budget)
    );
    const primary = checkpoints.find(
      (checkpoint) =>
        checkpoint.budget === SVI_SCORE_V5C_CONTRACT.search.primaryEvaluationsPerBusiness,
    );
    if (!primary) throw new Error(`${algorithm} has no V5C primary result.`);
    const receipt = checks.get(algorithm)!;
    return {
      algorithm,
      checkpoints,
      primary,
      primaryObjective: primary.selectionObjective,
      duplicateEvaluations: receipt.duplicates,
      incompleteTraces: receipt.incomplete,
    } satisfies V5CAlgorithmReceipt;
  });
  const primaryWinner = [...algorithms].sort(
    (left, right) =>
      left.primaryObjective - right.primaryObjective ||
      left.algorithm.localeCompare(right.algorithm),
  )[0].algorithm;
  const fixed = algorithms.find((algorithm) => algorithm.algorithm === "fixed-coverage")!;
  const adaptive = algorithms
    .filter((algorithm) =>
      algorithm.algorithm === "gp-ucb" || algorithm.algorithm === "hybrid-global-local"
    )
    .sort((left, right) => left.primaryObjective - right.primaryObjective)[0];
  const adaptivePrimaryRelativeDifference =
    (adaptive.primaryObjective - fixed.primaryObjective) /
    Math.max(Math.abs(fixed.primaryObjective), 1e-12);
  const earlier = adaptive.checkpoints.filter((checkpoint) => checkpoint.budget < 48);
  const bestEarlier = [...earlier].sort(
    (left, right) => left.selectionObjective - right.selectionObjective,
  )[0];
  const fullPool = adaptive.checkpoints.find((checkpoint) => checkpoint.budget === 48)!;
  const fullPoolRelativeDifferenceVersusBestEarlier =
    (fullPool.selectionObjective - bestEarlier.selectionObjective) /
    Math.max(Math.abs(bestEarlier.selectionObjective), 1e-12);
  const adaptivePrimaryNonInferior = adaptivePrimaryRelativeDifference <=
    SVI_SCORE_V5C_CONTRACT.acceptance.adaptivePrimaryNonInferiorityMargin;
  const fullPoolStable = fullPoolRelativeDifferenceVersusBestEarlier <=
    SVI_SCORE_V5C_CONTRACT.acceptance.fullPoolVersusBestEarlierCheckpointMargin;
  return {
    version: SVI_SCORE_V5C_VERSION,
    contract: SVI_SCORE_V5C_CONTRACT,
    data: {
      businesses: grouped.size,
      rows: rows.length,
      candidatesPerBusiness: universe.length,
      candidateRegionCounts,
      outerAssignmentHash: createHash("sha256")
        .update(JSON.stringify(assignments))
        .digest("hex"),
      selectorEnsembleHashes,
    },
    algorithms,
    primaryWinner,
    acceptance: {
      adaptiveAlgorithm: adaptive.algorithm,
      adaptivePrimaryNonInferior,
      adaptivePrimaryRelativeDifference,
      fullPoolStable,
      fullPoolRelativeDifferenceVersusBestEarlier,
      expandedSearchSafe: adaptivePrimaryNonInferior && fullPoolStable,
    },
  };
}
