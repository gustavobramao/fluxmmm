import { createHash } from "node:crypto";
import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";
import {
  metricsFromV4Selections,
  passesV4Safety,
  v4SelectionPool,
} from "../svi_score_v4/ranker";
import {
  V5A_COMMON_POOL_CONFIGURATION,
  type SviScoreV5ATrainingConfiguration,
} from "../svi_score_v5a/contract";
import {
  latentV5AScore,
  learnScoreV5A,
  v5aSelectionObjective,
} from "../svi_score_v5a/ranker";
import type { SviScoreV5AModel } from "../svi_score_v5a/types";
import {
  SVI_SCORE_V5B_CONTRACT,
  SVI_SCORE_V5B_VERSION,
  V5B_SEARCH_ALGORITHMS,
  type V5BSearchAlgorithm,
} from "./contract";
import { v5bCandidateUniverse } from "./embedding";
import { runV5BSearch } from "./search";
import type {
  V5BAlgorithmReceipt,
  V5BCheckpointMetrics,
  V5BCheckpointSelection,
  V5BObservedCandidate,
  V5BSearchCandidate,
} from "./types";

function groupRows(
  rows: readonly SviScoreV4CandidateRow[],
): Map<string, SviScoreV4CandidateRow[]> {
  const grouped = new Map<string, SviScoreV4CandidateRow[]>();
  rows.forEach((row) => {
    grouped.set(row.businessId, [...(grouped.get(row.businessId) ?? []), row]);
  });
  return grouped;
}

function average(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) /
    Math.max(values.length, 1);
}

function selectedRow(
  rows: readonly SviScoreV4CandidateRow[],
  model: SviScoreV5AModel,
): { row: SviScoreV4CandidateRow; reviewOnlyFallback: boolean } {
  const pool = v4SelectionPool(rows, V5A_COMMON_POOL_CONFIGURATION);
  const row = [...pool.rows].sort(
    (left, right) =>
      latentV5AScore(right, model) - latentV5AScore(left, model) ||
      left.candidateId.localeCompare(right.candidateId),
  )[0];
  if (!row) throw new Error("V5B cannot select from an empty evaluated pool.");
  return { row, reviewOnlyFallback: !pool.hasSafetyAcceptedCandidate };
}

function economicOracle(
  rows: readonly SviScoreV4CandidateRow[],
  model: SviScoreV5AModel,
): SviScoreV4CandidateRow {
  const pool = v4SelectionPool(rows, V5A_COMMON_POOL_CONFIGURATION);
  const row = [...pool.rows].sort(
    (left, right) =>
      left.decisionLoss - right.decisionLoss ||
      left.candidateId.localeCompare(right.candidateId),
  )[0];
  if (!row) throw new Error("V5B full pool has no economic oracle.");
  return row;
}

function selectionAtBudget(
  businessRows: readonly SviScoreV4CandidateRow[],
  model: SviScoreV5AModel,
  algorithm: V5BSearchAlgorithm,
  orderedCandidateIds: readonly string[],
  budget: number,
): V5BCheckpointSelection {
  const byCandidate = new Map(businessRows.map((row) => [row.candidateId, row]));
  const evaluated = orderedCandidateIds.slice(0, budget).map((candidateId) => {
    const row = byCandidate.get(candidateId);
    if (!row) throw new Error(`V5B trace contains unknown candidate ${candidateId}.`);
    return row;
  });
  const selected = selectedRow(evaluated, model);
  const fullChampion = selectedRow(businessRows, model).row;
  const oracle = economicOracle(businessRows, model);
  const excessLoss = Math.max(0, selected.row.decisionLoss - oracle.decisionLoss);
  return {
    businessId: selected.row.businessId,
    family: selected.row.family,
    algorithm,
    budget,
    candidateId: selected.row.candidateId,
    loss: selected.row.decisionLoss,
    excessLoss,
    normalizedExcessRegret: Math.min(excessLoss, 1),
    safetyAccepted: passesV4Safety(selected.row, V5A_COMMON_POOL_CONFIGURATION),
    reviewOnlyFallback: selected.reviewOnlyFallback,
    fullPoolChampionRecovered: selected.row.candidateId === fullChampion.candidateId,
    economicOracleRecovered: excessLoss <= 1e-10,
    selectorScoreGap:
      latentV5AScore(fullChampion, model) - latentV5AScore(selected.row, model),
  };
}

function checkpointMetrics(
  selections: readonly V5BCheckpointSelection[],
  budget: number,
): V5BCheckpointMetrics {
  const base = metricsFromV4Selections(selections.map((selection) => ({
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
    ...base,
    budget,
    selectionObjective: v5aSelectionObjective(base),
    fullPoolChampionRecall: average(
      selections.map((selection) => selection.fullPoolChampionRecovered ? 1 : 0),
    ),
    economicOracleRecall: average(
      selections.map((selection) => selection.economicOracleRecovered ? 1 : 0),
    ),
    meanSelectorScoreGap: average(
      selections.map((selection) => selection.selectorScoreGap),
    ),
  };
}

export interface V5BDevelopmentReceipt {
  version: typeof SVI_SCORE_V5B_VERSION;
  contract: typeof SVI_SCORE_V5B_CONTRACT;
  data: {
    businesses: number;
    rows: number;
    candidatesPerBusiness: number;
    parameterDimensions: number;
    assignmentHash: string;
    selectorModelHashes: Record<string, string>;
  };
  algorithms: V5BAlgorithmReceipt[];
  primaryWinner: V5BSearchAlgorithm;
}

export function evaluateV5BSearchAlgorithms(
  rows: readonly SviScoreV4CandidateRow[],
  assignments: readonly { businessId: string; family: string; fold: number }[],
  configuration: SviScoreV5ATrainingConfiguration,
): V5BDevelopmentReceipt {
  if (!rows.length) throw new Error("V5B requires development candidate rows.");
  if (rows.some((row) => row.split === "audit")) {
    throw new Error("The sealed V3 audit is forbidden in V5B development.");
  }
  const grouped = groupRows(rows);
  if (
    grouped.size !== SVI_SCORE_V5B_CONTRACT.development.businesses ||
    [...grouped.values()].some(
      (business) => business.length !== SVI_SCORE_V5B_CONTRACT.development.candidatesPerBusiness,
    )
  ) {
    throw new Error("V5B development cohort does not match the frozen 420 x 48 contract.");
  }
  const universe = v5bCandidateUniverse(rows.map((row) => row.candidateId));
  const foldByBusiness = new Map(
    assignments.map((assignment) => [assignment.businessId, assignment.fold]),
  );
  if (
    foldByBusiness.size !== grouped.size ||
    [...grouped.keys()].some((businessId) => !foldByBusiness.has(businessId))
  ) {
    throw new Error("V5B assignments do not cover each development business exactly once.");
  }
  const selections = new Map<V5BSearchAlgorithm, V5BCheckpointSelection[]>();
  const traceChecks = new Map<
    V5BSearchAlgorithm,
    { duplicates: number; incomplete: number }
  >();
  V5B_SEARCH_ALGORITHMS.forEach((algorithm) => {
    selections.set(algorithm, []);
    traceChecks.set(algorithm, { duplicates: 0, incomplete: 0 });
  });
  const selectorModelHashes: Record<string, string> = {};
  const folds = [...new Set(assignments.map((assignment) => assignment.fold))].sort(
    (left, right) => left - right,
  );
  folds.forEach((fold) => {
    const training = rows.filter((row) => foldByBusiness.get(row.businessId) !== fold);
    const model = learnScoreV5A(training, configuration);
    selectorModelHashes[String(fold)] = createHash("sha256")
      .update(JSON.stringify(model))
      .digest("hex");
    const assessmentBusinesses = assignments
      .filter((assignment) => assignment.fold === fold)
      .map((assignment) => assignment.businessId)
      .sort();
    assessmentBusinesses.forEach((businessId) => {
      const businessRows = grouped.get(businessId);
      if (!businessRows) throw new Error(`V5B is missing ${businessId}.`);
      const rowByCandidate = new Map(
        businessRows.map((row) => [row.candidateId, row]),
      );
      V5B_SEARCH_ALGORITHMS.forEach((algorithm) => {
        const revealed = new Set<string>();
        const reveal = (candidate: V5BSearchCandidate): V5BObservedCandidate => {
          if (revealed.has(candidate.candidateId)) {
            throw new Error(`${algorithm} evaluated ${candidate.candidateId} twice.`);
          }
          const row = rowByCandidate.get(candidate.candidateId);
          if (!row) throw new Error(`V5B cannot reveal ${candidate.candidateId}.`);
          revealed.add(candidate.candidateId);
          return {
            ...candidate,
            selectorScore: latentV5AScore(row, model),
            safetyAccepted: passesV4Safety(row, V5A_COMMON_POOL_CONFIGURATION),
          };
        };
        const trace = runV5BSearch(businessId, algorithm, universe, reveal);
        const check = traceChecks.get(algorithm)!;
        check.duplicates += trace.orderedCandidateIds.length -
          new Set(trace.orderedCandidateIds).size;
        check.incomplete += trace.orderedCandidateIds.length === universe.length ? 0 : 1;
        SVI_SCORE_V5B_CONTRACT.compute.checkpoints.forEach((budget) => {
          selections.get(algorithm)!.push(selectionAtBudget(
            businessRows,
            model,
            algorithm,
            trace.orderedCandidateIds,
            budget,
          ));
        });
      });
    });
  });
  const algorithms = V5B_SEARCH_ALGORITHMS.map((algorithm) => {
    const algorithmSelections = selections.get(algorithm)!;
    const checkpoints = SVI_SCORE_V5B_CONTRACT.compute.checkpoints.map((budget) =>
      checkpointMetrics(
        algorithmSelections.filter((selection) => selection.budget === budget),
        budget,
      )
    );
    const primary = checkpoints.find(
      (checkpoint) =>
        checkpoint.budget === SVI_SCORE_V5B_CONTRACT.compute.primaryEvaluationsPerBusiness,
    );
    if (!primary) throw new Error(`${algorithm} has no V5B primary checkpoint.`);
    const checks = traceChecks.get(algorithm)!;
    return {
      algorithm,
      checkpoints,
      primary,
      primaryObjective: v5aSelectionObjective(primary),
      primaryCandidateEvaluations:
        grouped.size * SVI_SCORE_V5B_CONTRACT.compute.primaryEvaluationsPerBusiness,
      curveCandidateEvaluations:
        grouped.size * Math.max(...SVI_SCORE_V5B_CONTRACT.compute.checkpoints),
      duplicateEvaluations: checks.duplicates,
      incompleteTraces: checks.incomplete,
    } satisfies V5BAlgorithmReceipt;
  });
  const primaryWinner = [...algorithms].sort(
    (left, right) =>
      left.primaryObjective - right.primaryObjective ||
      left.algorithm.localeCompare(right.algorithm),
  )[0].algorithm;
  return {
    version: SVI_SCORE_V5B_VERSION,
    contract: SVI_SCORE_V5B_CONTRACT,
    data: {
      businesses: grouped.size,
      rows: rows.length,
      candidatesPerBusiness: universe.length,
      parameterDimensions: universe[0]?.point.length ?? 0,
      assignmentHash: createHash("sha256")
        .update(JSON.stringify(assignments))
        .digest("hex"),
      selectorModelHashes,
    },
    algorithms,
    primaryWinner,
  };
}
