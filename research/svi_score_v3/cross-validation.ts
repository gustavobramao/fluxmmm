import { createHash } from "node:crypto";
import {
  evaluateScoreV6,
  evaluateRegretBound,
  learnScoreV6,
  scoreV6,
  selectScoreV6Candidates,
} from "../score_v6/learn";
import type {
  ScoreV6CandidateRow,
  ScoreV6FeatureWeights,
  ScoreV6Metrics,
} from "../score_v6/types";

export interface ScoreV3TrainingConfiguration {
  id: string;
  regularization: number;
  epochs: number;
}

export interface GroupedFoldAssignment {
  businessId: string;
  family: string;
  fold: number;
}

export interface GroupedFoldReceipt {
  fold: number;
  trainingBusinesses: number;
  assessmentBusinesses: number;
  assessmentRows: number;
  assessmentFamilies: Record<string, number>;
  metrics: ScoreV6Metrics;
}

export interface GroupedConfigurationReceipt {
  configuration: ScoreV3TrainingConfiguration;
  folds: GroupedFoldReceipt[];
  outOfFoldMetrics: ScoreV6Metrics;
  selection: {
    meanLossDifferenceFromBest: number;
    standardErrorOfPairedDifference: number;
    withinOneStandardError: boolean;
    withinEconomicNonInferiorityMargin: boolean;
    selectable: boolean;
  };
}

interface GroupedConfigurationWork extends GroupedConfigurationReceipt {
  _businessLosses: Record<string, number>;
}

export interface GroupedCrossValidationReceipt {
  version: "flux-svi-score-v3-grouped-cv-v2-regret-bound";
  contract: {
    folds: number;
    groupingUnit: "business";
    stratification: "generator-family";
    selectionMetric: "out-of-fold-mean-excess-decision-loss";
    selectionRule: "paired-one-standard-error-with-economic-non-inferiority";
    maximumRelativeMeanExcessLossIncrease: 0.01;
    tieBreakers: readonly ["stronger-regularization", "configuration-id"];
    heldOutSplitsPermitted: false;
    theoremTarget: "capped-normalized-selected-model-excess-regret";
    pairConstruction: "every-in-pool-oracle-versus-candidate-comparison";
    economicScale: "predeclared-normalized-opportunity-loss-1";
  };
  data: {
    businesses: number;
    rows: number;
    families: Record<string, number>;
    foldBusinesses: Record<string, number>;
    assignmentHash: string;
  };
  assignments: GroupedFoldAssignment[];
  configurations: GroupedConfigurationReceipt[];
  selectedConfiguration: ScoreV3TrainingConfiguration;
  finalFit: {
    weights: ScoreV6FeatureWeights;
    pairCount: number;
    descriptiveFullTrainingMetrics: ScoreV6Metrics;
    theoremAudit: ReturnType<typeof evaluateRegretBound>;
  };
}

export const PREDECLARED_SCORE_V3_TUNING_GRID: readonly ScoreV3TrainingConfiguration[] = [
  { id: "ridge-light", regularization: 0.005, epochs: 180 },
  { id: "ridge-balanced", regularization: 0.015, epochs: 180 },
  { id: "ridge-strong", regularization: 0.05, epochs: 180 },
] as const;

function countBy(values: string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  values.forEach((value) => {
    counts[value] = (counts[value] ?? 0) + 1;
  });
  return Object.fromEntries(
    Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)),
  );
}

function stableOrder(value: string): string {
  return createHash("sha256")
    .update(`flux-svi-score-v3-grouped-cv-v1:${value}`)
    .digest("hex");
}

function assertTrainingRows(rows: ScoreV6CandidateRow[]): Map<string, string> {
  if (!rows.length) throw new Error("Grouped cross-validation requires training rows.");
  const familyByBusiness = new Map<string, string>();
  rows.forEach((row) => {
    if (row.split !== "train") {
      throw new Error(
        `Held-out split '${row.split}' is not permitted in score tuning (${row.businessId}).`,
      );
    }
    if (!Number.isFinite(row.decisionLoss)) {
      throw new Error(`Training row ${row.businessId}/${row.candidateId} has no finite label.`);
    }
    const family = familyByBusiness.get(row.businessId);
    if (family !== undefined && family !== row.family) {
      throw new Error(`Business ${row.businessId} appears in multiple generator families.`);
    }
    familyByBusiness.set(row.businessId, row.family);
  });
  return familyByBusiness;
}

export function groupedBusinessFolds(
  rows: ScoreV6CandidateRow[],
  foldCount = 5,
): GroupedFoldAssignment[] {
  if (!Number.isInteger(foldCount) || foldCount < 2) {
    throw new Error("Grouped cross-validation requires at least two folds.");
  }
  const familyByBusiness = assertTrainingRows(rows);
  if (familyByBusiness.size < foldCount) {
    throw new Error("The number of training businesses must be at least the fold count.");
  }
  const businessesByFamily = new Map<string, string[]>();
  familyByBusiness.forEach((family, businessId) => {
    businessesByFamily.set(family, [
      ...(businessesByFamily.get(family) ?? []),
      businessId,
    ]);
  });
  const assignments: GroupedFoldAssignment[] = [];
  [...businessesByFamily.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .forEach(([family, businessIds]) => {
      businessIds
        .sort((left, right) =>
          stableOrder(left).localeCompare(stableOrder(right)) || left.localeCompare(right)
        )
        .forEach((businessId, index) => {
          assignments.push({ businessId, family, fold: index % foldCount });
        });
    });
  return assignments.sort((left, right) =>
    left.businessId.localeCompare(right.businessId)
  );
}

function validateConfigurations(
  configurations: readonly ScoreV3TrainingConfiguration[],
): void {
  if (!configurations.length) throw new Error("At least one training configuration is required.");
  const ids = new Set<string>();
  configurations.forEach((configuration) => {
    if (!configuration.id || ids.has(configuration.id)) {
      throw new Error(`Training configuration IDs must be unique: '${configuration.id}'.`);
    }
    ids.add(configuration.id);
    if (!Number.isFinite(configuration.regularization) || configuration.regularization < 0) {
      throw new Error(`Invalid regularization for ${configuration.id}.`);
    }
    if (!Number.isInteger(configuration.epochs) || configuration.epochs < 1) {
      throw new Error(`Invalid epoch count for ${configuration.id}.`);
    }
  });
}

function configurationOrder(
  left: GroupedConfigurationReceipt,
  right: GroupedConfigurationReceipt,
): number {
  return Number(right.selection.selectable) -
      Number(left.selection.selectable) ||
    right.configuration.regularization - left.configuration.regularization ||
    left.configuration.id.localeCompare(right.configuration.id);
}

function average(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) /
    Math.max(values.length, 1);
}

function standardError(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = average(values);
  const variance = values.reduce(
    (total, value) => total + (value - mean) ** 2,
    0,
  ) / (values.length - 1);
  return Math.sqrt(variance / values.length);
}

export function crossValidateScoreV3(
  trainRows: ScoreV6CandidateRow[],
  configurations: readonly ScoreV3TrainingConfiguration[] = PREDECLARED_SCORE_V3_TUNING_GRID,
  foldCount = 5,
): GroupedCrossValidationReceipt {
  validateConfigurations(configurations);
  const assignments = groupedBusinessFolds(trainRows, foldCount);
  const foldByBusiness = new Map(
    assignments.map((assignment) => [assignment.businessId, assignment.fold]),
  );
  const configurationsReceipts: GroupedConfigurationWork[] = configurations.map((configuration) => {
    const outOfFoldScores = new Map<ScoreV6CandidateRow, number>();
    const folds = Array.from({ length: foldCount }, (_, fold) => {
      const fittingRows = trainRows.filter(
        (row) => foldByBusiness.get(row.businessId) !== fold,
      );
      const assessmentRows = trainRows.filter(
        (row) => foldByBusiness.get(row.businessId) === fold,
      );
      const fit = learnScoreV6(
        fittingRows,
        configuration.regularization,
        configuration.epochs,
      );
      assessmentRows.forEach((row) => {
        outOfFoldScores.set(row, scoreV6(row, fit.weights));
      });
      const assessmentBusinessIds = [
        ...new Set(assessmentRows.map((row) => row.businessId)),
      ];
      return {
        fold,
        trainingBusinesses: new Set(fittingRows.map((row) => row.businessId)).size,
        assessmentBusinesses: assessmentBusinessIds.length,
        assessmentRows: assessmentRows.length,
        assessmentFamilies: countBy(
          assessmentBusinessIds.map(
            (businessId) => assignments.find(
              (assignment) => assignment.businessId === businessId,
            )!.family,
          ),
        ),
        metrics: evaluateScoreV6(
          assessmentRows,
          (row) => outOfFoldScores.get(row)!,
        ),
      } satisfies GroupedFoldReceipt;
    });
    if (outOfFoldScores.size !== trainRows.length) {
      throw new Error("Every training row must receive exactly one out-of-fold score.");
    }
    const selections = selectScoreV6Candidates(
      trainRows,
      (row) => outOfFoldScores.get(row)!,
    );
    return {
      configuration: { ...configuration },
      folds,
      outOfFoldMetrics: evaluateScoreV6(
        trainRows,
        (row) => outOfFoldScores.get(row)!,
      ),
      selection: {
        meanLossDifferenceFromBest: 0,
        standardErrorOfPairedDifference: 0,
        withinOneStandardError: false,
        withinEconomicNonInferiorityMargin: false,
        selectable: false,
      },
      _businessLosses: Object.fromEntries(
        selections.map((selection) => [selection.businessId, selection.loss]),
      ),
    } satisfies GroupedConfigurationWork;
  });
  const bestMean = [...configurationsReceipts].sort(
    (left, right) =>
      left.outOfFoldMetrics.meanExcessLoss - right.outOfFoldMetrics.meanExcessLoss ||
      left.configuration.id.localeCompare(right.configuration.id),
  )[0];
  const bestLosses = bestMean._businessLosses;
  const economicNonInferiorityMargin = Math.max(
    bestMean.outOfFoldMetrics.meanExcessLoss * 0.01,
    1e-12,
  );
  configurationsReceipts.forEach((receipt) => {
    const differences = Object.entries(receipt._businessLosses).map(
      ([businessId, loss]) => loss - bestLosses[businessId],
    );
    const meanDifference = average(differences);
    const differenceStandardError = standardError(differences);
    const withinOneStandardError =
      meanDifference <= differenceStandardError + 1e-12;
    const withinEconomicNonInferiorityMargin =
      meanDifference <= economicNonInferiorityMargin + 1e-12;
    receipt.selection = {
      meanLossDifferenceFromBest: meanDifference,
      standardErrorOfPairedDifference: differenceStandardError,
      withinOneStandardError,
      withinEconomicNonInferiorityMargin,
      selectable: withinOneStandardError && withinEconomicNonInferiorityMargin,
    };
  });
  const selected = [...configurationsReceipts].sort(configurationOrder)[0];
  const finalFit = learnScoreV6(
    trainRows,
    selected.configuration.regularization,
    selected.configuration.epochs,
  );
  const foldBusinesses = Object.fromEntries(
    Array.from({ length: foldCount }, (_, fold) => [
      `fold-${fold}`,
      assignments.filter((assignment) => assignment.fold === fold).length,
    ]),
  );
  const assignmentHash = createHash("sha256")
    .update(JSON.stringify(assignments))
    .digest("hex");
  return {
    version: "flux-svi-score-v3-grouped-cv-v2-regret-bound",
    contract: {
      folds: foldCount,
      groupingUnit: "business",
      stratification: "generator-family",
      selectionMetric: "out-of-fold-mean-excess-decision-loss",
      selectionRule: "paired-one-standard-error-with-economic-non-inferiority",
      maximumRelativeMeanExcessLossIncrease: 0.01,
      tieBreakers: ["stronger-regularization", "configuration-id"],
      heldOutSplitsPermitted: false,
      theoremTarget: "capped-normalized-selected-model-excess-regret",
      pairConstruction: "every-in-pool-oracle-versus-candidate-comparison",
      economicScale: "predeclared-normalized-opportunity-loss-1",
    },
    data: {
      businesses: assignments.length,
      rows: trainRows.length,
      families: countBy(assignments.map((assignment) => assignment.family)),
      foldBusinesses,
      assignmentHash,
    },
    assignments,
    configurations: configurationsReceipts.map(({ _businessLosses: _losses, ...receipt }) => receipt),
    selectedConfiguration: { ...selected.configuration },
    finalFit: {
      weights: finalFit.weights,
      pairCount: finalFit.pairCount,
      descriptiveFullTrainingMetrics: evaluateScoreV6(
        trainRows,
        (row) => scoreV6(row, finalFit.weights),
      ),
      theoremAudit: evaluateRegretBound(
        trainRows,
        (row) => scoreV6(row, finalFit.weights),
      ),
    },
  };
}
