import { createHash } from "node:crypto";
import {
  PREDECLARED_SCORE_V4_TUNING_GRID,
  type SviScoreV4TrainingConfiguration,
} from "./contract";
import {
  evaluateScoreV4,
  evaluateTopOneRegretBound,
  learnScoreV4,
  metricsFromV4Selections,
  selectScoreV4Candidates,
} from "./ranker";
import type {
  SviScoreV4CandidateRow,
  SviScoreV4Metrics,
  SviScoreV4Model,
  SviScoreV4Selection,
  SviScoreV4TopOneBoundReceipt,
} from "./types";

export interface SviScoreV4FoldAssignment {
  businessId: string;
  family: string;
  fold: number;
}

export interface SviScoreV4FoldReceipt {
  fold: number;
  trainingBusinesses: number;
  assessmentBusinesses: number;
  assessmentFamilies: Record<string, number>;
  metrics: SviScoreV4Metrics;
  theorem: SviScoreV4TopOneBoundReceipt;
}

export interface SviScoreV4ConfigurationReceipt {
  configuration: SviScoreV4TrainingConfiguration;
  folds: SviScoreV4FoldReceipt[];
  outOfFoldMetrics: SviScoreV4Metrics;
  selectionObjective: number;
}

export interface SviScoreV4CrossValidationReceipt {
  version: "flux-svi-score-v4-grouped-cv-v1";
  contract: {
    folds: number;
    groupingUnit: "business";
    stratification: "generator-family";
    auditRowsPermitted: false;
    selectionObjective:
      "mean-loss-plus-0.5-cvar90-loss-plus-0.5-worst-family-mean-loss";
    theoremTarget: "capped-normalized-top-one-excess-regret";
  };
  data: {
    businesses: number;
    rows: number;
    families: Record<string, number>;
    assignmentHash: string;
  };
  assignments: SviScoreV4FoldAssignment[];
  configurations: SviScoreV4ConfigurationReceipt[];
  selectedConfiguration: SviScoreV4TrainingConfiguration;
  finalFit: {
    model: SviScoreV4Model;
    descriptiveDevelopmentMetrics: SviScoreV4Metrics;
    theorem: SviScoreV4TopOneBoundReceipt;
  };
}

function stableOrder(value: string): string {
  return createHash("sha256")
    .update(`flux-svi-score-v4-grouped-cv-v1:${value}`)
    .digest("hex");
}

function countBy(values: string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  values.forEach((value) => {
    counts[value] = (counts[value] ?? 0) + 1;
  });
  return Object.fromEntries(
    Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)),
  );
}

function familyByBusiness(
  rows: readonly SviScoreV4CandidateRow[],
): Map<string, string> {
  const families = new Map<string, string>();
  rows.forEach((row) => {
    if (row.split === "audit") {
      throw new Error("The sealed V3 audit is forbidden in V4 cross-validation.");
    }
    if (!Number.isFinite(row.decisionLoss)) {
      throw new Error(`V4 label is not finite for ${row.businessId}/${row.candidateId}.`);
    }
    const existing = families.get(row.businessId);
    if (existing && existing !== row.family) {
      throw new Error(`${row.businessId} appears in multiple generator families.`);
    }
    families.set(row.businessId, row.family);
  });
  return families;
}

export function groupedV4BusinessFolds(
  rows: readonly SviScoreV4CandidateRow[],
  foldCount = 5,
): SviScoreV4FoldAssignment[] {
  if (!Number.isInteger(foldCount) || foldCount < 2) {
    throw new Error("V4 grouped cross-validation needs at least two folds.");
  }
  const families = familyByBusiness(rows);
  const byFamily = new Map<string, string[]>();
  families.forEach((family, businessId) => {
    byFamily.set(family, [...(byFamily.get(family) ?? []), businessId]);
  });
  const assignments = [...byFamily.entries()].flatMap(([family, businesses]) =>
    businesses
      .sort((left, right) =>
        stableOrder(left).localeCompare(stableOrder(right)) ||
        left.localeCompare(right)
      )
      .map((businessId, index) => ({
        businessId,
        family,
        fold: index % foldCount,
      }))
  );
  return assignments.sort((left, right) =>
    left.businessId.localeCompare(right.businessId)
  );
}

function selectionObjective(metrics: SviScoreV4Metrics): number {
  // Absolute decision loss is required here: otherwise a strict gate could
  // remove the global oracle, call the best remaining candidate its oracle,
  // and appear to have zero excess loss while making a worse decision.
  return metrics.meanLoss +
    0.5 * metrics.cvar90Loss +
    0.5 * metrics.worstFamilyMeanLoss;
}

function combineBounds(
  receipts: readonly SviScoreV4TopOneBoundReceipt[],
): SviScoreV4TopOneBoundReceipt {
  const businesses = receipts.reduce((sum, receipt) => sum + receipt.businesses, 0);
  const weighted = (
    key:
      | "meanSelectedNormalizedRegret"
      | "meanTopOneHingeBound"
      | "meanSmoothTopOneBound",
  ) => receipts.reduce(
    (sum, receipt) => sum + receipt.businesses * receipt[key],
    0,
  ) / Math.max(businesses, 1);
  return {
    businesses,
    violations: receipts.reduce((sum, receipt) => sum + receipt.violations, 0),
    meanSelectedNormalizedRegret: weighted("meanSelectedNormalizedRegret"),
    meanTopOneHingeBound: weighted("meanTopOneHingeBound"),
    meanSmoothTopOneBound: weighted("meanSmoothTopOneBound"),
    maximumNumericalSlack: Math.max(
      0,
      ...receipts.map((receipt) => receipt.maximumNumericalSlack),
    ),
  };
}

export function crossValidateScoreV4(
  rows: readonly SviScoreV4CandidateRow[],
  configurations: readonly SviScoreV4TrainingConfiguration[] =
    PREDECLARED_SCORE_V4_TUNING_GRID,
  foldCount = 5,
): SviScoreV4CrossValidationReceipt {
  if (!configurations.length) throw new Error("V4 requires a tuning grid.");
  const assignments = groupedV4BusinessFolds(rows, foldCount);
  const foldByBusiness = new Map(
    assignments.map((assignment) => [assignment.businessId, assignment.fold]),
  );
  const receipts = configurations.map((configuration) => {
    const outOfFoldSelections: SviScoreV4Selection[] = [];
    const folds = Array.from({ length: foldCount }, (_, fold) => {
      const training = rows.filter(
        (row) => foldByBusiness.get(row.businessId) !== fold,
      );
      const assessment = rows.filter(
        (row) => foldByBusiness.get(row.businessId) === fold,
      );
      const model = learnScoreV4(training, configuration);
      const selections = selectScoreV4Candidates(assessment, model);
      outOfFoldSelections.push(...selections);
      const assessmentAssignments = assignments.filter(
        (assignment) => assignment.fold === fold,
      );
      return {
        fold,
        trainingBusinesses: new Set(training.map((row) => row.businessId)).size,
        assessmentBusinesses: assessmentAssignments.length,
        assessmentFamilies: countBy(
          assessmentAssignments.map((assignment) => assignment.family),
        ),
        metrics: metricsFromV4Selections(selections),
        theorem: evaluateTopOneRegretBound(assessment, model),
      } satisfies SviScoreV4FoldReceipt;
    });
    const outOfFoldMetrics = metricsFromV4Selections(outOfFoldSelections);
    // Evaluated for its proof checks even though the combined receipt is not
    // persisted separately in the compact artifact.
    const combinedTheorem = combineBounds(folds.map((fold) => fold.theorem));
    if (combinedTheorem.violations > 0) {
      throw new Error(
        `${configuration.id} violated the finite-candidate top-one theorem.`,
      );
    }
    return {
      configuration: structuredClone(configuration),
      folds,
      outOfFoldMetrics,
      selectionObjective: selectionObjective(outOfFoldMetrics),
    } satisfies SviScoreV4ConfigurationReceipt;
  });
  const selected = [...receipts].sort(
    (left, right) =>
      left.selectionObjective - right.selectionObjective ||
      right.configuration.regularization - left.configuration.regularization ||
      left.configuration.id.localeCompare(right.configuration.id),
  )[0];
  const finalModel = learnScoreV4(rows, selected.configuration);
  const familyMap = familyByBusiness(rows);
  return {
    version: "flux-svi-score-v4-grouped-cv-v1",
    contract: {
      folds: foldCount,
      groupingUnit: "business",
      stratification: "generator-family",
      auditRowsPermitted: false,
      selectionObjective:
        "mean-loss-plus-0.5-cvar90-loss-plus-0.5-worst-family-mean-loss",
      theoremTarget: "capped-normalized-top-one-excess-regret",
    },
    data: {
      businesses: familyMap.size,
      rows: rows.length,
      families: countBy([...familyMap.values()]),
      assignmentHash: createHash("sha256")
        .update(JSON.stringify(assignments))
        .digest("hex"),
    },
    assignments,
    configurations: receipts,
    selectedConfiguration: structuredClone(selected.configuration),
    finalFit: {
      model: finalModel,
      descriptiveDevelopmentMetrics: evaluateScoreV4(rows, finalModel),
      theorem: evaluateTopOneRegretBound(rows, finalModel),
    },
  };
}
