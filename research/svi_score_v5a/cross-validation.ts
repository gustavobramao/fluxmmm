import { createHash } from "node:crypto";
import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";
import { groupedV4BusinessFolds } from "../svi_score_v4/cross-validation";
import { metricsFromV4Selections } from "../svi_score_v4/ranker";
import {
  PREDECLARED_SCORE_V5A_TUNING_GRID,
  type SviScoreV5ATrainingConfiguration,
} from "./contract";
import { v5aFeatureNames } from "./features";
import {
  evaluateScoreV5A,
  evaluateV5ATopOneRegretBound,
  learnScoreV5A,
  selectScoreV5ACandidates,
  v5aSelectionObjective,
} from "./ranker";
import type {
  SviScoreV5AConfigurationReceipt,
  SviScoreV5ACrossValidationReceipt,
  SviScoreV5ASelection,
  SviScoreV5ATheoremReceipt,
} from "./types";

function countBy(values: readonly string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  values.forEach((value) => {
    counts[value] = (counts[value] ?? 0) + 1;
  });
  return Object.fromEntries(
    Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)),
  );
}

function validateRows(rows: readonly SviScoreV4CandidateRow[]): Map<string, string> {
  if (!rows.length) throw new Error("V5A cross-validation requires candidate rows.");
  const families = new Map<string, string>();
  rows.forEach((row) => {
    if (row.split === "audit") {
      throw new Error("The sealed V3 audit is forbidden in V5A cross-validation.");
    }
    if (!Number.isFinite(row.decisionLoss)) {
      throw new Error(`V5A label is not finite for ${row.businessId}/${row.candidateId}.`);
    }
    const existing = families.get(row.businessId);
    if (existing && existing !== row.family) {
      throw new Error(`${row.businessId} appears in multiple generator families.`);
    }
    families.set(row.businessId, row.family);
  });
  return families;
}

function combineBounds(
  receipts: readonly SviScoreV5ATheoremReceipt[],
): SviScoreV5ATheoremReceipt {
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

export function crossValidateScoreV5A(
  rows: readonly SviScoreV4CandidateRow[],
  configurations: readonly SviScoreV5ATrainingConfiguration[] =
    PREDECLARED_SCORE_V5A_TUNING_GRID,
  foldCount = 5,
): SviScoreV5ACrossValidationReceipt {
  if (!configurations.length) throw new Error("V5A requires a tuning grid.");
  const familyMap = validateRows(rows);
  const assignments = groupedV4BusinessFolds(rows, foldCount);
  const foldByBusiness = new Map(
    assignments.map((assignment) => [assignment.businessId, assignment.fold]),
  );
  const configurationsReceipts = configurations.map((configuration) => {
    const outOfFoldSelections: SviScoreV5ASelection[] = [];
    const folds = Array.from({ length: foldCount }, (_, fold) => {
      const training = rows.filter(
        (row) => foldByBusiness.get(row.businessId) !== fold,
      );
      const assessment = rows.filter(
        (row) => foldByBusiness.get(row.businessId) === fold,
      );
      const model = learnScoreV5A(training, configuration);
      const selections = selectScoreV5ACandidates(assessment, model);
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
        theorem: evaluateV5ATopOneRegretBound(assessment, model),
      };
    });
    const outOfFoldMetrics = metricsFromV4Selections(outOfFoldSelections);
    const combinedTheorem = combineBounds(folds.map((fold) => fold.theorem));
    if (combinedTheorem.violations > 0) {
      throw new Error(`${configuration.id} violated the V5A top-one theorem.`);
    }
    return {
      configuration: structuredClone(configuration),
      folds,
      outOfFoldMetrics,
      selectionObjective: v5aSelectionObjective(outOfFoldMetrics),
    } satisfies SviScoreV5AConfigurationReceipt;
  });
  const selected = [...configurationsReceipts].sort(
    (left, right) =>
      left.selectionObjective - right.selectionObjective ||
      right.configuration.regularization - left.configuration.regularization ||
      left.configuration.id.localeCompare(right.configuration.id),
  )[0];
  const finalModel = learnScoreV5A(rows, selected.configuration);
  const candidatesByBusiness = countBy(rows.map((row) => row.businessId));
  const candidateCounts = Object.values(candidatesByBusiness);
  return {
    version: "flux-svi-score-v5a-grouped-cv-v1",
    contract: {
      folds: foldCount,
      groupingUnit: "business",
      stratification: "generator-family",
      commonCandidatePool: "v4.2-roi-safety-only",
      auditRowsPermitted: false,
      selectionObjective:
        "mean-loss-plus-0.5-cvar90-loss-plus-0.5-worst-family-mean-loss",
      theoremTarget: "capped-normalized-top-one-excess-regret",
    },
    data: {
      businesses: familyMap.size,
      rows: rows.length,
      candidatesPerBusiness: candidateCounts.every(
          (value) => value === candidateCounts[0]
        )
        ? candidateCounts[0]
        : 0,
      families: countBy([...familyMap.values()]),
      assignmentHash: createHash("sha256")
        .update(JSON.stringify(assignments))
        .digest("hex"),
      featureHash: createHash("sha256")
        .update(JSON.stringify(v5aFeatureNames(rows[0])))
        .digest("hex"),
    },
    assignments,
    configurations: configurationsReceipts,
    selectedConfiguration: structuredClone(selected.configuration),
    finalFit: {
      model: finalModel,
      descriptiveDevelopmentMetrics: evaluateScoreV5A(rows, finalModel),
      theorem: evaluateV5ATopOneRegretBound(rows, finalModel),
    },
  };
}
