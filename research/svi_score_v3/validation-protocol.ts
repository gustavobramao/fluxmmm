import {
  normalizedExcessRegret,
  scoreV6,
  selectScoreV6Candidates,
} from "../score_v6/learn";
import type {
  ScoreV6CandidateRow,
  ScoreV6FeatureWeights,
} from "../score_v6/types";

export const SVI_SCORE_V3_VALIDATION_PROTOCOL = {
  id: "flux-svi-score-v3-validation-v2-powered-screen",
  status: "predeclared-unopened",
  split: "validation",
  businesses: 120,
  candidatesPerBusiness: 48,
  rows: 5_760,
  families: {
    "delayed-tv": 60,
    "correlated-planning": 60,
  },
  primaryComparator: "active-v6-diagnostic-ranker",
  economicLabel:
    "uncapped SVI posterior-expected decision loss averaged over four predeclared budget decisions",
  bootstrap: {
    method: "generator-family-stratified-paired-percentile",
    replicates: 20_000,
    seed: 2_026_090_1,
    oneSidedConfidence: 0.95,
  },
  margins: {
    meanExcessLossRelativeNonInferiority: 0.02,
    p90ExcessLossRelativeNonInferiority: 0.02,
    p95ExcessLossRelativeNonInferiority: 0.02,
    familyMeanExcessLossRelativeNonInferiority: 0.02,
    lowestLossSelectionRateAbsoluteNonInferiority: 0.02,
  },
} as const;

export interface ValidationMethodReceipt {
  meanExcessLoss: number;
  p90ExcessLoss: number;
  p95ExcessLoss: number;
  lowestLossSelectionRate: number;
  decisionGradeBusinessShare: number;
  meanRegretWeightedMisrankingBound: number;
  familyMeanExcessLoss: Record<string, number>;
}

export interface SviScoreV3ValidationReceipt {
  protocolId: typeof SVI_SCORE_V3_VALIDATION_PROTOCOL.id;
  businesses: number;
  rows: number;
  challenger: ValidationMethodReceipt;
  comparator: ValidationMethodReceipt;
  pairedPrimary: {
    meanDifference: number;
    relativeMeanReduction: number;
    oneSided95UpperBound: number;
    nonInferiorityMargin: number;
    statisticallyNonInferior: boolean;
    pointEstimateImproves: boolean;
    statisticallySuperior: boolean;
  };
  gates: {
    completeCohort: boolean;
    primaryMeanImproves: boolean;
    theoremMisrankingBoundImproves: boolean;
    p90TailNonInferior: boolean;
    p95TailNonInferior: boolean;
    familyMeansNonInferior: boolean;
    oracleSelectionRateNonInferior: boolean;
    immutableEligibilityAgreement: boolean;
  };
  proceedToSealedAudit: boolean;
}

function average(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) /
    Math.max(values.length, 1);
}

function quantile(values: number[], probability: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  if (!sorted.length) return 0;
  const position = Math.max(0, Math.min(1, probability)) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function randomGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function stratifiedBootstrapUpperBound(
  differences: Array<{ family: string; value: number }>,
): number {
  const byFamily = new Map<string, number[]>();
  differences.forEach(({ family, value }) => {
    byFamily.set(family, [...(byFamily.get(family) ?? []), value]);
  });
  const random = randomGenerator(SVI_SCORE_V3_VALIDATION_PROTOCOL.bootstrap.seed);
  const estimates = Array.from(
    { length: SVI_SCORE_V3_VALIDATION_PROTOCOL.bootstrap.replicates },
    () => {
      const sample: number[] = [];
      [...byFamily.values()].forEach((values) => {
        for (let index = 0; index < values.length; index += 1) {
          sample.push(values[Math.floor(random() * values.length)]);
        }
      });
      return average(sample);
    },
  );
  return quantile(
    estimates,
    SVI_SCORE_V3_VALIDATION_PROTOCOL.bootstrap.oneSidedConfidence,
  );
}

function assertValidationCohort(rows: readonly ScoreV6CandidateRow[]): void {
  const protocol = SVI_SCORE_V3_VALIDATION_PROTOCOL;
  if (rows.length !== protocol.rows) {
    throw new Error(`Validation requires exactly ${protocol.rows} candidate rows.`);
  }
  if (rows.some((row) => row.split !== "validation")) {
    throw new Error("Training or audit rows are forbidden in validation.");
  }
  if (rows.some((row) => !Number.isFinite(row.decisionLoss))) {
    throw new Error("Every validation candidate requires a finite frozen SVI label.");
  }
  const rowsByBusiness = new Map<string, ScoreV6CandidateRow[]>();
  rows.forEach((row) => {
    rowsByBusiness.set(row.businessId, [
      ...(rowsByBusiness.get(row.businessId) ?? []),
      row,
    ]);
  });
  if (rowsByBusiness.size !== protocol.businesses) {
    throw new Error(`Validation requires exactly ${protocol.businesses} businesses.`);
  }
  rowsByBusiness.forEach((businessRows, businessId) => {
    if (businessRows.length !== protocol.candidatesPerBusiness) {
      throw new Error(
        `${businessId} requires exactly ${protocol.candidatesPerBusiness} candidates.`,
      );
    }
    if (new Set(businessRows.map((row) => row.candidateId)).size !== businessRows.length) {
      throw new Error(`${businessId} contains duplicate candidate IDs.`);
    }
  });
  const familyCounts = Object.fromEntries(
    [...rowsByBusiness.values()].reduce<Array<[string, number]>>((entries, business) => {
      const family = business[0].family;
      const existing = entries.find(([id]) => id === family);
      if (existing) existing[1] += 1;
      else entries.push([family, 1]);
      return entries;
    }, []),
  );
  if (
    JSON.stringify(Object.fromEntries(Object.entries(familyCounts).sort())) !==
    JSON.stringify(Object.fromEntries(Object.entries(protocol.families).sort()))
  ) {
    throw new Error("Validation generator-family counts do not match the predeclared cohort.");
  }
}

function methodReceipt(
  rows: readonly ScoreV6CandidateRow[],
  selections: ReturnType<typeof selectScoreV6Candidates>,
  weights: ScoreV6FeatureWeights,
): ValidationMethodReceipt {
  const rowsByBusiness = new Map<string, ScoreV6CandidateRow[]>();
  rows.forEach((row) => {
    rowsByBusiness.set(row.businessId, [
      ...(rowsByBusiness.get(row.businessId) ?? []),
      row,
    ]);
  });
  const enriched = selections.map((selection) => {
    const businessRows = rowsByBusiness.get(selection.businessId)!;
    const eligible = businessRows.filter((row) => row.eligible);
    const review = businessRows.filter((row) => row.reviewEligible);
    const pool = eligible.length ? eligible : review.length ? review : businessRows;
    const oracleLoss = Math.min(...pool.map((row) => row.decisionLoss));
    return { ...selection, excess: Math.max(0, selection.loss - oracleLoss) };
  });
  const familyMeanExcessLoss = Object.fromEntries(
    Object.keys(SVI_SCORE_V3_VALIDATION_PROTOCOL.families).map((family) => [
      family,
      average(enriched.filter((row) => row.family === family).map((row) => row.excess)),
    ]),
  );
  const excessLosses = enriched.map((selection) => selection.excess);
  const regretWeightedMisrankingBounds = [...rowsByBusiness.values()].map(
    (businessRows) => {
      const eligible = businessRows.filter((row) => row.eligible);
      const review = businessRows.filter((row) => row.reviewEligible);
      const pool = eligible.length ? eligible : review.length ? review : businessRows;
      const oracle = [...pool].sort(
        (left, right) =>
          left.decisionLoss - right.decisionLoss ||
          left.candidateId.localeCompare(right.candidateId),
      )[0];
      const oracleScore = scoreV6(oracle, weights);
      return pool.reduce(
        (total, candidate) =>
          candidate === oracle || scoreV6(candidate, weights) < oracleScore
            ? total
            : total + normalizedExcessRegret(
                candidate.decisionLoss,
                oracle.decisionLoss,
                candidate.economicScale ?? 1,
              ),
        0,
      );
    },
  );
  return {
    meanExcessLoss: average(excessLosses),
    p90ExcessLoss: quantile(excessLosses, 0.9),
    p95ExcessLoss: quantile(excessLosses, 0.95),
    lowestLossSelectionRate: average(enriched.map((selection) => selection.lowest ? 1 : 0)),
    decisionGradeBusinessShare: average(
      enriched.map((selection) => selection.decisionGrade ? 1 : 0),
    ),
    meanRegretWeightedMisrankingBound: average(
      regretWeightedMisrankingBounds,
    ),
    familyMeanExcessLoss,
  };
}

export function evaluateSviScoreV3Validation(
  rows: ScoreV6CandidateRow[],
  challengerWeights: ScoreV6FeatureWeights,
  comparatorWeights: ScoreV6FeatureWeights,
): SviScoreV3ValidationReceipt {
  assertValidationCohort(rows);
  const challengerSelections = selectScoreV6Candidates(
    rows,
    (row) => scoreV6(row, challengerWeights),
  );
  const comparatorSelections = selectScoreV6Candidates(
    rows,
    (row) => scoreV6(row, comparatorWeights),
  );
  const challenger = methodReceipt(rows, challengerSelections, challengerWeights);
  const comparator = methodReceipt(rows, comparatorSelections, comparatorWeights);
  const comparatorByBusiness = new Map(
    comparatorSelections.map((selection) => [selection.businessId, selection]),
  );
  const pairedDifferences = challengerSelections.map((selection) => {
    const comparatorSelection = comparatorByBusiness.get(selection.businessId)!;
    const businessRows = rows.filter((row) => row.businessId === selection.businessId);
    const eligible = businessRows.filter((row) => row.eligible);
    const review = businessRows.filter((row) => row.reviewEligible);
    const pool = eligible.length ? eligible : review.length ? review : businessRows;
    const oracleLoss = Math.min(...pool.map((row) => row.decisionLoss));
    return {
      family: selection.family,
      value:
        Math.max(0, selection.loss - oracleLoss) -
        Math.max(0, comparatorSelection.loss - oracleLoss),
    };
  });
  const meanDifference = average(pairedDifferences.map(({ value }) => value));
  const oneSided95UpperBound = stratifiedBootstrapUpperBound(pairedDifferences);
  const nonInferiorityMargin =
    comparator.meanExcessLoss *
    SVI_SCORE_V3_VALIDATION_PROTOCOL.margins.meanExcessLossRelativeNonInferiority;
  const relativeMeanReduction =
    (comparator.meanExcessLoss - challenger.meanExcessLoss) /
    Math.max(comparator.meanExcessLoss, 1e-12);
  const statisticallyNonInferior =
    oneSided95UpperBound <= nonInferiorityMargin + 1e-12;
  const pointEstimateImproves = meanDifference < 0;
  const statisticallySuperior = oneSided95UpperBound < 0;
  const relativeGate = (challengerValue: number, comparatorValue: number, margin: number) =>
    challengerValue <= comparatorValue * (1 + margin) + 1e-12;
  const gates = {
    completeCohort: true,
    primaryMeanImproves: pointEstimateImproves,
    theoremMisrankingBoundImproves:
      challenger.meanRegretWeightedMisrankingBound <
      comparator.meanRegretWeightedMisrankingBound,
    p90TailNonInferior: relativeGate(
      challenger.p90ExcessLoss,
      comparator.p90ExcessLoss,
      SVI_SCORE_V3_VALIDATION_PROTOCOL.margins.p90ExcessLossRelativeNonInferiority,
    ),
    p95TailNonInferior: relativeGate(
      challenger.p95ExcessLoss,
      comparator.p95ExcessLoss,
      SVI_SCORE_V3_VALIDATION_PROTOCOL.margins.p95ExcessLossRelativeNonInferiority,
    ),
    familyMeansNonInferior: Object.keys(
      SVI_SCORE_V3_VALIDATION_PROTOCOL.families,
    ).every((family) => relativeGate(
      challenger.familyMeanExcessLoss[family],
      comparator.familyMeanExcessLoss[family],
      SVI_SCORE_V3_VALIDATION_PROTOCOL.margins.familyMeanExcessLossRelativeNonInferiority,
    )),
    oracleSelectionRateNonInferior:
      challenger.lowestLossSelectionRate +
        SVI_SCORE_V3_VALIDATION_PROTOCOL.margins
          .lowestLossSelectionRateAbsoluteNonInferiority >=
      comparator.lowestLossSelectionRate,
    immutableEligibilityAgreement:
      Math.abs(
        challenger.decisionGradeBusinessShare - comparator.decisionGradeBusinessShare,
      ) < 1e-12,
  };
  return {
    protocolId: SVI_SCORE_V3_VALIDATION_PROTOCOL.id,
    businesses: SVI_SCORE_V3_VALIDATION_PROTOCOL.businesses,
    rows: rows.length,
    challenger,
    comparator,
    pairedPrimary: {
      meanDifference,
      relativeMeanReduction,
      oneSided95UpperBound,
      nonInferiorityMargin,
      statisticallyNonInferior,
      pointEstimateImproves,
      statisticallySuperior,
    },
    gates,
    proceedToSealedAudit: Object.values(gates).every(Boolean),
  };
}
