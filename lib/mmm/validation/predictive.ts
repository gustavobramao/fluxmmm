import { toNumber } from "../csv";
import { mean, rmse, std } from "../math";
import type { Dataset } from "../types";
import { fitValidationFold, type ValidationModelSpec } from "./adapter";
import { clampScore, weightedScore } from "./math";
import type {
  ValidationLayerResult,
  ValidationStatus,
  ValidationTest,
} from "./types";

function statusForScore(score: number): ValidationStatus {
  return score >= 75 ? "pass" : score >= 55 ? "review" : "fail";
}

function wape(actual: number[], predicted: number[]): number {
  return (
    actual.reduce(
      (total, value, index) =>
        total + Math.abs(value - predicted[index]),
      0,
    ) /
    Math.max(
      actual.reduce((total, value) => total + Math.abs(value), 0),
      1,
    )
  );
}

function outOfSampleR2(actual: number[], predicted: number[]): number {
  const average = mean(actual);
  const total = actual.reduce(
    (sum, value) => sum + (value - average) ** 2,
    0,
  );
  const residual = actual.reduce(
    (sum, value, index) => sum + (value - predicted[index]) ** 2,
    0,
  );
  return 1 - residual / Math.max(total, 1e-9);
}

function seasonalBaseline(
  outcome: number[],
  trainIndexes: number[],
  testIndexes: number[],
  periodsPerYear: number,
): number[] {
  const trainSet = new Set(trainIndexes);
  const fallback = mean(trainIndexes.map((index) => outcome[index]));
  const seasonalLag = Math.max(1, Math.round(periodsPerYear));
  return testIndexes.map((index) =>
    trainSet.has(index - seasonalLag) ? outcome[index - seasonalLag] : fallback,
  );
}

function metricTest(
  id: string,
  name: string,
  score: number,
  metric: string,
  detail: string,
  importance: ValidationTest["importance"],
): ValidationTest {
  return {
    id,
    name,
    score: clampScore(score),
    status: statusForScore(score),
    metric,
    detail,
    importance,
  };
}

export function runGeneralizationValidation(
  dataset: Dataset,
  spec: ValidationModelSpec,
): ValidationLayerResult {
  const rowCount = dataset.rows.length;
  const minimumHorizon = dataset.modelCadence === "monthly" ? 3 : 6;
  const maximumHorizon = dataset.modelCadence === "monthly" ? 6 : 13;
  const minimumTraining = dataset.modelCadence === "monthly" ? 18 : 36;
  const horizon = Math.max(
    minimumHorizon,
    Math.min(maximumHorizon, Math.floor(rowCount * 0.08)),
  );
  const endpoints = [0.58, 0.72, 0.86].map((fraction) =>
    Math.min(
      Math.max(Math.floor(rowCount * fraction), minimumTraining),
      rowCount - horizon,
    ),
  );
  const outcome = dataset.rows.map((row) =>
    toNumber(row[dataset.outcomeColumn]),
  );
  const foldResults = endpoints.map((endpoint, foldIndex) => {
    const trainIndexes = Array.from({ length: endpoint }, (_, index) => index);
    const testIndexes = Array.from(
      { length: Math.min(horizon, rowCount - endpoint) },
      (_, index) => endpoint + index,
    );
    const prediction = fitValidationFold(
      dataset,
      spec,
      trainIndexes,
      testIndexes,
    );
    const baseline = seasonalBaseline(
      outcome,
      trainIndexes,
      testIndexes,
      dataset.periodsPerYear,
    );
    const modelWape = wape(prediction.actual, prediction.predicted);
    const baselineWape = wape(prediction.actual, baseline);
    return {
      ...prediction,
      id: `fold-${foldIndex + 1}`,
      label: `Fold ${foldIndex + 1}`,
      dates: testIndexes.map((index) =>
        String(dataset.rows[index][dataset.dateColumn]),
      ),
      modelWape,
      skill: 1 - modelWape / Math.max(baselineWape, 1e-6),
      r2: outOfSampleR2(prediction.actual, prediction.predicted),
      nrmse:
        rmse(prediction.actual, prediction.predicted) /
        Math.max(std(prediction.actual), 1),
      coverage:
        prediction.actual.filter(
          (value, index) =>
            value >= prediction.lower[index] &&
            value <= prediction.upper[index],
        ).length / Math.max(prediction.actual.length, 1),
    };
  });
  const averageSkill = mean(foldResults.map((fold) => fold.skill));
  const averageWape = mean(foldResults.map((fold) => fold.modelWape));
  const averageR2 = mean(foldResults.map((fold) => fold.r2));
  const averageCoverage = mean(foldResults.map((fold) => fold.coverage));
  const foldWapes = foldResults.map((fold) => fold.modelWape);
  const stabilityCv = std(foldWapes) / Math.max(mean(foldWapes), 1e-6);
  const accuracyScore = clampScore(
    62 + averageSkill * 42 - Math.max(0, averageWape - 0.2) * 80,
  );
  const coverageScore = clampScore(
    100 - Math.abs(averageCoverage - 0.95) * 180,
  );
  const stabilityScore = clampScore(100 - stabilityCv * 110);

  const intensity = dataset.rows.map((_, rowIndex) =>
    dataset.mediaColumns.reduce(
      (total, column) =>
        total + Math.max(0, toNumber(dataset.rows[rowIndex][column])),
      0,
    ),
  );
  const minimumRegimeWindow = dataset.modelCadence === "monthly" ? 3 : 8;
  const maximumRegimeWindow = dataset.modelCadence === "monthly" ? 6 : 13;
  const regimeWindow = Math.max(
    minimumRegimeWindow,
    Math.min(maximumRegimeWindow, horizon),
  );
  const windowMeans = Array.from(
    { length: Math.max(rowCount - regimeWindow + 1, 1) },
    (_, start) => ({
      start,
      average: mean(intensity.slice(start, start + regimeWindow)),
    }),
  );
  const lowWindow = windowMeans.reduce((lowest, current) =>
    current.average < lowest.average ? current : lowest,
  );
  const highWindow = windowMeans.reduce((highest, current) =>
    current.average > highest.average ? current : highest,
  );
  const lowIndexes = Array.from(
    { length: regimeWindow },
    (_, index) => lowWindow.start + index,
  );
  const highIndexes = Array.from(
    { length: regimeWindow },
    (_, index) => highWindow.start + index,
  );
  const regimeScores = [
    { name: "Low spend", indexes: lowIndexes },
    { name: "High spend", indexes: highIndexes },
  ].map(({ name, indexes }) => {
    const heldOut = new Set(indexes);
    const purged = new Set<number>();
    indexes.forEach((index) => {
      for (let offset = -2; offset <= 2; offset += 1) {
        purged.add(index + offset);
      }
    });
    const trainIndexes = Array.from({ length: rowCount }, (_, index) => index)
      .filter((index) => !heldOut.has(index) && !purged.has(index));
    if (
      indexes.length < minimumRegimeWindow ||
      trainIndexes.length < rowCount * 0.45
    ) {
      return {
        name,
        dates: [] as string[],
        actual: [] as number[],
        predicted: [] as number[],
        score: 0,
        wape: 0,
        available: false,
      };
    }
    const prediction = fitValidationFold(
      dataset,
      spec,
      trainIndexes,
      indexes,
    );
    const regimeWape = wape(prediction.actual, prediction.predicted);
    return {
      name,
      dates: indexes.map((index) =>
        String(dataset.rows[index][dataset.dateColumn]),
      ),
      actual: prediction.actual,
      predicted: prediction.predicted,
      score: clampScore(100 - regimeWape * 210),
      wape: regimeWape,
      available: true,
    };
  });
  const availableRegimes = regimeScores.filter((regime) => regime.available);
  const regimeScore = availableRegimes.length
    ? mean(availableRegimes.map((regime) => regime.score))
    : 0;

  const tests: ValidationTest[] = [
    metricTest(
      "rolling-oos",
      "Three-fold rolling performance",
      accuracyScore,
      `${(averageWape * 100).toFixed(1)}% WAPE`,
      `${averageSkill >= 0 ? "Beats" : "Trails"} the seasonal benchmark by ${Math.abs(averageSkill * 100).toFixed(0)}%; mean out-of-sample R² is ${averageR2.toFixed(2)}.`,
      "critical",
    ),
    metricTest(
      "predictive-coverage",
      "Predictive interval coverage",
      coverageScore,
      `${(averageCoverage * 100).toFixed(0)}% covered`,
      "Checks whether unseen outcomes land inside nominal 95% predictive intervals.",
      "high",
    ),
    metricTest(
      "fold-stability",
      "Fold stability",
      stabilityScore,
      `${(stabilityCv * 100).toFixed(0)}% variation`,
      "Large changes between time folds indicate sensitivity to the selected history.",
      "high",
    ),
    availableRegimes.length
      ? metricTest(
          "spend-regimes",
          "High and low spend regimes",
          regimeScore,
          availableRegimes
            .map(
              (regime) =>
                `${regime.name} ${(regime.wape * 100).toFixed(0)}%`,
            )
            .join(" · "),
          `Refits the model without each observed spend regime and tests those held-out ${dataset.periodUnit}s with an adstock purge buffer.`,
          "critical",
        )
      : {
          id: "spend-regimes",
          name: "High and low spend regimes",
          score: 0,
          status: "incomplete",
          metric: "Insufficient support",
          detail:
            "The history does not contain enough distinct regime observations for a defensible holdout.",
          importance: "critical",
        },
  ];
  const score = weightedScore([
    { score: accuracyScore, weight: 45 },
    { score: coverageScore, weight: 20 },
    { score: regimeScore, weight: 25 },
    { score: stabilityScore, weight: 10 },
  ]);
  const status =
    availableRegimes.length < 2 ? "incomplete" : statusForScore(score);
  return {
    id: "generalization",
    score,
    status,
    summary:
      status === "pass"
        ? "The model generalizes across time and observed spend regimes."
        : status === "incomplete"
          ? "Temporal performance is available, but regime evidence is incomplete."
          : "Out-of-sample behavior needs review before relying on extrapolation.",
    tests,
    evidence: {
      kind: "generalization",
      folds: foldResults.map((fold) => ({
        id: fold.id,
        label: fold.label,
        dates: fold.dates,
        actual: fold.actual,
        predicted: fold.predicted,
        lower: fold.lower,
        upper: fold.upper,
        wape: fold.modelWape,
        coverage: fold.coverage,
      })),
      regimes: regimeScores.map((regime) => ({
        name: regime.name,
        dates: regime.dates,
        actual: regime.actual,
        predicted: regime.predicted,
        wape: regime.wape,
        available: regime.available,
      })),
    },
  };
}
