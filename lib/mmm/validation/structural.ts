import { toNumber } from "../csv";
import { mean, normalise, std } from "../math";
import { responseForChannel, responseTransform } from "../response";
import type { Dataset, ModelResult } from "../types";
import type { ValidationModelSpec } from "./adapter";
import {
  clampScore,
  correlation,
  excessKurtosis,
  skewness,
  varianceInflationFactors,
  weightedScore,
} from "./math";
import type {
  ValidationLayerResult,
  ValidationStatus,
  ValidationTest,
} from "./types";

function statusForScore(score: number): ValidationStatus {
  return score >= 75 ? "pass" : score >= 55 ? "review" : "fail";
}

function test(
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

function varianceRatio(
  residuals: number[],
  fitted: number[],
): number {
  const order = fitted
    .map((value, index) => ({ value, index }))
    .sort((a, b) => a.value - b.value);
  const groupSize = Math.max(Math.floor(order.length / 4), 1);
  const low = order
    .slice(0, groupSize)
    .map(({ index }) => residuals[index]);
  const high = order
    .slice(-groupSize)
    .map(({ index }) => residuals[index]);
  const lowVariance = std(low) ** 2;
  const highVariance = std(high) ** 2;
  return Math.max(lowVariance, highVariance) /
    Math.max(Math.min(lowVariance, highVariance), 1e-9);
}

function residualAutocorrelations(
  residuals: number[],
  dataset: Dataset,
): { lag: number; value: number }[] {
  const lags = (dataset.modelCadence === "monthly"
    ? [1, 2, 3, 6, 12]
    : [1, 2, 4, 8, 13, 26, 52]
  ).filter(
    (lag) => lag < residuals.length / 2,
  );
  return lags.map((lag) => ({
    lag,
    value: correlation(residuals.slice(lag), residuals.slice(0, -lag)),
  }));
}

function inverseNormal(probability: number): number {
  const a = [
    -39.69683028665376,
    220.9460984245205,
    -275.9285104469687,
    138.357751867269,
    -30.66479806614716,
    2.506628277459239,
  ];
  const b = [
    -54.47609879822406,
    161.5858368580409,
    -155.6989798598866,
    66.80131188771972,
    -13.28068155288572,
  ];
  const c = [
    -0.007784894002430293,
    -0.3223964580411365,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
  ];
  const d = [
    0.007784695709041462,
    0.3224671290700398,
    2.445134137142996,
    3.754408661907416,
  ];
  const low = 0.02425;
  const high = 1 - low;
  if (probability < low) {
    const q = Math.sqrt(-2 * Math.log(probability));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) *
        q +
        c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (probability > high) {
    const q = Math.sqrt(-2 * Math.log(1 - probability));
    return -(
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) *
        q +
        c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  const q = probability - 0.5;
  const r = q * q;
  return (
    (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) *
      r +
      a[5]) *
    q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) *
      r +
      1)
  );
}

function residualQuantiles(
  residuals: number[],
): { expected: number; observed: number }[] {
  const scale = Math.max(std(residuals), 1e-9);
  const centered = residuals
    .map((value) => (value - mean(residuals)) / scale)
    .sort((a, b) => a - b);
  const step = Math.max(1, Math.floor(centered.length / 80));
  return centered
    .map((observed, index) => ({
      expected: inverseNormal((index + 0.5) / centered.length),
      observed,
    }))
    .filter((_, index) => index % step === 0);
}

function transformedColumns(
  dataset: Dataset,
  spec: ValidationModelSpec,
): { label: string; values: number[] }[] {
  const media = dataset.mediaColumns.map((column) => {
    const values = dataset.rows.map((row) =>
      Math.max(0, toNumber(row[column])),
    );
    return {
      label: column,
      values: normalise(
        responseTransform(
          values,
          responseForChannel(spec.config, column),
        ).transformed,
      ),
    };
  });
  const controls = dataset.controlColumns.slice(0, 4).map((column) => ({
    label: column,
    values: normalise(dataset.rows.map((row) => toNumber(row[column]))),
  }));
  return [...media, ...controls].filter(
    (column) => std(column.values) > 1e-8,
  );
}

function pathStability(dataset: Dataset, result: ModelResult): {
  score: number;
  metric: string;
  detail: string;
} {
  const paths = result.advanced?.coefficientPaths ?? [];
  if (!paths.length || !result.advanced?.timeVarying) {
    return {
      score: 100,
      metric: "Static coefficients",
      detail: "No dynamic coefficient path was selected.",
    };
  }
  const roughness = paths.map((path) => {
    const scale = Math.max(Math.abs(mean(path.values)), 1e-9);
    const secondDifferences = path.values
      .slice(2)
      .map(
        (value, index) =>
          Math.abs(
            value - 2 * path.values[index + 1] + path.values[index],
          ) / scale,
      );
    return mean(secondDifferences);
  });
  const averageRoughness = mean(roughness);
  return {
    score: clampScore(100 - averageRoughness * 420),
    metric: `${(averageRoughness * 100).toFixed(1)}% curvature`,
    detail:
      `Measures ${dataset.periodUnit}-to-${dataset.periodUnit} curvature relative to each channel’s average coefficient; excessive movement suggests over-flexibility.`,
  };
}

export function runStructuralValidation(
  dataset: Dataset,
  spec: ValidationModelSpec,
  result: ModelResult,
): ValidationLayerResult {
  const isLogNormal =
    spec.kind === "advanced" &&
    spec.advancedConfig.likelihoodDistribution === "log-normal";
  const isStudentT =
    spec.kind === "advanced" &&
    spec.advancedConfig.likelihoodDistribution === "student-t";
  const residuals = isLogNormal
    ? result.actual.map(
        (value, index) =>
          Math.log(Math.max(value, 1e-6)) -
          Math.log(Math.max(result.predicted[index], 1e-6)),
      )
    : result.residuals;
  const fitted = isLogNormal
    ? result.predicted.map((value) => Math.log(Math.max(value, 1e-6)))
    : result.predicted;
  const autocorrelation = residualAutocorrelations(residuals, dataset);
  const maxAcf = Math.max(
    ...autocorrelation.map((point) => Math.abs(point.value)),
    0,
  );
  const independenceScore = clampScore(100 - maxAcf * 190);
  const absoluteResiduals = residuals.map((value) => Math.abs(value));
  const varianceCorrelation = Math.abs(
    correlation(absoluteResiduals, fitted),
  );
  const spreadRatio = varianceRatio(residuals, fitted);
  const varianceScore = clampScore(
    100 -
      varianceCorrelation * 120 -
      Math.max(0, Math.log(spreadRatio)) * 25,
  );
  const skew = skewness(residuals);
  const kurtosis = excessKurtosis(residuals);
  const distributionScore = isStudentT
    ? clampScore(100 - Math.abs(skew) * 28)
    : clampScore(
        100 - Math.abs(skew) * 27 - Math.abs(kurtosis) * 10,
  );
  const columns = transformedColumns(dataset, spec);
  const vifValues = varianceInflationFactors(
    columns.map((column) => column.values),
  );
  const vifs = columns.map((column, index) => ({
    label: column.label,
    value: vifValues[index] ?? 1,
  }));
  const maxVif = Math.max(...vifValues, 1);
  const identificationScore = clampScore(
    100 - Math.max(0, maxVif - 3) * 11,
  );
  const fittedCorrelation = Math.abs(correlation(residuals, fitted));
  const timeCorrelation = Math.abs(
    correlation(
      residuals,
      Array.from({ length: residuals.length }, (_, index) => index),
    ),
  );
  const functionalScore = clampScore(
    100 - Math.max(fittedCorrelation, timeCorrelation) * 180,
  );
  const maximumStandardizedResidual =
    Math.max(...absoluteResiduals) / Math.max(std(residuals), 1e-9);
  const influenceScore = clampScore(
    100 - Math.max(0, maximumStandardizedResidual - 3) * 18,
  );
  const path = pathStability(dataset, result);
  const stabilityScore = weightedScore([
    { score: influenceScore, weight: 0.55 },
    { score: path.score, weight: 0.45 },
  ]);
  const clipping = result.numerical?.clipping;
  const materiallyAffectedIntervals =
    clipping?.channels.filter((channel) => channel.roiIntervalMaterial).length ?? 0;
  const severeClipping = Boolean(
    clipping?.applied &&
      (clipping.predictionShiftShare >= 0.05 ||
        clipping.channels.length >= 3 ||
        materiallyAffectedIntervals >= 2),
  );
  const clippingScore = !clipping?.applied
    ? 100
    : !clipping.material
      ? 82
      : severeClipping
        ? 42
        : 58;

  const tests = [
    test(
      "residual-independence",
      "Residual independence",
      independenceScore,
      `max |ACF| ${maxAcf.toFixed(2)}`,
      "Checks short- and seasonal-lag residual correlation. Remaining temporal structure can be mistaken for media response.",
      "critical",
    ),
    test(
      "variance-structure",
      isLogNormal ? "Log-scale variance structure" : "Homoscedasticity",
      varianceScore,
      `${spreadRatio.toFixed(1)}× spread ratio`,
      isLogNormal
        ? "Tests whether residual variance is stable after moving to the selected log scale."
        : "Compares residual spread across low and high fitted outcomes and checks residual magnitude against fitted level.",
      "high",
    ),
    test(
      "likelihood-shape",
      isStudentT ? "Heavy-tail symmetry" : "Likelihood residual shape",
      distributionScore,
      `skew ${skew.toFixed(2)} · excess kurtosis ${kurtosis.toFixed(2)}`,
      isStudentT
        ? "Student-t permits heavy tails, so symmetry matters more than Gaussian kurtosis."
        : "Assesses whether the selected Gaussian-scale residual shape is plausible; this supports intervals more than point identification.",
      "supporting",
    ),
    test(
      "multicollinearity",
      "Channel identification",
      identificationScore,
      `max VIF ${maxVif.toFixed(1)}`,
      "VIF is computed on transformed media and controls. High values signal weak separation of channel effects even when aggregate fit is strong.",
      "critical",
    ),
    test(
      "functional-form",
      "Functional form",
      functionalScore,
      `max residual correlation ${Math.max(fittedCorrelation, timeCorrelation).toFixed(2)}`,
      "Checks whether residuals still move with fitted level or time, indicating an omitted trend, nonlinearity, or baseline component.",
      "high",
    ),
    test(
      "influence-stability",
      result.advanced?.timeVarying
        ? "Influence and coefficient smoothness"
        : "Influential observations",
      stabilityScore,
      result.advanced?.timeVarying
        ? `${path.metric} · max residual ${maximumStandardizedResidual.toFixed(1)}σ`
        : `max residual ${maximumStandardizedResidual.toFixed(1)}σ`,
      result.advanced?.timeVarying
        ? path.detail
        : "Flags periods capable of disproportionately changing estimates.",
      "supporting",
    ),
    test(
      "non-negative-boundary",
      "Non-negative boundary",
      clippingScore,
      !clipping?.applied
        ? "No media clipping"
        : `${clipping.channels.length} channel${clipping.channels.length === 1 ? "" : "s"} · ${(clipping.predictionShiftShare * 100).toFixed(1)}% prediction shift`,
      !clipping?.applied
        ? "The unconstrained Gaussian mode already respected the non-negative media contract."
        : clipping.material
          ? `${materiallyAffectedIntervals} ROI interval${materiallyAffectedIntervals === 1 ? " was" : "s were"} materially affected. Clipping is retained, but the local Gaussian approximation requires structural review.`
          : "A small boundary correction was applied. The effect is disclosed, while the local Gaussian approximation remains usable.",
      "supporting",
    ),
  ];
  const baseScore = weightedScore([
    { score: identificationScore, weight: 30 },
    { score: independenceScore, weight: 25 },
    {
      score: weightedScore([
        { score: varianceScore, weight: 0.65 },
        { score: distributionScore, weight: 0.35 },
      ]),
      weight: 20,
    },
    { score: functionalScore, weight: 15 },
    { score: stabilityScore, weight: 10 },
  ]);
  const score = clipping?.applied
    ? weightedScore([
        { score: baseScore, weight: 95 },
        { score: clippingScore, weight: 5 },
      ])
    : baseScore;
  const status = statusForScore(score);
  return {
    id: "structure",
    score,
    status,
    summary:
      status === "pass"
        ? "The selected model structure is broadly consistent with its assumptions."
        : status === "review"
          ? "Most assumptions are usable, with material diagnostics to review."
          : "Structural violations weaken attribution and interval reliability.",
    tests,
    evidence: {
      kind: "structure",
      dates: dataset.rows.map((row) =>
        String(row[dataset.dateColumn]),
      ),
      residuals,
      fitted,
      residualScale: std(residuals),
      autocorrelation,
      vifs,
      quantiles: residualQuantiles(residuals),
    },
  };
}
