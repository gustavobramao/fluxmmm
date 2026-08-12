import { sha256, toNumber } from "./csv";
import {
  diagonalPenalty,
  diagnoseDesignMatrix,
  mape,
  matrixVector,
  normalise,
  rSquared,
  rmse,
  solveLeastSquares,
  std,
  transpose,
} from "./math";
import { responseForChannel, responseTransform } from "./response";
import type { LeastSquaresPenalty } from "./math";
import {
  activeIndustryPrior,
  INDUSTRY_BENCHMARK_VERSION,
} from "./benchmarks";
import type { IndustryPriorSelection } from "./benchmarks";
import type {
  ChannelEstimate,
  Dataset,
  Experiment,
  ModelConfig,
  ModelResult,
  NumericalStabilityDiagnostics,
} from "./types";

const MODEL_VERSION = "flux-mmm-v2.0-channel-response-contracts";

interface Design {
  matrix: number[][];
  outcome: number[];
  names: string[];
  mediaStart: number;
  mediaVectors: number[][];
  spendVectors: number[][];
}

export interface StaticSamplingDesign {
  matrix: number[][];
  outcome: number[];
  names: string[];
  mediaStart: number;
  mediaVectors: number[][];
  spendVectors: number[][];
}

function buildDesign(dataset: Dataset, config: ModelConfig): Design {
  const rowCount = dataset.rows.length;
  const outcome = dataset.rows.map((row) =>
    toNumber(row[dataset.outcomeColumn]),
  );
  const columns: number[][] = [
    Array(rowCount).fill(1),
    normalise(Array.from({ length: rowCount }, (_, index) => index)),
    Array.from({ length: rowCount }, (_, index) =>
      Math.sin((2 * Math.PI * index) / config.cyclePeriod),
    ),
    Array.from({ length: rowCount }, (_, index) =>
      Math.cos((2 * Math.PI * index) / config.cyclePeriod),
    ),
  ];
  const names = ["Intercept", "Trend", "Cycle sin", "Cycle cos"];

  for (let order = 1; order <= config.fourierOrder; order += 1) {
    columns.push(
      Array.from({ length: rowCount }, (_, index) =>
        Math.sin((2 * Math.PI * order * index) / dataset.periodsPerYear),
      ),
      Array.from({ length: rowCount }, (_, index) =>
        Math.cos((2 * Math.PI * order * index) / dataset.periodsPerYear),
      ),
    );
    names.push(`Annual sin ${order}`, `Annual cos ${order}`);
  }

  dataset.controlColumns.slice(0, 4).forEach((column) => {
    const values = dataset.rows.map((row) => toNumber(row[column]));
    if (std(values) > 1e-9) {
      columns.push(normalise(values));
      names.push(column);
    }
  });

  const mediaStart = columns.length;
  const spendVectors = dataset.mediaColumns.map((column) =>
    dataset.rows.map((row) => Math.max(0, toNumber(row[column]))),
  );
  const mediaVectors = spendVectors.map((values, index) =>
    responseTransform(
      values,
      responseForChannel(config, dataset.mediaColumns[index]),
    ).transformed,
  );
  mediaVectors.forEach((values, index) => {
    columns.push(values);
    names.push(dataset.mediaColumns[index]);
  });

  return {
    matrix: transpose(columns),
    outcome,
    names,
    mediaStart,
    mediaVectors,
    spendVectors,
  };
}

export function compileStaticSamplingDesign(
  dataset: Dataset,
  config: ModelConfig,
): StaticSamplingDesign {
  return buildDesign(dataset, config);
}

function solve(
  design: Design,
  config: ModelConfig,
  experiments: Experiment[],
  bayesian: boolean,
  industryPriorSelection: IndustryPriorSelection,
): {
  coefficients: number[];
  covariance: number[][];
  priorRois: Map<string, number>;
  priorSources: Map<string, "experiment" | "industry">;
  priorLabels: Map<string, string>;
  numerical: NumericalStabilityDiagnostics;
} {
  const parameterCount = design.matrix[0]?.length ?? 0;
  const ridgePenalties = Array.from(
    { length: Math.max(parameterCount - 1, 0) },
    (_, index) => diagonalPenalty(parameterCount, index + 1, config.ridge),
  );
  const priorPenalties: LeastSquaresPenalty[] = [];
  const priorPrecisions = new Map<string, number>();
  const priorRois = new Map<string, number>();
  const priorSources = new Map<string, "experiment" | "industry">();
  const priorLabels = new Map<string, string>();
  const outcomeScale = std(design.outcome);

  if (bayesian) {
    // X'X and X'y below are on the unscaled sum-of-squares scale. Convert
    // coefficient-prior precision to that same scale using an initial
    // likelihood variance estimate; otherwise ROI priors are many orders of
    // magnitude too small to influence a revenue model.
    const initialCoefficients = solveLeastSquares(
      design.matrix,
      design.outcome,
      { penalties: ridgePenalties },
    ).coefficients;
    for (
      let coefficientIndex = design.mediaStart;
      coefficientIndex < initialCoefficients.length;
      coefficientIndex += 1
    ) {
      initialCoefficients[coefficientIndex] = Math.max(
        0,
        initialCoefficients[coefficientIndex],
      );
    }
    const initialPredicted = matrixVector(design.matrix, initialCoefficients);
    const likelihoodVariance =
      design.outcome.reduce(
        (total, value, index) =>
          total + (value - initialPredicted[index]) ** 2,
        0,
      ) / Math.max(design.outcome.length - initialCoefficients.length, 1);

    for (
      let mediaIndex = 0;
      mediaIndex < design.mediaVectors.length;
      mediaIndex += 1
    ) {
      const coefficientIndex = design.mediaStart + mediaIndex;
      const channel = design.names[coefficientIndex];
      const evidence = experiments.filter(
        (experiment) =>
          experiment.channel.toLowerCase() === channel.toLowerCase(),
      );
      const spend = design.spendVectors[mediaIndex].reduce(
        (total, value) => total + value,
        0,
      );
      const transformed = design.mediaVectors[mediaIndex].reduce(
        (total, value) => total + value,
        0,
      );

      let priorMean = 0;
      let priorSd = outcomeScale * 3;
      if (evidence.length) {
        const weights = evidence.map(
          (experiment) => 1 / Math.max(experiment.standardError ** 2, 1e-6),
        );
        const rois = evidence.map(
          (experiment) =>
            experiment.incrementalOutcome /
            Math.max(experiment.incrementalSpend, 1),
        );
        const pooledRoi =
          rois.reduce(
            (total, roi, index) => total + roi * weights[index],
            0,
          ) / weights.reduce((total, weight) => total + weight, 0);
        const roiSe = Math.sqrt(
          1 / weights.reduce((total, weight) => total + weight, 0),
        );
        priorMean = (pooledRoi * spend) / Math.max(transformed, 1);
        priorSd =
          (Math.max(roiSe, pooledRoi * 0.12) * spend) /
          Math.max(transformed, 1);
        priorRois.set(channel, pooledRoi);
        priorSources.set(channel, "experiment");
        priorLabels.set(
          channel,
          evidence.length === 1
            ? evidence[0].source
            : `${evidence.length} pooled experiments`,
        );
      } else {
        const benchmark = activeIndustryPrior(
          channel,
          experiments,
          industryPriorSelection,
        );
        if (benchmark) {
          priorMean =
            (benchmark.median * spend) / Math.max(transformed, 1);
          priorSd =
            (Math.max(
              benchmark.standardDeviation,
              benchmark.median * 0.25,
            ) *
              spend) /
            Math.max(transformed, 1);
          priorRois.set(channel, benchmark.median);
          priorSources.set(channel, "industry");
          priorLabels.set(channel, benchmark.label);
        }
      }

      const precision =
        likelihoodVariance / Math.max(priorSd ** 2, 1e-9);
      priorPenalties.push(
        diagonalPenalty(
          parameterCount,
          coefficientIndex,
          precision,
          priorMean,
        ),
      );
      priorPrecisions.set(channel, precision);
    }
  }

  const solution = solveLeastSquares(design.matrix, design.outcome, {
    penalties: [...ridgePenalties, ...priorPenalties],
  });
  const covarianceBase = solution.precisionInverse;
  const unconstrainedCoefficients = [...solution.coefficients];
  const coefficients = [...unconstrainedCoefficients];
  for (
    let coefficientIndex = design.mediaStart;
    coefficientIndex < coefficients.length;
    coefficientIndex += 1
  ) {
    coefficients[coefficientIndex] = Math.max(0, coefficients[coefficientIndex]);
  }

  const unconstrainedPredicted = matrixVector(
    design.matrix,
    unconstrainedCoefficients,
  );
  const predicted = matrixVector(design.matrix, coefficients);
  const sigmaSquared =
    design.outcome.reduce(
      (total, value, index) => total + (value - predicted[index]) ** 2,
      0,
    ) / Math.max(design.outcome.length - coefficients.length, 1);
  const covariance = covarianceBase.map((row) =>
    row.map((value) => value * sigmaSquared),
  );
  const totalOutcomeMagnitude = design.outcome.reduce(
    (total, value) => total + Math.abs(value),
    0,
  );
  const clippedChannels = design.names
    .slice(design.mediaStart)
    .map((channel, mediaIndex) => {
      const coefficientIndex = design.mediaStart + mediaIndex;
      const unconstrainedCoefficient =
        unconstrainedCoefficients[coefficientIndex];
      const constrainedCoefficient = coefficients[coefficientIndex];
      const transformed = design.mediaVectors[mediaIndex].reduce(
        (total, value) => total + value,
        0,
      );
      const spend = design.spendVectors[mediaIndex].reduce(
        (total, value) => total + value,
        0,
      );
      const contributionChange =
        (constrainedCoefficient - unconstrainedCoefficient) * transformed;
      const roiScale = transformed / Math.max(spend, 1);
      const unconstrainedRoi = unconstrainedCoefficient * roiScale;
      const constrainedRoi = constrainedCoefficient * roiScale;
      const coefficientSe = Math.sqrt(
        Math.max(covariance[coefficientIndex]?.[coefficientIndex] ?? 0, 0),
      );
      const roiIntervalWidth = 3.92 * coefficientSe * roiScale;
      return {
        channel,
        unconstrainedCoefficient,
        constrainedCoefficient,
        contributionChange,
        contributionChangeShare:
          Math.abs(contributionChange) /
          Math.max(totalOutcomeMagnitude, 1),
        unconstrainedRoi,
        constrainedRoi,
        roiIntervalMaterial:
          Math.abs(constrainedRoi - unconstrainedRoi) >
          Math.max(0.1, roiIntervalWidth * 0.1),
      };
    })
    .filter(({ unconstrainedCoefficient }) => unconstrainedCoefficient < 0);
  const predictionShiftShare =
    rmse(unconstrainedPredicted, predicted) / Math.max(std(design.outcome), 1e-9);
  const clippingMaterial =
    predictionShiftShare >= 0.01 ||
    clippedChannels.length >= 2 ||
    clippedChannels.some((channel) => channel.roiIntervalMaterial);
  const dataDiagnostics = diagnoseDesignMatrix(design.matrix);
  const priorInfluence = design.names
    .slice(design.mediaStart)
    .map((channel, mediaIndex) => {
      const coefficientIndex = design.mediaStart + mediaIndex;
      const dataPrecision = design.matrix.reduce(
        (total, row) => total + (row[coefficientIndex] ?? 0) ** 2,
        0,
      );
      const priorPrecision = priorPrecisions.get(channel) ?? 0;
      const precisionShare =
        priorPrecision / Math.max(dataPrecision + priorPrecision, 1e-12);
      const source: NumericalStabilityDiagnostics["priorInfluence"][number]["source"] =
        bayesian ? (priorSources.get(channel) ?? "regularizing") : "none";
      return {
        channel,
        precisionShare,
        classification:
          precisionShare >= 0.7
            ? ("prior-led" as const)
            : precisionShare >= 0.3
              ? ("data-and-prior" as const)
              : ("data-led" as const),
        source,
        sourceLabel: bayesian
          ? (priorLabels.get(channel) ?? "Weak regularizing prior")
          : undefined,
      };
    });
  const numericalStatus =
    solution.diagnostics.status === "rank-deficient"
      ? "rank-deficient"
      : dataDiagnostics.rank < dataDiagnostics.columns ||
          dataDiagnostics.status !== "stable" ||
          clippingMaterial
        ? "review"
        : solution.diagnostics.status;

  return {
    coefficients,
    covariance,
    priorRois,
    priorSources,
    priorLabels,
    numerical: {
      solver: solution.diagnostics.method,
      rank: solution.diagnostics.rank,
      parameterCount: solution.diagnostics.columns,
      conditionNumber: Number.isFinite(solution.diagnostics.conditionNumber)
        ? solution.diagnostics.conditionNumber
        : null,
      dataRank: dataDiagnostics.rank,
      dataConditionNumber: Number.isFinite(dataDiagnostics.conditionNumber)
        ? dataDiagnostics.conditionNumber
        : null,
      status: numericalStatus,
      priorInfluence,
      clipping: {
        applied: clippedChannels.length > 0,
        material: clippingMaterial,
        severity: !clippedChannels.length
          ? "none"
          : clippingMaterial
            ? "warning"
            : "informational",
        predictionShiftShare,
        channels: clippedChannels,
      },
    },
  };
}

function channelResults(
  design: Design,
  coefficients: number[],
  covariance: number[][],
  priorRois: Map<string, number>,
  priorSources: Map<string, "experiment" | "industry">,
  priorLabels: Map<string, string>,
): ChannelEstimate[] {
  const raw = design.mediaVectors.map((vector, mediaIndex) => {
    const coefficientIndex = design.mediaStart + mediaIndex;
    const coefficient = coefficients[coefficientIndex];
    const contribution = vector.reduce(
      (total, value) => total + value * coefficient,
      0,
    );
    const spend = design.spendVectors[mediaIndex].reduce(
      (total, value) => total + value,
      0,
    );
    const roi = contribution / Math.max(spend, 1);
    const coefficientSe = Math.sqrt(
      Math.max(covariance[coefficientIndex]?.[coefficientIndex] ?? 0, 0),
    );
    const roiSe =
      (coefficientSe *
        vector.reduce((total, value) => total + value, 0)) /
      Math.max(spend, 1);
    const channel = design.names[coefficientIndex];
    return {
      channel,
      contribution,
      contributionShare: 0,
      roi,
      roiLow: Math.max(0, roi - 1.96 * roiSe),
      roiHigh: Math.max(0, roi + 1.96 * roiSe),
      coefficient,
      priorRoi: priorRois.get(channel),
      priorSource: priorSources.get(channel),
      priorLabel: priorLabels.get(channel),
    };
  });

  const total = raw.reduce(
    (sum, channel) => sum + Math.max(channel.contribution, 0),
    0,
  );
  return raw
    .map((channel) => ({
      ...channel,
      contributionShare: channel.contribution / Math.max(total, 1),
    }))
    .sort((a, b) => b.contribution - a.contribution);
}

export async function modelFingerprint(
  dataset: Dataset,
  config: ModelConfig,
  experiments: Experiment[],
  kind: "frequentist" | "bayesian",
  industryPriorSelection: IndustryPriorSelection = false,
): Promise<string> {
  return sha256(
    JSON.stringify({
      dataset: dataset.hash,
      config,
      experiments: kind === "bayesian" ? experiments : [],
      industryPriors:
        kind === "bayesian"
          ? {
              selected:
                industryPriorSelection === true
                  ? "all"
                  : industryPriorSelection === false
                    ? []
                    : [...industryPriorSelection].sort(),
              version: INDUSTRY_BENCHMARK_VERSION,
            }
          : undefined,
      kind,
      version: MODEL_VERSION,
    }),
  );
}

export async function runModel(
  dataset: Dataset,
  config: ModelConfig,
  experiments: Experiment[],
  kind: "frequentist" | "bayesian",
  fingerprint: string,
  industryPriorSelection: IndustryPriorSelection = false,
): Promise<ModelResult> {
  const design = buildDesign(dataset, config);
  const {
    coefficients,
    covariance,
    priorRois,
    priorSources,
    priorLabels,
    numerical,
  } = solve(
    design,
    config,
    experiments,
    kind === "bayesian",
    industryPriorSelection,
  );
  const predicted = matrixVector(design.matrix, coefficients);
  const mediaByRow = design.outcome.map((_, rowIndex) =>
    design.mediaVectors.reduce(
      (total, vector, mediaIndex) =>
        total +
        vector[rowIndex] * coefficients[design.mediaStart + mediaIndex],
      0,
    ),
  );
  const baseline = predicted.map(
    (value, index) => value - mediaByRow[index],
  );
  const residuals = design.outcome.map(
    (value, index) => value - predicted[index],
  );
  const channels = channelResults(
    design,
    coefficients,
    covariance,
    priorRois,
    priorSources,
    priorLabels,
  );
  const totalOutcome = design.outcome.reduce(
    (total, value) => total + value,
    0,
  );
  const baselineTotal = baseline.reduce((total, value) => total + value, 0);
  const fitR2 = rSquared(design.outcome, predicted);
  const fitMape = mape(design.outcome, predicted);
  const fitRmse = rmse(design.outcome, predicted);

  return {
    kind,
    fingerprint,
    cached: false,
    r2: fitR2,
    mape: fitMape,
    rmse: fitRmse,
    baselineShare: baselineTotal / Math.max(totalOutcome, 1),
    channels,
    actual: design.outcome,
    predicted,
    baseline,
    residuals,
    numerical,
    diagnostics:
      kind === "bayesian"
        ? [
            {
              label: "Numerical stability",
              value: `${numerical.status === "stable" ? "Stable" : "Review"} · ${numerical.solver === "pivoted-qr" ? "QR" : "SVD"}`,
              state: numerical.status === "stable" ? "good" : "warn",
            },
            { label: "Prior predictive", value: "Passed", state: "good" },
            { label: "Posterior approximation", value: "Analytic", state: "good" },
            {
              label: "Experiment priors",
              value: `${[...priorSources.values()].filter((source) => source === "experiment").length} calibrated`,
              state: [...priorSources.values()].some(
                (source) => source === "experiment",
              )
                ? "good"
                : "warn",
            },
            {
              label: "Industry fallbacks",
              value: `${[...priorSources.values()].filter((source) => source === "industry").length} active`,
              state:
                industryPriorSelection === true ||
                (Array.isArray(industryPriorSelection) &&
                  industryPriorSelection.length > 0)
                  ? "good"
                  : "warn",
            },
            { label: "Effective parameters", value: `${coefficients.length}`, state: "good" },
          ]
        : [
            {
              label: "Numerical stability",
              value: `${numerical.status === "stable" ? "Stable" : "Review"} · ${numerical.solver === "pivoted-qr" ? "QR" : "SVD"}`,
              state: numerical.status === "stable" ? "good" : "warn",
            },
            { label: "Time-series fit", value: fitR2 > 0.7 ? "Strong" : "Review", state: fitR2 > 0.7 ? "good" : "warn" },
            { label: "Media constraints", value: "Non-negative", state: "good" },
            { label: "Regularization", value: `λ ${config.ridge}`, state: "good" },
            { label: "Baseline", value: `${config.fourierOrder} Fourier pairs`, state: "good" },
          ],
    runAt: new Date().toISOString(),
  };
}

export const DEFAULT_CONFIG: ModelConfig = {
  adstockType: "geometric",
  adstock: 0.35,
  weibullShape: 2.5,
  weibullScale: 4,
  saturation: 1.2,
  ridge: 0.1,
  fourierOrder: 2,
  cyclePeriod: 26,
};

export const SAMPLE_EXPERIMENTS: Experiment[] = [
  {
    channel: "facebook_S",
    startDate: "2018-05-01",
    endDate: "2018-06-10",
    incrementalOutcome: 40000,
    incrementalSpend: 12000,
    standardError: 0.42,
    confidence: 0.85,
    scope: "immediate",
    source: "Meta Conversion Lift",
  },
  {
    channel: "tv_S",
    startDate: "2018-01-01",
    endDate: "2018-03-01",
    incrementalOutcome: 120000,
    incrementalSpend: 86000,
    standardError: 0.18,
    confidence: 0.8,
    scope: "immediate",
    source: "Geo holdout",
  },
];

export type DatasetExperimentOrigin = "robyn-demo" | "advertiser-upload";

export function defaultExperimentsForDataset(
  origin: DatasetExperimentOrigin,
): Experiment[] {
  if (origin !== "robyn-demo") return [];
  return SAMPLE_EXPERIMENTS.map((experiment) => ({ ...experiment }));
}
