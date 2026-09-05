import { compileAdvancedSamplingDesign } from "./advanced";
import {
  activeIndustryPrior,
  inferIndustryPrior,
  type IndustryPriorOverrides,
} from "./benchmarks";
import { sha256 } from "./csv";
import { mean, std } from "./math";
import {
  carryoverForResponse,
  hasChannelResponseContracts,
  positiveQuantile,
  responseForChannel,
} from "./response";
import { compileStaticSamplingDesign } from "./models";
import { experimentResponseContrast } from "./experiment-window";
import {
  passesAgenticEligibility,
  type AgenticCandidateRun,
} from "./agentic";
import type {
  AdvancedModelConfig,
  Dataset,
  Experiment,
  ModelConfig,
  ModelResult,
  MediaResponseConfig,
} from "./types";

export const SAMPLING_ENGINE_VERSION =
  "flux-pymc-nuts-v2.2.0-full-response-same-window-experiment-roi";

export type SamplingPreset = "production" | "robust" | "diagnostic" | "custom";

export interface SamplingContract {
  preset: SamplingPreset;
  chains: 4 | 6 | 8;
  tune: 500 | 1000 | 2000 | 3000 | 4000;
  draws: 500 | 1000 | 2000 | 4000;
  targetAccept: 0.8 | 0.9 | 0.95 | 0.99;
  maxTreeDepth: 10 | 12 | 14 | 16;
  seed: number;
}

export const SAMPLING_PRESETS: Record<
  Exclude<SamplingPreset, "custom">,
  SamplingContract
> = {
  production: {
    preset: "production",
    chains: 4,
    tune: 1000,
    draws: 1000,
    targetAccept: 0.9,
    maxTreeDepth: 10,
    seed: 202603,
  },
  robust: {
    preset: "robust",
    chains: 4,
    tune: 2000,
    draws: 2000,
    targetAccept: 0.95,
    maxTreeDepth: 12,
    seed: 202603,
  },
  diagnostic: {
    preset: "diagnostic",
    chains: 6,
    tune: 3000,
    draws: 2000,
    targetAccept: 0.95,
    maxTreeDepth: 14,
    seed: 202603,
  },
};

export const DEFAULT_SAMPLING_CONTRACT = SAMPLING_PRESETS.production;

export interface CompiledSamplingPrior {
  kind: "normal" | "half-normal" | "log-normal" | "truncated-normal";
  mean: number;
  standardDeviation: number;
  initialValue: number;
}

export interface CompiledSamplingCalibration {
  label: string;
  channel: string;
  route: "prior" | "likelihood";
  basis: "experiment-window";
  indexes: number[];
  weights: number[];
  observedRoi: number;
  standardError: number;
  rows: number[];
  spendRows: number[];
  outcomeRows: number[];
  spend: number;
}

export interface CompiledSamplingEvidence {
  mean: number;
  standardDeviation: number;
  source: "experiment" | "industry";
  label: string;
  rows: number[];
}

export interface CompiledSamplingMedia {
  channel: string;
  indexes: number[];
  spend: number;
  roiWeights: number[];
  rawSpend: number[];
  halfSaturation: number;
  response: MediaResponseConfig;
  screeningRoi: number;
  screeningLow: number;
  screeningHigh: number;
  plausibleUpperRoi: number;
  priorEvidence?: CompiledSamplingEvidence;
}

export interface CompiledSamplingResponseContract {
  adstockType: ModelConfig["adstockType"];
  adstockDecay: number;
  weibullShape: number;
  weibullScale: number;
  hillShape: number;
  timeVarying: boolean;
  kernelKnots: number;
  kernelBandwidth: number;
  planningIntensity: boolean;
  planningParameterIndex: number | null;
  planningInitial: number[];
  channelSpecific: boolean;
}

export interface CompiledSamplingModel {
  version: string;
  promotedId: string;
  promotedFingerprint: string;
  promotedFamily: "bayesian" | "advanced";
  validationScore: number | null;
  validationEligible: boolean;
  matrix: number[][];
  outcome: number[];
  target: number[];
  dates: string[];
  parameterNames: string[];
  priors: CompiledSamplingPrior[];
  media: CompiledSamplingMedia[];
  calibrations: CompiledSamplingCalibration[];
  response: CompiledSamplingResponseContract;
  screeningPredicted: number[];
  ridge: number;
  likelihood: "gaussian" | "student-t" | "log-normal";
  studentTDegreesFreedom: number;
}

export interface SamplingDiagnosticGate {
  id:
    | "rhat"
    | "ess"
    | "divergences"
    | "treedepth"
    | "bfmi"
    | "mcse"
    | "validation"
    | "roi-plausibility"
    | "roi-precision"
    | "predictive-coverage";
  label: string;
  value: string;
  threshold: string;
  passed: boolean;
  detail: string;
}

export interface SamplingParameterDiagnostic {
  name: string;
  label: string;
  mean: number;
  standardDeviation: number;
  low: number;
  median: number;
  high: number;
  rhat: number;
  bulkEss: number;
  tailEss: number;
  mcse: number;
  traces: number[][];
  ranks: number[][];
}

export interface SamplingChannelPosterior {
  channel: string;
  mapRoi: number;
  mapLow: number;
  mapHigh: number;
  posteriorMean: number;
  posteriorMedian: number;
  posteriorLow: number;
  posteriorHigh: number;
  difference: number;
  uncertaintyRatio: number;
  materialShift: boolean;
  posteriorSamples?: number[];
  assessment?: "compatible" | "center" | "shape" | "center-shape";
  plausibleUpperRoi?: number;
  implausibleProbability?: number;
  relativeIntervalWidth?: number;
  evidenceSource?: "experiment" | "industry" | "regularizing";
  explanation?: {
    density: {
      scale: "linear" | "log1p";
      grid: number[];
      laplace: number[];
      posterior: number[];
    };
    locationShiftSd: number;
    intervalOverlap: number;
    posteriorSkewness: number;
    nearZeroProbability: number;
    topTradeoff?: {
      label: string;
      kind: "channel" | "baseline";
      correlation: number;
    };
  };
}

/**
 * A compact, aligned posterior draw used by offline decision research.
 * Every channel and response value in one item comes from the same NUTS draw.
 * These draws are intentionally separate from the display-oriented marginal
 * summaries above so downstream research cannot accidentally pair unrelated
 * quantiles.
 */
export interface SamplingDecisionDraw {
  channels: {
    channel: string;
    roi: number;
    contribution: number;
    response: MediaResponseConfig;
  }[];
  kernelBandwidth: number;
}

export interface SamplingResult {
  kind: "mcmc";
  fingerprint: string;
  promotedFingerprint: string;
  promotedId: string;
  cached: boolean;
  engine: string;
  inferenceContract: {
    version: string;
    sharedPosterior: false;
    sharedPriorContract: true;
    mapEngine: "Analytic screening MAP / local Laplace";
    samplingEngine: "PyMC NUTS";
    sharedRoiTransform: true;
    responseUncertainty: "Sampled continuous response parameters";
    planningFactor: "Probabilistic latent basis factor" | "Not included";
  };
  contract: SamplingContract;
  status: "ready" | "review";
  gates: SamplingDiagnosticGate[];
  diagnostics: {
    maxRhat: number;
    minBulkEss: number;
    minTailEss: number;
    divergences: number;
    treeDepthHits: number;
    minBfmi: number;
    maxMcseRatio: number;
    predictiveCoverage?: number;
    maxImplausibleProbability?: number;
    maxRelativeRoiWidth?: number;
  };
  channels: SamplingChannelPosterior[];
  decisionDraws?: SamplingDecisionDraw[];
  predictive: {
    dates: string[];
    actual: number[];
    map: number[];
    median: number[];
    low: number[];
    high: number[];
    meanMedian: number[];
    meanLow: number[];
    meanHigh: number[];
  };
  parameters: SamplingParameterDiagnostic[];
  retryRecommendation?: {
    title: string;
    detail: string;
    contract: SamplingContract;
  };
  artifact: {
    id: string;
    retainedDraws: number;
    stored: boolean;
  };
  runtimeSeconds: number;
  runAt: string;
}

export interface SamplingJobProgress {
  stage: "queued" | "compiling" | "warmup" | "sampling" | "diagnostics" | "complete" | "error";
  completed: number;
  total: number;
  chain: number;
  detail: string;
}

export interface SamplingJobSnapshot {
  id: string;
  status: "queued" | "running" | "complete" | "error";
  progress: SamplingJobProgress;
  result?: SamplingResult;
  error?: string;
}

function matchingExperimentRows(
  dataset: Dataset,
  experiment: Experiment,
): number[] {
  const start = Date.parse(experiment.startDate);
  const end = Date.parse(experiment.endDate);
  const matches = dataset.rows
    .map((row, index) => ({
      index,
      date: Date.parse(String(row[dataset.dateColumn])),
    }))
    .filter(
      ({ date }) =>
        Number.isFinite(date) &&
        (!Number.isFinite(start) || date >= start) &&
        (!Number.isFinite(end) || date <= end),
    )
    .map(({ index }) => index);
  return matches.length
    ? matches
    : Array.from({ length: dataset.rows.length }, (_, index) => index);
}

function pooledExperimentPrior(
  experiments: Experiment[],
): { roi: number; standardError: number } | undefined {
  if (!experiments.length) return undefined;
  const weights = experiments.map(
    (experiment) => 1 / Math.max(experiment.standardError ** 2, 1e-6),
  );
  const totalWeight = weights.reduce((total, value) => total + value, 0);
  return {
    roi:
      experiments.reduce(
        (total, experiment, index) =>
          total +
          (experiment.incrementalOutcome /
            Math.max(experiment.incrementalSpend, 1)) *
            weights[index],
        0,
      ) / totalWeight,
    standardError: Math.sqrt(1 / totalWeight),
  };
}

function parameterNamesForAdvanced(
  dataset: Dataset,
  config: ModelConfig,
  advancedConfig: AdvancedModelConfig,
  baselineCount: number,
  channelCoefficientIndexes: number[][],
): string[] {
  const names = ["Intercept", "Trend", "Cycle sin", "Cycle cos"];
  for (let order = 1; order <= config.fourierOrder; order += 1) {
    names.push(`Annual sin ${order}`, `Annual cos ${order}`);
  }
  dataset.controlColumns.slice(0, 4).forEach((column) => {
    const values = dataset.rows.map((row) => Number(row[column]) || 0);
    if (std(values) > 1e-9) names.push(column);
  });
  if (advancedConfig.planningIntensity) names.push("Planning intensity");
  while (names.length < baselineCount) names.push(`Baseline ${names.length + 1}`);
  dataset.mediaColumns.forEach((channel, mediaIndex) => {
    const indexes = channelCoefficientIndexes[mediaIndex] ?? [];
    indexes.forEach((_, knotIndex) =>
      names.push(
        indexes.length === 1
          ? `${channel} effect`
          : `${channel} · knot ${knotIndex + 1}`,
      ),
    );
  });
  return names;
}

function baselinePrior(
  index: number,
  target: number[],
  initialValue: number,
  ridge: number,
): CompiledSamplingPrior {
  const targetScale = Math.max(std(target), 1e-3);
  return {
    kind: "normal",
    mean: index === 0 ? mean(target) : 0,
    standardDeviation:
      index === 0
        ? targetScale * 5
        : targetScale / Math.sqrt(Math.max(ridge, 1e-4)),
    initialValue,
  };
}

function roiPriorEvidence(
  channel: string,
  experiments: Experiment[],
  industryPriorChannels: string[],
  industryPriorOverrides?: IndustryPriorOverrides,
): {
  mean: number;
  standardDeviation: number;
  source: "experiment" | "industry";
} | undefined {
  const experimentPrior = pooledExperimentPrior(
    experiments.filter(
      (experiment) =>
        experiment.channel.toLowerCase() === channel.toLowerCase(),
    ),
  );
  if (experimentPrior) {
    return {
      mean: experimentPrior.roi,
      standardDeviation: Math.max(
        experimentPrior.standardError,
        Math.abs(experimentPrior.roi) * 0.12,
      ),
      source: "experiment",
    };
  }
  const benchmark = activeIndustryPrior(
    channel,
    experiments,
    industryPriorChannels,
    industryPriorOverrides,
  );
  if (!benchmark) return undefined;
  return {
    mean: benchmark.median,
    standardDeviation: Math.max(
      benchmark.standardDeviation,
      benchmark.median * 0.25,
    ),
    source: "industry",
  };
}

function mediaPrior(
  family: "bayesian" | "advanced",
  distribution: AdvancedModelConfig["priorDistribution"],
  initialValue: number,
  targetScale: number,
  ridge: number,
): CompiledSamplingPrior {
  const safeInitial = Math.max(initialValue, targetScale * 0.01, 1e-6);
  const logNormalMean = Math.max(targetScale * 0.12, 1e-4);
  const weakScale = Math.max(
    distribution === "log-normal"
      ? targetScale * 0.35
      : targetScale / Math.sqrt(Math.max(ridge, 1e-4)),
    1e-4,
  );
  return {
    kind:
      family === "bayesian"
        ? "half-normal"
        : distribution === "log-normal"
          ? "log-normal"
          : "half-normal",
    mean: distribution === "log-normal" ? logNormalMean : 0,
    standardDeviation: weakScale,
    initialValue: safeInitial,
  };
}

function plausibleUpperRoi(channel: string): number {
  const benchmark = inferIndustryPrior(channel);
  const logMedian = Math.log(Math.max(benchmark.median, 1e-6));
  const z90 = 1.281551565545;
  const lowerSigma =
    (logMedian - Math.log(Math.max(benchmark.low, 1e-6))) / z90;
  const upperSigma =
    (Math.log(Math.max(benchmark.high, 1e-6)) - logMedian) / z90;
  const logSigma = Math.max((lowerSigma + upperSigma) / 2, 0.1);
  return Math.exp(logMedian + 1.644853626951 * logSigma);
}

export function compileSamplingModel(
  dataset: Dataset,
  run: AgenticCandidateRun,
  experiments: Experiment[],
  industryPriorChannels: string[],
  industryPriorOverrides?: IndustryPriorOverrides,
): CompiledSamplingModel {
  if (!run.model || (run.spec.family !== "bayesian" && run.spec.family !== "advanced")) {
    throw new Error(
      "Production sampling requires a promoted Bayesian or Advanced specification.",
    );
  }
  const model = run.model;
  const family = run.spec.family;
  const dates = dataset.rows.map((row) => String(row[dataset.dateColumn]));
  const advancedConfig = run.spec.advancedConfig;
  let matrix: number[][];
  let outcome: number[];
  let target: number[];
  let parameterNames: string[];
  let mediaIndexes: number[][];
  let mediaVectors: number[][];
  let spendVectors: number[][];
  let baselineCount: number;
  let kernelWeights: number[][];
  let effectMultiplier = 1;

  if (family === "advanced") {
    const design = compileAdvancedSamplingDesign(
      dataset,
      run.spec.config,
      advancedConfig,
    );
    matrix = design.matrix;
    outcome = design.outcome;
    target = design.target;
    mediaIndexes = design.channelCoefficientIndexes;
    mediaVectors = design.mediaVectors;
    spendVectors = design.spendVectors;
    baselineCount = design.baselineCount;
    kernelWeights = design.kernelWeights;
    effectMultiplier = design.effectMultiplier;
    parameterNames = parameterNamesForAdvanced(
      dataset,
      run.spec.config,
      advancedConfig,
      baselineCount,
      mediaIndexes,
    );
  } else {
    const design = compileStaticSamplingDesign(dataset, run.spec.config);
    matrix = design.matrix;
    outcome = design.outcome;
    target = design.outcome;
    baselineCount = design.mediaStart;
    parameterNames = design.names;
    mediaIndexes = dataset.mediaColumns.map((_, mediaIndex) => [
      design.mediaStart + mediaIndex,
    ]);
    mediaVectors = design.mediaVectors;
    spendVectors = design.spendVectors;
    kernelWeights = Array.from({ length: dataset.rows.length }, () => [1]);
  }

  const targetScale = Math.max(std(target), 1e-3);
  const initialValues = Array(parameterNames.length).fill(0);
  initialValues[0] = mean(target);
  dataset.mediaColumns.forEach((channel, mediaIndex) => {
    const estimate = model.channels.find((item) => item.channel === channel);
    (mediaIndexes[mediaIndex] ?? []).forEach((index) => {
      initialValues[index] = Math.max(estimate?.coefficient ?? 0, 1e-6);
    });
  });
  const priors = parameterNames.map((_, index) =>
    baselinePrior(index, target, initialValues[index], run.spec.config.ridge),
  );
  const calibrations: CompiledSamplingCalibration[] = [];
  const planningParameterIndex = parameterNames.indexOf("Planning intensity");
  const planningInitial =
    planningParameterIndex >= 0
      ? matrix.map((row) => row[planningParameterIndex] ?? 0)
      : [];

  const media = dataset.mediaColumns.map((channel, mediaIndex) => {
    const indexes = mediaIndexes[mediaIndex] ?? [];
    const spend = spendVectors[mediaIndex].reduce(
      (total, value) => total + value,
      0,
    );
    const experimentEvidence = experiments.filter(
      (experiment) =>
        experiment.channel.toLowerCase() === channel.toLowerCase(),
    );
    const useExperimentPrior =
      family === "bayesian" || advancedConfig.calibrationMode === "prior";
    const evidence = roiPriorEvidence(
      channel,
      useExperimentPrior ? experiments : [],
      industryPriorChannels,
      industryPriorOverrides,
    );
    const benchmark = activeIndustryPrior(
      channel,
      experiments,
      industryPriorChannels,
      industryPriorOverrides,
    );
    const estimate = model.channels.find((item) => item.channel === channel);
    indexes.forEach((index) => {
      priors[index] = mediaPrior(
        family,
        advancedConfig.priorDistribution,
        estimate?.coefficient ?? 0,
        targetScale,
        run.spec.config.ridge,
      );
    });

    experimentEvidence.forEach((experiment, experimentIndex) => {
      const contrast = experimentResponseContrast(
        dataset,
        run.spec.config,
        channel,
        experiment,
      );
      const fallbackRows = matchingExperimentRows(dataset, experiment);
      const spendRows = contrast?.spendRows ?? fallbackRows;
      const outcomeRows = contrast?.outcomeRows ?? fallbackRows;
      const experimentSpend = contrast?.spend ?? spendRows.reduce(
        (total, rowIndex) =>
          total + spendVectors[mediaIndex][rowIndex],
        0,
      );
      if (experimentSpend <= 0) return;
      calibrations.push({
        label: `${channel} · ${experiment.source || `experiment ${experimentIndex + 1}`}`,
        channel,
        route: useExperimentPrior ? "prior" : "likelihood",
        basis: "experiment-window",
        indexes,
        weights: indexes.map((_, knotIndex) =>
          outcomeRows.reduce(
            (total, rowIndex) =>
              total +
              (contrast?.deltaTransformed[rowIndex] ??
                mediaVectors[mediaIndex][rowIndex]) *
                kernelWeights[rowIndex][knotIndex] *
                effectMultiplier,
            0,
          ) / experimentSpend,
        ),
        observedRoi:
          experiment.incrementalOutcome /
          Math.max(experiment.incrementalSpend, 1),
        standardError: Math.max(experiment.standardError, 1e-4),
        rows: outcomeRows,
        spendRows,
        outcomeRows,
        spend: experimentSpend,
      });
    });

    const roiWeights = indexes.map((_, knotIndex) =>
      mediaVectors[mediaIndex].reduce(
        (total, value, rowIndex) =>
          total + value * kernelWeights[rowIndex][knotIndex],
        0,
      ) / Math.max(spend, 1),
    );
    const channelResponse = responseForChannel(run.spec.config, channel);
    return {
      channel,
      indexes,
      spend,
      roiWeights,
      rawSpend: spendVectors[mediaIndex],
      halfSaturation: positiveQuantile(
        carryoverForResponse(
          spendVectors[mediaIndex],
          channelResponse,
        ),
        channelResponse.halfSaturationQuantile,
      ),
      response: channelResponse,
      screeningRoi: estimate?.roi ?? 0,
      screeningLow: estimate?.roiLow ?? 0,
      screeningHigh: estimate?.roiHigh ?? 0,
      plausibleUpperRoi: Math.max(
        plausibleUpperRoi(channel),
        evidence ? evidence.mean + 3 * evidence.standardDeviation : 0,
      ),
      priorEvidence: evidence
        ? {
            ...evidence,
            rows:
              evidence.source === "experiment"
                ? Array.from(
                    new Set(
                      calibrations
                        .filter(
                          (calibration) =>
                            calibration.channel.toLowerCase() ===
                              channel.toLowerCase(),
                        )
                        .flatMap((calibration) => calibration.outcomeRows),
                    ),
                  ).sort((a, b) => a - b)
                : dates.map((_, rowIndex) => rowIndex),
            label:
              evidence.source === "experiment"
                ? experimentEvidence.length === 1
                  ? experimentEvidence[0].source
                  : `${experimentEvidence.length} pooled experiments`
                : benchmark?.label ?? "Industry benchmark",
          }
        : undefined,
    };
  });

  return {
    version: SAMPLING_ENGINE_VERSION,
    promotedId: run.spec.id,
    promotedFingerprint: model.fingerprint,
    promotedFamily: family,
    validationScore: run.validation?.finalScore ?? null,
    validationEligible: passesAgenticEligibility(run),
    matrix,
    outcome,
    target,
    dates,
    parameterNames,
    priors,
    media,
    calibrations,
    response: {
      adstockType: run.spec.config.adstockType,
      adstockDecay: run.spec.config.adstock,
      weibullShape: run.spec.config.weibullShape,
      weibullScale: run.spec.config.weibullScale,
      hillShape: run.spec.config.saturation,
      timeVarying: family === "advanced" && advancedConfig.timeVarying,
      kernelKnots:
        family === "advanced" && advancedConfig.timeVarying
          ? advancedConfig.kernelKnots
          : 1,
      kernelBandwidth: advancedConfig.kernelBandwidth,
      planningIntensity:
        family === "advanced" && advancedConfig.planningIntensity,
      planningParameterIndex:
        planningParameterIndex >= 0 ? planningParameterIndex : null,
      planningInitial,
      channelSpecific: hasChannelResponseContracts(run.spec.config),
    },
    screeningPredicted: model.predicted,
    ridge: run.spec.config.ridge,
    likelihood:
      family === "advanced"
        ? advancedConfig.likelihoodDistribution
        : "gaussian",
    studentTDegreesFreedom: advancedConfig.studentTDegreesFreedom,
  };
}

export async function samplingFingerprint(
  model: CompiledSamplingModel,
  contract: SamplingContract,
): Promise<string> {
  return sha256(
    JSON.stringify({
      version: SAMPLING_ENGINE_VERSION,
      promotedFingerprint: model.promotedFingerprint,
      model,
      contract,
    }),
  );
}

export function samplingContractLabel(contract: SamplingContract): string {
  return `${contract.chains} chains · ${contract.tune.toLocaleString()} warmup · ${contract.draws.toLocaleString()} draws · ${contract.targetAccept.toFixed(2)} target`;
}

export function sameSamplingContract(
  left: SamplingContract,
  right: SamplingContract,
): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

export function samplingResultForModel(
  result: SamplingResult | undefined,
  model: ModelResult | undefined,
): boolean {
  return Boolean(result && model && result.promotedFingerprint === model.fingerprint);
}
