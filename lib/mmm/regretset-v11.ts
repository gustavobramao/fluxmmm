import selectorJson from "./artifacts/regretset-v11-selector.json";
import candidateJson from "./artifacts/regretset-v11-candidates.json";
import type { AgenticCandidateRun, AgenticCandidateSpec } from "./agentic";
import { validationDiagnosticValues } from "./score-diagnostics";
import type { Dataset, Experiment, MediaResponseConfig } from "./types";
import type { ScoreV6CandidateRow, ScoreV6EvidenceArm } from "../../research/score_v6/types";
import { v5dFeatureNames, v5dFeatureVector } from "../../research/svi_score_v5d/features";
import type { SviScoreV4CandidateRow } from "../../research/svi_score_v4/types";
import {
  temporalEvidenceForCandidate,
  wholeFlightGeneralization,
} from "../../research/svi_score_v4/temporal";
import type { ValidationModelSpec } from "./validation/adapter";
import type { ValidationOptions } from "./validation/types";
import { SVI_SCORE_V3_CONTRACT } from "../../research/svi_score_v3/contract";
import type { SviApproximationResult } from "../../research/svi_score_v3/types";

export const REGRETSET_V11_VERSION =
  "flux-regretset-relative-log-regret-1.0.0";
export const REGRETSET_V11_SELECTOR_SHA256 =
  "8096f31fd7a30728796de77937c79f9f4d2263ebc4c4f54d741a45b667ff1a33";
export const REGRETSET_V11_CANDIDATE_SHA256 =
  "ec40ca53b0f94243694a16ae5e1b4d1779bad284e42aa8cc38f7d4d529ef6270";
export const REGRETSET_V11_CANDIDATE_COUNT = 48;
export const REGRETSET_V11_BASE_SPECIFICATION_COUNT = 24;
export const REGRETSET_V11_INFERENCE_CONTRACT =
  SVI_SCORE_V3_CONTRACT.inference;

export const REGRETSET_V11_EXPERTS = [
  "predictive-generalization",
  "causal-identification",
  "posterior-decision",
  "structural-specification",
] as const;
export type RegretSetV11Expert = (typeof REGRETSET_V11_EXPERTS)[number];

export interface RegretSetV11CandidateReceipt {
  version: typeof REGRETSET_V11_VERSION;
  selectorSha256: typeof REGRETSET_V11_SELECTOR_SHA256;
  candidateId: string;
  rank: number;
  selected: boolean;
  posteriorStatus: "labelled" | "review";
  posteriorCached: boolean;
  predictedMean: number;
  predictedMedian: number;
  predictedP90: number;
  predictedP95: number;
  predictedCvar90: number;
  predictedScenarios: {
    budgetReduction: number;
    fixedBudgetMix: number;
    budgetGrowth: number;
    economicCeiling: number;
  };
  predictedDanger: number;
  predictedRisk: number;
  ensembleUncertainty: number;
  adjustedRisk: number;
  runnerUpMargin: number;
  confidenceRisk: number;
}

interface FrozenModel {
  means: number[];
  scales: number[];
  contextMeans: number[];
  contextScales: number[];
  featureIndexes: number[];
  parameters: Record<string, number[] | number[][]>;
}

interface FrozenSelector {
  artifactId: string;
  version: string;
  activation: string;
  architecture: string;
  rawFeatureNames: string[];
  featureNames: string[];
  contextNames: string[];
  policy: {
    mean_weight: number;
    p90_weight: number;
    danger_penalty: number;
    uncertainty_penalty: number;
  };
  models: FrozenModel[];
}

const SELECTOR = selectorJson as unknown as FrozenSelector;
const CANDIDATES = candidateJson as unknown as AgenticCandidateSpec[];

const SUMMARY_METRICS = [
  "diagnostic:rolling-oos",
  "diagnostic:predictive-coverage",
  "diagnostic:fold-stability",
  "diagnostic:spend-regimes",
  "diagnostic:multicollinearity",
  "diagnostic:roi-identification",
  "diagnostic:evidence-source-quality",
  "diagnostic:evidence-compatibility",
  "diagnostic:evidence-decision-dependence",
  "diagnostic:roi-posterior-plausibility",
  "diagnostic:roi-decision-stability",
] as const;

const PAIRED_METRICS = [
  "diagnostic:rolling-oos",
  "diagnostic:predictive-coverage",
  "diagnostic:roi-identification",
  "diagnostic:evidence-source-quality",
  "diagnostic:evidence-compatibility",
  "diagnostic:evidence-decision-dependence",
  "diagnostic:roi-posterior-plausibility",
  "diagnostic:roi-decision-stability",
] as const;

const POSTERIOR_CHANNEL_METRICS = [
  "log-roi-median",
  "log-roi-mean",
  "log-roi-low",
  "log-roi-high",
  "log-relative-roi-width",
  "log-roi-draw-sd",
  "implausible-probability",
  "near-zero-probability",
  "posterior-skewness",
  "location-shift-sd",
  "interval-overlap",
  "evidence-experiment",
  "evidence-benchmark",
  "response-adstock-mean",
  "response-adstock-sd",
  "response-log-weibull-scale-mean",
  "response-weibull-shape-mean",
  "response-saturation-mean",
  "response-saturation-sd",
  "response-half-saturation-mean",
  "response-weibull-share",
  "response-sum-normalization-share",
  "log-contribution-cv",
] as const;

function finite(value: unknown, fallback = 0): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function average(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) /
    Math.max(values.length, 1);
}

function standardDeviation(values: readonly number[], sample = false): number {
  if (!values.length || (sample && values.length < 2)) return 0;
  const center = average(values);
  const divisor = sample ? values.length - 1 : values.length;
  return Math.sqrt(
    values.reduce((sum, value) => sum + (value - center) ** 2, 0) /
      Math.max(divisor, 1),
  );
}

function correlation(left: readonly number[], right: readonly number[]): number {
  const length = Math.min(left.length, right.length);
  if (length < 2) return 0;
  const x = left.slice(0, length);
  const y = right.slice(0, length);
  const xMean = average(x);
  const yMean = average(y);
  const numerator = x.reduce(
    (sum, value, index) => sum + (value - xMean) * (y[index] - yMean),
    0,
  );
  const denominator = Math.sqrt(
    x.reduce((sum, value) => sum + (value - xMean) ** 2, 0) *
      y.reduce((sum, value) => sum + (value - yMean) ** 2, 0),
  );
  return denominator > 1e-12 ? numerator / denominator : 0;
}

function aggregate(values: readonly number[]): number[] {
  const safe = values.map((value) => finite(value));
  return [
    average(safe),
    standardDeviation(safe),
    Math.min(...safe),
    Math.max(...safe),
  ];
}

function logPositive(value: unknown): number {
  return Math.log(Math.max(finite(value), 1e-6));
}

function evidenceArm(candidateId: string): ScoreV6EvidenceArm {
  return candidateId.endsWith("benchmark-gap-fill")
    ? "benchmark-gap-fill"
    : "experiments-only";
}

function baseCandidateId(candidateId: string): string {
  return candidateId.split(" · ")[0];
}

function roleForChannel(channel: string): "social" | "search" | "tv" | undefined {
  const normalized = channel.toLowerCase().replace(/[^a-z0-9]+/g, "_");
  if (/(?:^|_)(?:ctv|tv|television)(?:_|$)/.test(normalized)) return "tv";
  if (/(?:non_?brand|generic|paid_search|sem)(?:_|$)/.test(normalized)) return "search";
  if (/(?:^|_)(?:meta|facebook|instagram|paid_social|social|prospect)(?:_|$)/.test(normalized)) {
    return "social";
  }
  return undefined;
}

function canonicalResponse(
  responses: ModelConfigLike["channelResponses"],
  role: "social" | "search" | "tv",
): Partial<MediaResponseConfig> | undefined {
  const key = role === "social"
    ? "meta_acquisition_spend"
    : role === "search"
      ? "google_search_nonbrand_spend"
      : "ctv_spend";
  return responses?.[key];
}

type ModelConfigLike = AgenticCandidateSpec["config"];

/**
 * The frozen research grid uses three archetypal channel names. Production
 * data may use advertiser-specific names, so overrides are mapped only when a
 * channel has an unambiguous social, non-brand-search, or television role.
 * All other channels retain the frozen specification's global response.
 */
export function adaptRegretSetV11Specification(
  specification: AgenticCandidateSpec,
  mediaColumns: readonly string[],
): AgenticCandidateSpec {
  const mapped = Object.fromEntries(
    mediaColumns.flatMap((channel) => {
      const role = roleForChannel(channel);
      const response = role
        ? canonicalResponse(specification.config.channelResponses, role)
        : undefined;
      return response ? [[channel, structuredClone(response)]] : [];
    }),
  );
  return {
    ...structuredClone(specification),
    config: {
      ...structuredClone(specification.config),
      channelResponses: Object.keys(mapped).length ? mapped : undefined,
    },
  };
}

export function regretSetV11CandidateSpecifications(
  dataset: Dataset,
  experiments: readonly Experiment[],
  benchmarkScreeningEnabled: boolean,
): AgenticCandidateSpec[] {
  if (CANDIDATES.length !== REGRETSET_V11_BASE_SPECIFICATION_COUNT) {
    throw new Error("The frozen V11 candidate library is incomplete.");
  }
  const experimentChannels = new Set(
    experiments.map((experiment) => experiment.channel.toLowerCase()),
  );
  return (["experiments-only", "benchmark-gap-fill"] as const).flatMap((arm) => {
    const industryPriorChannels = arm === "benchmark-gap-fill" && benchmarkScreeningEnabled
      ? dataset.mediaColumns.filter(
          (channel) => !experimentChannels.has(channel.toLowerCase()),
        )
      : [];
    return CANDIDATES.map((candidate) => {
      const adapted = adaptRegretSetV11Specification(candidate, dataset.mediaColumns);
      return {
        ...adapted,
        id: `${candidate.id} · ${arm}`,
        label: `${candidate.label} · ${arm === "experiments-only" ? "experiments" : "benchmark gap-fill"}`,
        evidencePriorChannels: industryPriorChannels,
      };
    });
  });
}

export function observableRegretSetV11Row(
  dataset: Dataset,
  run: AgenticCandidateRun,
  experiments: Experiment[],
  validationOptions: ValidationOptions,
): SviScoreV4CandidateRow {
  if (!run.model || !run.validation) {
    throw new Error(`Candidate ${run.spec.id} has no observable fit and validation evidence.`);
  }
  const validation = run.validation;
  const row: ScoreV6CandidateRow = {
    businessId: dataset.hash,
    family: "advertiser-upload",
    split: "validation",
    candidateId: run.spec.id,
    evidenceArm: evidenceArm(run.spec.id),
    modelFamily: run.spec.family,
    eligible: validation.eligible,
    reviewEligible: validation.gates.every((gate) => !gate.applicable || gate.passed),
    eligibilityTier: validation.eligible ? "decision-grade" : "review",
    failedGateCount: validation.gates.filter((gate) => gate.applicable && !gate.passed).length,
    layerScores: Object.fromEntries(
      Object.entries(validation.layers).map(([id, layer]) => [id, layer.score]),
    ) as ScoreV6CandidateRow["layerScores"],
    diagnostics: validationDiagnosticValues(validation.layers),
    heuristicScore: validation.heuristicScore ?? 0,
    decisionLoss: 0,
    cappedRegret: 0,
    roiError: 0,
    contributionError: 0,
  };
  const temporal = temporalEvidenceForCandidate(dataset, run.model, run.spec.config);
  const flight = wholeFlightGeneralization(dataset, {
    kind: run.model.kind,
    config: run.spec.config,
    advancedConfig: run.spec.advancedConfig,
    experiments,
    validationOptions: {
      ...validationOptions,
      industryPriorChannels:
        validationOptions.industryPriorChannels ??
        run.spec.evidencePriorChannels ??
        [],
    },
  } satisfies ValidationModelSpec);
  return {
    ...row,
    temporal: {
      ...temporal,
      wholeFlightGeneralization: flight.score,
      temporalIdentification: 100 * Math.sqrt(
        Math.max(0.01, temporal.carryoverSupport / 100) *
          Math.max(0.01, flight.score / 100),
      ),
      detail: `${temporal.detail} ${flight.folds} forward whole-flight holdout fold${flight.folds === 1 ? "" : "s"}.`,
    },
  };
}

function rolePosterior(
  result: SviApproximationResult,
  role: "social" | "search" | "tv",
): number {
  const matches = result.channels.filter(
    (channel) => roleForChannel(channel.channel) === role,
  );
  if (!matches.length) return 1e-4;
  return average(matches.map((channel) => finite(channel.posteriorMedian, 1e-4)));
}

function convergenceTokens(result: SviApproximationResult): number[] {
  const diagnostics = result.diagnostics;
  const channelLogs = (["social", "search", "tv"] as const).map((role) =>
    logPositive(rolePosterior(result, role)),
  );
  const allLogs = result.channels.map((channel) => logPositive(channel.posteriorMedian));
  const minimum = Math.min(...allLogs);
  const maximum = Math.max(...allLogs);
  return [
    result.status === "labelled" ? 1 : 0,
    diagnostics.finite ? 1 : 0,
    diagnostics.elboStable ? 1 : 0,
    diagnostics.seedAgreement ? 1 : 0,
    diagnostics.adjudicationUsed ? 1 : 0,
    Math.log1p(Math.max(0, finite(diagnostics.maximumElboDrift))),
    finite(diagnostics.maximumSeedLogRoiDifference),
    finite(diagnostics.predictiveCoverage),
    finite(diagnostics.maximumImplausibleProbability),
    Math.log1p(Math.max(0, finite(diagnostics.maximumRelativeRoiWidth))),
    finite(diagnostics.predictiveCoverage) - 0.8,
    0.2 - finite(diagnostics.maximumImplausibleProbability),
    3 - finite(diagnostics.maximumRelativeRoiWidth),
    ...channelLogs,
    average(allLogs),
    standardDeviation(allLogs),
    minimum,
    maximum,
    maximum - minimum,
  ];
}

function channelMetricVector(
  channel: SviApproximationResult["channels"][number],
  draws: SviApproximationResult["decisionDraws"][number]["channels"],
): number[] {
  const roiDraws = (channel.posteriorSamples ?? []).map(logPositive);
  const contributions = draws.map((draw) => Math.max(finite(draw.contribution), 0));
  const contributionMean = average(contributions);
  const source = (channel.evidenceSource ?? "").toLowerCase();
  const explanation = (channel.explanation ?? {}) as {
    nearZeroProbability?: number;
    posteriorSkewness?: number;
    locationShiftSd?: number;
    intervalOverlap?: number;
  };
  const responses = draws.map((draw) => draw.response);
  const adstock = responses.map((response) => finite(response.adstock));
  const weibullScale = responses.map((response) => logPositive(response.weibullScale));
  const weibullShape = responses.map((response) => finite(response.weibullShape));
  const saturation = responses.map((response) => finite(response.saturation));
  const halfSaturation = responses.map((response) => finite(response.halfSaturationQuantile));
  return [
    logPositive(channel.posteriorMedian),
    logPositive(channel.posteriorMean),
    logPositive(channel.posteriorLow),
    logPositive(channel.posteriorHigh),
    Math.log1p(
      Math.max(0, finite(channel.posteriorHigh) - finite(channel.posteriorLow)) /
        Math.max(finite(channel.posteriorMedian), 1e-6),
    ),
    standardDeviation(roiDraws),
    finite(channel.implausibleProbability),
    finite(explanation.nearZeroProbability),
    finite(explanation.posteriorSkewness),
    finite(explanation.locationShiftSd),
    finite(explanation.intervalOverlap),
    source.includes("experiment") ? 1 : 0,
    source.includes("benchmark") || source.includes("industry") ? 1 : 0,
    average(adstock),
    standardDeviation(adstock),
    average(weibullScale),
    average(weibullShape),
    average(saturation),
    standardDeviation(saturation),
    average(halfSaturation),
    average(responses.map((response) => response.adstockType === "weibull" ? 1 : 0)),
    average(responses.map((response) => response.kernelNormalization === "sum" ? 1 : 0)),
    Math.log1p(
      standardDeviation(contributions) / Math.max(Math.abs(contributionMean), 1e-6),
    ),
  ];
}

function pairwiseAbsoluteCorrelations(series: number[][]): [number, number] {
  const values: number[] = [];
  for (let left = 0; left < series.length; left += 1) {
    for (let right = left + 1; right < series.length; right += 1) {
      values.push(Math.abs(correlation(series[left], series[right])));
    }
  }
  return values.length ? [average(values), Math.max(...values)] : [0, 0];
}

function posteriorDecisionTokens(result: SviApproximationResult): number[] {
  if (!result.channels.length || !result.decisionDraws.length) {
    throw new Error(`V11 posterior ${result.fingerprint} has no channel draws.`);
  }
  const drawChannels = new Map<string, SviApproximationResult["decisionDraws"][number]["channels"]>();
  result.decisionDraws.forEach((draw) => {
    draw.channels.forEach((channel) => {
      drawChannels.set(channel.channel, [
        ...(drawChannels.get(channel.channel) ?? []),
        channel,
      ]);
    });
  });
  const channelVectors = result.channels.map((channel) =>
    channelMetricVector(channel, drawChannels.get(channel.channel) ?? []),
  );
  const tokens: number[] = [];
  for (let metric = 0; metric < POSTERIOR_CHANNEL_METRICS.length; metric += 1) {
    tokens.push(...aggregate(channelVectors.map((vector) => vector[metric])));
  }
  const roiSeries = result.channels.map((channel) =>
    (channel.posteriorSamples ?? []).map(logPositive),
  );
  const contributionSeries = result.channels.map((channel) =>
    (drawChannels.get(channel.channel) ?? []).map((draw) =>
      Math.log1p(Math.max(finite(draw.contribution), 0)),
    ),
  );
  tokens.push(
    ...pairwiseAbsoluteCorrelations(roiSeries),
    ...pairwiseAbsoluteCorrelations(contributionSeries),
  );
  if (tokens.length !== 96 || tokens.some((value) => !Number.isFinite(value))) {
    throw new Error(`V11 posterior token contract failed for ${result.fingerprint}.`);
  }
  return tokens;
}

export function regretSetV11FeatureVector(
  row: SviScoreV4CandidateRow,
  posterior: SviApproximationResult,
): { names: string[]; values: number[] } {
  const names = [
    ...v5dFeatureNames(row),
    "svi:status:labelled",
    "svi:convergence:finite",
    "svi:convergence:elbo-stable",
    "svi:convergence:seed-agreement",
    "svi:convergence:adjudication-used",
    "svi:convergence:log-elbo-drift",
    "svi:convergence:seed-log-roi-difference",
    "svi:predictive:coverage",
    "svi:decision:implausible-probability",
    "svi:decision:log-relative-roi-width",
    "svi:margin:predictive-coverage",
    "svi:margin:roi-plausibility",
    "svi:margin:roi-precision",
    "svi:roi:log-center:paid-social",
    "svi:roi:log-center:nonbrand-search",
    "svi:roi:log-center:ctv",
    "svi:roi:log-center:mean",
    "svi:roi:log-center:standard-deviation",
    "svi:roi:log-center:minimum",
    "svi:roi:log-center:maximum",
    "svi:roi:log-center:range",
    ...POSTERIOR_CHANNEL_METRICS.flatMap((metric) =>
      ["mean", "sd", "min", "max"].map(
        (aggregation) => `posterior:channel-set:${metric}:${aggregation}`,
      ),
    ),
    "posterior:channel-pair:log-roi-correlation-mean-absolute",
    "posterior:channel-pair:log-roi-correlation-maximum-absolute",
    "posterior:channel-pair:log-contribution-correlation-mean-absolute",
    "posterior:channel-pair:log-contribution-correlation-maximum-absolute",
  ];
  const values = [
    ...v5dFeatureVector(row),
    ...convergenceTokens(posterior),
    ...posteriorDecisionTokens(posterior),
  ];
  if (
    names.length !== 232 ||
    values.length !== 232 ||
    names.some((name, index) => name !== SELECTOR.rawFeatureNames[index]) ||
    values.some((value) => !Number.isFinite(value))
  ) {
    throw new Error("The runtime V11 candidate token registry does not match the frozen selector.");
  }
  return { names, values };
}

function summary(values: readonly number[]): [number, number, number] {
  return [average(values), standardDeviation(values), Math.max(...values)];
}

export function regretSetV11Context(
  candidates: readonly { candidateId: string; features: number[] }[],
): number[] {
  const featureIndex = new Map(SELECTOR.rawFeatureNames.map((name, index) => [name, index]));
  const experimentIndex = featureIndex.get("posterior:channel-set:evidence-experiment:mean");
  const benchmarkIndex = featureIndex.get("posterior:channel-set:evidence-benchmark:mean");
  if (experimentIndex === undefined || benchmarkIndex === undefined) {
    throw new Error("The frozen V11 selector is missing evidence-source tokens.");
  }
  const experimentCoverage = Math.max(0, ...candidates.map((row) => row.features[experimentIndex]));
  const benchmarkCoverage = Math.max(0, ...candidates.map((row) => row.features[benchmarkIndex]));
  const experimentAvailable = experimentCoverage > 1e-12 ? 1 : 0;
  const benchmarkAvailable = benchmarkCoverage > 1e-12 ? 1 : 0;
  const externalAvailable = experimentAvailable || benchmarkAvailable ? 1 : 0;
  const values = [
    experimentAvailable,
    experimentCoverage,
    benchmarkAvailable,
    benchmarkCoverage,
    externalAvailable,
    Math.min(1, experimentCoverage + benchmarkCoverage),
    externalAvailable,
    externalAvailable,
    externalAvailable,
    1,
    0,
    0,
    candidates.length / REGRETSET_V11_CANDIDATE_COUNT,
  ];
  SUMMARY_METRICS.forEach((metric) => {
    const index = featureIndex.get(metric);
    if (index === undefined) throw new Error(`Missing V11 context token ${metric}.`);
    const metrics = candidates.map((candidate) => candidate.features[index]);
    values.push(...summary(metrics));
  });
  values.push(1);
  const pairs = new Map<string, Partial<Record<ScoreV6EvidenceArm, number[]>>>();
  candidates.forEach((candidate) => {
    const base = baseCandidateId(candidate.candidateId);
    pairs.set(base, {
      ...(pairs.get(base) ?? {}),
      [evidenceArm(candidate.candidateId)]: candidate.features,
    });
  });
  if (
    pairs.size !== REGRETSET_V11_BASE_SPECIFICATION_COUNT ||
    [...pairs.values()].some((pair) => !pair["experiments-only"] || !pair["benchmark-gap-fill"])
  ) {
    throw new Error("V11 requires all 24 paired evidence specifications before scoring.");
  }
  PAIRED_METRICS.forEach((metric) => {
    const index = featureIndex.get(metric);
    if (index === undefined) throw new Error(`Missing V11 paired token ${metric}.`);
    const differences = [...pairs.values()].map((pair) =>
      pair["benchmark-gap-fill"]![index] - pair["experiments-only"]![index],
    );
    values.push(average(differences), standardDeviation(differences));
  });
  if (
    values.length !== SELECTOR.contextNames.length ||
    values.some((value) => !Number.isFinite(value))
  ) {
    throw new Error("The runtime V11 business-context contract is invalid.");
  }
  return values;
}

function sigmoid(value: number): number {
  const clipped = Math.max(-40, Math.min(40, value));
  return 1 / (1 + Math.exp(-clipped));
}

function softplus(value: number): number {
  return Math.log1p(Math.exp(-Math.abs(value))) + Math.max(value, 0);
}

function silu(value: number): number {
  return value * sigmoid(value);
}

function vectorMatrix(vector: readonly number[], matrix: number[][]): number[] {
  return matrix[0].map((_, column) =>
    vector.reduce((sum, value, row) => sum + value * matrix[row][column], 0),
  );
}

function add(left: readonly number[], right: readonly number[]): number[] {
  return left.map((value, index) => value + right[index]);
}

function quantile(values: readonly number[], probability: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const position = Math.max(0, Math.min(sorted.length - 1, probability * (sorted.length - 1)));
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const fraction = position - lower;
  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}

function candidateRelativeFeatures(
  features: readonly number[][],
  valid: readonly boolean[],
): number[][] {
  const active = features.map((_, index) => index).filter((index) => valid[index]);
  if (!active.length) throw new Error("RegretSet-MMM has no active candidates for relative tokenization.");
  const ranks = features.map(() => Array(features[0].length).fill(0));
  const robust = features.map(() => Array(features[0].length).fill(0));
  for (let feature = 0; feature < features[0].length; feature += 1) {
    const ordered = active
      .map((index) => ({ index, value: features[index][feature] }))
      .sort((left, right) => left.value - right.value || left.index - right.index);
    let cursor = 0;
    while (cursor < ordered.length) {
      let end = cursor + 1;
      while (end < ordered.length && ordered[end].value === ordered[cursor].value) end += 1;
      const averageRank = (cursor + end - 1) / 2;
      for (let position = cursor; position < end; position += 1) {
        ranks[ordered[position].index][feature] = ordered.length === 1
          ? 0.5
          : averageRank / (ordered.length - 1);
      }
      cursor = end;
    }
    const values = active.map((index) => features[index][feature]);
    const median = quantile(values, 0.5);
    const scale = quantile(values, 0.75) - quantile(values, 0.25);
    active.forEach((index) => {
      robust[index][feature] = scale > 1e-8
        ? Math.max(-8, Math.min(8, (features[index][feature] - median) / scale))
        : 0;
    });
  }
  return features.map((row, index) => [...row, ...ranks[index], ...robust[index]]);
}

function modelPrediction(
  model: FrozenModel,
  features: readonly number[][],
  context: readonly number[],
  valid: readonly boolean[],
): { heads: number[][] } {
  const standardized = features.map((row) =>
    row.map((value, index) => Math.max(-8, Math.min(8, (value - model.means[index]) / model.scales[index]))),
  );
  const standardizedContext = context.map((value, index) =>
    Math.max(-8, Math.min(8, (value - model.contextMeans[index]) / model.contextScales[index])),
  );
  const parameters = model.parameters;
  const candidate = standardized.map((row) =>
    add(
      vectorMatrix(
        [...model.featureIndexes.map((index) => row[index]), ...standardizedContext],
        parameters.w1 as number[][],
      ),
      parameters.b1 as number[],
    ).map(silu),
  );
  const validCount = Math.max(valid.filter(Boolean).length, 1);
  const pooled = candidate[0].map((_, column) =>
    candidate.reduce(
      (sum, row, index) => sum + (valid[index] ? row[column] : 0),
      0,
    ) / validCount,
  );
  const heads = candidate.map((row) => {
    const decoder = [
      ...row,
      ...pooled,
      ...row.map((value, index) => value - pooled[index]),
      ...row.map((value, index) => value * pooled[index]),
    ];
    const second = add(
      vectorMatrix(decoder, parameters.w2 as number[][]),
      parameters.b2 as number[],
    ).map(silu);
    const raw = add(
      vectorMatrix(second, parameters.wo as number[][]),
      parameters.bo as number[],
    );
    const output = Array.from({ length: 10 }, () => 0);
    output[0] = softplus(raw[0]);
    output[1] = softplus(raw[1]);
    output[2] = output[1] + softplus(raw[2]);
    output[3] = output[2] + softplus(raw[3]);
    output[4] = output[3] + softplus(raw[4]);
    for (let index = 5; index <= 8; index += 1) output[index] = softplus(raw[index]);
    output[9] = sigmoid(raw[9]);
    return output;
  });
  return { heads };
}

export function scoreRegretSetV11(
  candidates: readonly {
    candidateId: string;
    features: number[];
    valid: boolean;
    posteriorStatus: "labelled" | "review";
    posteriorCached?: boolean;
  }[],
): RegretSetV11CandidateReceipt[] {
  if (candidates.length !== REGRETSET_V11_CANDIDATE_COUNT) {
    throw new Error(`V11 requires exactly ${REGRETSET_V11_CANDIDATE_COUNT} candidates.`);
  }
  const context = regretSetV11Context(candidates);
  const valid = candidates.map((candidate) => candidate.valid);
  if (!valid.some(Boolean)) throw new Error("V11 has no finite posterior candidate to rank.");
  const rawFeatures = candidates.map((candidate) => candidate.features);
  const features = candidateRelativeFeatures(rawFeatures, valid);
  if (features[0].length !== SELECTOR.featureNames.length) {
    throw new Error("The runtime relative-token contract does not match the frozen selector.");
  }
  const predictions = SELECTOR.models.map((model) =>
    modelPrediction(model, features, context, valid),
  );
  const candidatePredictions = candidates.map((candidate, index) => {
    const memberHeads = predictions.map((prediction) => prediction.heads[index]);
    const heads = memberHeads[0].map((_, head) =>
      average(memberHeads.map((member) => member[head])),
    );
    const memberRisks = memberHeads.map((member) =>
      SELECTOR.policy.mean_weight * member[0] + SELECTOR.policy.p90_weight * member[2]
    );
    const predictedRisk = average(memberRisks);
    const ensembleUncertainty = standardDeviation(memberRisks, true);
    const adjustedRisk = predictedRisk;
    return {
      candidate,
      heads,
      predictedRisk,
      ensembleUncertainty,
      adjustedRisk,
    };
  });
  const order = candidatePredictions
    .map((prediction, index) => ({ prediction, index }))
    .filter(({ index }) => valid[index])
    .sort((left, right) =>
      left.prediction.adjustedRisk - right.prediction.adjustedRisk ||
      left.prediction.candidate.candidateId.localeCompare(right.prediction.candidate.candidateId),
    );
  const rank = new Map(order.map((item, index) => [item.index, index + 1]));
  const winner = order[0];
  const runnerUpMargin = order.length > 1
    ? order[1].prediction.adjustedRisk - winner.prediction.adjustedRisk
    : Number.POSITIVE_INFINITY;
  const selectedRelativeUncertainty = winner.prediction.ensembleUncertainty /
    Math.max(Math.abs(winner.prediction.predictedRisk) + 0.1, 0.1);
  const ambiguity = Math.exp(
    -Math.max(runnerUpMargin, 0) /
      Math.max(Math.abs(winner.prediction.adjustedRisk) + 0.1, 0.1),
  );
  const confidenceRisk = 0.4 * winner.prediction.heads[9] +
    0.3 * Math.min(selectedRelativeUncertainty, 1) +
    0.3 * ambiguity;
  return candidatePredictions.map((prediction, index) => ({
    version: REGRETSET_V11_VERSION,
    selectorSha256: REGRETSET_V11_SELECTOR_SHA256,
    candidateId: prediction.candidate.candidateId,
    rank: rank.get(index) ?? REGRETSET_V11_CANDIDATE_COUNT + 1,
    selected: index === winner.index,
    posteriorStatus: prediction.candidate.posteriorStatus,
    posteriorCached: Boolean(prediction.candidate.posteriorCached),
    predictedMean: prediction.heads[0],
    predictedMedian: prediction.heads[1],
    predictedP90: prediction.heads[2],
    predictedP95: prediction.heads[3],
    predictedCvar90: prediction.heads[4],
    predictedScenarios: {
      budgetReduction: prediction.heads[5],
      fixedBudgetMix: prediction.heads[6],
      budgetGrowth: prediction.heads[7],
      economicCeiling: prediction.heads[8],
    },
    predictedDanger: prediction.heads[9],
    predictedRisk: prediction.predictedRisk,
    ensembleUncertainty: prediction.ensembleUncertainty,
    adjustedRisk: prediction.adjustedRisk,
    runnerUpMargin: index === winner.index ? runnerUpMargin : 0,
    confidenceRisk: index === winner.index ? confidenceRisk : 0,
  }));
}

export function regretSetV11SelectorMetadata() {
  return {
    version: SELECTOR.version,
    artifactId: SELECTOR.artifactId,
    activation: SELECTOR.activation,
    architecture: SELECTOR.architecture,
    rawFeatureNames: [...SELECTOR.rawFeatureNames],
    featureNames: [...SELECTOR.featureNames],
    contextNames: [...SELECTOR.contextNames],
    modelCount: SELECTOR.models.length,
  };
}
