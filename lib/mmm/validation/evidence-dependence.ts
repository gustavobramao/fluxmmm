import { advancedModelFingerprint, runAdvancedModel } from "../advanced";
import { activeIndustryPrior, type IndustryPriorSelection } from "../benchmarks";
import { modelFingerprint, runModel } from "../models";
import type {
  AdvancedModelConfig,
  Dataset,
  Experiment,
  ModelConfig,
  ModelResult,
} from "../types";
import type {
  ChannelEvidenceCoherence,
  EvidenceCoherenceAssessment,
  ValidationOptions,
} from "./types";

export interface EvidenceQualityComponents {
  independence: number;
  precision: number;
  relevance: number;
  transportability: number;
}

export interface EvidenceDecisionDependence {
  source: "experiment" | "benchmark";
  influence: number;
  quality: number;
  qualityComponents: EvidenceQualityComponents;
  standardizedConflict: number;
  sourceRoi: number;
  sourceStandardError: number;
  leaveSourceOutRoi: number;
  roiLocationShift: number;
  intervalWidthShift: number;
  contributionShift: number;
  allocationProbabilityShift: number;
  marginalProfitIndexShift: number;
}

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function geometricMean(values: number[]): number {
  return Math.exp(
    values.reduce(
      (total, value) => total + Math.log(Math.max(value, 0.01)),
      0,
    ) / Math.max(values.length, 1),
  );
}

function pooledExperiment(experiments: Experiment[]): {
  roi: number;
  standardError: number;
  confidence: number;
} {
  const weights = experiments.map(
    (experiment) => 1 / Math.max(experiment.standardError ** 2, 1e-9),
  );
  const totalWeight = weights.reduce((total, value) => total + value, 0);
  return {
    roi: experiments.reduce(
      (total, experiment, index) =>
        total +
        (experiment.incrementalOutcome /
          Math.max(experiment.incrementalSpend, 1)) *
          weights[index],
      0,
    ) / Math.max(totalWeight, 1e-9),
    standardError: Math.sqrt(1 / Math.max(totalWeight, 1e-9)),
    confidence: experiments.reduce(
      (total, experiment) =>
        total + clamp(experiment.confidence > 1 ? experiment.confidence / 100 : experiment.confidence),
      0,
    ) / Math.max(experiments.length, 1),
  };
}

function withoutIndustryChannel(
  selection: IndustryPriorSelection | undefined,
  dataset: Dataset,
  channel: string,
): IndustryPriorSelection {
  if (selection === false || selection === undefined) return false;
  const selected = selection === true ? dataset.mediaColumns : selection;
  return selected.filter(
    (candidate) => candidate.toLowerCase() !== channel.toLowerCase(),
  );
}

async function refitWithoutSource(
  dataset: Dataset,
  model: ModelResult,
  config: ModelConfig,
  advancedConfig: AdvancedModelConfig,
  experiments: Experiment[],
  options: ValidationOptions,
  channel: string,
  source: "experiment" | "benchmark",
): Promise<ModelResult> {
  const refitExperiments = source === "experiment"
    ? experiments.filter(
        (experiment) => experiment.channel.toLowerCase() !== channel.toLowerCase(),
      )
    : experiments;
  const refitIndustrySelection = source === "benchmark"
    ? withoutIndustryChannel(options.industryPriorChannels, dataset, channel)
    : options.industryPriorChannels ?? false;
  if (model.kind === "advanced") {
    const fingerprint = await advancedModelFingerprint(
      dataset,
      config,
      advancedConfig,
      refitExperiments,
      refitIndustrySelection,
      options.industryPriorOverrides,
    );
    return runAdvancedModel(
      dataset,
      config,
      advancedConfig,
      refitExperiments,
      fingerprint,
      refitIndustrySelection,
      options.industryPriorOverrides,
    );
  }
  const fingerprint = await modelFingerprint(
    dataset,
    config,
    refitExperiments,
    model.kind,
    refitIndustrySelection,
    options.industryPriorOverrides,
  );
  return runModel(
    dataset,
    config,
    refitExperiments,
    model.kind,
    fingerprint,
    refitIndustrySelection,
    options.industryPriorOverrides,
  );
}

function allocationProbabilities(model: ModelResult, grossMargin = 0.4) {
  const logits = model.channels.map(
    (channel) => clamp(channel.roi * grossMargin - 1, -8, 8),
  );
  const maximum = Math.max(...logits, 0);
  const values = logits.map((value) => Math.exp((value - maximum) / 0.35));
  const total = values.reduce((sum, value) => sum + value, 0);
  return new Map(
    model.channels.map((channel, index) => [
      channel.channel.toLowerCase(),
      values[index] / Math.max(total, 1e-12),
    ]),
  );
}

function attributionInfluence(
  model: ModelResult,
  channel: string,
  source: "experiment" | "benchmark",
): number {
  const attribution = model.numerical?.evidenceAttribution?.find(
    (item) => item.channel.toLowerCase() === channel.toLowerCase(),
  );
  return attribution?.uncertaintyShare[source] ?? 0;
}

function experimentQuality(
  evidence: Experiment[],
  pooled: ReturnType<typeof pooledExperiment>,
  independenceConfirmed: boolean,
): { quality: number; components: EvidenceQualityComponents } {
  const relativeError = pooled.standardError / Math.max(Math.abs(pooled.roi), 0.1);
  const components = {
    independence: independenceConfirmed ? 1 : 0.35,
    precision: 1 / (1 + relativeError),
    relevance: pooled.confidence,
    transportability: evidence.every(
      (experiment) => experiment.startDate && experiment.endDate,
    ) ? 0.8 : 0.5,
  };
  return { quality: geometricMean(Object.values(components)), components };
}

function benchmarkQuality(
  benchmark: NonNullable<ReturnType<typeof activeIndustryPrior>>,
): { quality: number; components: EvidenceQualityComponents } {
  const confidence = benchmark.confidence === "Medium"
    ? 0.7
    : benchmark.confidence === "Low"
      ? 0.45
      : 0.25;
  const relevance = benchmark.matchQuality === "Exact tactic"
    ? 1
    : benchmark.matchQuality === "Channel only"
      ? 0.7
      : 0.35;
  const relativeError = benchmark.standardDeviation /
    Math.max(Math.abs(benchmark.median), 0.1);
  const components = {
    independence: 0.65,
    precision: 1 / (1 + relativeError),
    relevance,
    transportability: confidence,
  };
  return { quality: geometricMean(Object.values(components)), components };
}

export async function assessEvidenceDependence(
  dataset: Dataset,
  model: ModelResult,
  config: ModelConfig,
  advancedConfig: AdvancedModelConfig,
  experiments: Experiment[],
  options: ValidationOptions,
): Promise<Record<string, EvidenceDecisionDependence>> {
  if (model.kind === "frequentist") return {};
  const output: Record<string, EvidenceDecisionDependence> = {};
  for (const estimate of model.channels) {
    const matching = experiments.filter(
      (experiment) =>
        experiment.channel.toLowerCase() === estimate.channel.toLowerCase(),
    );
    const benchmark = activeIndustryPrior(
      estimate.channel,
      experiments,
      options.industryPriorChannels ?? false,
      options.industryPriorOverrides,
    );
    const source = matching.length
      ? "experiment" as const
      : benchmark
        ? "benchmark" as const
        : undefined;
    if (!source) continue;
    const refit = await refitWithoutSource(
      dataset,
      model,
      config,
      advancedConfig,
      experiments,
      options,
      estimate.channel,
      source,
    );
    const leaveOut = refit.channels.find(
      (channel) => channel.channel.toLowerCase() === estimate.channel.toLowerCase(),
    );
    if (!leaveOut) continue;
    const sourceEvidence = source === "experiment"
      ? pooledExperiment(matching)
      : {
          roi: benchmark!.median,
          standardError: benchmark!.standardDeviation,
          confidence: 0,
        };
    const quality = source === "experiment"
      ? experimentQuality(
          matching,
          sourceEvidence,
          options.anchorIndependenceConfirmed,
        )
      : benchmarkQuality(benchmark!);
    const leaveOutSe = Math.max(
      (leaveOut.roiHigh - leaveOut.roiLow) / 3.92,
      Math.abs(leaveOut.roi) * 0.02,
      0.01,
    );
    const fullWidth = Math.max(estimate.roiHigh - estimate.roiLow, 0.01);
    const leaveOutWidth = Math.max(leaveOut.roiHigh - leaveOut.roiLow, 0.01);
    const fullAllocation = allocationProbabilities(model);
    const leaveOutAllocation = allocationProbabilities(refit);
    const key = estimate.channel.toLowerCase();
    output[key] = {
      source,
      influence: attributionInfluence(model, estimate.channel, source),
      quality: quality.quality,
      qualityComponents: quality.components,
      standardizedConflict:
        Math.abs(leaveOut.roi - sourceEvidence.roi) /
        Math.sqrt(leaveOutSe ** 2 + sourceEvidence.standardError ** 2),
      sourceRoi: sourceEvidence.roi,
      sourceStandardError: sourceEvidence.standardError,
      leaveSourceOutRoi: leaveOut.roi,
      roiLocationShift: Math.abs(
        Math.log((Math.max(estimate.roi, 0) + 0.05) /
          (Math.max(leaveOut.roi, 0) + 0.05)),
      ),
      intervalWidthShift: Math.abs(fullWidth - leaveOutWidth) / leaveOutWidth,
      contributionShift:
        Math.abs(estimate.contribution - leaveOut.contribution) /
        Math.max(Math.abs(estimate.contribution), Math.abs(leaveOut.contribution), 1),
      allocationProbabilityShift: Math.abs(
        (fullAllocation.get(key) ?? 0) - (leaveOutAllocation.get(key) ?? 0),
      ),
      marginalProfitIndexShift: Math.abs(
        (estimate.roi - leaveOut.roi) * 0.4,
      ),
    };
  }
  return output;
}

export function attachEvidenceDependence(
  coherence: EvidenceCoherenceAssessment,
  dependence: Record<string, EvidenceDecisionDependence>,
): EvidenceCoherenceAssessment {
  return {
    ...coherence,
    channels: coherence.channels.map((channel): ChannelEvidenceCoherence => ({
      ...channel,
      evidenceDependence: dependence[channel.channel.toLowerCase()],
    })),
  };
}
