import {
  V9_POSTERIOR_CHANNEL_METRICS,
  V9_POSTERIOR_TOKEN_NAMES,
} from "./contract";
import type { SamplingDecisionDraw } from "../../lib/mmm/sampling";

export interface V9PosteriorChannel {
  channel: string;
  posteriorMean: number;
  posteriorMedian: number;
  posteriorLow: number;
  posteriorHigh: number;
  posteriorSamples: number[];
  implausibleProbability?: number;
  evidenceSource?: string;
  explanation?: {
    locationShiftSd?: number;
    intervalOverlap?: number;
    posteriorSkewness?: number;
    nearZeroProbability?: number;
  };
}

export interface V9DecisionDrawChannel {
  channel: string;
  roi: number;
  contribution: number;
  response: {
    adstockType: "geometric" | "weibull";
    adstock: number;
    weibullShape: number;
    weibullScale: number;
    saturation: number;
    halfSaturationQuantile: number;
    kernelNormalization: "peak" | "sum";
  };
}

export interface V9SviResult {
  fingerprint: string;
  contract: unknown;
  status: "labelled" | "review";
  diagnostics: {
    finite: boolean;
    elboStable: boolean;
    seedAgreement: boolean;
    adjudicationUsed: boolean;
    maximumElboDrift: number;
    maximumSeedLogRoiDifference: number;
    predictiveCoverage: number;
    maximumImplausibleProbability: number;
    maximumRelativeRoiWidth: number;
  };
  seeds: Array<{
    seed: number;
    iterations: number;
    finalLoss: number;
    elboDrift: number;
    finite: boolean;
    runtimeSeconds: number;
  }>;
  channels: V9PosteriorChannel[];
  decisionDraws: SamplingDecisionDraw[];
}

const finite = (value: unknown, fallback = 0) => {
  const result = Number(value);
  return Number.isFinite(result) ? result : fallback;
};

const average = (values: readonly number[]) =>
  values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);

const standardDeviation = (values: readonly number[]) => {
  const center = average(values);
  return Math.sqrt(average(values.map((value) => (value - center) ** 2)));
};

const logPositive = (value: unknown) => Math.log(Math.max(finite(value), 1e-6));

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
  return [average(safe), standardDeviation(safe), Math.min(...safe), Math.max(...safe)];
}

function channelMetricVector(
  channel: V9PosteriorChannel,
  draws: V9DecisionDrawChannel[],
): number[] {
  const roiDraws = channel.posteriorSamples.map((value) => logPositive(value));
  const contributions = draws.map((draw) => Math.max(finite(draw.contribution), 0));
  const contributionMean = average(contributions);
  const source = (channel.evidenceSource ?? "").toLowerCase();
  const explanation = channel.explanation ?? {};
  const responses = draws.map((draw) => draw.response);
  const adstock = responses.map((response) => finite(response.adstock));
  const weibullScale = responses.map((response) => logPositive(response.weibullScale));
  const weibullShape = responses.map((response) => finite(response.weibullShape));
  const saturation = responses.map((response) => finite(response.saturation));
  const halfSaturation = responses.map((response) =>
    finite(response.halfSaturationQuantile)
  );
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

export function v9PosteriorDecisionTokens(result: V9SviResult): number[] {
  if (!result.channels.length || !result.decisionDraws.length) {
    throw new Error(`V9 posterior ${result.fingerprint} has no channel draws.`);
  }
  const drawChannels = new Map<string, V9DecisionDrawChannel[]>();
  result.decisionDraws.forEach((draw) => {
    draw.channels.forEach((channel) => {
      drawChannels.set(channel.channel, [
        ...(drawChannels.get(channel.channel) ?? []),
        channel,
      ]);
    });
  });
  const channelVectors = result.channels.map((channel) =>
    channelMetricVector(channel, drawChannels.get(channel.channel) ?? [])
  );
  const aggregated: number[] = [];
  for (let metric = 0; metric < V9_POSTERIOR_CHANNEL_METRICS.length; metric += 1) {
    aggregated.push(...aggregate(channelVectors.map((vector) => vector[metric])));
  }
  const roiSeries = result.channels.map((channel) =>
    channel.posteriorSamples.map((value) => logPositive(value))
  );
  const contributionSeries = result.channels.map((channel) =>
    (drawChannels.get(channel.channel) ?? []).map((draw) =>
      Math.log1p(Math.max(finite(draw.contribution), 0))
    )
  );
  aggregated.push(
    ...pairwiseAbsoluteCorrelations(roiSeries),
    ...pairwiseAbsoluteCorrelations(contributionSeries),
  );
  if (
    aggregated.length !== V9_POSTERIOR_TOKEN_NAMES.length ||
    aggregated.some((value) => !Number.isFinite(value))
  ) {
    throw new Error(`V9 posterior token contract failed for ${result.fingerprint}.`);
  }
  return aggregated;
}
