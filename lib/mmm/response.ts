import { adstock, hill, weibullAdstock } from "./math";
import type { MediaResponseConfig, ModelConfig } from "./types";

const clamp = (value: number, low: number, high: number) =>
  Math.min(high, Math.max(low, value));

function channelOverride(
  config: ModelConfig,
  channel: string,
): Partial<MediaResponseConfig> | undefined {
  const exact = config.channelResponses?.[channel];
  if (exact) return exact;
  const target = channel.toLowerCase();
  const match = Object.entries(config.channelResponses ?? {}).find(
    ([key]) => key.toLowerCase() === target,
  );
  return match?.[1];
}

export function responseForChannel(
  config: ModelConfig,
  channel: string,
): MediaResponseConfig {
  const override = channelOverride(config, channel);
  return {
    adstockType: override?.adstockType ?? config.adstockType,
    adstock: clamp(override?.adstock ?? config.adstock, 0.01, 0.99),
    weibullShape: clamp(
      override?.weibullShape ?? config.weibullShape,
      0.2,
      12,
    ),
    weibullScale: clamp(
      override?.weibullScale ?? config.weibullScale,
      0.5,
      52,
    ),
    saturation: clamp(override?.saturation ?? config.saturation, 0.25, 5),
    halfSaturationQuantile: clamp(
      override?.halfSaturationQuantile ?? 0.5,
      0.1,
      0.9,
    ),
    kernelNormalization: override?.kernelNormalization ?? "peak",
  };
}

export function positiveQuantile(values: number[], quantile: number): number {
  const positive = values
    .filter((value) => Number.isFinite(value) && value > 0)
    .sort((a, b) => a - b);
  if (!positive.length) return 1;
  const position = clamp(quantile, 0, 1) * (positive.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return positive[lower] * (1 - weight) + positive[upper] * weight;
}

function normaliseCarryover(
  source: number[],
  transformed: number[],
  mode: MediaResponseConfig["kernelNormalization"],
): number[] {
  if (mode !== "sum") return transformed;
  const sourceTotal = source.reduce((total, value) => total + value, 0);
  const transformedTotal = transformed.reduce(
    (total, value) => total + value,
    0,
  );
  if (sourceTotal <= 0 || transformedTotal <= 0) return transformed;
  const scale = sourceTotal / transformedTotal;
  return transformed.map((value) => value * scale);
}

export function carryoverForResponse(
  values: number[],
  response: MediaResponseConfig,
): number[] {
  const transformed =
    response.adstockType === "weibull"
      ? weibullAdstock(values, response.weibullShape, response.weibullScale)
      : adstock(values, response.adstock);
  return normaliseCarryover(values, transformed, response.kernelNormalization);
}

export function responseTransform(
  values: number[],
  response: MediaResponseConfig,
  fixedHalfSaturation?: number,
): { carryover: number[]; transformed: number[]; halfSaturation: number } {
  const carryover = carryoverForResponse(values, response);
  const halfSaturation =
    fixedHalfSaturation ??
    positiveQuantile(carryover, response.halfSaturationQuantile);
  return {
    carryover,
    transformed: hill(carryover, response.saturation, halfSaturation),
    halfSaturation,
  };
}

export function hasChannelResponseContracts(config: ModelConfig): boolean {
  return Object.keys(config.channelResponses ?? {}).length > 0;
}
