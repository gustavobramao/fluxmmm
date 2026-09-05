import type { AgenticCandidateSpec } from "../../lib/mmm/agentic";
import { AGENTIC_PARAMETER_BOUNDS } from "../../lib/mmm/agentic-search";
import type { MediaResponseConfig } from "../../lib/mmm/types";
import { SCORE_V6_CANDIDATES } from "../score_v6/cohort";
import type { V5BSearchCandidate } from "./types";

const SPEC_BY_ID = new Map(SCORE_V6_CANDIDATES.map((spec) => [spec.id, spec]));
const CHANNELS = [
  "meta_acquisition_spend",
  "google_search_nonbrand_spend",
  "ctv_spend",
] as const;

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function linear(value: number, minimum: number, maximum: number): number {
  return clamp((value - minimum) / Math.max(maximum - minimum, 1e-12));
}

function logarithmic(value: number, minimum: number, maximum: number): number {
  return linear(Math.log(Math.max(value, minimum)), Math.log(minimum), Math.log(maximum));
}

function oneHot<T extends string | number>(value: T, values: readonly T[]): number[] {
  return values.map((candidate) => candidate === value ? 1 : 0);
}

export function v5bBaseCandidateId(candidateId: string): string {
  return candidateId.split(" · ")[0]?.trim() ?? candidateId;
}

export function v5bEvidenceArm(candidateId: string):
  | "experiments-only"
  | "benchmark-gap-fill" {
  return candidateId.includes("benchmark-gap-fill")
    ? "benchmark-gap-fill"
    : "experiments-only";
}

export function v5bSpecification(candidateId: string): AgenticCandidateSpec {
  const id = v5bBaseCandidateId(candidateId);
  const spec = SPEC_BY_ID.get(id);
  if (!spec) throw new Error(`V5B has no declared specification for ${candidateId}.`);
  return spec;
}

function response(
  spec: AgenticCandidateSpec,
  channel: (typeof CHANNELS)[number],
): MediaResponseConfig {
  return {
    adstockType: spec.config.adstockType,
    adstock: spec.config.adstock,
    weibullShape: spec.config.weibullShape,
    weibullScale: spec.config.weibullScale,
    saturation: spec.config.saturation,
    halfSaturationQuantile: 0.5,
    kernelNormalization: "peak",
    ...spec.config.channelResponses?.[channel],
  };
}

/**
 * Search-space coordinates contain declared model settings only. Proposal
 * phase, generator identity, diagnostics, selector score, and economic labels
 * are deliberately excluded.
 */
export function v5bParameterPoint(candidateId: string): number[] {
  const spec = v5bSpecification(candidateId);
  const config = spec.config;
  const advanced = spec.advancedConfig;
  const point = [
    ...oneHot(spec.family, ["frequentist", "bayesian", "advanced"] as const),
    v5bEvidenceArm(candidateId) === "benchmark-gap-fill" ? 1 : 0,
    ...oneHot(config.adstockType, ["geometric", "weibull"] as const),
    linear(config.adstock, AGENTIC_PARAMETER_BOUNDS.adstock.min, AGENTIC_PARAMETER_BOUNDS.adstock.max),
    linear(config.weibullShape, AGENTIC_PARAMETER_BOUNDS.weibullShape.min, AGENTIC_PARAMETER_BOUNDS.weibullShape.max),
    linear(config.weibullScale, AGENTIC_PARAMETER_BOUNDS.weibullScale.min, AGENTIC_PARAMETER_BOUNDS.weibullScale.max),
    linear(config.saturation, AGENTIC_PARAMETER_BOUNDS.saturation.min, AGENTIC_PARAMETER_BOUNDS.saturation.max),
    logarithmic(config.ridge, AGENTIC_PARAMETER_BOUNDS.ridge.min, AGENTIC_PARAMETER_BOUNDS.ridge.max),
    ...oneHot(config.fourierOrder, AGENTIC_PARAMETER_BOUNDS.fourierOrder),
    ...oneHot(config.cyclePeriod, [13, 26, 52] as const),
    advanced.timeVarying ? 1 : 0,
    linear(advanced.kernelKnots, AGENTIC_PARAMETER_BOUNDS.kernelKnots.min, AGENTIC_PARAMETER_BOUNDS.kernelKnots.max),
    linear(advanced.kernelBandwidth, AGENTIC_PARAMETER_BOUNDS.kernelBandwidth.min, AGENTIC_PARAMETER_BOUNDS.kernelBandwidth.max),
    advanced.planningIntensity ? 1 : 0,
    ...oneHot(advanced.calibrationMode, ["prior", "likelihood"] as const),
    ...oneHot(advanced.priorDistribution, ["half-normal", "log-normal"] as const),
    ...oneHot(
      advanced.likelihoodDistribution,
      ["gaussian", "student-t", "log-normal"] as const,
    ),
    linear(
      advanced.studentTDegreesFreedom,
      AGENTIC_PARAMETER_BOUNDS.studentTDegreesFreedom.min,
      AGENTIC_PARAMETER_BOUNDS.studentTDegreesFreedom.max,
    ),
  ];
  CHANNELS.forEach((channel) => {
    const item = response(spec, channel);
    point.push(
      ...oneHot(item.adstockType, ["geometric", "weibull"] as const),
      linear(item.adstock, AGENTIC_PARAMETER_BOUNDS.adstock.min, AGENTIC_PARAMETER_BOUNDS.adstock.max),
      linear(item.weibullShape, AGENTIC_PARAMETER_BOUNDS.weibullShape.min, AGENTIC_PARAMETER_BOUNDS.weibullShape.max),
      linear(item.weibullScale, AGENTIC_PARAMETER_BOUNDS.weibullScale.min, AGENTIC_PARAMETER_BOUNDS.weibullScale.max),
      linear(item.saturation, AGENTIC_PARAMETER_BOUNDS.saturation.min, AGENTIC_PARAMETER_BOUNDS.saturation.max),
      linear(
        item.halfSaturationQuantile,
        AGENTIC_PARAMETER_BOUNDS.halfSaturationQuantile.min,
        AGENTIC_PARAMETER_BOUNDS.halfSaturationQuantile.max,
      ),
      item.kernelNormalization === "sum" ? 1 : 0,
    );
  });
  return point;
}

export function v5bCandidateUniverse(candidateIds: readonly string[]): V5BSearchCandidate[] {
  const unique = [...new Set(candidateIds)].sort((left, right) =>
    v5bBaseCandidateId(left).localeCompare(v5bBaseCandidateId(right), undefined, {
      numeric: true,
    }) || v5bEvidenceArm(left).localeCompare(v5bEvidenceArm(right))
  );
  const candidates = unique.map((candidateId) => ({
    candidateId,
    point: v5bParameterPoint(candidateId),
  }));
  const dimensions = candidates[0]?.point.length ?? 0;
  if (!dimensions || candidates.some((candidate) => candidate.point.length !== dimensions)) {
    throw new Error("V5B candidate parameter vectors do not share one contract.");
  }
  return candidates;
}

export function v5bSquaredDistance(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length || !left.length) {
    throw new Error("V5B distance requires equal non-empty parameter vectors.");
  }
  return left.reduce((sum, value, index) => {
    const difference = value - right[index];
    return sum + difference * difference;
  }, 0) / left.length;
}
