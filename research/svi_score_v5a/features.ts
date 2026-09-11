import type { AgenticCandidateSpec } from "../../lib/mmm/agentic";
import type { MediaResponseConfig } from "../../lib/mmm/types";
import candidateJson from "../../lib/mmm/artifacts/regretset-v11-candidates.json";
import {
  SVI_SCORE_V4_FEATURES,
  v4FeatureValue,
} from "../svi_score_v4/features";
import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";

interface FeatureTerm {
  name: string;
  value: number;
  base: boolean;
}

const SPEC_BY_ID = new Map(
  (candidateJson as AgenticCandidateSpec[]).map((spec) => [spec.id, spec]),
);

const DEFICIT_DIAGNOSTICS = [
  "rolling-oos",
  "residual-independence",
  "multicollinearity",
  "anchor-recovery",
  "confounder-sensitivity",
  "future-media-placebo",
  "roi-identification",
  "roi-posterior-plausibility",
  "roi-decision-stability",
  "whole-flight-generalization",
  "carryover-support",
] as const;

const TEMPORAL_CONTEXT_DIAGNOSTICS = [
  "whole-flight-generalization",
  "carryover-support",
  "post-flight-residual-stability",
  "kernel-distinguishability",
  "rolling-oos",
  "new-data-stability",
] as const;

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function normalizedDiagnostic(
  row: SviScoreV4CandidateRow,
  feature: (typeof SVI_SCORE_V4_FEATURES)[number],
): number {
  return clamp(v4FeatureValue(row, feature) / 100, 0.01, 1);
}

function logDiagnostic(
  row: SviScoreV4CandidateRow,
  feature: (typeof SVI_SCORE_V4_FEATURES)[number],
): number {
  return Math.log(normalizedDiagnostic(row, feature));
}

function baseCandidateId(candidateId: string): string {
  return candidateId.split(" · ")[0]?.trim() ?? candidateId;
}

function specification(row: SviScoreV4CandidateRow): AgenticCandidateSpec {
  const spec = SPEC_BY_ID.get(baseCandidateId(row.candidateId));
  if (!spec) {
    throw new Error(`V5A has no declared specification for ${row.candidateId}.`);
  }
  return spec;
}

function response(
  spec: AgenticCandidateSpec,
  channel: "meta_acquisition_spend" | "google_search_nonbrand_spend" | "ctv_spend",
): MediaResponseConfig {
  const override = spec.config.channelResponses?.[channel];
  return {
    adstockType: spec.config.adstockType,
    adstock: spec.config.adstock,
    weibullShape: spec.config.weibullShape,
    weibullScale: spec.config.weibullScale,
    saturation: spec.config.saturation,
    halfSaturationQuantile: 0.5,
    kernelNormalization: "peak",
    ...override,
  };
}

function binaryTerms(
  row: SviScoreV4CandidateRow,
  spec: AgenticCandidateSpec,
): Array<[string, number]> {
  const ctv = response(spec, "ctv_spend");
  return [
    ["spec:family:advanced", spec.family === "advanced" ? 1 : 0],
    ["spec:family:bayesian", spec.family === "bayesian" ? 1 : 0],
    ["spec:evidence:benchmark-gap-fill", row.evidenceArm === "benchmark-gap-fill" ? 1 : 0],
    ["spec:evidence:experiments-only", row.evidenceArm === "experiments-only" ? 1 : 0],
    ["spec:global-adstock:weibull", spec.config.adstockType === "weibull" ? 1 : 0],
    ["spec:global-adstock:geometric", spec.config.adstockType === "geometric" ? 1 : 0],
    ["spec:time-varying:on", spec.advancedConfig.timeVarying ? 1 : 0],
    ["spec:time-varying:off", spec.advancedConfig.timeVarying ? 0 : 1],
    ["spec:planning:on", spec.advancedConfig.planningIntensity ? 1 : 0],
    ["spec:planning:off", spec.advancedConfig.planningIntensity ? 0 : 1],
    ["spec:calibration:likelihood", spec.advancedConfig.calibrationMode === "likelihood" ? 1 : 0],
    ["spec:calibration:prior", spec.advancedConfig.calibrationMode === "prior" ? 1 : 0],
    ["spec:prior:log-normal", spec.advancedConfig.priorDistribution === "log-normal" ? 1 : 0],
    ["spec:prior:half-normal", spec.advancedConfig.priorDistribution === "half-normal" ? 1 : 0],
    ["spec:likelihood:gaussian", spec.advancedConfig.likelihoodDistribution === "gaussian" ? 1 : 0],
    ["spec:likelihood:student-t", spec.advancedConfig.likelihoodDistribution === "student-t" ? 1 : 0],
    ["spec:likelihood:log-normal", spec.advancedConfig.likelihoodDistribution === "log-normal" ? 1 : 0],
    ["spec:search-phase:seed", spec.searchPhase === "seed" ? 1 : 0],
    ["spec:search-phase:response-coverage", spec.searchPhase === "response-coverage" ? 1 : 0],
    ["spec:ctv-adstock:weibull", ctv.adstockType === "weibull" ? 1 : 0],
    ["spec:ctv-adstock:geometric", ctv.adstockType === "geometric" ? 1 : 0],
    ["spec:ctv-normalization:sum", ctv.kernelNormalization === "sum" ? 1 : 0],
    ["spec:ctv-normalization:peak", ctv.kernelNormalization === "peak" ? 1 : 0],
  ];
}

function interaction(
  row: SviScoreV4CandidateRow,
  diagnostic: (typeof SVI_SCORE_V4_FEATURES)[number],
  multiplier: number,
): number {
  return multiplier * logDiagnostic(row, diagnostic);
}

/**
 * Candidate features are advertiser-visible diagnostics, observed temporal
 * context, and declared model settings only. Business id, generator family,
 * split, SVI loss, and injected truth are deliberately inaccessible here.
 */
export function v5aFeatureTerms(row: SviScoreV4CandidateRow): FeatureTerm[] {
  const spec = specification(row);
  const ctv = response(spec, "ctv_spend");
  const terms: FeatureTerm[] = SVI_SCORE_V4_FEATURES.map((feature) => ({
    name: `diagnostic:${feature}`,
    value: logDiagnostic(row, feature),
    base: true,
  }));

  DEFICIT_DIAGNOSTICS.forEach((feature) => {
    const value = normalizedDiagnostic(row, feature);
    [0.5, 0.7].forEach((threshold) => {
      terms.push({
        name: `deficit:${feature}:${Math.round(threshold * 100)}`,
        value: -Math.max(0, threshold - value),
        base: false,
      });
    });
  });

  const temporalContext = clamp(row.temporal.context);
  TEMPORAL_CONTEXT_DIAGNOSTICS.forEach((feature) => {
    terms.push({
      name: `context:temporal:${feature}`,
      value: interaction(row, feature, temporalContext),
      base: false,
    });
  });

  const binary = Object.fromEntries(binaryTerms(row, spec));
  Object.entries(binary).forEach(([name, value]) => {
    terms.push({ name, value, base: false });
    terms.push({
      name: `context:temporal:${name}`,
      value: temporalContext * value,
      base: false,
    });
  });

  const ctvLongMemory = ctv.adstockType === "geometric"
    ? clamp((ctv.adstock - 0.25) / 0.55)
    : clamp((ctv.weibullScale - 3) / 8);
  const highRidge = clamp((Math.log10(Math.max(spec.config.ridge, 0.01)) + 2) / 2.3);
  const highSaturation = clamp((Math.max(
    spec.config.saturation,
    ...Object.values(spec.config.channelResponses ?? {}).map(
      (item) => item.saturation ?? spec.config.saturation,
    ),
  ) - 0.5) / 2.5);
  const declaredInteractions: Array<[string, (typeof SVI_SCORE_V4_FEATURES)[number], number]> = [
    ["weibull:carryover", "carryover-support", binary["spec:global-adstock:weibull"]],
    ["weibull:whole-flight", "whole-flight-generalization", binary["spec:global-adstock:weibull"]],
    ["ctv-long:carryover", "carryover-support", ctvLongMemory],
    ["ctv-long:whole-flight", "whole-flight-generalization", ctvLongMemory],
    ["time-varying:new-data", "new-data-stability", binary["spec:time-varying:on"]],
    ["time-varying:fold", "fold-stability", binary["spec:time-varying:on"]],
    ["planning:confounder", "confounder-sensitivity", binary["spec:planning:on"]],
    ["planning:placebo", "future-media-placebo", binary["spec:planning:on"]],
    ["benchmark:compatibility", "evidence-compatibility", binary["spec:evidence:benchmark-gap-fill"]],
    ["benchmark:quality", "evidence-source-quality", binary["spec:evidence:benchmark-gap-fill"]],
    ["benchmark:dependence", "evidence-decision-dependence", binary["spec:evidence:benchmark-gap-fill"]],
    ["student-t:shape", "likelihood-shape", binary["spec:likelihood:student-t"]],
    ["student-t:variance", "variance-structure", binary["spec:likelihood:student-t"]],
    ["log-normal:shape", "likelihood-shape", binary["spec:likelihood:log-normal"]],
    ["ridge:collinearity", "multicollinearity", highRidge],
    ["ridge:identification", "roi-identification", highRidge],
    ["saturation:spend-regimes", "spend-regimes", highSaturation],
    ["saturation:roi-resolution", "roi-resolution", highSaturation],
  ];
  declaredInteractions.forEach(([name, feature, multiplier]) => {
    terms.push({
      name: `interaction:${name}`,
      value: interaction(row, feature, multiplier),
      base: false,
    });
  });
  return terms;
}

export function v5aFeatureNames(row: SviScoreV4CandidateRow): string[] {
  return v5aFeatureTerms(row).map((term) => term.name);
}

export function v5aFeatureVector(row: SviScoreV4CandidateRow): number[] {
  return v5aFeatureTerms(row).map((term) => term.value);
}

export function v5aPriorWeights(row: SviScoreV4CandidateRow): number[] {
  const terms = v5aFeatureTerms(row);
  const baseCount = terms.filter((term) => term.base).length;
  const extensionCount = Math.max(1, terms.length - baseCount);
  return terms.map((term) =>
    term.base ? 0.72 / Math.max(baseCount, 1) : 0.28 / extensionCount
  );
}
