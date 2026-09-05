import type { AuditSplit } from "../score_v3/types";

export const SVI_SCORE_V3_VERSION =
  "flux-svi-score-v3.1.0-unbiased-500-business-design";

export const SVI_SCORE_V3_SPLITS: Record<AuditSplit, number> = {
  train: 300,
  validation: 120,
  audit: 80,
};

export interface SviInferenceContract {
  method: "fullrank-advi";
  iterations: number;
  draws: number;
  primarySeeds: readonly [number, number];
  adjudicationSeed: number;
  learningRate: number;
  initialScale: number;
  gradientNorm: number;
  elboWindow: number;
  maximumElboDrift: number;
  maximumSeedLogRoiDifference: number;
}

export const SVI_SCORE_V3_CONTRACT = {
  designBusinesses: 500,
  candidateSpecifications: 24,
  evidenceArms: 2,
  candidatesPerBusiness: 48,
  targetSviFits: 24_000,
  splits: SVI_SCORE_V3_SPLITS,
  inference: {
    method: "fullrank-advi",
    iterations: 5_000,
    draws: 256,
    primarySeeds: [30_071, 81_119],
    adjudicationSeed: 190_081,
    learningRate: 0.001,
    initialScale: 0.01,
    gradientNorm: 10,
    elboWindow: 250,
    maximumElboDrift: 0.05,
    maximumSeedLogRoiDifference: 0.25,
  } satisfies SviInferenceContract,
  goldAudit: {
    businesses: 24,
    candidatesPerBusiness: 4,
    targetMcmcFits: 96,
    businessAllocation: {
      train: 12,
      validation: 6,
      audit: 6,
    } satisfies Record<AuditSplit, number>,
    strata: [
      "bayesian:experiments-only",
      "bayesian:benchmark-gap-fill",
      "advanced:experiments-only",
      "advanced:benchmark-gap-fill",
    ],
  },
  label:
    "Posterior-expected budget decision loss from every candidate; MAP may initialize inference but cannot select labels",
} as const;

