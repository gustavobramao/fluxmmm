import type { SviScoreV4TrainingConfiguration } from "../svi_score_v4/contract";

export const SVI_SCORE_V5A_VERSION =
  "flux-svi-score-v5a.0.1-common-pool-monotonic-interactions";

export const SVI_SCORE_V5A_CONTRACT = {
  purpose:
    "selector-only comparison on the unchanged V4.2 420-business/48-candidate pool",
  development: {
    businesses: 420,
    candidatesPerBusiness: 48,
    groupedFolds: 5,
    groupingUnit: "business",
    stratification: "generator-family",
    sealedAuditBusinesses: 0,
  },
  labels: {
    inference: "stored-svi-posterior-decision-surfaces",
    target: "capped-normalized-within-business-excess-decision-loss",
    syntheticTruthAvailableToFeatures: false,
  },
  candidatePool: {
    policy: "v4.2-roi-safety-only",
    temporalIdentificationIsGate: false,
    roiPlausibilityThreshold: 40,
    roiDecisionStabilityThreshold: 40,
  },
  model: {
    family: "monotonic-top-one-spline-interaction-ranker",
    nonnegativeWeights: true,
    nonlinearities: [
      "diagnostic-deficit-hinges",
      "observed-temporal-context-interactions",
      "predeclared-specification-diagnostic-interactions",
    ],
    forbiddenFeatures: [
      "business-id",
      "generator-family",
      "split-label",
      "synthetic-truth",
      "decision-loss",
      "roi-truth-error",
      "contribution-truth-error",
    ],
  },
  evaluation: {
    selectionObjective:
      "mean-loss-plus-0.5-cvar90-loss-plus-0.5-worst-family-mean-loss",
    primaryComparator: "v4.2-out-of-fold-common-pool",
    historicalComparators: ["active-v6", "v4.0", "v4.1"],
  },
  governance: {
    researchOnly: true,
    earlierArtifactsMayChange: false,
    auditMayOpen: false,
    nextStep:
      "freeze selector before any expanded search or fresh validation cohort",
  },
} as const;

export interface SviScoreV5ATrainingConfiguration {
  id: string;
  regularization: number;
  epochs: number;
  cvarWeight: number;
  familyDroWeight: number;
  featureCap: number;
  learningRate: number;
}

export const V5A_COMMON_POOL_CONFIGURATION: SviScoreV4TrainingConfiguration = {
  id: "v5a-common-pool",
  regularization: 0,
  epochs: 0,
  cvarWeight: 0,
  familyDroWeight: 0,
  safety: {
    roiPlausibility: 40,
    decisionStability: 40,
  },
};

export const PREDECLARED_SCORE_V5A_TUNING_GRID:
  readonly SviScoreV5ATrainingConfiguration[] = [
    {
      id: "interaction-balanced",
      regularization: 0.02,
      epochs: 110,
      cvarWeight: 0.4,
      familyDroWeight: 0.85,
      featureCap: 0.08,
      learningRate: 0.14,
    },
    {
      id: "interaction-regularized",
      regularization: 0.06,
      epochs: 140,
      cvarWeight: 0.4,
      familyDroWeight: 0.85,
      featureCap: 0.06,
      learningRate: 0.12,
    },
    {
      id: "interaction-tail-protective",
      regularization: 0.03,
      epochs: 125,
      cvarWeight: 0.8,
      familyDroWeight: 0.85,
      featureCap: 0.07,
      learningRate: 0.12,
    },
  ] as const;
