export const SVI_SCORE_V4_VERSION =
  "flux-svi-score-v4.2.0-top-one-risk-temporal-diagnostic";

/**
 * V4 is a new research method. The failed V3 validation result and its sealed
 * audit are immutable inputs to the scientific history, never tuning data.
 */
export const SVI_SCORE_V4_CONTRACT = {
  development: {
    businesses: 420,
    sources: {
      originalTraining: 300,
      retiredV3Validation: 120,
      sealedV3Audit: 0,
    },
    groupedFolds: 5,
    groupingUnit: "business",
    stratification: "generator-family",
  },
  target: {
    primary: "posterior-expected-budget-decision-loss",
    theorem: "capped-normalized-top-one-excess-regret",
    economicScale: 1,
    cvarProbability: 0.9,
    configurationSelectionObjective: {
      meanDecisionLoss: 1,
      cvar90DecisionLoss: 0.5,
      worstFamilyMeanDecisionLoss: 0.5,
    },
    reportUncappedP90: true,
    reportUncappedP95: true,
  },
  model: {
    family: "two-expert-monotonic-linear-ranker",
    context: "truth-blind-observed-temporal-complexity",
    experts: ["ordinary-response", "temporally-complex"] as const,
    featureFloor: 0.01,
    featureCap: 0.18,
  },
  safety: {
    behavior: "non-compensatory-when-a-safe-candidate-exists",
    diagnostics: [
      "roi-posterior-plausibility",
      "roi-decision-stability",
    ] as const,
    noSafeCandidateBehavior: "return-review-only-selection",
  },
  governance: {
    frozenV3FilesMayChange: false,
    v3AuditMayOpen: false,
    validationRule:
      "generate-and-freeze-a-fresh-independent-cohort-after-development-freeze",
  },
} as const;

export interface SviScoreV4TrainingConfiguration {
  id: string;
  regularization: number;
  epochs: number;
  cvarWeight: number;
  familyDroWeight: number;
  safety: {
    roiPlausibility: number;
    decisionStability: number;
  };
}

export const PREDECLARED_SCORE_V4_TUNING_GRID: readonly SviScoreV4TrainingConfiguration[] = [
  {
    id: "balanced-risk",
    regularization: 0.01,
    epochs: 260,
    cvarWeight: 0.5,
    familyDroWeight: 0.5,
    safety: {
      roiPlausibility: 40,
      decisionStability: 40,
    },
  },
  {
    id: "tail-protective",
    regularization: 0.01,
    epochs: 260,
    cvarWeight: 0.85,
    familyDroWeight: 0.4,
    safety: {
      roiPlausibility: 40,
      decisionStability: 40,
    },
  },
  {
    id: "family-robust",
    regularization: 0.01,
    epochs: 260,
    cvarWeight: 0.4,
    familyDroWeight: 0.85,
    safety: {
      roiPlausibility: 40,
      decisionStability: 40,
    },
  },
  {
    id: "conservative-balanced",
    regularization: 0.02,
    epochs: 300,
    cvarWeight: 0.6,
    familyDroWeight: 0.6,
    safety: {
      roiPlausibility: 50,
      decisionStability: 50,
    },
  },
] as const;
