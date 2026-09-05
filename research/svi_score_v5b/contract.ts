export const SVI_SCORE_V5B_VERSION =
  "flux-svi-score-v5b.0.1-common-universe-equal-budget-search";

export const V5B_SEARCH_ALGORITHMS = [
  "fixed-coverage",
  "uniform-random",
  "maximin-space-filling",
  "gp-ucb",
  "hybrid-global-local",
] as const;

export type V5BSearchAlgorithm = (typeof V5B_SEARCH_ALGORITHMS)[number];

export const SVI_SCORE_V5B_CONTRACT = {
  purpose:
    "search-only comparison with the V5A selector and V4.2 safety pool held fixed",
  development: {
    businesses: 420,
    candidatesPerBusiness: 48,
    candidateSpecifications: 24,
    evidenceArms: 2,
    groupedFolds: 5,
    groupingUnit: "business",
    sealedAuditBusinesses: 0,
  },
  informationTiming: {
    visibleBeforeEvaluation: [
      "declared-model-parameters",
      "evidence-arm",
      "candidate-identity",
    ],
    visibleAfterEvaluation: [
      "v5a-selector-score",
      "v4.2-roi-safety-status",
    ],
    hiddenUntilAssessment: [
      "svi-decision-loss",
      "synthetic-truth",
      "roi-truth-error",
      "contribution-truth-error",
      "generator-family",
    ],
  },
  selector: {
    version: "flux-svi-score-v5a.0.1-common-pool-monotonic-interactions",
    fitting: "out-of-fold-by-business",
    configuration: "frozen-v5a-selected-configuration",
    mayChangeDuringSearch: false,
  },
  candidatePool: {
    policy: "v4.2-roi-safety-only",
    temporalIdentificationIsGate: false,
    roiPlausibilityThreshold: 40,
    roiDecisionStabilityThreshold: 40,
  },
  compute: {
    primaryEvaluationsPerBusiness: 16,
    checkpoints: [4, 8, 12, 16, 24, 32, 48],
    costUnit: "one-candidate-model-plus-validation-evaluation",
    earlyStoppingPermitted: false,
    duplicateEvaluationsPermitted: false,
  },
  algorithms: V5B_SEARCH_ALGORITHMS,
  adaptivePolicy: {
    gpInitialDesign: 6,
    hybridInitialDesign: 8,
    gpLengthScale: 0.42,
    gpNoise: 0.0001,
    gpExploration: 0.7,
    unsafeUtilityPenalty: 0.25,
    hybridLocalFrequency: 3,
  },
  evaluation: {
    primaryEndpoint:
      "mean-loss-plus-0.5-cvar90-loss-plus-0.5-worst-family-mean-loss-at-16-evaluations",
    secondaryEndpoints: [
      "mean-loss",
      "p90-loss",
      "p95-loss",
      "cvar90-loss",
      "worst-family-mean-loss",
      "full-pool-selector-champion-recall",
      "economic-oracle-recall",
      "anytime-loss-curve",
    ],
  },
  governance: {
    researchOnly: true,
    v5aArtifactMayChange: false,
    auditMayOpen: false,
    freshValidationMayOpen: false,
    limitation:
      "V5B compares search efficiency inside the existing 48-candidate universe; it does not establish a global optimum over the continuous parameter space.",
  },
} as const;
