import { V5B_SEARCH_ALGORITHMS } from "../svi_score_v5b/contract";

export const SVI_SCORE_V5C_VERSION =
  "flux-svi-score-v5c.0.1-search-robust-two-axis-crossfit";

export interface V5CTrainingConfiguration {
  id: string;
  regularization: number;
  epochs: number;
  cvarWeight: number;
  familyDroWeight: number;
  featureCap: number;
  learningRate: number;
  economicScale: number;
}

export const V5C_TRAINING_CONFIGURATION: V5CTrainingConfiguration = {
  id: "uncapped-search-robust",
  regularization: 0.06,
  epochs: 160,
  cvarWeight: 0.8,
  familyDroWeight: 0.85,
  featureCap: 0.06,
  learningRate: 0.1,
  economicScale: 1,
};

export const SVI_SCORE_V5C_CONTRACT = {
  purpose:
    "correct selector optimization pressure before any expanded parameter search",
  development: {
    businesses: 420,
    candidatesPerBusiness: 48,
    outerBusinessFolds: 5,
    candidateRegions: 4,
    uncertaintyMembersPerRegion: 3,
    sealedAuditBusinesses: 0,
  },
  featureContract: {
    source: "v5a-monotonic-basis-minus-proposal-metadata",
    forbidden: [
      "candidate-id",
      "proposal-method",
      "search-phase",
      "restart",
      "generator-family",
      "split-label",
      "synthetic-truth",
      "decision-loss",
      "roi-truth-error",
      "contribution-truth-error",
    ],
    proposalInvariant: true,
  },
  target: {
    name: "uncapped-normalized-within-business-excess-economic-loss",
    formula: "max(0, candidate-loss - safe-oracle-loss) / declared-economic-scale",
    capped: false,
    economicScale: V5C_TRAINING_CONFIGURATION.economicScale,
    tailProtection: "CVaR90 hard-negative emphasis",
    familyProtection: "worst-generator-family emphasis",
  },
  crossFitting: {
    outerAxis: "unseen-business",
    innerAxis: "unseen-candidate-region",
    regionConstruction:
      "balanced round-robin assignment over a maximin parameter-space traversal; evidence arms remain paired",
    assessmentPrediction:
      "each candidate is scored only by models that excluded its parameter region",
  },
  uncertainty: {
    source: "three business-subfold selector fits excluding the candidate region",
    robustScoreStandardErrors: 1,
    promotionStandardErrors: 1,
    incumbentRule:
      "replace only when challenger mean minus incumbent mean exceeds the joint uncertainty margin; safe candidates dominate unsafe candidates",
  },
  search: {
    algorithms: V5B_SEARCH_ALGORITHMS,
    primaryEvaluationsPerBusiness: 16,
    checkpoints: [4, 8, 12, 16, 24, 32, 48],
    earlyStoppingPermitted: false,
    duplicateEvaluationsPermitted: false,
    adaptiveSearchRunsInsideOuterAssessment: true,
  },
  safety: {
    policy: "v4.2-roi-safety-only",
    temporalIdentificationIsGate: false,
    roiPlausibilityThreshold: 40,
    roiDecisionStabilityThreshold: 40,
  },
  acceptance: {
    adaptivePrimaryNonInferiorityMargin: 0.02,
    fullPoolVersusBestEarlierCheckpointMargin: 0.02,
    requireNoAuditAccess: true,
    globalOptimumClaimPermitted: false,
  },
  governance: {
    researchOnly: true,
    v5aAndV5bArtifactsMayChange: false,
    freshValidationMayOpen: false,
    auditMayOpen: false,
  },
} as const;
