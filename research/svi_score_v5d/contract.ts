import { V5B_SEARCH_ALGORITHMS } from "../svi_score_v5b/contract";

export const SVI_SCORE_V5D_VERSION =
  "flux-svi-score-v5d.0.1-direct-economic-loss-ranker";

export interface V5DTrainingConfiguration {
  id: string;
  epochs: number;
  learningRate: number;
  regularization: number;
  huberDelta: number;
  quantile: number;
  pairwiseWeight: number;
  tailRowWeight: number;
  worstFamilyRowWeight: number;
  riskAversion: number;
  economicScale: number;
}

export const V5D_TRAINING_CONFIGURATION: V5DTrainingConfiguration = {
  id: "direct-loss-ranking-tail",
  epochs: 28,
  learningRate: 0.055,
  regularization: 0.012,
  huberDelta: 2.5,
  quantile: 0.9,
  pairwiseWeight: 0.55,
  tailRowWeight: 0.8,
  worstFamilyRowWeight: 0.7,
  riskAversion: 0.35,
  economicScale: 1,
};

export const SVI_SCORE_V5D_CONTRACT = {
  purpose:
    "replace proxy-score optimization with direct out-of-fold prediction of economic loss and tail risk",
  development: {
    businesses: 420,
    composition: "300 original training plus 120 previously opened validation businesses, all now treated as development",
    candidatesPerBusiness: 48,
    outerBusinessFolds: 5,
    candidateRegions: 4,
    sealedAuditBusinesses: 0,
  },
  features: {
    source: "v5c-observable-diagnostics-and-predeclared-model-settings",
    forbidden: [
      "business-id",
      "generator-family",
      "split-label",
      "candidate-id",
      "proposal-method",
      "search-phase",
      "restart",
      "synthetic-truth",
      "decision-loss",
      "roi-truth-error",
      "contribution-truth-error",
    ],
  },
  target: {
    mean:
      "uncapped within-business excess economic loss relative to the safe oracle",
    tail: "conditional P90 of uncapped excess economic loss",
    ranking:
      "economically weighted hard-negative oracle-versus-candidate misranking",
    economicScale: V5D_TRAINING_CONFIGURATION.economicScale,
    capped: false,
  },
  crossFitting: {
    outerAxis: "unseen-business",
    innerAxis: "unseen-candidate-region",
    assessmentPrediction:
      "each candidate is predicted only by the model that excluded its parameter region",
  },
  selection: {
    formula: "predicted-mean-loss + risk-aversion * (predicted-p90-loss - predicted-mean-loss)",
    orderIndependent: true,
    sequentialIncumbentPermitted: false,
    safetyPool: "v4.2-roi-safety-only",
  },
  search: {
    algorithms: V5B_SEARCH_ALGORITHMS,
    primaryEvaluationsPerBusiness: 16,
    checkpoints: [4, 8, 12, 16, 24, 32, 48],
    equalCandidateEvaluationCost: true,
    duplicateEvaluationsPermitted: false,
  },
  acceptance: {
    adaptivePrimaryNonInferiorityMargin: 0.02,
    fullPoolAlgorithmAgreementRequired: true,
    fullPoolVersusBestEarlierCheckpointMargin: 0.02,
    globalOptimumClaimPermitted: false,
  },
  governance: {
    researchOnly: true,
    priorArtifactsMayChange: false,
    priorOpenedValidationIsDevelopment: true,
    freshValidationMayOpen: false,
    auditMayOpen: false,
  },
} as const;
