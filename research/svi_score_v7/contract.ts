export const SVI_SCORE_V7_VERSION =
  "flux-svi-score-v7.0.1-multitask-economic-surrogate";

export const SVI_SCORE_V7_CONTRACT = {
  purpose:
    "test whether a nonlinear multi-task model predicts out-of-fold economic loss better than V5D",
  development: {
    businesses: 420,
    candidatesPerBusiness: 48,
    composition:
      "300 original training plus 120 previously opened validation businesses, all treated as development",
    outerBusinessFolds: 5,
    candidateRegions: 4,
  },
  inputs: {
    source: "the exact 115 observable V5D features",
    channelTokensAvailable: false,
    hiddenTruthAvailableToInputs: false,
    forbidden: [
      "business-id",
      "generator-family",
      "split-label",
      "candidate-id",
      "decision-loss",
      "roi-truth-error",
      "contribution-truth-error",
      "synthetic-truth",
    ],
  },
  model: {
    family: "deterministic-multitask-mlp",
    hiddenLayers: [48, 24],
    activation: "silu",
    epochs: 24,
    batchSize: 512,
    learningRate: 0.003,
    weightDecay: 0.001,
    outputs: [
      "mean-excess-economic-loss",
      "p90-excess-economic-loss",
      "roi-recovery-error",
      "contribution-recovery-error",
    ],
    lossWeights: {
      meanHuber: 1,
      p90Pinball: 0.5,
      roiHuber: 0.15,
      contributionHuber: 0.15,
      hardNegativeRanking: 0.45,
    },
    p90Quantile: 0.9,
    riskPreference: {
      mean: 0.65,
      p90: 0.35,
      interpretation: "subjective business risk preference, not learned scientific truth",
    },
  },
  crossFitting: {
    outerAxis: "unseen-business",
    innerAxis: "unseen-candidate-region",
    models: 20,
    earlyStopping: false,
    hyperparameterTuningOnAssessmentFolds: false,
  },
  evaluation: {
    comparator: "V5D under identical folds, candidate regions, safety pool, and search traces",
    orderIndependentFinalSelection: true,
    freshValidation: false,
    sealedAudit: false,
    sotaClaimPermitted: false,
    developmentImprovementChecks: [
      "higher candidate-level loss correlation than V5D",
      "higher economically weighted pairwise accuracy than V5D",
      "lower full-pool economic objective than V5D",
      "lower best-at-16 economic objective than V5D",
      "P90 coverage between 87% and 93%",
      "all exhaustive search algorithms agree",
    ],
  },
  governance: {
    researchOnly: true,
    priorOpenedValidationIsDevelopment: true,
    freshValidationMayOpen: false,
    auditMayOpen: false,
  },
} as const;
