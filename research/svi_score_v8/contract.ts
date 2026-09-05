import { SVI_SCORE_V3_CONTRACT } from "../svi_score_v3/contract";

export const SVI_SCORE_V8_VERSION =
  "flux-svi-score-v8.0.0-selection-aware-development";

export const V8_DECISION_SCENARIOS = [
  "budget-reduction",
  "fixed-budget-mix",
  "budget-growth",
  "economic-ceiling",
] as const;

export const V8_RISK_PREFERENCE = {
  mean: 0.65,
  p90: 0.35,
} as const;

export const V8_DANGEROUS_EXCESS_LOSS = 1;

export const SVI_SCORE_V8_CONTRACT = {
  purpose:
    "learn selection-aware economic risk from truth-blind SVI posterior tokens and choose one candidate per unseen advertiser",
  inference: {
    engine: "PyMC 6.2 FullRankADVI",
    contract: SVI_SCORE_V3_CONTRACT.inference,
    mapPosteriorTokensPermitted: false,
    nutsPermittedInResearch: false,
    statement:
      "V8 uses scalable variational posterior inference during research and candidate evaluation. Its posterior-token and decision-risk framework is inference-engine agnostic in principle and can be applied to NUTS posterior draws.",
  },
  cohort: {
    developmentBusinesses: 420,
    sealedAuditBusinesses: 80,
    candidatesPerBusiness: 48,
    developmentRows: 20_160,
    outerBusinessFolds: 5,
    innerBusinessFolds: 4,
    auditMayOpenDuringDevelopment: false,
  },
  tokens: {
    count: 136,
    source:
      "V5D observable validation, temporal, specification, interaction, and SVI posterior tokens",
    hiddenTruthPermitted: false,
    forbidden: [
      "business-id",
      "generator-family",
      "split-label",
      "candidate-id",
      "proposal-method",
      "search-phase",
      "synthetic-truth",
      "decision-loss",
      "scenario-decision-loss",
      "roi-truth-error",
      "contribution-truth-error",
    ],
  },
  targets: {
    scenarios: V8_DECISION_SCENARIOS,
    scenarioSource:
      "one posterior-expected action per scenario evaluated against hidden synthetic truth after action selection",
    economicScale: 1,
    normalization:
      "min(max(candidate-risk-minus-valid-oracle-risk,0)/business-economic-scale,1)",
    outputs: [
      "mean-scenario-excess-loss",
      "p90-scenario-excess-loss",
      "four-scenario-excess-losses",
      "dangerous-false-champion-probability",
    ],
    uncappedLossReportedSeparately: true,
  },
  objective: {
    components: [
      "mean-huber",
      "p90-pinball",
      "four-scenario-huber",
      "dangerous-champion-binary-cross-entropy",
      "all-oracle-versus-candidate-normalized-regret-weighted-logistic-ranking",
    ],
    theoremTarget: "capped-normalized-selected-model-excess-regret",
    pairSamplingPermitted: false,
  },
  learners: [
    "regularized-linear-multi-head",
    "regularized-additive-hinge-multi-head",
    "controlled-depth-gradient-boosted-multi-head",
  ],
  selection: {
    riskPreference: V8_RISK_PREFERENCE,
    formula:
      "predicted-risk + uncertainty-penalty * ensemble-disagreement + danger-penalty * predicted-danger-probability",
    orderIndependent: true,
    abstentionPermitted: true,
    abstentionMustReportCoverageAndPolicyLoss: true,
  },
  validity: {
    hardGates: [
      "development-only-cohort",
      "finite-SVI-posterior",
      "complete-token-contract",
      "complete-scenario-label-contract",
    ],
    softRiskSignals: [
      "ELBO stability",
      "seed agreement",
      "predictive coverage",
      "identification",
      "evidence availability",
      "evidence conflict",
      "ROI plausibility",
    ],
  },
  governance: {
    activation: "development-only",
    sealedAuditAccessed: false,
    productionActivationPermitted: false,
    sotaClaimPermitted: false,
    renamePermittedOnlyAfterEvaluation: true,
  },
} as const;
