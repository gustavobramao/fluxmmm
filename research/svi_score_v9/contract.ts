import { SVI_SCORE_V3_CONTRACT } from "../svi_score_v3/contract";
import {
  V8_DECISION_SCENARIOS,
  V8_RISK_PREFERENCE,
} from "../svi_score_v8/contract";

export const SVI_SCORE_V9_VERSION =
  "flux-svi-score-v9.0.0-setwise-distributional-development";

export const V9_DECISION_SCENARIOS = V8_DECISION_SCENARIOS;
export const V9_RISK_PREFERENCE = V8_RISK_PREFERENCE;

export const V9_POSTERIOR_CHANNEL_METRICS = [
  "log-roi-median",
  "log-roi-mean",
  "log-roi-low",
  "log-roi-high",
  "log-relative-roi-width",
  "log-roi-draw-sd",
  "implausible-probability",
  "near-zero-probability",
  "posterior-skewness",
  "location-shift-sd",
  "interval-overlap",
  "evidence-experiment",
  "evidence-benchmark",
  "response-adstock-mean",
  "response-adstock-sd",
  "response-log-weibull-scale-mean",
  "response-weibull-shape-mean",
  "response-saturation-mean",
  "response-saturation-sd",
  "response-half-saturation-mean",
  "response-weibull-share",
  "response-sum-normalization-share",
  "log-contribution-cv",
] as const;

export const V9_CHANNEL_AGGREGATIONS = ["mean", "sd", "min", "max"] as const;

export const V9_PAIRWISE_POSTERIOR_TOKENS = [
  "posterior:channel-pair:log-roi-correlation-mean-absolute",
  "posterior:channel-pair:log-roi-correlation-maximum-absolute",
  "posterior:channel-pair:log-contribution-correlation-mean-absolute",
  "posterior:channel-pair:log-contribution-correlation-maximum-absolute",
] as const;

export const V9_POSTERIOR_TOKEN_NAMES = [
  ...V9_POSTERIOR_CHANNEL_METRICS.flatMap((metric) =>
    V9_CHANNEL_AGGREGATIONS.map(
      (aggregation) => `posterior:channel-set:${metric}:${aggregation}`,
    )
  ),
  ...V9_PAIRWISE_POSTERIOR_TOKENS,
] as const;

export const SVI_SCORE_V9_CONTRACT = {
  purpose:
    "select one MMM candidate by learning the joint candidate set and the distribution of downstream economic loss",
  inference: {
    engine: "PyMC 6.2 FullRankADVI",
    contract: SVI_SCORE_V3_CONTRACT.inference,
    mapPosteriorTokensPermitted: false,
    nutsPermittedInResearch: false,
  },
  cohort: {
    developmentBusinesses: 420,
    candidatesPerBusiness: 48,
    developmentRows: 20_160,
    outerBusinessFolds: 5,
    innerBusinessFolds: 4,
    v8SealedAuditPermitted: false,
    newSealedAuditRequiredAfterDevelopment: true,
  },
  tokens: {
    baseCount: 136,
    posteriorDecisionCount: V9_POSTERIOR_TOKEN_NAMES.length,
    count: 136 + V9_POSTERIOR_TOKEN_NAMES.length,
    channelOrderInvariant: true,
    candidateOrderInvariant: true,
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
    scenarios: V9_DECISION_SCENARIOS,
    heads: [
      "mean-decision-loss",
      "median-decision-loss",
      "p90-decision-loss",
      "p95-decision-loss",
      "cvar90-decision-loss",
      "budget-reduction-decision-loss",
      "fixed-budget-mix-decision-loss",
      "budget-growth-decision-loss",
      "economic-ceiling-decision-loss",
      "dangerous-false-champion-probability",
    ],
    dangerousThreshold: 1,
    economicScale: 1,
  },
  model: {
    family: "small permutation-invariant DeepSets multi-head selector",
    candidateEncoder: "shared nonlinear encoder",
    businessContext: "mean-pooled candidate embedding",
    decoderInputs: "candidate, context, difference, and interaction embeddings",
    directSelectionObjective:
      "softmax expected capped normalized regret over all 48 candidates",
    theoremRegularizer:
      "all oracle-versus-candidate exact regret-weighted logistic comparisons",
  },
  evaluation: {
    groupedNestedCrossValidation: true,
    leaveOneGeneratorFamilyOut: true,
    riskCoverageRange: [0.6, 1],
    riskCoverageStep: 0.05,
    operationalCoverage: 0.7,
    comparators: ["V8 cross-fitted selector", "V5D-SVI"],
    requiredEndpoints: [
      "100%-coverage mean, P90, and P95 promoted excess loss",
      "matched-coverage mean, P90, and P95 promoted excess loss",
      "area under the 60%-to-100% selective-risk curve",
      "dangerous false-champion share",
      "oracle recall",
      "theorem violations",
    ],
  },
  governance: {
    activation: "development-only",
    v8AuditDereferenced: false,
    productionActivationPermitted: false,
    sotaClaimPermitted: false,
  },
} as const;
