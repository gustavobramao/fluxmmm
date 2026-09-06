import { V9_DECISION_SCENARIOS, V9_RISK_PREFERENCE } from "../svi_score_v9/contract";

export const SVI_SCORE_V11_EA_VERSION =
  "flux-svi-score-v11.0.0-evidence-adaptive-regretset-development";

export const V11_EA_DECISION_SCENARIOS = V9_DECISION_SCENARIOS;
export const V11_EA_RISK_PREFERENCE = V9_RISK_PREFERENCE;

export const V11_EA_REGIMES = [
  "full-candidate-set",
  "experiments-only",
  "benchmark-gap-fill",
] as const;

export type V11EaRegime = (typeof V11_EA_REGIMES)[number];

export const V11_EA_SUMMARY_METRICS = [
  "diagnostic:rolling-oos",
  "diagnostic:predictive-coverage",
  "diagnostic:fold-stability",
  "diagnostic:spend-regimes",
  "diagnostic:multicollinearity",
  "diagnostic:roi-identification",
  "diagnostic:evidence-source-quality",
  "diagnostic:evidence-compatibility",
  "diagnostic:evidence-decision-dependence",
  "diagnostic:roi-posterior-plausibility",
  "diagnostic:roi-decision-stability",
] as const;

export const V11_EA_PAIRED_METRICS = [
  "diagnostic:rolling-oos",
  "diagnostic:predictive-coverage",
  "diagnostic:roi-identification",
  "diagnostic:evidence-source-quality",
  "diagnostic:evidence-compatibility",
  "diagnostic:evidence-decision-dependence",
  "diagnostic:roi-posterior-plausibility",
  "diagnostic:roi-decision-stability",
] as const;

export const V11_EA_CONTEXT_NAMES = [
  "evidence:experiment:available-mask",
  "evidence:experiment:channel-coverage",
  "evidence:benchmark:available-mask",
  "evidence:benchmark:channel-coverage",
  "evidence:external:any-available-mask",
  "evidence:external:channel-coverage",
  "evidence:source-quality:observed-mask",
  "evidence:compatibility:observed-mask",
  "evidence:decision-dependence:observed-mask",
  "candidate-set:regime:full",
  "candidate-set:regime:experiments-only",
  "candidate-set:regime:benchmark-gap-fill",
  "candidate-set:relative-size",
  ...V11_EA_SUMMARY_METRICS.flatMap((name) => [
    `candidate-set:${name}:mean`,
    `candidate-set:${name}:sd`,
    `candidate-set:${name}:best`,
  ]),
  "paired-evidence-arm:available-mask",
  ...V11_EA_PAIRED_METRICS.flatMap((name) => [
    `paired-evidence-arm:benchmark-minus-experiment:${name}:mean`,
    `paired-evidence-arm:benchmark-minus-experiment:${name}:sd`,
  ]),
] as const;

export type V11EaExpert =
  | "predictive-generalization"
  | "causal-identification"
  | "posterior-decision"
  | "structural-specification";

export const V11_EA_EXPERTS: readonly V11EaExpert[] = [
  "predictive-generalization",
  "causal-identification",
  "posterior-decision",
  "structural-specification",
] as const;

/**
 * Assign every truth-blind candidate token to exactly one interpretable expert.
 * The fallback is deliberate: specification and residual diagnostics remain
 * available even when a new token is added to an upstream frozen registry.
 */
export function v11EaExpertForToken(name: string): V11EaExpert {
  if (
    name.includes("rolling-oos") ||
    name.includes("predictive-coverage") ||
    name.includes("fold-stability") ||
    name.includes("spend-regimes") ||
    name.includes("whole-flight-generalization") ||
    name.includes("carryover-support") ||
    name.includes("post-flight-residual-stability") ||
    name.includes("kernel-distinguishability") ||
    name.startsWith("context:temporal")
  ) {
    return "predictive-generalization";
  }
  if (
    name.includes("anchor-recovery") ||
    name.includes("confounder-sensitivity") ||
    name.includes("future-media-placebo") ||
    name.includes("roi-identification") ||
    name.includes("evidence-source-quality") ||
    name.includes("evidence-compatibility") ||
    name.includes("evidence-decision-dependence") ||
    name.includes("roi-economic-consistency") ||
    name.startsWith("spec:evidence") ||
    name.startsWith("spec:calibration") ||
    name.startsWith("interaction:benchmark") ||
    name.startsWith("interaction:planning") ||
    name.includes("evidence-experiment") ||
    name.includes("evidence-benchmark")
  ) {
    return "causal-identification";
  }
  if (
    name.includes("roi-posterior-plausibility") ||
    name.includes("roi-decision-stability") ||
    name.includes("roi-resolution") ||
    name.startsWith("svi:decision") ||
    name.startsWith("svi:margin:roi") ||
    name.startsWith("svi:roi") ||
    name.startsWith("posterior:channel-set:log-roi") ||
    name.includes("implausible-probability") ||
    name.includes("near-zero-probability") ||
    name.includes("posterior-skewness") ||
    name.includes("location-shift-sd") ||
    name.includes("interval-overlap") ||
    name.includes("log-contribution-cv") ||
    name.startsWith("posterior:channel-pair")
  ) {
    return "posterior-decision";
  }
  return "structural-specification";
}

export const SVI_SCORE_V11_EA_CONTRACT = {
  purpose:
    "learn economic decision risk with evidence-conditional reliance on predictive, identification, posterior, and specification signals",
  status: "post-AMSS development; no confirmatory cohort accessed",
  source: {
    developmentBusinesses: 420,
    candidateMmmFits: 20_160,
    posteriorRefitsRequired: 0,
    frozenV9CandidateTokens: 232,
    evidenceArms: ["experiments-only", "benchmark-gap-fill"],
  },
  pairedRegimes: {
    setsPerBusiness: V11_EA_REGIMES.length,
    totalDevelopmentSets: 420 * V11_EA_REGIMES.length,
    fullCandidates: 48,
    armCandidates: 24,
    sameBusinessFoldRequired: true,
    sameObservedOutcomeRequired: true,
    recalculatesOracleWithinRegime: true,
    experimentPresenceCounterfactualGenerated: false,
    note:
      "The cached design identifies benchmark gap-fill value within business. Removing an existing experiment would change the posterior and requires a future refit experiment.",
  },
  context: {
    count: V11_EA_CONTEXT_NAMES.length,
    explicitMissingnessMasks: true,
    candidateIndependentWithinSet: true,
    hiddenTruthPermitted: false,
  },
  experts: V11_EA_EXPERTS,
  model: {
    family:
      "joint-plus-evidence-gated permutation-invariant DeepSets multi-head selector",
    jointPath:
      "an all-token candidate encoder preserves cross-pillar interactions",
    gate:
      "business evidence context -> softmax reliance over four residual candidate-token experts",
    selection:
      "direct 65% predicted mean plus 35% predicted P90 economic risk",
    training:
      "nested advertiser-grouped cross-validation; paired regimes never cross folds",
  },
  targets: {
    scenarios: V11_EA_DECISION_SCENARIOS,
    riskPreference: V11_EA_RISK_PREFERENCE,
    theoremTarget: "capped normalized within-regime excess economic loss",
  },
  governance: {
    activation: "development-only",
    amssConfirmatoryCohortAccessPermitted: false,
    previousAuditRetuningPermitted: false,
    freshIndependentAuditRequiredBeforeClaim: true,
    productionActivationPermitted: false,
  },
} as const;
