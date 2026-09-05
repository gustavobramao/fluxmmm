export const V10_AMSS_VERSION =
  "flux-svi-score-v10-amss.0.1.1-external-transport";

export const V10_AMSS_CHANNELS = [
  "meta_acquisition_spend",
  "google_search_nonbrand_spend",
  "ctv_spend",
] as const;

export const V10_AMSS_EVIDENCE_GROUPS = [
  "paid-social-experiment",
  "search-experiment",
  "tv-experiment",
  "no-experiment",
] as const;

export const V10_AMSS_MODULE_ORDERS = [
  "paid_social_proxy|search|tv",
  "paid_social_proxy|tv|search",
  "search|paid_social_proxy|tv",
  "search|tv|paid_social_proxy",
  "tv|paid_social_proxy|search",
  "tv|search|paid_social_proxy",
] as const;

export const V10_AMSS_DECISION_SCENARIOS = [
  {
    id: "budget-reduction",
    label: "Budget reduction",
    minimumTotalShare: 0.65,
    maximumTotalShare: 1,
  },
  {
    id: "fixed-budget-mix",
    label: "Fixed-budget reallocation",
    minimumTotalShare: 1,
    maximumTotalShare: 1,
  },
  {
    id: "budget-growth",
    label: "Budget growth",
    minimumTotalShare: 1,
    maximumTotalShare: 1.5,
  },
  {
    id: "economic-ceiling",
    label: "Economic ceiling",
    minimumTotalShare: 0.5,
    maximumTotalShare: 1.8,
  },
] as const;

export const V10_AMSS_CONTRACT = {
  version: V10_AMSS_VERSION,
  purpose:
    "one-time external-transport validation of the frozen V9 RegretSet-MMM selector in Google AMSS",
  cohort: {
    businesses: 100,
    candidateSpecifications: 24,
    evidenceArms: ["experiments-only", "benchmark-gap-fill"],
    candidatesPerBusiness: 48,
    candidateFits: 4_800,
    design: "deterministic 100-point Latin-hypercube-style marginal design",
    businessSeedStart: 710_001,
    truthSeedStart: 910_001,
    moduleOrders: V10_AMSS_MODULE_ORDERS,
    moduleOrderAllocation:
      "all six AMSS media-module permutations; four receive 17 businesses and two receive 16; every evidence group receives 4-5 businesses per order",
    evidenceGroups: V10_AMSS_EVIDENCE_GROUPS,
    evidenceAllocation: "25 businesses per evidence group",
  },
  channelAdapter: {
    rawToFlux: {
      paid_social_proxy: "meta_acquisition_spend",
      search: "google_search_nonbrand_spend",
      tv: "ctv_spend",
    },
    paidSocialClaim:
      "AMSS-generated exposure-media stress proxy using the unchanged DefaultTraditionalMediaModule",
    searchClaim:
      "AMSS native search mechanism mapped to the nonbrand-search archetype",
    tvClaim:
      "AMSS traditional reach mechanism mapped to the long-memory CTV archetype",
  },
  time: {
    cadence: "weekly",
    totalWeeks: 268,
    burnInWeeks: 52,
    visibleStartWeek: 53,
    visibleEndWeek: 208,
    visibleWeeks: 156,
    futureActionStartWeek: 209,
    futureActionEndWeek: 260,
    futureActionWeeks: 52,
    postActionCarryoverEndWeek: 268,
    postActionCarryoverWeeks: 8,
    experimentStartWeek: 170,
    experimentEndWeek: 195,
    experimentOutcomeEndWeek: 202,
  },
  experiments: {
    availability: {
      "paid-social-experiment": "meta_acquisition_spend",
      "search-experiment": "google_search_nonbrand_spend",
      "tv-experiment": "ctv_spend",
      "no-experiment": null,
    },
    estimand:
      "incremental revenue divided by incremental realized spend over weeks 170-195, with outcomes through week 202",
    construction:
      "paired AMSS treatment and 20%-lower-budget control paths from the same week-169 state using common random-number seeds",
    replicates: 4,
    relativeStandardErrorFloor: 0.15,
    absoluteRoiStandardErrorFloor: 0.05,
    reportedRoiFloor: 0.05,
    multiplicativeBiasRange: [-0.1, 0.1],
    candidateVisible: true,
    hiddenEconomicDecisionTruth: false,
  },
  evidenceArms: {
    "experiments-only":
      "use the predeclared period-matched AMSS experiment when assigned; use no industry benchmark",
    "benchmark-gap-fill":
      "use the same experiment when assigned and activate Flux's frozen built-in tactic benchmark only for channels without an experiment",
    benchmarkTruthOverridePermitted: false,
  },
  decisions: {
    scenarios: V10_AMSS_DECISION_SCENARIOS,
    baseline:
      "realized year-4 annual media spend, visible as business decision metadata",
    amssBudgetTranslation:
      "candidate intended spend divided by the channel's year-4 realized-spend-to-budget utilization, bounded to 5%-200%, yields the AMSS budget cap",
    totalBudgetGridPoints: 21,
    allocationSimplexUnits: 20,
    perChannelMinimumShareOfBaseline: 0,
    perChannelMaximumShareOfBaseline: 2.5,
    actionRoundingRelativeToBaseline: 0.005,
    posteriorDecisionDraws: 64,
    posteriorResponseSurfacePoints: 201,
    candidateAction:
      "maximize posterior-expected incremental contribution times declared gross margin minus incremental intended media spend",
    counterfactualTruth:
      "mean AMSS profit during weeks 209-268 under the frozen action, using four common-random-number replicates",
    truthReplicates: 4,
    oracle:
      "best realized action among the 48 truth-blind candidate actions for the same business and decision scenario",
    economicLoss:
      "(oracle profit minus candidate profit) divided by max(abs(oracle profit minus baseline profit), 2% of baseline annual budget, 1)",
    reportedRisk: "65% mean scenario loss plus 35% P90 scenario loss",
    uncappedLossReported: true,
    normalizedLossCap: 1,
  },
  selection: {
    selector: "frozen V9 DeepSets ensemble",
    retrainingPermitted: false,
    tokens: 232,
    riskPreference: { mean: 0.65, p90: 0.35 },
    uncertaintyPenalty: 0.25,
    dangerPenalty: 0,
    comparators: [
      "highest original Flux heuristic score among valid candidates",
      "uniform random valid candidate expectation",
      "in-pool oracle among the 48 frozen candidate actions",
    ],
  },
  inference: {
    engine: "PyMC 6.2 FullRankADVI",
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
    nutsPermitted: false,
  },
  evaluation: {
    unit: "business",
    primary:
      "V9 full-coverage realized excess economic loss across the four declared decisions",
    uncertainty: "10,000-replicate advertiser-level paired bootstrap",
    subgroupInferencePermitted: false,
    subgroupResults: "descriptive only",
    sampleSizeInterpretation:
      "100 businesses support a first external-transport test, not narrow subgroup claims",
  },
  orchestration: {
    platform: "Google Cloud Run Jobs",
    region: "europe-west1",
    tasks: 4_800,
    parallelism: 180,
    computePerTask: "1 vCPU and 2 GiB",
    maxRetries: 0,
    timeoutSeconds: 1_800,
    containerImageDigest:
      "sha256:3276faca285998efdf40db81ec8b881d4102ab5b3eb528e4dba529e53c9e893e",
    projectedGrossCostUsd: [25, 37],
    operationalCostStopUsd: 45,
  },
  governance: {
    freezeBeforeCohortGeneration: true,
    truthUnavailableToCandidateFit: true,
    truthUnavailableToSelector: true,
    candidateActionsFrozenBeforeTruthOpening: true,
    openTruthExactlyOnce: true,
    postOpenTuningPermitted: false,
    technicalFailureRerunPermittedOnlyUnderIdenticalFingerprint: true,
    businessExclusionAfterGenerationPermitted: false,
    externalRealAdvertiserClaimPermitted: false,
    sotaClaimPermitted: false,
    abandonedPreGenerationProtocols: [
      {
        version: "0.1.0",
        completedBusinesses: 7,
        hiddenEconomicOutcomesInspected: false,
        reason:
          "AMSS search keyword coverage reached the floating-point binomial boundary at one; revision 0.1.1 caps coverage at 1-1e-9",
      },
    ],
  },
} as const;
