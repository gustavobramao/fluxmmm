import { SVI_SCORE_V9_VERSION } from "./contract";

export const V9_AUDIT_FAMILY_IDS = [
  "balanced-dtc",
  "search-demand-harvesting",
  "social-frequency-pressure",
  "delayed-tv",
  "correlated-planning",
  "wrong-evidence",
  "swapped-channel-mechanics",
] as const;

export const SVI_SCORE_V9_AUDIT_CONTRACT = {
  version: SVI_SCORE_V9_VERSION,
  auditId: "003",
  cohort: {
    businessesPerFamily: 20,
    families: V9_AUDIT_FAMILY_IDS,
    businesses: 140,
    candidatesPerBusiness: 48,
    candidateRows: 6_720,
    tokens: 232,
    seedIndexStart: 90_000,
    seedIndexEndInclusive: 90_019,
    developmentSeedIndexStart: 12_000,
    developmentSeedIndexEndInclusive: 12_099,
  },
  inference: {
    engine: "PyMC 6.2 FullRankADVI",
    nutsPermitted: false,
    cachedDevelopmentPosteriorPermitted: false,
    newPosteriorRequiredForEveryCandidate: true,
  },
  orchestration: {
    platform: "Google Cloud Run Jobs",
    region: "europe-west1",
    grouping: "one immutable candidate fit per task",
    tasks: 6_720,
    parallelism: 180,
    candidateFitsPerTask: 1,
    candidateConcurrencyPerTask: 1,
    computePerTask: "1 vCPU and 2 GiB",
    maximumConcurrentVcpus: 180,
    availableRegionalVcpuQuota: 200,
    compilerCacheScope: "isolated per candidate task; no cross-task compile lock",
    linearAlgebraRuntime: "OpenBLAS with one thread per SVI worker",
    scientificRuntime: "scripts/svi_batch.py --backend c",
    containerImageDigest:
      "sha256:3276faca285998efdf40db81ec8b881d4102ab5b3eb528e4dba529e53c9e893e",
    expectedWallClockHours: "2.5-3.7",
    wallClockLimitHours: 4,
    expectedGrossCostUsd: "32-42 before any free-tier credit",
    operationalCostStopUsd: 45,
  },
  comparison: {
    challenger: "frozen V9 set-wise distributional selector",
    comparator: "frozen V8 selection-aware selector",
    pairedByBusiness: true,
    bothSelectorsSeeOnlyTheirFrozenTruthBlindTokens: true,
  },
  endpoints: {
    primary:
      "full-coverage selection objective: 65% mean selected excess loss plus 35% P90 selected excess loss",
    primaryInference: {
      method: "family-stratified paired bootstrap",
      resamples: 10_000,
      seed: 9_031_771,
      confidenceLevel: 0.95,
      superiorityRule:
        "V9 minus V8 primary-objective point estimate is below zero and its one-sided 95% bootstrap upper bound is below zero",
    },
    requiredSafety: [
      "V9 full-coverage mean selected excess loss below V8",
      "V9 full-coverage P90 selected excess loss below V8",
      "V9 full-coverage P95 selected excess loss no more than 2% above V8",
      "V9 dangerous false-champion share below V8",
      "V9 60%-to-100% area under selection-risk curve below V8",
      "V9 matched 70% coverage selection objective below V8",
      "zero V9 selected-model theorem-bound violations",
      "SVI only; no NUTS",
    ],
    descriptive: [
      "family-specific mean, P90, P95, dangerous false-champion share, and oracle recall",
      "paired business win, tie, and loss rates",
      "60%-to-100% risk-coverage curves",
    ],
  },
  governance: {
    freezeBeforeCohortGeneration: true,
    openExactlyOnce: true,
    tuningAfterOpenPermitted: false,
    failureRepairOnSameCohortPermitted: false,
    failedTechnicalFitsMayBeRerunUnderIdenticalInferenceContract: true,
    newMethodAfterAuditRequiresNewSeedWindow: true,
    externalRealWorldGeneralizationClaimPermitted: false,
    sotaClaimPermitted: false,
    abandonedPriorAudits: [
      {
        auditId: "001",
        posteriorResultsAtAbandonment: 171,
        hiddenEconomicOutcomesInspected: false,
      },
      {
        auditId: "002",
        posteriorResultsAtAbandonment: 0,
        hiddenEconomicOutcomesInspected: false,
      },
    ],
    orchestrationMayReuseCompilerArtifacts: true,
  },
} as const;
