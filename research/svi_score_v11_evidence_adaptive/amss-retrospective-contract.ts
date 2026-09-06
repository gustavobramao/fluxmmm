export const V11_AMSS_RETROSPECTIVE_VERSION =
  "flux-svi-score-v11.0.0-amss-retrospective-reuse";

export const V11_AMSS_RETROSPECTIVE_CONTRACT = {
  purpose:
    "screen the frozen V11 evidence-adaptive selector on the previously opened AMSS cohort without posterior refitting",
  scientificLabel: "retrospective external benchmark; not confirmatory",
  cohort: {
    businesses: 100,
    candidatesPerBusiness: 48,
    cachedSviFits: 4_800,
    evidenceGroups: [
      "no-experiment",
      "paid-social-experiment",
      "search-experiment",
      "tv-experiment",
    ],
  },
  selection: {
    selectorMayRetrain: false,
    posteriorMayRefit: false,
    truthMayEnterTokens: false,
    choicesMustBeFrozenBeforeRetrospectiveScoring: true,
    comparatorChoices: ["frozen-v9", "prediction-only", "in-pool-oracle"],
  },
  endpoints: {
    primary:
      "65% cohort mean candidate risk plus 35% cohort P90 candidate risk",
    secondary: [
      "cohort mean candidate risk",
      "cohort P90 candidate risk",
      "cohort P95 candidate risk",
      "mean excess risk over the in-pool oracle",
      "in-pool oracle recall",
      "paired business wins and stratified bootstrap interval",
    ],
    subgroupStatus: "descriptive only",
    bootstrapResamples: 10_000,
    bootstrapSeed: 11_100_510,
  },
  screeningRule: {
    proceedToFreshAuditOnlyIf: [
      "V11 primary objective is lower than frozen V9",
      "V11 primary objective is no higher than prediction-only",
    ],
    note:
      "The rule is operational only because the AMSS outcomes were already known before V11 was designed.",
  },
  governance: {
    confirmatory: false,
    cloudComputePermitted: false,
    newPosteriorFitsPermitted: false,
    existingAmssTruthWasPreviouslyOpened: true,
    publicationClaim:
      "retrospective transport evidence only; a new untouched cohort is required for a V11 confirmatory claim",
  },
} as const;
