import { SVI_SCORE_V8_VERSION } from "./contract";

export const SVI_SCORE_V8_AUDIT_CONTRACT = {
  version: SVI_SCORE_V8_VERSION,
  cohort: {
    businesses: 80,
    candidatesPerBusiness: 48,
    candidateRows: 3_840,
    tokens: 136,
  },
  comparison: "frozen V8 selector versus frozen V5D-SVI comparator",
  endpoints: {
    primary: "65% mean plus 35% P90 policy loss, including declared abstention cost",
    required: [
      "selection objective lower than V5D-SVI",
      "mean policy loss lower than V5D-SVI",
      "P90 policy loss lower than V5D-SVI",
      "P95 policy loss no more than 2% above V5D-SVI",
      "dangerous false-champion share lower than V5D-SVI",
      "promotion coverage at least 60%",
      "zero selected-model theorem-bound violations",
      "SVI only; no NUTS",
    ],
  },
  governance: {
    openExactlyOnce: true,
    tuningAfterOpenPermitted: false,
    failureRepairOnSameCohortPermitted: false,
    newMethodAfterAuditRequiresNewSealedCohort: true,
    externalGeneralizationClaimPermitted: false,
    sotaClaimPermitted: false,
  },
} as const;
