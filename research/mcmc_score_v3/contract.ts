import {
  SAMPLING_PRESETS,
  type SamplingContract,
} from "../../lib/mmm/sampling";
import type { AuditSplit } from "../score_v3/types";

export const MCMC_SCORE_V3_VERSION =
  "flux-mcmc-score-v3.0.0-posterior-decision-research";

export const MCMC_SCORE_V3_TARGET_SPLITS: Record<AuditSplit, number> = {
  train: 3_000,
  validation: 1_000,
  audit: 1_000,
};

export const MCMC_SCORE_V3_CONTRACT = {
  targetBusinesses: 5_000,
  candidatePoolPerBusiness: 48,
  targetCandidatesPerBusiness: 10,
  targetMcmcFits: 50_000,
  pilotBusinesses: 10,
  pilotCandidatesPerBusiness: 10,
  targetSplits: MCMC_SCORE_V3_TARGET_SPLITS,
  label:
    "One posterior-expected budget action per candidate, evaluated against hidden simulator truth",
  sampler: {
    ...SAMPLING_PRESETS.production,
    draws: 500,
    tune: 500,
  } satisfies SamplingContract,
  goldSampler: SAMPLING_PRESETS.production,
  selection: {
    v6Leaders: 3,
    familyRepresentatives: 2,
    diagnosticDiversity: 2,
    approximationRisk: 1,
    randomAudit: 2,
  },
} as const;
