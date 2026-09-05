import type { AgenticCandidateSpec } from "../../lib/mmm/agentic";
import type { SamplingContract } from "../../lib/mmm/sampling";
import type {
  ValidationDiagnosticValues,
  ValidationDiagnosticWeights,
} from "../../lib/mmm/score-diagnostics";
import type { ValidationLayerId } from "../../lib/mmm/validation";
import type { AuditSplit } from "../score_v3/types";

export type McmcSelectionReason =
  | "v6-leader"
  | "family-representative"
  | "diagnostic-diversity"
  | "approximation-risk"
  | "random-audit";

export interface McmcScoreV3PreparedCandidate {
  businessId: string;
  family: string;
  split: AuditSplit;
  candidateId: string;
  evidenceArm: "experiments-only" | "benchmark-gap-fill";
  modelFamily: AgenticCandidateSpec["family"];
  eligible: boolean;
  reviewEligible: boolean;
  eligibilityTier: "decision-grade" | "review";
  failedGateCount: number;
  layerScores: Record<ValidationLayerId, number>;
  diagnostics: ValidationDiagnosticValues;
  v6Score: number;
  mapDecisionLoss: number;
  selectionReason: McmcSelectionReason;
  selectionProbability: number;
  samplingFingerprint: string;
  samplingContract: SamplingContract;
}

export interface McmcScoreV3SampleRecord extends McmcScoreV3PreparedCandidate {
  status: "prepared" | "sampling" | "labelled" | "non-converged" | "error";
  samplerStatus?: "ready" | "review";
  convergencePassed?: boolean;
  mcmcDecisionLoss?: number;
  mcmcProfitRegret?: number;
  mcmcRoiError?: number;
  mcmcContributionError?: number;
  mapToMcmcLossShift?: number;
  posteriorRoi?: Record<string, number>;
  diagnosticsReceipt?: {
    maxRhat: number;
    minBulkEss: number;
    minTailEss: number;
    divergences: number;
    treeDepthHits: number;
    minBfmi: number;
    maxMcseRatio: number;
  };
  runtimeSeconds?: number;
  attempts?: number;
  error?: string;
}

export interface McmcScoreV3Artifact {
  artifactId: string;
  version: string;
  generatedAt: string;
  activation: "research-only";
  stage: "design" | "prepared" | "pilot-labelled" | "pilot-insufficient";
  contract: {
    targetBusinesses: number;
    targetCandidatesPerBusiness: number;
    targetMcmcFits: number;
    pilotBusinesses: number;
    pilotCandidatesPerBusiness: number;
    candidatePoolPerBusiness: number;
    label: string;
    sampler: SamplingContract;
    goldSampler: SamplingContract;
    splits: Record<AuditSplit, number>;
  };
  progress: {
    businessesPrepared: number;
    candidatesPrepared: number;
    candidatesLabelled: number;
    convergedCandidates: number;
    failedCandidates: number;
    completionShare: number;
  };
  pairedEvidence?: {
    pairs: number;
    businesses: number;
    mapMeanLoss: number;
    mcmcMeanLoss: number;
    meanLossShift: number;
    meanLossShiftLow: number;
    meanLossShiftHigh: number;
    mcmcWinRate: number;
    mapMedianLoss: number;
    mcmcMedianLoss: number;
    mapP90Loss: number;
    mcmcP90Loss: number;
    interpretation: "mcmc-lower" | "map-lower" | "inconclusive";
  };
  samplerEvidence?: {
    attemptedFits: number;
    convergenceRate: number;
    retryRate: number;
    medianRuntimeSeconds: number;
    p90RuntimeSeconds: number;
  };
  comparison: {
    baselineVersion: string;
    baselineWeights: ValidationDiagnosticWeights;
    learnedWeights?: ValidationDiagnosticWeights;
    weightDelta?: Partial<Record<keyof ValidationDiagnosticWeights, number>>;
    groupWeights?: Record<ValidationLayerId, number>;
    trainBusinesses?: number;
    validationBusinesses?: number;
    auditBusinesses?: number;
    note: string;
  };
  safeguards: string[];
  sampleRecords: McmcScoreV3SampleRecord[];
}
