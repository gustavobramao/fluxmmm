import type { SamplingChannelPosterior, SamplingDecisionDraw } from "../../lib/mmm/sampling";
import type { ScoreV6CandidateRow } from "../score_v6/types";
import type {
  ScoreV6FeatureWeights,
} from "../score_v6/types";
import type { SviInferenceContract } from "./contract";

export interface SviSeedDiagnostic {
  seed: number;
  iterations: number;
  finalLoss: number;
  elboDrift: number;
  finite: boolean;
  runtimeSeconds: number;
}

export interface SviApproximationResult {
  kind: "svi";
  fingerprint: string;
  promotedFingerprint: string;
  promotedId: string;
  engine: "PyMC 6.2 · FullRankADVI";
  executionBackend?: "numba-default" | "cvm-retry";
  contract: SviInferenceContract;
  status: "labelled" | "review";
  diagnostics: {
    finite: boolean;
    elboStable: boolean;
    maximumElboDrift: number;
    seedAgreement: boolean;
    maximumSeedLogRoiDifference: number;
    adjudicationUsed: boolean;
    predictiveCoverage: number;
    maximumImplausibleProbability: number;
    maximumRelativeRoiWidth: number;
  };
  seeds: SviSeedDiagnostic[];
  channels: SamplingChannelPosterior[];
  decisionDraws: SamplingDecisionDraw[];
  runtimeSeconds: number;
  runAt: string;
}

export interface SviScoreV3Record extends ScoreV6CandidateRow {
  status: "prepared" | "labelled" | "review" | "error";
  inferenceFingerprint: string;
  inferenceContract: SviInferenceContract;
  sviDecisionLoss?: number;
  sviProfitRegret?: number;
  sviRoiError?: number;
  sviContributionError?: number;
  mapToSviLossShift?: number;
  posteriorRoi?: Record<string, number>;
  diagnosticsReceipt?: SviApproximationResult["diagnostics"];
  seedDiagnostics?: SviSeedDiagnostic[];
  runtimeSeconds?: number;
  executionBackend?: SviApproximationResult["executionBackend"];
  error?: string;
}

export interface SviGoldAuditSelection {
  businessId: string;
  family: string;
  split: ScoreV6CandidateRow["split"];
  candidateId: string;
  evidenceArm: ScoreV6CandidateRow["evidenceArm"];
  modelFamily: ScoreV6CandidateRow["modelFamily"];
  businessInclusionProbability: number;
  candidateInclusionProbability: number;
  jointInclusionProbability: number;
  stratum: string;
}

export interface SviScoreV3Artifact {
  artifactId: string;
  version: string;
  generatedAt: string;
  activation: "research-only";
  stage: "design" | "smoke" | "running" | "complete";
  contract: {
    businesses: number;
    candidatesPerBusiness: number;
    targetSviFits: number;
    splits: Record<ScoreV6CandidateRow["split"], number>;
    inference: SviInferenceContract;
    targetMcmcAuditFits: number;
  };
  progress: {
    businessesStarted: number;
    businessesComplete: number;
    candidateAttempts: number;
    labelled: number;
    review: number;
    errors: number;
    completionShare: number;
  };
  feasibility: {
    medianRuntimeSeconds?: number;
    p90RuntimeSeconds?: number;
    projectedSerialHours?: number;
    projectedEightWorkerHours?: number;
    observedFits: number;
  };
  learning: {
    status: "awaiting-complete-cohort" | "ready-for-cross-validation";
    labelledCoverage: number;
    trainingBusinesses: number;
    baselineWeights: ScoreV6FeatureWeights;
    note: string;
  };
  safeguards: string[];
  goldAuditSelection: SviGoldAuditSelection[];
  records: SviScoreV3Record[];
}
