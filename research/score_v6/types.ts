import type { AgenticCandidateSpec } from "../../lib/mmm/agentic";
import type { ValidationLayerId } from "../../lib/mmm/validation";
import {
  SCORE_DIAGNOSTIC_GROUPS,
  type ValidationDiagnosticValues,
  type ValidationDiagnosticWeights,
  type ValidationScoreDiagnostic,
} from "../../lib/mmm/score-diagnostics";
import type { AuditSplit } from "../score_v3/types";

export const SCORE_V6_DIAGNOSTIC_GROUPS = SCORE_DIAGNOSTIC_GROUPS;

export type ScoreV6Feature = ValidationScoreDiagnostic;
export type ScoreV6FeatureValues = ValidationDiagnosticValues;
export type ScoreV6FeatureWeights = ValidationDiagnosticWeights;

export type ScoreV6EvidenceArm =
  | "experiments-only"
  | "benchmark-gap-fill";

export interface ScoreV6CandidateRow {
  businessId: string;
  family: string;
  split: AuditSplit;
  candidateId: string;
  evidenceArm: ScoreV6EvidenceArm;
  modelFamily: AgenticCandidateSpec["family"];
  eligible: boolean;
  reviewEligible: boolean;
  eligibilityTier: "decision-grade" | "review";
  failedGateCount: number;
  layerScores: Record<ValidationLayerId, number>;
  diagnostics: ScoreV6FeatureValues;
  heuristicScore: number;
  /**
   * Predeclared candidate-independent scale for the economic loss. The
   * current simulator already divides opportunity loss by a business/scenario
   * economic scale, so the frozen V3 cohort uses 1 for every candidate.
   */
  economicScale?: number;
  decisionLoss: number;
  cappedRegret: number;
  roiError: number;
  contributionError: number;
}

export interface ScoreV6Metrics {
  businesses: number;
  decisionGradeBusinessShare: number;
  meanLoss: number;
  medianLoss: number;
  p90Loss: number;
  p95Loss: number;
  meanExcessLoss: number;
  meanNormalizedExcessRegret: number;
  lowestLossSelectionRate: number;
  familyMeanLoss: Record<string, number>;
}

export interface ScoreV6Artifact {
  artifactId: string;
  version: string;
  trainedAt: string;
  activation: "active" | "rejected";
  activationReason: string;
  runtimeContractChanged: boolean;
  simulatorVersion: string;
  cohort: {
    businesses: number;
    candidates: number;
    pairs: number;
    splits: Record<AuditSplit, number>;
  };
  model: {
    family: "monotonic-pairwise-logistic-ranker";
    target: "uncapped-economic-decision-loss";
    weights: ScoreV6FeatureWeights;
    groupWeights: Record<ValidationLayerId, number>;
    regularization: number;
    epochs: number;
    formula: string;
  };
  performance: Record<
    AuditSplit,
    {
      heuristic: ScoreV6Metrics;
      learned: ScoreV6Metrics;
      relativeMeanLossReduction: number;
    }
  >;
  gates: Record<string, boolean>;
  pipelineReadiness: {
    decisionGradeBusinessShare: number;
    requiredShare: number;
    ready: boolean;
    detail: string;
  };
  guardrails: string[];
}
