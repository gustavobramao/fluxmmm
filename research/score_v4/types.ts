import type { ValidationLayerId } from "../../lib/mmm/validation";
import type { AuditSplit } from "../score_v3/types";

export type LearnedScoreWeights = Record<ValidationLayerId, number>;

export interface LearnedScoreCandidateRow {
  businessId: string;
  family: string;
  split: AuditSplit;
  candidateId: string;
  evidenceArm: "experiments-only" | "benchmark-gap-fill";
  eligible: boolean;
  failedGateCount: number;
  layers: Record<ValidationLayerId, number>;
  heuristicScore: number;
  profitRegret: number;
  roiError: number;
  contributionError: number;
}

export interface LearnedScoreSplitMetrics {
  businesses: number;
  eligibleBusinessShare: number;
  meanRegret: number;
  medianRegret: number;
  p90Regret: number;
  meanExcessRegret: number;
  lowestRegretSelectionRate: number;
  familyMeanRegret: Record<string, number>;
}

export interface LearnedScoreComparison {
  heuristic: LearnedScoreSplitMetrics;
  learned: LearnedScoreSplitMetrics;
  relativeMeanRegretReduction: number;
}

export interface LearnedScoreArtifact {
  artifactId: string;
  version: string;
  trainedAt: string;
  activation: "active" | "fallback";
  activationReason: string;
  simulatorVersion: string;
  cohort: {
    businesses: number;
    candidates: number;
    splits: Record<AuditSplit, number>;
    generatorFamilies: Record<AuditSplit, string[]>;
  };
  model: {
    family: "constrained-geometric-ensemble";
    target: "mean-four-decision-profit-regret";
    weights: LearnedScoreWeights;
    heuristicWeights: LearnedScoreWeights;
    minimumLayerWeight: number;
    gridStep: number;
    candidatesConsidered: number;
    formula: string;
  };
  performance: Record<AuditSplit, LearnedScoreComparison>;
  guardrails: string[];
}
