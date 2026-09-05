import type { ScoreV6CandidateRow, ScoreV6Feature } from "../score_v6/types";
import type { SviScoreV4TrainingConfiguration } from "./contract";

export const V4_TEMPORAL_FEATURES = [
  "whole-flight-generalization",
  "carryover-support",
  "post-flight-residual-stability",
  "kernel-distinguishability",
] as const;

export type SviScoreV4TemporalFeature =
  (typeof V4_TEMPORAL_FEATURES)[number];
export type SviScoreV4Feature = ScoreV6Feature | SviScoreV4TemporalFeature;
export type SviScoreV4FeatureWeights = Partial<
  Record<SviScoreV4Feature, number>
>;

export interface SviScoreV4TemporalReceipt {
  /** Candidate-invariant, truth-blind business context in [0, 1]. */
  context: number;
  wholeFlightGeneralization: number;
  carryoverSupport: number;
  postFlightResidualStability: number;
  kernelDistinguishability: number;
  temporalIdentification: number;
  maximumEffectiveLag: number;
  maximumLag95: number;
  materialFlightCount: number;
  detail: string;
}

export interface SviScoreV4CandidateRow extends ScoreV6CandidateRow {
  temporal: SviScoreV4TemporalReceipt;
}

export interface SviScoreV4Model {
  ordinaryWeights: SviScoreV4FeatureWeights;
  temporalWeights: SviScoreV4FeatureWeights;
  configuration: SviScoreV4TrainingConfiguration;
}

export interface SviScoreV4Selection {
  businessId: string;
  candidateId: string;
  family: string;
  loss: number;
  excessLoss: number;
  normalizedExcessRegret: number;
  oracle: boolean;
  decisionGrade: boolean;
  safetyAccepted: boolean;
  reviewOnlyFallback: boolean;
}

export interface SviScoreV4Metrics {
  businesses: number;
  meanLoss: number;
  medianLoss: number;
  p90Loss: number;
  p95Loss: number;
  cvar90Loss: number;
  worstFamilyMeanLoss: number;
  familyMeanLoss: Record<string, number>;
  meanExcessLoss: number;
  medianExcessLoss: number;
  p90ExcessLoss: number;
  p95ExcessLoss: number;
  cvar90ExcessLoss: number;
  worstFamilyMeanExcessLoss: number;
  familyMeanExcessLoss: Record<string, number>;
  oracleSelectionRate: number;
  safetyAcceptedShare: number;
  reviewOnlyFallbackShare: number;
}

export interface SviScoreV4TopOneBoundReceipt {
  businesses: number;
  violations: number;
  meanSelectedNormalizedRegret: number;
  meanTopOneHingeBound: number;
  meanSmoothTopOneBound: number;
  maximumNumericalSlack: number;
}
