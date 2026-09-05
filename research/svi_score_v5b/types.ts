import type { SviScoreV4Metrics } from "../svi_score_v4/types";
import type { V5BSearchAlgorithm } from "./contract";

export interface V5BSearchCandidate {
  candidateId: string;
  point: number[];
}

export interface V5BObservedCandidate extends V5BSearchCandidate {
  selectorScore: number;
  safetyAccepted: boolean;
}

export interface V5BSearchTrace {
  businessId: string;
  algorithm: V5BSearchAlgorithm;
  orderedCandidateIds: string[];
}

export interface V5BCheckpointSelection {
  businessId: string;
  family: string;
  algorithm: V5BSearchAlgorithm;
  budget: number;
  candidateId: string;
  loss: number;
  excessLoss: number;
  normalizedExcessRegret: number;
  safetyAccepted: boolean;
  reviewOnlyFallback: boolean;
  fullPoolChampionRecovered: boolean;
  economicOracleRecovered: boolean;
  selectorScoreGap: number;
}

export interface V5BCheckpointMetrics extends SviScoreV4Metrics {
  budget: number;
  selectionObjective: number;
  fullPoolChampionRecall: number;
  economicOracleRecall: number;
  meanSelectorScoreGap: number;
}

export interface V5BAlgorithmReceipt {
  algorithm: V5BSearchAlgorithm;
  checkpoints: V5BCheckpointMetrics[];
  primary: V5BCheckpointMetrics;
  primaryObjective: number;
  primaryCandidateEvaluations: number;
  curveCandidateEvaluations: number;
  duplicateEvaluations: number;
  incompleteTraces: number;
}
