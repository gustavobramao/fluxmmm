import type { DecisionScenarioId } from "../score_v2/types";

export interface V8ScenarioTargetRow {
  businessId: string;
  candidateId: string;
  scenarioDecisionLoss: Record<DecisionScenarioId, number>;
  scenarioProfitRegret: Record<DecisionScenarioId, number>;
  scenarioRecommendedSpend: Record<DecisionScenarioId, number>;
  aggregateDecisionLoss: number;
}

export interface V8DatasetRow {
  businessId: string;
  family: string;
  candidateId: string;
  fold: number;
  features: number[];
  valid: boolean;
  validityReasons: string[];
  scenarioLoss: Record<DecisionScenarioId, number>;
  scenarioExcessLoss: Record<DecisionScenarioId, number>;
  meanExcessLoss: number;
  p90ExcessLoss: number;
  economicRisk: number;
  excessEconomicRisk: number;
  normalizedExcessEconomicRisk: number;
  dangerous: boolean;
  roiError: number;
  contributionError: number;
}

export interface V8PredictionRow {
  businessId: string;
  candidateId: string;
  fold: number;
  modelFamily: string;
  predictedMean: number;
  predictedP90: number;
  predictedScenarios: Record<DecisionScenarioId, number>;
  predictedDanger: number;
  predictedRisk: number;
  uncertainty: number;
  adjustedRisk: number;
}
