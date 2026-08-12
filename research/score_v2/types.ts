import type { ModelConfig, ModelResult } from "../../lib/mmm/types";
import type { ValidationResult } from "../../lib/mmm/validation";

export interface SyntheticChannelConfig {
  channel: "paid_social" | "search" | "tv";
  spendColumn: string;
  targetRoi: number;
  averageWeeklySpend: number;
  spendVolatility: number;
  demandCoupling: number;
  planningCoupling: number;
  flightProbability?: number;
  response:
    | {
        family: "geometric";
        decay: number;
        hillShape: number;
        halfSaturationQuantile: number;
      }
    | {
        family: "weibull";
        shape: number;
        scale: number;
        hillShape: number;
        halfSaturationQuantile: number;
      };
}

export interface SyntheticScenarioConfig {
  id: string;
  label: string;
  description: string;
  seed: number;
  periods: number;
  latentDemandPersistence: number;
  planningPersistence: number;
  noiseShare: number;
  observeDemandProxy: boolean;
  channels: SyntheticChannelConfig[];
  industryBenchmarks: Record<string, number>;
}

export interface ChannelTruth {
  channel: string;
  spendColumn: string;
  targetRoi: number;
  realizedRoi: number;
  coefficient: number;
  totalSpend: number;
  totalContribution: number;
  halfSaturation: number;
  spend: number[];
  transformed: number[];
  contribution: number[];
}

export interface AllocationTruth {
  extraBudget: number;
  currentContribution: number;
  optimalContribution: number;
  optimalIncrementalOutcome: number;
  optimalAdditionalBudget: Record<string, number>;
  gridUnits: number;
}

export interface SyntheticTruth {
  scenarioId: string;
  seed: number;
  latentDemand: number[];
  planningIntensity: number[];
  baseline: number[];
  deterministicOutcome: number[];
  channels: ChannelTruth[];
  allocation: AllocationTruth;
  industryBenchmarks: Record<string, number>;
}

export interface SyntheticBusiness {
  scenario: SyntheticScenarioConfig;
  observedCsv: string;
  observedRows: Record<string, string | number>[];
  truth: SyntheticTruth;
}

export interface CandidateEvaluation {
  id: string;
  config: ModelConfig;
  model: ModelResult;
  validation: ValidationResult;
  weightedLogRoiError: number;
  weightedLogBenchmarkAgreement: number;
  contributionError: number;
  budgetRegret: number;
  recommendedAdditionalBudget: Record<string, number>;
  trueOutcomeUnderRecommendation: number;
}
