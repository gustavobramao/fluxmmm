import type {
  Experiment,
  ModelConfig,
  ModelResult,
} from "../../lib/mmm/types";
import type { ValidationResult } from "../../lib/mmm/validation";

export type SyntheticChannel = "paid_social" | "search" | "tv";

export type DeliveryContract =
  | {
      kind: "auction";
      priceLabel: "CPC" | "CPM";
      averageUnitPrice: number;
      priceVolatility: number;
      inventoryDemandCoupling: number;
    }
  | {
      kind: "reach";
      priceLabel: "CPM";
      averageUnitPrice: number;
      priceVolatility: number;
      frequencyInflation: number;
    }
  | {
      kind: "flighted-grp";
      priceLabel: "CPP";
      averageUnitPrice: number;
      priceVolatility: number;
      flightStartProbability: number;
      flightContinuationProbability: number;
    };

export interface SyntheticChannelConfig {
  channel: SyntheticChannel;
  spendColumn: string;
  roiEvidenceId: string;
  /** Optional fixed override used only in explicit stress-test scenarios. */
  targetRoi?: number;
  averageWeeklySpend: number;
  spendVolatility: number;
  demandCoupling: number;
  planningCoupling: number;
  delivery: DeliveryContract;
  experiment:
    | {
        design: "geo" | "platform-holdout";
        spend: number;
        standardErrorShare: number;
        biasShare: number;
        /** Independent, non-overlapping studies available to the model. */
        replicates?: number;
      }
    | { design: "none" };
  allocation: {
    maximumShareOfIncrement: number;
  };
  response:
    | {
        family: "geometric";
        decay: number;
        hillShape: number;
        halfSaturationQuantile: number;
        kernelNormalization: "sum";
      }
    | {
        family: "weibull";
        shape: number;
        scale: number;
        hillShape: number;
        halfSaturationQuantile: number;
        kernelNormalization: "sum";
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
  commercialPersistence: number;
  noiseShare: number;
  eventShockProbability: number;
  observeDemandProxy: boolean;
  grossMargin: number;
  ltvRevenueMultiplier: number;
  maximumIncrementShare: number;
  decisionBudgetShares: number[];
  channels: SyntheticChannelConfig[];
  industryBenchmarks: Record<SyntheticChannel, number>;
}

export interface SimulatedExperimentTruth {
  channel: SyntheticChannel;
  design: "geo" | "platform-holdout";
  /** Causal ROI attributable to spend in the declared test window. */
  trueRoi: number;
  fullHistoryRoi: number;
  sourceWindowSpend: number;
  trueIncrementalOutcome: number;
  observedRoi: number;
  standardError: number;
  biasShare: number;
  experiment: Experiment;
}

export interface ChannelTruth {
  channel: SyntheticChannel;
  spendColumn: string;
  evidenceId: string;
  targetRoi: number;
  realizedRoi: number;
  marginalRoiAtObserved: number;
  coefficient: number;
  totalSpend: number;
  totalContribution: number;
  halfSaturation: number;
  spend: number[];
  deliveryUnits: number[];
  deliveryEfficiency: number[];
  transformed: number[];
  contribution: number[];
}

export interface AllocationTruth {
  maximumAdditionalBudget: number;
  currentContribution: number;
  optimalContribution: number;
  optimalIncrementalOutcome: number;
  optimalIncrementalProfit: number;
  optimalAdditionalBudget: Record<string, number>;
  optimalSpend: number;
  effectiveRevenueMargin: number;
  evaluatedBudgetShares: number[];
  gridUnits: number;
}

export type DecisionScenarioId =
  | "budget-reduction"
  | "fixed-budget-mix"
  | "budget-growth"
  | "economic-ceiling";

export interface DecisionScenarioContract {
  id: DecisionScenarioId;
  label: string;
  minimumBudgetShare: number;
  maximumBudgetShare: number;
  fixedBudgetShare?: number;
}

export interface DecisionScenarioTruth {
  id: DecisionScenarioId;
  label: string;
  minimumSpend: number;
  maximumSpend: number;
  optimalSpend: number;
  optimalIncrementalProfit: number;
  optimalAdditionalBudget: Record<string, number>;
  evaluations: number;
}

export interface SyntheticTruth {
  simulatorVersion: string;
  evidenceRegistryVersion: string;
  scenarioId: string;
  seed: number;
  latentDemand: number[];
  planningIntensity: number[];
  commercialIntensity: number[];
  baseline: number[];
  deterministicOutcome: number[];
  channels: ChannelTruth[];
  experiments: SimulatedExperimentTruth[];
  allocation: AllocationTruth;
  decisionScenarios: DecisionScenarioTruth[];
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
  evidenceArm: "experiments-only" | "benchmark-gap-fill";
  config: ModelConfig;
  model: ModelResult;
  validation: ValidationResult;
  weightedLogRoiError: number;
  weightedLogBenchmarkAgreement: number;
  contributionError: number;
  budgetRegret: number;
  profitRegret: number;
  /** Uncapped economic opportunity loss used by V6 research. */
  decisionLoss: number;
  revenueRegret: number;
  scenarioProfitRegret: Record<DecisionScenarioId, number>;
  scenarioDecisionLoss: Record<DecisionScenarioId, number>;
  scenarioRecommendedSpend: Record<DecisionScenarioId, number>;
  recommendedAdditionalBudget: Record<string, number>;
  recommendedSpend: number;
  trueOutcomeUnderRecommendation: number;
  trueProfitUnderRecommendation: number;
}
