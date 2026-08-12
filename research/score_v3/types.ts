import type {
  DecisionScenarioId,
  SyntheticChannel,
} from "../score_v2/types";

export type AuditSplit = "train" | "validation" | "audit";

export interface GeneratorFamilyContract {
  id: string;
  label: string;
  split: AuditSplit;
  businessCount: number;
  description: string;
  heldOutReason?: string;
}

export interface AuditBusinessSummary {
  id: string;
  family: string;
  split: AuditSplit;
  seed: number;
  periods: number;
  noiseShare: number;
  effectiveRevenueMargin: number;
  hiddenDemand: boolean;
  experimentCount: number;
  benchmarkLogError: number;
  channels: Record<
    SyntheticChannel,
    {
      roi: number;
      marginalRoi: number;
      family: "geometric" | "weibull";
      memory: number;
      hillShape: number;
      halfSaturationQuantile: number;
      demandCoupling: number;
      planningCoupling: number;
      spendShare: number;
    }
  >;
  decisions: Record<
    DecisionScenarioId,
    {
      optimalBudgetShare: number;
      optimalIncrementalProfit: number;
      mixShiftShare: number;
    }
  >;
}

export interface DistributionSummary {
  minimum: number;
  p10: number;
  median: number;
  p90: number;
  maximum: number;
}

export interface SimulatorAuditV3Artifact {
  artifactId: string;
  simulatorVersion: string;
  evidenceRegistryVersion: string;
  businessCount: number;
  splits: Record<AuditSplit, number>;
  families: GeneratorFamilyContract[];
  coverage: {
    historyWeeks: DistributionSummary;
    noiseShare: DistributionSummary;
    effectiveRevenueMargin: DistributionSummary;
    benchmarkLogError: DistributionSummary;
    experimentCoverage: number;
    channel: Record<
      SyntheticChannel,
      {
        roi: DistributionSummary;
        marginalRoi: DistributionSummary;
        memory: DistributionSummary;
        hillShape: DistributionSummary;
        halfSaturationQuantile: DistributionSummary;
        demandCoupling: DistributionSummary;
        planningCoupling: DistributionSummary;
        weibullShare: number;
      }
    >;
  };
  decisions: Record<
    DecisionScenarioId,
    {
      label: string;
      optimalBudgetShare: DistributionSummary;
      profitOpportunity: DistributionSummary;
      mixShiftShare: DistributionSummary;
      decreaseShare: number;
      unchangedShare: number;
      increaseShare: number;
    }
  >;
  businesses: AuditBusinessSummary[];
}

export type SimulatorAuditV3PublicArtifact = Omit<
  SimulatorAuditV3Artifact,
  "businesses"
> & {
  sampleBusinesses: AuditBusinessSummary[];
};
