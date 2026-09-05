import type { V5DTrainingConfiguration } from "./contract";

export interface V5DModel {
  featureNames: string[];
  means: number[];
  scales: number[];
  meanWeights: number[];
  p90Weights: number[];
  configuration: V5DTrainingConfiguration;
  receipt: {
    trainingRows: number;
    trainingBusinesses: number;
    meanTarget: number;
    p90Target: number;
    hardNegativeUpdates: number;
  };
}

export interface V5DLossPrediction {
  mean: number;
  p90: number;
  tailWidth: number;
  risk: number;
}

export interface V5DRegionModel {
  region: number;
  excludedCandidateIds: string[];
  model: V5DModel;
}

export interface V5DCrossFittedModel {
  regions: V5DRegionModel[];
  regionByCandidateId: Record<string, number>;
}
