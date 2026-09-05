import type { V5CTrainingConfiguration } from "./contract";

export interface V5CModel {
  featureNames: string[];
  weights: number[];
  configuration: V5CTrainingConfiguration;
}

export interface V5CScoreDistribution {
  mean: number;
  standardDeviation: number;
  robust: number;
  members: number[];
}

export interface V5CRegionEnsemble {
  region: number;
  models: V5CModel[];
  trainingBusinesses: number;
  excludedCandidateIds: string[];
}

export interface V5CSelectorEnsemble {
  regions: V5CRegionEnsemble[];
  regionByCandidateId: Record<string, number>;
}
