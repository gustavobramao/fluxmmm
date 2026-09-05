import type { SviScoreV4Metrics, SviScoreV4Selection } from "../svi_score_v4/types";
import type { SviScoreV5ATrainingConfiguration } from "./contract";

export interface SviScoreV5AModel {
  featureNames: string[];
  weights: number[];
  configuration: SviScoreV5ATrainingConfiguration;
}

export interface SviScoreV5ATheoremReceipt {
  businesses: number;
  violations: number;
  meanSelectedNormalizedRegret: number;
  meanTopOneHingeBound: number;
  meanSmoothTopOneBound: number;
  maximumNumericalSlack: number;
}

export interface SviScoreV5AFoldReceipt {
  fold: number;
  trainingBusinesses: number;
  assessmentBusinesses: number;
  assessmentFamilies: Record<string, number>;
  metrics: SviScoreV4Metrics;
  theorem: SviScoreV5ATheoremReceipt;
}

export interface SviScoreV5AConfigurationReceipt {
  configuration: SviScoreV5ATrainingConfiguration;
  folds: SviScoreV5AFoldReceipt[];
  outOfFoldMetrics: SviScoreV4Metrics;
  selectionObjective: number;
}

export interface SviScoreV5ACrossValidationReceipt {
  version: "flux-svi-score-v5a-grouped-cv-v1";
  contract: {
    folds: number;
    groupingUnit: "business";
    stratification: "generator-family";
    commonCandidatePool: "v4.2-roi-safety-only";
    auditRowsPermitted: false;
    selectionObjective:
      "mean-loss-plus-0.5-cvar90-loss-plus-0.5-worst-family-mean-loss";
    theoremTarget: "capped-normalized-top-one-excess-regret";
  };
  data: {
    businesses: number;
    rows: number;
    candidatesPerBusiness: number;
    families: Record<string, number>;
    assignmentHash: string;
    featureHash: string;
  };
  assignments: Array<{ businessId: string; family: string; fold: number }>;
  configurations: SviScoreV5AConfigurationReceipt[];
  selectedConfiguration: SviScoreV5ATrainingConfiguration;
  finalFit: {
    model: SviScoreV5AModel;
    descriptiveDevelopmentMetrics: SviScoreV4Metrics;
    theorem: SviScoreV5ATheoremReceipt;
  };
}

export type SviScoreV5ASelection = SviScoreV4Selection;
