export type DataRow = Record<string, string | number>;

export type ColumnRole =
  | "date"
  | "outcome"
  | "media_spend"
  | "media_exposure"
  | "control"
  | "categorical"
  | "ignore";

export interface ColumnSpec {
  name: string;
  role: ColumnRole;
  channel?: string;
  numeric: boolean;
}

export interface Dataset {
  name: string;
  rawCsv: string;
  sourceRows: DataRow[];
  sourceRowCount: number;
  rows: DataRow[];
  columns: string[];
  specs: ColumnSpec[];
  dateColumn: string;
  outcomeColumn: string;
  mediaColumns: string[];
  controlColumns: string[];
  sourceHash: string;
  hash: string;
  chronologyReordered: boolean;
  automaticRepairs: DatasetRepair[];
  sourceCadence: DatasetCadence;
  modelCadence: ModelCadence;
  periodsPerYear: number;
  periodUnit: "week" | "month";
}

export type ModelCadence = "weekly" | "monthly";

export interface DatasetRepair {
  id: "date-format" | "daily-aggregation";
  column: string;
  count: number;
  title: string;
  detail: string;
}

export type DatasetCadence =
  | "weekly"
  | "daily"
  | "monthly"
  | "irregular"
  | "unknown";

export interface ValidationIssue {
  level: "error" | "warning" | "info";
  title: string;
  detail: string;
}

export interface ValidationResult {
  score: number;
  status: "ready" | "review" | "blocked";
  issues: ValidationIssue[];
  startDate: string;
  endDate: string;
  frequency: string;
  cadence: DatasetCadence;
  sourceCadence: DatasetCadence;
  modelCadence: ModelCadence;
  intervalDays: number | null;
  gapCount: number;
  invalidDates: number;
  modeledMissingCells: number;
  excludedControlColumns: string[];
  automaticRepairs: DatasetRepair[];
  missingCells: number;
  duplicateDates: number;
}

export interface ChannelSummary {
  channel: string;
  spend: number;
  share: number;
  activePeriods: number;
  zeroShare: number;
  correlation: number;
}

export interface EdaResult {
  totalOutcome: number;
  totalSpend: number;
  outcomeTrend: number[];
  spendTrend: number[];
  channels: ChannelSummary[];
  correlations: number[][];
  readiness: string[];
}

export interface Experiment {
  channel: string;
  startDate: string;
  endDate: string;
  incrementalOutcome: number;
  incrementalSpend: number;
  standardError: number;
  confidence: number;
  scope: "immediate" | "total";
  source: string;
}

export interface MediaResponseConfig {
  adstockType: "geometric" | "weibull";
  adstock: number;
  weibullShape: number;
  weibullScale: number;
  saturation: number;
  halfSaturationQuantile: number;
  kernelNormalization: "peak" | "sum";
}

export interface ModelConfig {
  adstockType: "geometric" | "weibull";
  adstock: number;
  weibullShape: number;
  weibullScale: number;
  saturation: number;
  ridge: number;
  fourierOrder: number;
  cyclePeriod: number;
  /**
   * Optional channel-level response contracts. Missing channels inherit the
   * global settings above, so existing saved specifications remain valid.
   */
  channelResponses?: Record<string, Partial<MediaResponseConfig>>;
}

export interface AdvancedModelConfig {
  timeVarying: boolean;
  kernelKnots: number;
  kernelBandwidth: number;
  planningIntensity: boolean;
  calibrationMode: "prior" | "likelihood";
  priorDistribution: "half-normal" | "log-normal";
  likelihoodDistribution: "gaussian" | "student-t" | "log-normal";
  studentTDegreesFreedom: number;
}

export interface ChannelEstimate {
  channel: string;
  contribution: number;
  contributionShare: number;
  roi: number;
  roiLow: number;
  roiHigh: number;
  coefficient: number;
  priorRoi?: number;
  priorSource?: "experiment" | "industry";
  priorLabel?: string;
}

export interface ModelCalibrationReceipt {
  channel: string;
  source: "experiment" | "industry";
  route: "prior" | "likelihood";
  comparisonBasis: "experiment-window" | "full-history";
  evidenceLabel: string;
  targetRoi: number;
  targetLow: number;
  targetHigh: number;
  modelRoi: number;
  modelLow: number;
  modelHigh: number;
  startDate?: string;
  endDate?: string;
  scope?: Experiment["scope"];
}

export interface NumericalStabilityDiagnostics {
  solver: "pivoted-qr" | "svd";
  rank: number;
  parameterCount: number;
  conditionNumber: number | null;
  dataRank: number;
  dataConditionNumber: number | null;
  status: "stable" | "review" | "rank-deficient";
  priorInfluence: {
    channel: string;
    precisionShare: number;
    classification: "data-led" | "data-and-prior" | "prior-led";
    source: "none" | "experiment" | "industry" | "regularizing";
    sourceLabel?: string;
  }[];
  clipping: {
    applied: boolean;
    material: boolean;
    severity: "none" | "informational" | "warning";
    predictionShiftShare: number;
    channels: {
      channel: string;
      unconstrainedCoefficient: number;
      constrainedCoefficient: number;
      contributionChange: number;
      contributionChangeShare: number;
      unconstrainedRoi: number;
      constrainedRoi: number;
      roiIntervalMaterial: boolean;
    }[];
  };
}

export interface ModelResult {
  kind: "frequentist" | "bayesian" | "advanced";
  fingerprint: string;
  cached: boolean;
  r2: number;
  mape: number;
  rmse: number;
  baselineShare: number;
  channels: ChannelEstimate[];
  actual: number[];
  predicted: number[];
  baseline: number[];
  residuals: number[];
  diagnostics: {
    label: string;
    value: string;
    state: "good" | "warn";
  }[];
  numerical?: NumericalStabilityDiagnostics;
  advanced?: {
    timeVarying: boolean;
    calibrationMode: AdvancedModelConfig["calibrationMode"];
    priorDistribution: AdvancedModelConfig["priorDistribution"];
    likelihoodDistribution: AdvancedModelConfig["likelihoodDistribution"];
    coefficientPaths: {
      channel: string;
      values: number[];
    }[];
    planningIntensity: number[];
    calibratedExperiments: number;
    calibratedIndustryPriors: number;
    calibrationReceipts: ModelCalibrationReceipt[];
  };
  runAt: string;
}
