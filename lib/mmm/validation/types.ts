import type { ModelResult } from "../types";
import type { IndustryPriorOverrides } from "../benchmarks";
import type { ValidationScoreContract } from "../score-contract";

export type ValidationModelKind = ModelResult["kind"];
export type ValidationLayerId =
  | "generalization"
  | "structure"
  | "causal"
  | "decision";
export type ValidationStatus =
  | "pass"
  | "review"
  | "fail"
  | "incomplete";

export interface ValidationOptions {
  anchorIndependenceConfirmed: boolean;
  industryPriorChannels?: string[];
  industryBenchmarkScreeningEnabled?: boolean;
  materialSpendShareThreshold?: number;
  industryPriorOverrides?: IndustryPriorOverrides;
}

export type ChannelEvidenceStatus =
  | "aligned"
  | "experiment-conflict"
  | "benchmark-tension"
  | "boundary-collapse"
  | "unidentified"
  | "unanchored";

export interface ChannelEvidenceCoherence {
  channel: string;
  spendShare: number;
  material: boolean;
  roi: number;
  roiLow: number;
  roiHigh: number;
  evidenceSource: "experiment" | "industry" | "none";
  evidenceLabel: string;
  evidenceCenter?: number;
  evidenceLow?: number;
  evidenceHigh?: number;
  comparisonBasis: "experiment-window" | "full-history" | "unavailable";
  comparisonLabel: string;
  percentile?: number;
  standardizedGap?: number;
  priorUse: "calibration" | "screening-only" | "none";
  identification: "data-led" | "data-and-prior" | "prior-led" | "unknown";
  clipped: boolean;
  status: ChannelEvidenceStatus;
  blocking: boolean;
  detail: string;
}

export interface EvidenceCoherenceAssessment {
  status: "pass" | "review" | "fail";
  robustnessScore?: number;
  decisionScore?: number;
  causalScoreCap: number;
  materialSpendShareThreshold: number;
  blockingChannels: string[];
  rescueChannels: string[];
  summary: string;
  channels: ChannelEvidenceCoherence[];
}

export interface ValidationTest {
  id: string;
  name: string;
  score: number;
  status: ValidationStatus;
  metric: string;
  detail: string;
  importance: "critical" | "high" | "supporting";
}

export interface GeneralizationEvidence {
  kind: "generalization";
  folds: {
    id: string;
    label: string;
    dates: string[];
    actual: number[];
    predicted: number[];
    lower: number[];
    upper: number[];
    wape: number;
    coverage: number;
  }[];
  regimes: {
    name: string;
    dates: string[];
    actual: number[];
    predicted: number[];
    wape: number;
    available: boolean;
  }[];
}

export interface StructuralEvidence {
  kind: "structure";
  dates: string[];
  residuals: number[];
  fitted: number[];
  residualScale: number;
  autocorrelation: {
    lag: number;
    value: number;
  }[];
  vifs: {
    label: string;
    value: number;
  }[];
  quantiles: {
    expected: number;
    observed: number;
  }[];
}

export interface CausalEvidence {
  kind: "causal";
  anchorAssessment: {
    status: "qualified" | "needs-confirmation" | "not-testable";
    qualifiedCount: number;
    totalCount: number;
    summary: string;
    scoreWeight: number;
    criteria: {
      label: string;
      state: "pass" | "fail" | "confirm";
      detail: string;
    }[];
  };
  anchors: {
    channel: string;
    experimentRoi: number;
    experimentLow: number;
    experimentHigh: number;
    modelRoi: number;
    modelLow: number;
    modelHigh: number;
    covered: boolean;
    comparisonBasis?: "experiment-window" | "full-history";
    comparisonLabel?: string;
  }[];
  stability: {
    channel: string;
    points: {
      historyShare: number;
      roi: number;
    }[];
  }[];
  confounders: {
    channel: string;
    points: {
      strength: number;
      normalizedRoi: number;
    }[];
  }[];
  placebo: {
    lead: number;
    correlation: number;
  }[];
}

export interface DecisionEvidence {
  kind: "decision";
  channels: {
    channel: string;
    spendShare: number;
    material: boolean;
    roi: number;
    roiLow: number;
    roiHigh: number;
    evidenceLabel: string;
    evidenceLow?: number;
    evidenceHigh?: number;
    posteriorMass?: number;
    priorUse: ChannelEvidenceCoherence["priorUse"];
    identification: ChannelEvidenceCoherence["identification"];
    plausibilityScore: number;
    stabilityScore: number;
    resolutionScore: number;
    identificationScore: number;
    economicScore: number;
    score: number;
    status: ValidationStatus;
    blocking: boolean;
  }[];
}

export type ValidationEvidence =
  | GeneralizationEvidence
  | StructuralEvidence
  | CausalEvidence
  | DecisionEvidence;

export interface ValidationLayerResult {
  id: ValidationLayerId;
  score: number;
  status: ValidationStatus;
  summary: string;
  tests: ValidationTest[];
  evidence: ValidationEvidence;
}

export interface ValidationGate {
  id: string;
  label: string;
  applicable: boolean;
  passed: boolean;
  detail: string;
}

export interface ValidationResult {
  kind: "validation";
  modelKind: ValidationModelKind;
  modelFingerprint: string;
  fingerprint: string;
  cached: boolean;
  finalScore: number | null;
  heuristicScore: number | null;
  scoreContract: ValidationScoreContract;
  eligible: boolean;
  evidenceGrade: "A" | "B" | "C" | "Incomplete";
  recommendation:
    | "Decision-grade candidate"
    | "Use with caution"
    | "Not decision-grade"
    | "Incomplete evidence";
  layers: Record<ValidationLayerId, ValidationLayerResult>;
  evidenceCoherence: EvidenceCoherenceAssessment;
  gates: ValidationGate[];
  runAt: string;
}

export interface ValidationProgress {
  stage:
    | "structure"
    | "generalization"
    | "causal"
    | "decision"
    | "complete";
  detail: string;
  layers: Partial<Record<ValidationLayerId, ValidationLayerResult>>;
}
