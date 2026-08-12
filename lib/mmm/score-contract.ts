import learnedScoreArtifact from "../../research/score_v4/artifacts/learned-score-v4.json";
import type { ValidationLayerId } from "./validation/types";

export type ValidationScoreWeights = Record<ValidationLayerId, number>;

interface LearnedScoreRuntimeArtifact {
  artifactId: string;
  version: string;
  activation: "active" | "fallback";
  activationReason: string;
  model: {
    family: string;
    target: string;
    weights: ValidationScoreWeights;
    heuristicWeights: ValidationScoreWeights;
    formula: string;
    minimumLayerWeight: number;
    gridStep: number;
    candidatesConsidered: number;
  };
  cohort: {
    businesses: number;
    candidates: number;
    splits: Record<"train" | "validation" | "audit", number>;
    generatorFamilies: Record<"train" | "validation" | "audit", string[]>;
  };
  performance: Record<"train" | "validation" | "audit", {
    heuristic: {
      meanRegret: number;
      p90Regret: number;
      familyMeanRegret: Record<string, number>;
    };
    learned: {
      meanRegret: number;
      p90Regret: number;
      familyMeanRegret: Record<string, number>;
    };
    relativeMeanRegretReduction: number;
  }>;
  guardrails: string[];
}

export interface ValidationScoreContract {
  kind: "learned" | "heuristic";
  version: string;
  artifactId: string;
  weights: ValidationScoreWeights;
  target: string;
  activationReason: string;
}

export const SCORE_LAYER_ORDER: ValidationLayerId[] = [
  "generalization",
  "structure",
  "causal",
  "decision",
];

export const LEARNED_SCORE_ARTIFACT =
  learnedScoreArtifact as LearnedScoreRuntimeArtifact;

export const HEURISTIC_SCORE_WEIGHTS: ValidationScoreWeights =
  LEARNED_SCORE_ARTIFACT.model.heuristicWeights;

export const ACTIVE_SCORE_CONTRACT: ValidationScoreContract =
  LEARNED_SCORE_ARTIFACT.activation === "active"
    ? {
        kind: "learned",
        version: LEARNED_SCORE_ARTIFACT.version,
        artifactId: LEARNED_SCORE_ARTIFACT.artifactId,
        weights: LEARNED_SCORE_ARTIFACT.model.weights,
        target: LEARNED_SCORE_ARTIFACT.model.target,
        activationReason: LEARNED_SCORE_ARTIFACT.activationReason,
      }
    : {
        kind: "heuristic",
        version: "flux-score-heuristic-v1",
        artifactId: LEARNED_SCORE_ARTIFACT.artifactId,
        weights: HEURISTIC_SCORE_WEIGHTS,
        target: "transparent-four-layer-evidence-score",
        activationReason: LEARNED_SCORE_ARTIFACT.activationReason,
      };

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

export function scoreValidationLayers(
  layers: Record<ValidationLayerId, number>,
  weights: ValidationScoreWeights = ACTIVE_SCORE_CONTRACT.weights,
): number {
  const logScore = SCORE_LAYER_ORDER.reduce(
    (total, layer) =>
      total + weights[layer] * Math.log(clamp(layers[layer] / 100, 1e-6, 1)),
    0,
  );
  return 100 * Math.exp(logScore);
}

export function scoreWeightPercent(layer: ValidationLayerId): number {
  return Math.round(ACTIVE_SCORE_CONTRACT.weights[layer] * 1000) / 10;
}

export function activeScoreFormula(): string {
  const weights = ACTIVE_SCORE_CONTRACT.weights;
  return `100 × G^${weights.generalization.toFixed(3)} × S^${weights.structure.toFixed(3)} × C^${weights.causal.toFixed(3)} × D^${weights.decision.toFixed(3)}`;
}
