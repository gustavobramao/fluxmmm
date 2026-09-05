import { sha256 } from "../csv";
import type {
  AdvancedModelConfig,
  Dataset,
  Experiment,
  ModelConfig,
  ModelResult,
} from "../types";
import type { ValidationModelSpec } from "./adapter";
import { runCausalValidation } from "./causal";
import { runDecisionValidation } from "./decision";
import { assessEvidenceCoherence } from "./evidence";
import {
  assessEvidenceDependence,
  attachEvidenceDependence,
} from "./evidence-dependence";
import { runGeneralizationValidation } from "./predictive";
import { scoreValidation } from "./scoring";
import { runStructuralValidation } from "./structural";
import type {
  ValidationProgress,
  ValidationResult,
  ValidationOptions,
} from "./types";

const VALIDATION_VERSION =
  "flux-validation-v1.8.0-experiment-window-roi";
export const DEFAULT_VALIDATION_OPTIONS: ValidationOptions = {
  anchorIndependenceConfirmed: false,
  industryPriorChannels: [],
  industryBenchmarkScreeningEnabled: true,
  materialSpendShareThreshold: 0.02,
};

export async function validationFingerprint(
  dataset: Dataset,
  model: ModelResult,
  config: ModelConfig,
  advancedConfig: AdvancedModelConfig,
  experiments: Experiment[],
  validationOptions: ValidationOptions = DEFAULT_VALIDATION_OPTIONS,
): Promise<string> {
  return sha256(
    JSON.stringify({
      dataset: dataset.hash,
      model: model.fingerprint,
      config,
      advancedConfig: model.kind === "advanced" ? advancedConfig : undefined,
      experiments,
      validationOptions,
      folds: 3,
      objective: "decision-grade-causal",
      version: VALIDATION_VERSION,
    }),
  );
}

export async function runModelValidation(
  dataset: Dataset,
  model: ModelResult,
  config: ModelConfig,
  advancedConfig: AdvancedModelConfig,
  experiments: Experiment[],
  fingerprint: string,
  onProgress?: (progress: ValidationProgress) => void,
  validationOptions: ValidationOptions = DEFAULT_VALIDATION_OPTIONS,
): Promise<ValidationResult> {
  const spec: ValidationModelSpec = {
    kind: model.kind,
    config,
    advancedConfig,
    experiments,
    validationOptions,
  };
  const structure = runStructuralValidation(dataset, spec, model);
  onProgress?.({
    stage: "structure",
    detail: "Structural diagnostics complete; running temporal holdouts.",
    layers: { structure },
  });
  if (onProgress) {
    await new Promise<void>((resolve) => setTimeout(resolve, 40));
  }

  const generalization = runGeneralizationValidation(dataset, spec);
  onProgress?.({
    stage: "generalization",
    detail: "Rolling and spend-regime holdouts complete; testing causal stability.",
    layers: { structure, generalization },
  });
  if (onProgress) {
    await new Promise<void>((resolve) => setTimeout(resolve, 40));
  }

  const causalRobustness = await runCausalValidation(dataset, spec, model);
  const causal: typeof causalRobustness = {
    ...causalRobustness,
    summary: `${causalRobustness.summary} Business plausibility is scored independently in ROI Decision Coherence.`,
  };
  onProgress?.({
    stage: "causal",
    detail: "Causal stress tests complete; scoring material-channel ROI coherence.",
    layers: { structure, generalization, causal },
  });
  if (onProgress) {
    await new Promise<void>((resolve) => setTimeout(resolve, 24));
  }

  const baseCoherence = assessEvidenceCoherence(
    dataset,
    model,
    experiments,
    validationOptions,
    config,
  );
  const dependence = await assessEvidenceDependence(
    dataset,
    model,
    config,
    advancedConfig,
    experiments,
    validationOptions,
  );
  const assessedCoherence = attachEvidenceDependence(
    baseCoherence,
    dependence,
  );
  const decision = runDecisionValidation(model, assessedCoherence, causal);
  const evidenceCoherence = {
    ...assessedCoherence,
    robustnessScore: causalRobustness.score,
    decisionScore: decision.score,
  };
  onProgress?.({
    stage: "decision",
    detail: "ROI Decision Coherence complete; assembling the winner score.",
    layers: { structure, generalization, causal, decision },
  });
  const layers = { generalization, structure, causal, decision };
  const scoring = scoreValidation(layers, evidenceCoherence);
  const result: ValidationResult = {
    kind: "validation",
    modelKind: model.kind,
    modelFingerprint: model.fingerprint,
    fingerprint,
    cached: false,
    ...scoring,
    layers,
    evidenceCoherence,
    runAt: new Date().toISOString(),
  };
  onProgress?.({
    stage: "complete",
    detail: "All validation layers complete.",
    layers,
  });
  return result;
}

export type {
  CausalEvidence,
  DecisionEvidence,
  GeneralizationEvidence,
  StructuralEvidence,
  ValidationEvidence,
  EvidenceCoherenceAssessment,
  ChannelEvidenceCoherence,
  ValidationLayerId,
  ValidationLayerResult,
  ValidationModelKind,
  ValidationOptions,
  ValidationProgress,
  ValidationResult,
  ValidationStatus,
  ValidationTest,
} from "./types";
export { assessExternalAnchorEligibility } from "./causal";
export {
  assessEvidenceCoherence,
  rescueIndustryPriorChannels,
} from "./evidence";
