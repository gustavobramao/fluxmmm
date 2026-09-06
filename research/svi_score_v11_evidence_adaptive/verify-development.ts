import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  SVI_SCORE_V11_EA_CONTRACT,
  SVI_SCORE_V11_EA_VERSION,
  V11_EA_CONTEXT_NAMES,
  V11_EA_EXPERTS,
} from "./contract";

function invariant(condition: unknown, detail: string): asserts condition {
  if (!condition) throw new Error(`Evidence-adaptive verification failed: ${detail}.`);
}

const sha256 = (value: Buffer) => createHash("sha256").update(value).digest("hex");

const datasetPath = resolve(
  ".flux-artifacts/svi-score-v11-evidence-adaptive/development-dataset.json",
);
const predictionsPath = resolve(
  ".flux-artifacts/svi-score-v11-evidence-adaptive/cross-fitted-predictions.json",
);
const modelPath = resolve(
  ".flux-artifacts/svi-score-v11-evidence-adaptive/development-selector.json",
);
const artifactPath = resolve(
  "research/svi_score_v11_evidence_adaptive/artifacts/svi-score-v11-evidence-adaptive-development.json",
);
const [datasetSource, predictionsSource, modelSource, artifactSource] = await Promise.all([
  readFile(datasetPath),
  readFile(predictionsPath),
  readFile(modelPath),
  readFile(artifactPath),
]);
const dataset = JSON.parse(datasetSource.toString("utf8"));
const predictions = JSON.parse(predictionsSource.toString("utf8"));
const model = JSON.parse(modelSource.toString("utf8"));
const artifact = JSON.parse(artifactSource.toString("utf8"));

invariant(dataset.version === SVI_SCORE_V11_EA_VERSION, "dataset version changed");
invariant(model.version === SVI_SCORE_V11_EA_VERSION, "model version changed");
invariant(artifact.version === SVI_SCORE_V11_EA_VERSION, "result version changed");
invariant(dataset.activation === "development-only", "dataset escaped development");
invariant(artifact.activation === "development-only", "result escaped development");
invariant(dataset.summary.businesses === 420, "business count changed");
invariant(dataset.summary.candidateMmmFits === 20_160, "candidate fit count changed");
invariant(dataset.summary.developmentSets === 1_260, "paired set count changed");
invariant(dataset.summary.pairedArmSets === 840, "paired arm count changed");
invariant(dataset.summary.contextTokens === V11_EA_CONTEXT_NAMES.length, "context count changed");
invariant(dataset.summary.noExperimentBusinesses === 24, "no-experiment count changed");
invariant(predictions.selections.length === 420, "full sets were not all cross-fitted");
invariant(model.models.length === 5, "final ensemble size changed");
invariant(
  Object.keys(dataset.expertGroups).sort().join("|") === [...V11_EA_EXPERTS].sort().join("|"),
  "expert registry changed",
);
const expertTokens = Object.values(dataset.expertGroups).flat() as Array<{ index: number }>;
invariant(expertTokens.length === 232, "expert token assignment is incomplete");
invariant(new Set(expertTokens.map((item) => item.index)).size === 232, "expert groups overlap");
invariant(artifact.provenance.datasetSha256 === sha256(datasetSource), "dataset hash changed");
invariant(
  artifact.provenance.predictionsSha256 === sha256(predictionsSource),
  "prediction hash changed",
);
invariant(artifact.provenance.modelSha256 === sha256(modelSource), "model hash changed");
invariant(artifact.checks.theoremVerified, "selection theorem failed");
invariant(artifact.checks.evidenceGatesFinite, "evidence gate is non-finite");
invariant(artifact.checks.evidenceGatesSumToOne, "evidence gate does not sum to one");
invariant(dataset.provenance.posteriorRefits === 0, "posterior refits entered development");
invariant(
  !dataset.provenance.amssConfirmatoryCohortAccessed &&
    !artifact.provenance.amssConfirmatoryCohortAccessed,
  "AMSS confirmatory cohort was accessed",
);
invariant(!artifact.conclusion.confirmatoryClaimPermitted, "development was called confirmatory");
invariant(artifact.conclusion.freshIndependentAuditRequired, "fresh audit requirement removed");
invariant(!artifact.conclusion.amssCohortMayBeReused, "AMSS reuse was permitted");
invariant(!artifact.conclusion.productionActivationPermitted, "selector activated prematurely");
invariant(
  SVI_SCORE_V11_EA_CONTRACT.pairedRegimes.experimentPresenceCounterfactualGenerated === false,
  "experiment-removal counterfactual was overstated",
);

process.stdout.write(`${JSON.stringify({
  verified: true,
  version: artifact.version,
  businesses: dataset.summary.businesses,
  cachedCandidateFits: dataset.summary.candidateMmmFits,
  pairedDevelopmentSets: dataset.summary.developmentSets,
  contextTokens: dataset.summary.contextTokens,
  evidenceAdaptiveGate: artifact.evidenceAdaptiveRegretSet.averageEvidenceGate,
  developmentImprovedVsV9: artifact.conclusion.developmentImprovedVsV9,
  developmentImprovedVsPredictionOnly:
    artifact.conclusion.developmentImprovedVsPredictionOnly,
  posteriorRefits: 0,
  amssConfirmatoryCohortAccessed: false,
  freshIndependentAuditRequired: true,
}, null, 2)}\n`);
