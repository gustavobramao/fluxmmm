import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { loadSviScoreV4DevelopmentRows } from "../svi_score_v4/development-rows";
import type { V5DDevelopmentReceipt } from "../svi_score_v5d/evaluate";
import { evaluateV7, type V7Prediction } from "./evaluate";

interface PredictionArtifact {
  activation: string;
  provenance: {
    datasetSha256: string;
    freshValidationAccessed: boolean;
    auditAccessed: boolean;
  };
  architecture: unknown;
  modelReceipts: unknown[];
  predictions: V7Prediction[];
}

interface V5DArtifact {
  activation: string;
  provenance: {
    freshValidationAccessed: boolean;
    auditAccessed: boolean;
  };
  result: V5DDevelopmentReceipt;
}

function reduction(candidate: number, comparator: number): number {
  return (comparator - candidate) / Math.max(Math.abs(comparator), 1e-12);
}

const datasetPath = resolve(".flux-artifacts/svi-score-v7/dataset.json");
const predictionsPath = resolve(".flux-artifacts/svi-score-v7/predictions.json");
const v5dPath = resolve("research/svi_score_v5d/artifacts/svi-score-v5d-development.json");
const [datasetSource, predictionSource, v5dSource] = await Promise.all([
  readFile(datasetPath),
  readFile(predictionsPath),
  readFile(v5dPath),
]);
const predictions = JSON.parse(predictionSource.toString("utf8")) as PredictionArtifact;
const v5d = JSON.parse(v5dSource.toString("utf8")) as V5DArtifact;
if (
  predictions.activation !== "research-only" ||
  predictions.provenance.datasetSha256 !==
    createHash("sha256").update(datasetSource).digest("hex") ||
  predictions.provenance.freshValidationAccessed ||
  predictions.provenance.auditAccessed ||
  v5d.activation !== "research-only" ||
  v5d.provenance.freshValidationAccessed ||
  v5d.provenance.auditAccessed
) {
  throw new Error("V7 requires matching research-only data with validation/audit closed.");
}
const rows = await loadSviScoreV4DevelopmentRows();
const result = evaluateV7(rows, predictions.predictions);
const v5dBestPrimary = [...v5d.result.algorithms].sort(
  (left, right) => left.primaryObjective - right.primaryObjective,
)[0];
const v7BestPrimary = [...result.algorithms].sort(
  (left, right) => left.primaryObjective - right.primaryObjective,
)[0];
const v5dFull = v5d.result.algorithms[0].checkpoints.find(
  (checkpoint) => checkpoint.budget === 48,
)!;
const v7Full = result.algorithms[0].checkpoints.find(
  (checkpoint) => checkpoint.budget === 48,
)!;
const checks = {
  candidateCorrelationImproved:
    result.proxyAssessment.candidateLevelCorrelation >
      v5d.result.proxyAssessment.candidateLevelCorrelation,
  economicallyWeightedPairwiseAccuracyImproved:
    result.proxyAssessment.economicallyWeightedPairwiseAccuracy >
      v5d.result.proxyAssessment.economicallyWeightedPairwiseAccuracy,
  fullPoolObjectiveImproved:
    v7Full.selectionObjective < v5dFull.selectionObjective,
  bestPrimaryObjectiveImproved:
    v7BestPrimary.primaryObjective < v5dBestPrimary.primaryObjective,
  p90Calibrated:
    result.proxyAssessment.p90Coverage >= 0.87 &&
      result.proxyAssessment.p90Coverage <= 0.93,
  fullPoolAlgorithmAgreement: result.fullPoolAlgorithmAgreement,
};
const comparisonVersusV5D = {
  proxy: {
    v7Correlation: result.proxyAssessment.candidateLevelCorrelation,
    v5dCorrelation: v5d.result.proxyAssessment.candidateLevelCorrelation,
    v7WeightedPairwiseAccuracy:
      result.proxyAssessment.economicallyWeightedPairwiseAccuracy,
    v5dWeightedPairwiseAccuracy:
      v5d.result.proxyAssessment.economicallyWeightedPairwiseAccuracy,
    v7MeanAbsoluteError: result.proxyAssessment.meanAbsoluteError,
    v5dMeanAbsoluteError: v5d.result.proxyAssessment.meanAbsoluteError,
    v7P90Coverage: result.proxyAssessment.p90Coverage,
    v5dP90Coverage: v5d.result.proxyAssessment.p90Coverage,
    v7DangerousFalseChampionShare:
      result.proxyAssessment.dangerousFalseChampionShare,
    v5dDangerousFalseChampionShare:
      v5d.result.proxyAssessment.dangerousFalseChampionShare,
  },
  bestAt16: {
    v7Algorithm: v7BestPrimary.algorithm,
    v7Objective: v7BestPrimary.primaryObjective,
    v5dAlgorithm: v5dBestPrimary.algorithm,
    v5dObjective: v5dBestPrimary.primaryObjective,
    relativeObjectiveReduction: reduction(
      v7BestPrimary.primaryObjective,
      v5dBestPrimary.primaryObjective,
    ),
  },
  fullPool: {
    v7Objective: v7Full.selectionObjective,
    v5dObjective: v5dFull.selectionObjective,
    relativeObjectiveReduction: reduction(
      v7Full.selectionObjective,
      v5dFull.selectionObjective,
    ),
  },
};
const artifact = {
  artifactId: "flux-svi-score-v7-multitask-economic-surrogate-development-v1",
  generatedAt: new Date().toISOString(),
  activation: "research-only",
  stage: "multitask-economic-surrogate-development-complete",
  provenance: {
    datasetSha256: createHash("sha256").update(datasetSource).digest("hex"),
    predictionsSha256: createHash("sha256").update(predictionSource).digest("hex"),
    v5dSha256: createHash("sha256").update(v5dSource).digest("hex"),
    priorOpenedValidationReusedAsDevelopment: true,
    freshValidationAccessed: false,
    auditAccessed: false,
  },
  modelReceipts: predictions.modelReceipts,
  result,
  comparisonVersusV5D,
  checks,
  conclusion: {
    allDevelopmentChecksPassed: Object.values(checks).every(Boolean),
    unseenBusinessEvidence: "grouped-cross-fitted-development-only",
    freshGeneralizationEstablished: false,
    publicationReady: false,
    sotaClaimPermitted: false,
  },
  nextGate: Object.values(checks).every(Boolean)
    ? "Freeze V7 and generate a completely fresh multi-simulator validation cohort. Do not open the sealed audit."
    : "Keep V7 in development. Diagnose failed checks without opening fresh validation or the sealed audit.",
};
const directory = resolve("research/svi_score_v7/artifacts");
const artifactPath = resolve(directory, "svi-score-v7-development.json");
await mkdir(directory, { recursive: true });
const temporary = `${artifactPath}.tmp`;
await writeFile(temporary, JSON.stringify(artifact, null, 2));
await rename(temporary, artifactPath);
process.stdout.write(`${JSON.stringify({
  artifactPath,
  proxyAssessment: result.proxyAssessment,
  primaryWinner: result.primaryWinner,
  primary: Object.fromEntries(result.algorithms.map((algorithm) => [
    algorithm.algorithm,
    {
      objective: algorithm.primaryObjective,
      meanLoss: algorithm.primary.meanLoss,
      cvar90Loss: algorithm.primary.cvar90Loss,
      worstFamilyMeanLoss: algorithm.primary.worstFamilyMeanLoss,
      oracleRecall: algorithm.primary.economicOracleRecall,
    },
  ])),
  comparisonVersusV5D,
  checks,
  conclusion: artifact.conclusion,
}, null, 2)}\n`);
