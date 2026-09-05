import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { SVI_SCORE_V8_CONTRACT, SVI_SCORE_V8_VERSION } from "./contract";

function invariant(condition: unknown, detail: string): asserts condition {
  if (!condition) throw new Error(`V8 freeze verification failed: ${detail}.`);
}

function sha256(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

const artifactPath = resolve(
  "research/svi_score_v8/artifacts/svi-score-v8-development.json",
);
const predictionPath = resolve(
  ".flux-artifacts/svi-score-v8/cross-fitted-predictions.json",
);
const modelPath = resolve(
  ".flux-artifacts/svi-score-v8/frozen-development-selector.json",
);
const datasetPath = resolve(
  ".flux-artifacts/svi-score-v8/development-dataset.json",
);
const [artifactSource, predictionSource, modelSource, datasetSource] =
  await Promise.all([
    readFile(artifactPath),
    readFile(predictionPath),
    readFile(modelPath),
    readFile(datasetPath),
  ]);
const artifact = JSON.parse(artifactSource.toString("utf8")) as {
  version: string;
  activation: string;
  provenance: {
    datasetSha256: string;
    predictionsSha256: string;
    frozenModelSha256: string;
    auditAccessed: boolean;
    nutsUsed: boolean;
  };
  development: { theoremViolations: number; promotionCoverage: number };
  conclusion: {
    sealedAuditMayOpen: boolean;
    sealedAuditAccessed: boolean;
    productionActivationPermitted: boolean;
  };
};
const dataset = JSON.parse(datasetSource.toString("utf8")) as {
  version: string;
  featureNames: string[];
  summary: { businesses: number; rows: number };
  provenance: { auditAccessed: boolean; nutsUsed: boolean };
};
invariant(artifact.version === SVI_SCORE_V8_VERSION, "artifact version changed");
invariant(dataset.version === SVI_SCORE_V8_VERSION, "dataset version changed");
invariant(artifact.activation === "development-only", "activation escaped development");
invariant(dataset.summary.businesses === 420, "development business count changed");
invariant(dataset.summary.rows === 20_160, "development row count changed");
invariant(dataset.featureNames.length === SVI_SCORE_V8_CONTRACT.tokens.count, "token count changed");
invariant(artifact.provenance.datasetSha256 === sha256(datasetSource), "dataset hash changed");
invariant(artifact.provenance.predictionsSha256 === sha256(predictionSource), "prediction hash changed");
invariant(artifact.provenance.frozenModelSha256 === sha256(modelSource), "model hash changed");
invariant(!artifact.provenance.auditAccessed && !dataset.provenance.auditAccessed, "audit was accessed");
invariant(!artifact.provenance.nutsUsed && !dataset.provenance.nutsUsed, "NUTS entered research");
invariant(!artifact.conclusion.sealedAuditAccessed, "audit conclusion changed");
invariant(!artifact.conclusion.productionActivationPermitted, "development was promoted");
invariant(artifact.development.theoremViolations === 0, "regret theorem failed");
process.stdout.write(`${JSON.stringify({
  verified: true,
  version: artifact.version,
  developmentBusinesses: dataset.summary.businesses,
  developmentRows: dataset.summary.rows,
  tokens: dataset.featureNames.length,
  promotionCoverage: artifact.development.promotionCoverage,
  sealedAuditMayOpen: artifact.conclusion.sealedAuditMayOpen,
  sealedAuditAccessed: false,
  nutsUsed: false,
}, null, 2)}\n`);
