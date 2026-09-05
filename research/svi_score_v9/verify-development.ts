import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { SVI_SCORE_V9_CONTRACT, SVI_SCORE_V9_VERSION } from "./contract";

function invariant(condition: unknown, detail: string): asserts condition {
  if (!condition) throw new Error(`V9 development verification failed: ${detail}.`);
}

const sha256 = (value: Buffer) => createHash("sha256").update(value).digest("hex");

const artifactPath = resolve(
  "research/svi_score_v9/artifacts/svi-score-v9-development.json",
);
const datasetPath = resolve(".flux-artifacts/svi-score-v9/development-dataset.json");
const predictionsPath = resolve(
  ".flux-artifacts/svi-score-v9/cross-fitted-predictions.json",
);
const modelPath = resolve(".flux-artifacts/svi-score-v9/development-selector.json");
const mechanismHoldoutPath = resolve(
  "research/svi_score_v9/artifacts/svi-score-v9-mechanism-holdout.json",
);
const [artifactSource, datasetSource, predictionsSource, modelSource, mechanismHoldoutSource] =
  await Promise.all([
    readFile(artifactPath),
    readFile(datasetPath),
    readFile(predictionsPath),
    readFile(modelPath),
    readFile(mechanismHoldoutPath),
  ]);
const artifact = JSON.parse(artifactSource.toString("utf8")) as {
  version: string;
  activation: string;
  provenance: {
    datasetSha256: string;
    predictionsSha256: string;
    modelSha256: string;
    v8AuditDereferenced: boolean;
    nutsUsed: boolean;
  };
  v9: {
    at100PercentCoverage: { theoremViolations: number };
    riskCoverageCurve: { points: unknown[] };
  };
  checks: Record<string, boolean>;
  conclusion: {
    allDevelopmentChecksPassed: boolean;
    eligibleToFreezeForNewAudit: boolean;
    v8AuditReused: boolean;
    newSealedAuditRequired: boolean;
    productionActivationPermitted: boolean;
  };
};
const dataset = JSON.parse(datasetSource.toString("utf8")) as {
  version: string;
  activation: string;
  featureNames: string[];
  rows: unknown[];
  summary: { businesses: number; candidatesPerBusiness: number };
  provenance: { v8AuditDereferenced: boolean; nutsUsed: boolean };
};
const mechanismHoldout = JSON.parse(mechanismHoldoutSource.toString("utf8")) as {
  version: string;
  activation: string;
  allChecksPassed: boolean;
  families: Array<{ heldOutFamily: string }>;
  checks: {
    allFiveMechanismsHeldOutExactlyOnce: boolean;
    theoremVerified: boolean;
    v8AuditUntouched: boolean;
    nutsExcluded: boolean;
  };
  v9: { at100PercentCoverage: { theoremViolations: number } };
};
invariant(artifact.version === SVI_SCORE_V9_VERSION, "artifact version changed");
invariant(dataset.version === SVI_SCORE_V9_VERSION, "dataset version changed");
invariant(
  mechanismHoldout.version === SVI_SCORE_V9_VERSION,
  "mechanism-holdout version changed",
);
invariant(artifact.activation === "development-only", "artifact escaped development");
invariant(dataset.activation === "development-only", "dataset escaped development");
invariant(
  mechanismHoldout.activation === "development-only",
  "mechanism holdout escaped development",
);
invariant(dataset.summary.businesses === 420, "business count changed");
invariant(dataset.summary.candidatesPerBusiness === 48, "candidate count changed");
invariant(dataset.rows.length === 20_160, "row count changed");
invariant(
  dataset.featureNames.length === SVI_SCORE_V9_CONTRACT.tokens.count,
  "token count changed",
);
invariant(artifact.provenance.datasetSha256 === sha256(datasetSource), "dataset hash changed");
invariant(
  artifact.provenance.predictionsSha256 === sha256(predictionsSource),
  "prediction hash changed",
);
invariant(artifact.provenance.modelSha256 === sha256(modelSource), "model hash changed");
invariant(
  !artifact.provenance.v8AuditDereferenced && !dataset.provenance.v8AuditDereferenced,
  "V8 audit was dereferenced",
);
invariant(!artifact.provenance.nutsUsed && !dataset.provenance.nutsUsed, "NUTS entered V9");
invariant(!artifact.conclusion.v8AuditReused, "V8 audit was reused");
invariant(artifact.conclusion.newSealedAuditRequired, "new audit requirement was removed");
invariant(!artifact.conclusion.productionActivationPermitted, "V9 was promoted prematurely");
invariant(
  artifact.v9.at100PercentCoverage.theoremViolations === 0,
  "selection theorem failed",
);
invariant(
  artifact.v9.riskCoverageCurve.points.length === 9,
  "risk-coverage curve is incomplete",
);
invariant(mechanismHoldout.allChecksPassed, "mechanism-holdout checks failed");
invariant(
  mechanismHoldout.families.length === 5 &&
    mechanismHoldout.checks.allFiveMechanismsHeldOutExactlyOnce,
  "mechanism-family holdout is incomplete",
);
invariant(
  mechanismHoldout.v9.at100PercentCoverage.theoremViolations === 0 &&
    mechanismHoldout.checks.theoremVerified,
  "mechanism-holdout selection theorem failed",
);
invariant(
  mechanismHoldout.checks.v8AuditUntouched,
  "mechanism holdout touched the V8 audit",
);
invariant(mechanismHoldout.checks.nutsExcluded, "NUTS entered mechanism holdout");
invariant(
  artifact.conclusion.allDevelopmentChecksPassed ===
    Object.values(artifact.checks).every(Boolean),
  "development conclusion does not match checks",
);
process.stdout.write(`${JSON.stringify({
  verified: true,
  version: artifact.version,
  developmentBusinesses: dataset.summary.businesses,
  candidatesPerBusiness: dataset.summary.candidatesPerBusiness,
  tokens: dataset.featureNames.length,
  allDevelopmentChecksPassed: artifact.conclusion.allDevelopmentChecksPassed,
  eligibleToFreezeForNewAudit: artifact.conclusion.eligibleToFreezeForNewAudit,
  v8AuditReused: false,
  newSealedAuditRequired: true,
  nutsUsed: false,
  mechanismFamiliesHeldOut: mechanismHoldout.families.length,
  mechanismHoldoutChecksPassed: mechanismHoldout.allChecksPassed,
}, null, 2)}\n`);
