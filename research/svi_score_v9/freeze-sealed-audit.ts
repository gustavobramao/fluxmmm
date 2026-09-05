import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { SVI_SCORE_V9_AUDIT_CONTRACT } from "./audit-contract";
import { SVI_SCORE_V9_VERSION } from "./contract";

const ROOT = resolve(".flux-artifacts/svi-score-v9/audit");
const OUTPUT = resolve(ROOT, "sealed-audit-protocol-freeze.json");
const OPEN = resolve(ROOT, "sealed-audit-open-receipt.json");
const AUDIT_DATASET = resolve(ROOT, "sealed-audit-dataset.json");
const AUDIT_RESULT = resolve(
  "research/svi_score_v9/artifacts/svi-score-v9-sealed-audit.json",
);
const DEVELOPMENT = resolve(
  "research/svi_score_v9/artifacts/svi-score-v9-development.json",
);
const MECHANISM = resolve(
  "research/svi_score_v9/artifacts/svi-score-v9-mechanism-holdout.json",
);
const V9_MODEL_SOURCE = resolve(
  ".flux-artifacts/svi-score-v9/development-selector.json",
);
const V8_MODEL_SOURCE = resolve(
  ".flux-artifacts/svi-score-v8/frozen-development-selector.json",
);
const V9_MODEL_FROZEN = resolve(ROOT, "frozen-v9-selector.json");
const V8_MODEL_FROZEN = resolve(ROOT, "frozen-v8-selector.json");

const SOURCE_INPUTS = {
  v9Contract: "research/svi_score_v9/contract.ts",
  v9AuditContract: "research/svi_score_v9/audit-contract.ts",
  v9AuditArguments: "research/svi_score_v9/arguments.ts",
  v9AuditDesign: "research/svi_score_v9/audit-design.ts",
  v9AuditTokens: "research/svi_score_v9/audit-tokens.ts",
  v9AuditPreparer: "research/svi_score_v9/prepare-sealed-audit.ts",
  v9AuditDatasetBuilder: "research/svi_score_v9/build-sealed-audit-dataset.ts",
  v9AuditScorer: "research/svi_score_v9/score-sealed-audit.py",
  v9AuditVerifier: "research/svi_score_v9/verify-sealed-audit.ts",
  v9Learner: "research/svi_score_v9/train_selector.py",
  v8Learner: "research/svi_score_v8/train_selector.py",
  candidateCohort: "research/mcmc_score_v3/cohort.ts",
  generatorPopulation: "research/score_v3/population.ts",
  syntheticSimulator: "research/score_v2/simulator.ts",
  truthEvaluator: "research/score_v2/evaluate.ts",
  temporalFeatures: "research/svi_score_v4/temporal.ts",
  posteriorFeatures: "research/svi_score_v5d_svi/posterior.ts",
  inferenceCompiler: "lib/mmm/sampling.ts",
  sviRuntime: "scripts/svi_batch.py",
  cloudRuntime: "scripts/cloud_svi_shard_task.py",
  cloudManifestSharder: "scripts/build_cloud_svi_shards.py",
  cloudContainer: "research/svi_score_v3/cloud/Dockerfile.sharded",
  cloudBuild: "research/svi_score_v3/cloud/cloudbuild-sharded.yaml",
  cloudJob: "research/svi_score_v9/cloud/sealed-audit-003-cloud-run.yaml",
  cloudProtocol: "research/svi_score_v9/AUDIT_003_PROTOCOL.md",
  pythonDependencies: "research/svi_score_v3/cloud/requirements.lock",
  nodeLockfile: "pnpm-lock.yaml",
} as const;

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

async function exists(path: string): Promise<boolean> {
  try {
    await readFile(path);
    return true;
  } catch (error) {
    if (
      error instanceof Error && "code" in error &&
      (error as NodeJS.ErrnoException).code === "ENOENT"
    ) return false;
    throw error;
  }
}

async function immutableCopy(sourcePath: string, targetPath: string): Promise<Buffer> {
  const source = await readFile(sourcePath);
  try {
    await writeFile(targetPath, source, { flag: "wx" });
  } catch (error) {
    if (!(
      error instanceof Error && "code" in error &&
      (error as NodeJS.ErrnoException).code === "EEXIST"
    )) throw error;
    const existing = await readFile(targetPath);
    if (sha256(existing) !== sha256(source)) {
      throw new Error(`Frozen selector already exists with different bytes: ${targetPath}.`);
    }
  }
  return source;
}

await mkdir(ROOT, { recursive: true });
if (
  await exists(OUTPUT) || await exists(OPEN) || await exists(AUDIT_DATASET) ||
  await exists(AUDIT_RESULT)
) {
  throw new Error("Cannot freeze: the V9 sealed audit is already frozen, opened, or scored.");
}

const [developmentSource, mechanismSource] = await Promise.all([
  readFile(DEVELOPMENT),
  readFile(MECHANISM),
]);
const development = JSON.parse(developmentSource.toString("utf8")) as {
  version: string;
  checks: Record<string, boolean>;
  conclusion: {
    allDevelopmentChecksPassed: boolean;
    eligibleToFreezeForNewAudit: boolean;
    v8AuditReused: boolean;
  };
  provenance: { v8AuditDereferenced: boolean; nutsUsed: boolean };
};
const mechanism = JSON.parse(mechanismSource.toString("utf8")) as {
  version: string;
  allChecksPassed: boolean;
  checks: Record<string, boolean>;
};
if (
  development.version !== SVI_SCORE_V9_VERSION ||
  mechanism.version !== SVI_SCORE_V9_VERSION ||
  !development.conclusion.allDevelopmentChecksPassed ||
  !development.conclusion.eligibleToFreezeForNewAudit ||
  development.conclusion.v8AuditReused ||
  development.provenance.v8AuditDereferenced ||
  development.provenance.nutsUsed ||
  !Object.values(development.checks).every(Boolean) ||
  !mechanism.allChecksPassed ||
  !Object.values(mechanism.checks).every(Boolean)
) {
  throw new Error("V9 did not satisfy the pre-audit development contract.");
}

const [v9ModelSource, v8ModelSource] = await Promise.all([
  immutableCopy(V9_MODEL_SOURCE, V9_MODEL_FROZEN),
  immutableCopy(V8_MODEL_SOURCE, V8_MODEL_FROZEN),
]);
const sources = await Promise.all(
  Object.entries(SOURCE_INPUTS).map(async ([id, path]) => {
    const source = await readFile(resolve(path));
    return [id, { path, sha256: sha256(source) }] as const;
  }),
);
const payload = {
  artifactId: "flux-svi-score-v9-sealed-audit-protocol-freeze-v1",
  version: SVI_SCORE_V9_VERSION,
  frozenAt: new Date().toISOString(),
  status: "frozen-sealed",
  contract: SVI_SCORE_V9_AUDIT_CONTRACT,
  developmentSha256: sha256(developmentSource),
  mechanismHoldoutSha256: sha256(mechanismSource),
  frozenV9SelectorSha256: sha256(v9ModelSource),
  frozenV8SelectorSha256: sha256(v8ModelSource),
  sourceSha256: Object.fromEntries(sources),
  auditAccessed: false,
  hiddenTruthDereferenced: false,
  nutsUsed: false,
};
await writeFile(OUTPUT, JSON.stringify(payload, null, 2), { flag: "wx" });
process.stdout.write(`${JSON.stringify({
  path: OUTPUT,
  status: payload.status,
  businesses: payload.contract.cohort.businesses,
  candidateFits: payload.contract.cohort.candidateRows,
  sourceFilesFrozen: sources.length,
  auditAccessed: false,
  hiddenTruthDereferenced: false,
  nutsUsed: false,
}, null, 2)}\n`);
