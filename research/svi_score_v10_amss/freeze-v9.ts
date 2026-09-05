import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import { SVI_SCORE_V3_CONTRACT } from "../svi_score_v3/contract";
import {
  SVI_SCORE_V8_CONTRACT,
  V8_RISK_PREFERENCE,
} from "../svi_score_v8/contract";
import {
  SVI_SCORE_V9_CONTRACT,
  SVI_SCORE_V9_VERSION,
  V9_POSTERIOR_TOKEN_NAMES,
} from "../svi_score_v9/contract";

const SNAPSHOT_VERSION = "flux-svi-score-v10-amss-v9-freeze-v1";
const SNAPSHOT_ROOT = resolve("research/svi_score_v10_amss/frozen-v9");
const MANIFEST_PATH = resolve(SNAPSHOT_ROOT, "manifest.json");

const SELECTOR_SOURCE = ".flux-artifacts/svi-score-v9/development-selector.json";
const DEVELOPMENT_SOURCE =
  "research/svi_score_v9/artifacts/svi-score-v9-development.json";

const SCIENTIFIC_SOURCE_PATHS = [
  "research/svi_score_v3/contract.ts",
  "research/svi_score_v8/contract.ts",
  "research/svi_score_v9/contract.ts",
  "research/svi_score_v9/audit-tokens.ts",
  "research/svi_score_v9/prepare-dataset.ts",
  "research/svi_score_v9/train_selector.py",
] as const;

interface FrozenFile {
  path: string;
  bytes: number;
  sha256: string;
}

function invariant(condition: unknown, detail: string): asserts condition {
  if (!condition) throw new Error(`V10 AMSS V9 freeze failed: ${detail}.`);
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

async function atomicWrite(path: string, value: Buffer | string): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp`;
  await writeFile(temporary, value);
  await rename(temporary, path);
}

async function mustNotExist(path: string): Promise<void> {
  try {
    await readFile(path);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    throw error;
  }
  throw new Error(
    "V10 AMSS V9 freeze already exists; verify it instead of overwriting it",
  );
}

await mustNotExist(MANIFEST_PATH);

const [selectorSource, developmentSource] = await Promise.all([
  readFile(resolve(SELECTOR_SOURCE)),
  readFile(resolve(DEVELOPMENT_SOURCE)),
]);
const selector = JSON.parse(selectorSource.toString("utf8")) as {
  artifactId: string;
  version: string;
  activation: string;
  featureNames: string[];
  model: { id: string };
  models: unknown[];
  policy: { danger_penalty: number; uncertainty_penalty: number };
  provenance: { datasetSha256: string; v8AuditDereferenced: boolean; nutsUsed: boolean };
};
const development = JSON.parse(developmentSource.toString("utf8")) as {
  version: string;
  activation: string;
  contract: unknown;
  developmentSelector: { model: string; policy: string; path: string };
  provenance: { modelSha256: string; v8AuditDereferenced: boolean; nutsUsed: boolean };
  conclusion: {
    allDevelopmentChecksPassed: boolean;
    eligibleToFreezeForNewAudit: boolean;
    productionActivationPermitted: boolean;
  };
};

invariant(selector.version === SVI_SCORE_V9_VERSION, "selector version changed");
invariant(development.version === SVI_SCORE_V9_VERSION, "development version changed");
invariant(selector.activation === "development-only", "selector activation changed");
invariant(development.activation === "development-only", "development activation changed");
invariant(
  development.provenance.modelSha256 === sha256(selectorSource),
  "selector bytes no longer match the verified development artifact",
);
invariant(
  selector.featureNames.length === SVI_SCORE_V9_CONTRACT.tokens.count,
  "selector token count changed",
);
invariant(
  new Set(selector.featureNames).size === selector.featureNames.length,
  "selector token registry contains duplicates",
);
invariant(
  selector.model.id === development.developmentSelector.model,
  "selector model does not match the development winner",
);
invariant(
  selector.policy.danger_penalty === 0 && selector.policy.uncertainty_penalty === 0.25,
  "selector promotion policy changed",
);
invariant(
  V8_RISK_PREFERENCE.mean === 0.65 &&
    V8_RISK_PREFERENCE.p90 === 0.35 &&
    V8_RISK_PREFERENCE.mean + V8_RISK_PREFERENCE.p90 === 1,
  "economic risk preference changed",
);
invariant(
  development.conclusion.allDevelopmentChecksPassed &&
    development.conclusion.eligibleToFreezeForNewAudit &&
    !development.conclusion.productionActivationPermitted,
  "V9 is not eligible for a frozen external audit",
);
invariant(
  !selector.provenance.v8AuditDereferenced &&
    !selector.provenance.nutsUsed &&
    !development.provenance.v8AuditDereferenced &&
    !development.provenance.nutsUsed,
  "V9 provenance boundary changed",
);

const forbidden = SVI_SCORE_V9_CONTRACT.tokens.forbidden;
invariant(
  selector.featureNames.every((name) =>
    forbidden.every((term) => !name.toLowerCase().includes(term))
  ),
  "truth or identity entered the frozen token registry",
);
invariant(
  V9_POSTERIOR_TOKEN_NAMES.every((name) => selector.featureNames.includes(name)),
  "posterior token registry is incomplete",
);

const selectorTarget = "selector.json";
const tokenTarget = "token-registry.json";
const contractTarget = "scientific-contract.json";
const developmentTarget = "development-verification.json";

const tokenRegistry = {
  snapshotVersion: SNAPSHOT_VERSION,
  selectorVersion: SVI_SCORE_V9_VERSION,
  count: selector.featureNames.length,
  orderedNamesSha256: sha256(JSON.stringify(selector.featureNames)),
  names: selector.featureNames,
};
const scientificContract = {
  snapshotVersion: SNAPSHOT_VERSION,
  selectorVersion: SVI_SCORE_V9_VERSION,
  candidateContract: {
    candidatesPerBusiness: SVI_SCORE_V9_CONTRACT.cohort.candidatesPerBusiness,
    candidateOrderInvariant: SVI_SCORE_V9_CONTRACT.tokens.candidateOrderInvariant,
    channelOrderInvariant: SVI_SCORE_V9_CONTRACT.tokens.channelOrderInvariant,
  },
  inference: SVI_SCORE_V3_CONTRACT.inference,
  tokens: SVI_SCORE_V9_CONTRACT.tokens,
  targets: SVI_SCORE_V9_CONTRACT.targets,
  model: SVI_SCORE_V9_CONTRACT.model,
  promotion: {
    riskPreference: V8_RISK_PREFERENCE,
    selectorPolicy: selector.policy,
    hardGates: SVI_SCORE_V8_CONTRACT.validity.hardGates,
    softRiskSignals: SVI_SCORE_V8_CONTRACT.validity.softRiskSignals,
  },
  externalAuditBoundary: {
    selectorRetrainingPermitted: false,
    tokenAdditionOrRemovalPermitted: false,
    thresholdTuningPermitted: false,
    hiddenAmssTruthPermittedBeforeSelection: false,
    technicalAdapterRepairsPermittedBeforeOutcomeOpening: true,
  },
};
const developmentVerification = {
  snapshotVersion: SNAPSHOT_VERSION,
  selectorVersion: SVI_SCORE_V9_VERSION,
  selectorArtifactId: selector.artifactId,
  selectorModel: selector.model.id,
  ensembleMembers: selector.models.length,
  sourceModelSha256: sha256(selectorSource),
  developmentConclusion: development.conclusion,
  sourceDevelopmentArtifactSha256: sha256(developmentSource),
};

await atomicWrite(resolve(SNAPSHOT_ROOT, selectorTarget), selectorSource);
await atomicWrite(
  resolve(SNAPSHOT_ROOT, tokenTarget),
  `${JSON.stringify(tokenRegistry, null, 2)}\n`,
);
await atomicWrite(
  resolve(SNAPSHOT_ROOT, contractTarget),
  `${JSON.stringify(scientificContract, null, 2)}\n`,
);
await atomicWrite(
  resolve(SNAPSHOT_ROOT, developmentTarget),
  `${JSON.stringify(developmentVerification, null, 2)}\n`,
);

const sourceInputs: FrozenFile[] = [];
for (const sourcePath of SCIENTIFIC_SOURCE_PATHS) {
  const source = await readFile(resolve(sourcePath));
  const target = `source/${sourcePath}`;
  await atomicWrite(resolve(SNAPSHOT_ROOT, target), source);
  sourceInputs.push({ path: target, bytes: source.length, sha256: sha256(source) });
}

const frozenPaths = [
  selectorTarget,
  tokenTarget,
  contractTarget,
  developmentTarget,
  ...sourceInputs.map(({ path }) => path),
];
const frozenFiles: FrozenFile[] = [];
for (const path of frozenPaths) {
  const value = await readFile(resolve(SNAPSHOT_ROOT, path));
  frozenFiles.push({ path, bytes: value.length, sha256: sha256(value) });
}

const manifest = {
  snapshotVersion: SNAPSHOT_VERSION,
  createdAt: new Date().toISOString(),
  purpose:
    "immutable V9 selector input to the AMSS external-transport validation",
  status: "frozen-before-amss-installation-or-data-generation",
  selector: {
    version: SVI_SCORE_V9_VERSION,
    artifactId: selector.artifactId,
    sha256: sha256(selectorSource),
    model: selector.model.id,
    ensembleMembers: selector.models.length,
    featureCount: selector.featureNames.length,
  },
  risk: {
    meanWeight: V8_RISK_PREFERENCE.mean,
    p90Weight: V8_RISK_PREFERENCE.p90,
    dangerPenalty: selector.policy.danger_penalty,
    uncertaintyPenalty: selector.policy.uncertainty_penalty,
  },
  gates: SVI_SCORE_V8_CONTRACT.validity.hardGates,
  truthBoundary: {
    amssObservedInputsPermitted: true,
    amssGroundTruthBeforeSelection: false,
    v9RetrainingPermitted: false,
  },
  sourceProvenance: {
    selector: SELECTOR_SOURCE,
    development: DEVELOPMENT_SOURCE,
    selectorSha256: sha256(selectorSource),
    developmentSha256: sha256(developmentSource),
  },
  frozenFiles,
};
const manifestSource = `${JSON.stringify(manifest, null, 2)}\n`;
await atomicWrite(MANIFEST_PATH, manifestSource);
await atomicWrite(resolve(SNAPSHOT_ROOT, "manifest.sha256"), `${sha256(manifestSource)}\n`);

process.stdout.write(`${JSON.stringify({
  frozen: true,
  snapshotVersion: SNAPSHOT_VERSION,
  selectorVersion: SVI_SCORE_V9_VERSION,
  selectorSha256: sha256(selectorSource),
  tokens: selector.featureNames.length,
  candidatesPerBusiness: SVI_SCORE_V9_CONTRACT.cohort.candidatesPerBusiness,
  riskPreference: V8_RISK_PREFERENCE,
  frozenFiles: frozenFiles.length,
  manifest: "research/svi_score_v10_amss/frozen-v9/manifest.json",
}, null, 2)}\n`);
