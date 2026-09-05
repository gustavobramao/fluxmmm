import { spawn } from "node:child_process";
import { cpus } from "node:os";
import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { gzipSync, gunzipSync } from "node:zlib";
import { compileSamplingModel } from "../../lib/mmm/sampling";
import { sha256 as modelSha256 } from "../../lib/mmm/csv";
import { prepareMcmcResearchBusiness } from "../mcmc_score_v3/cohort";
import { syntheticIndustryPriorOverrides } from "../score_v6/evidence";
import type { SviInferenceContract } from "../svi_score_v3/contract";
import { sviDesignBusinesses } from "../svi_score_v3/design";
import { integerArgument } from "./arguments";
import { SVI_SCORE_V9_AUDIT_CONTRACT } from "./audit-contract";
import { v9SealedAuditBusinesses } from "./audit-design";
import { SVI_SCORE_V9_CONTRACT, SVI_SCORE_V9_VERSION } from "./contract";

const ROOT = resolve(".flux-artifacts/svi-score-v9/audit");
const PAYLOADS = resolve(ROOT, "payloads");
const PARTS = resolve(ROOT, "preparation-parts");
const FREEZE_PATH = resolve(ROOT, "sealed-audit-protocol-freeze.json");
const OPEN_PATH = resolve(ROOT, "sealed-audit-open-receipt.json");
const MANIFEST_PATH = resolve(ROOT, "sealed-audit-inference-manifest.json");
const CLOUD_MANIFEST_PATH = resolve(ROOT, "sealed-audit-cloud-manifest.json");
const REMOTE_PREFIX = "v9-sealed-audit-003";

interface FrozenSource {
  path: string;
  sha256: string;
}

interface ProtocolFreeze {
  version: string;
  status: string;
  contract: typeof SVI_SCORE_V9_AUDIT_CONTRACT;
  sourceSha256: Record<string, FrozenSource>;
  frozenV9SelectorSha256: string;
  frozenV8SelectorSha256: string;
}

export interface V9AuditInferenceItem {
  businessId: string;
  family: string;
  sourceSplit: string;
  seed: number;
  auditFamilyIndex: number;
  candidateId: string;
  evidenceArm: "experiments-only" | "benchmark-gap-fill";
  inferenceFingerprint: string;
  payloadPath: string;
  payloadObject: string;
  payloadSha256: string;
  outputObject: string;
}

function digest(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.tmp`;
  await writeFile(temporary, JSON.stringify(value));
  await rename(temporary, path);
}

async function verifyFrozenSources(freeze: ProtocolFreeze): Promise<void> {
  if (
    freeze.version !== SVI_SCORE_V9_VERSION ||
    freeze.status !== "frozen-sealed" ||
    JSON.stringify(freeze.contract) !== JSON.stringify(SVI_SCORE_V9_AUDIT_CONTRACT)
  ) throw new Error("V9 audit protocol freeze is invalid.");
  for (const source of Object.values(freeze.sourceSha256)) {
    if (digest(await readFile(resolve(source.path))) !== source.sha256) {
      throw new Error(`A frozen V9 audit input changed: ${source.path}.`);
    }
  }
  const [v9, v8] = await Promise.all([
    readFile(resolve(ROOT, "frozen-v9-selector.json")),
    readFile(resolve(ROOT, "frozen-v8-selector.json")),
  ]);
  if (
    digest(v9) !== freeze.frozenV9SelectorSha256 ||
    digest(v8) !== freeze.frozenV8SelectorSha256
  ) throw new Error("A frozen audit selector changed after protocol freeze.");
}

async function writePayload(path: string, value: unknown): Promise<Buffer> {
  const bytes = gzipSync(JSON.stringify(value));
  try {
    await writeFile(path, bytes, { flag: "wx" });
  } catch (error) {
    if (!(
      error instanceof Error && "code" in error &&
      (error as NodeJS.ErrnoException).code === "EEXIST"
    )) throw error;
    const existing = await readFile(path);
    if (digest(existing) !== digest(bytes)) {
      throw new Error(`Immutable V9 audit payload changed: ${path}.`);
    }
    const parsed = JSON.parse(gunzipSync(existing).toString("utf8")) as {
      fingerprint: string;
    };
    if (!parsed.fingerprint) throw new Error(`Invalid cached V9 payload: ${path}.`);
    return existing;
  }
  return bytes;
}

async function worker(shard: number, shards: number): Promise<void> {
  const freezeSource = await readFile(FREEZE_PATH);
  const freeze = JSON.parse(freezeSource.toString("utf8")) as ProtocolFreeze;
  await verifyFrozenSources(freeze);
  const protocolFreezeSha256 = digest(freezeSource);
  const design = v9SealedAuditBusinesses().filter((_, index) => index % shards === shard);
  const items: V9AuditInferenceItem[] = [];
  await mkdir(PAYLOADS, { recursive: true });
  for (let businessIndex = 0; businessIndex < design.length; businessIndex += 1) {
    const item = design[businessIndex];
    const prepared = await prepareMcmcResearchBusiness(
      item.scenario,
      item.family.id,
      "audit",
      undefined,
      { includeHiddenTruthLabels: false },
    );
    for (const candidate of prepared.candidates) {
      const model = compileSamplingModel(
        prepared.dataset,
        candidate.run,
        prepared.experiments,
        candidate.industryPriorChannels,
        syntheticIndustryPriorOverrides(prepared.business),
      );
      const contract = SVI_SCORE_V9_CONTRACT.inference.contract as SviInferenceContract;
      const inferenceFingerprint = await modelSha256(JSON.stringify({
        version: SVI_SCORE_V9_VERSION,
        protocolFreezeSha256,
        model,
        contract,
      }));
      const payloadPath = resolve(PAYLOADS, `${inferenceFingerprint}.payload.json.gz`);
      const payload = await writePayload(payloadPath, {
        fingerprint: inferenceFingerprint,
        model,
        contract,
      });
      items.push({
        businessId: item.scenario.id,
        family: item.family.id,
        sourceSplit: item.sourceSplit,
        seed: item.scenario.seed,
        auditFamilyIndex: item.auditFamilyIndex,
        candidateId: candidate.row.candidateId,
        evidenceArm: candidate.row.evidenceArm,
        inferenceFingerprint,
        payloadPath,
        payloadObject: `${REMOTE_PREFIX}/payloads/${inferenceFingerprint}.payload.json.gz`,
        payloadSha256: digest(payload),
        outputObject: `${REMOTE_PREFIX}/results/${inferenceFingerprint}.result.json.gz`,
      });
    }
    process.stdout.write(
      `V9 audit preparation ${shard + 1}/${shards}: ${businessIndex + 1}/${design.length} businesses\n`,
    );
  }
  await mkdir(PARTS, { recursive: true });
  await atomicJson(resolve(PARTS, `part-${shard}-of-${shards}.json`), {
    version: SVI_SCORE_V9_VERSION,
    protocolFreezeSha256,
    shard,
    shards,
    hiddenTruthDereferenced: false,
    items,
  });
}

async function runChild(shard: number, shards: number): Promise<void> {
  await new Promise<void>((done, reject) => {
    const child = spawn(process.execPath, [
      "--import",
      "tsx",
      resolve("research/svi_score_v9/prepare-sealed-audit.ts"),
      "--worker",
      `--shard=${shard}`,
      `--shards=${shards}`,
    ], { cwd: process.cwd(), env: process.env, stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code) => code === 0 ? done() : reject(
      new Error(`V9 audit preparation worker ${shard} exited ${code}.`),
    ));
  });
}

async function orchestrate(): Promise<void> {
  const freezeSource = await readFile(FREEZE_PATH);
  const freeze = JSON.parse(freezeSource.toString("utf8")) as ProtocolFreeze;
  await verifyFrozenSources(freeze);
  const protocolFreezeSha256 = digest(freezeSource);
  const design = v9SealedAuditBusinesses();
  const designReceipt = design.map((item) => ({
    businessId: item.scenario.id,
    family: item.family.id,
    sourceSplit: item.sourceSplit,
    seed: item.scenario.seed,
    auditFamilyIndex: item.auditFamilyIndex,
  }));
  try {
    await writeFile(OPEN_PATH, JSON.stringify({
      artifactId: "flux-svi-score-v9-sealed-audit-open-receipt-v1",
      version: SVI_SCORE_V9_VERSION,
      openedAt: new Date().toISOString(),
      protocolFreezeSha256,
      designSha256: digest(JSON.stringify(designReceipt)),
      businesses: designReceipt.length,
      oneTimeOpening: true,
      hiddenTruthDereferenced: false,
      nutsUsed: false,
    }, null, 2), { flag: "wx" });
  } catch (error) {
    if (!(
      error instanceof Error && "code" in error &&
      (error as NodeJS.ErrnoException).code === "EEXIST"
    )) throw error;
    const existing = JSON.parse(await readFile(OPEN_PATH, "utf8")) as {
      protocolFreezeSha256: string;
      designSha256: string;
      hiddenTruthDereferenced: boolean;
    };
    if (
      existing.protocolFreezeSha256 !== protocolFreezeSha256 ||
      existing.designSha256 !== digest(JSON.stringify(designReceipt)) ||
      existing.hiddenTruthDereferenced
    ) throw new Error("V9 audit was previously opened under another protocol.");
  }

  const workers = Math.min(
    16,
    integerArgument(process.argv, "workers", Math.min(8, cpus().length)),
  );
  await Promise.all(Array.from({ length: workers }, (_, shard) => runChild(shard, workers)));
  const items: V9AuditInferenceItem[] = [];
  for (let shard = 0; shard < workers; shard += 1) {
    const part = JSON.parse(
      await readFile(resolve(PARTS, `part-${shard}-of-${workers}.json`), "utf8"),
    ) as {
      version: string;
      protocolFreezeSha256: string;
      hiddenTruthDereferenced: boolean;
      items: V9AuditInferenceItem[];
    };
    if (
      part.version !== SVI_SCORE_V9_VERSION ||
      part.protocolFreezeSha256 !== protocolFreezeSha256 ||
      part.hiddenTruthDereferenced
    ) throw new Error(`Stale or invalid V9 audit preparation part ${shard}.`);
    items.push(...part.items);
  }
  items.sort((left, right) => left.businessId.localeCompare(right.businessId) ||
    left.candidateId.localeCompare(right.candidateId));
  if (
    items.length !== SVI_SCORE_V9_AUDIT_CONTRACT.cohort.candidateRows ||
    new Set(items.map((item) => item.businessId)).size !==
      SVI_SCORE_V9_AUDIT_CONTRACT.cohort.businesses ||
    new Set(items.map((item) => item.inferenceFingerprint)).size !== items.length
  ) throw new Error("V9 audit inference manifest is incomplete or duplicated.");
  await atomicJson(MANIFEST_PATH, {
    artifactId: "flux-svi-score-v9-sealed-audit-inference-manifest-v1",
    version: SVI_SCORE_V9_VERSION,
    activation: "sealed-audit-opened-inference-only",
    protocolFreezeSha256,
    openReceiptSha256: digest(await readFile(OPEN_PATH)),
    designSha256: digest(JSON.stringify(designReceipt)),
    inferenceContract: SVI_SCORE_V9_CONTRACT.inference.contract,
    hiddenTruthDereferenced: false,
    nutsUsed: false,
    items,
  });
  await atomicJson(CLOUD_MANIFEST_PATH, {
    version: "flux-svi-cvm-retry-v1",
    sourceCheckpoint: "v9-sealed-audit-inference-manifest.json",
    failureCount: items.length,
    remotePrefix: REMOTE_PREFIX,
    protocolFreezeSha256,
    auditVersion: SVI_SCORE_V9_VERSION,
    items: items.map((item) => ({
      fingerprint: item.inferenceFingerprint,
      businessId: item.businessId,
      candidateId: item.candidateId,
      payloadObject: item.payloadObject,
      payloadSha256: item.payloadSha256,
      outputObject: item.outputObject,
    })),
  });
  process.stdout.write(`${JSON.stringify({
    manifest: MANIFEST_PATH,
    cloudManifest: CLOUD_MANIFEST_PATH,
    businesses: design.length,
    candidateFits: items.length,
    payloads: items.length,
    protocolFrozen: true,
    auditOpenedExactlyOnce: true,
    hiddenTruthDereferenced: false,
    nutsUsed: false,
  }, null, 2)}\n`);
}

async function developmentSmoke(): Promise<void> {
  const design = sviDesignBusinesses().find(({ family }) => family.split !== "audit");
  if (!design) throw new Error("No development business exists for the V9 audit smoke test.");
  const prepared = await prepareMcmcResearchBusiness(
    design.scenario,
    design.family.id,
    design.family.split,
    undefined,
    { includeHiddenTruthLabels: false },
  );
  const candidate = prepared.candidates[0];
  const model = compileSamplingModel(
    prepared.dataset,
    candidate.run,
    prepared.experiments,
    candidate.industryPriorChannels,
    syntheticIndustryPriorOverrides(prepared.business),
  );
  const fingerprint = await modelSha256(JSON.stringify({
    version: SVI_SCORE_V9_VERSION,
    smoke: true,
    model,
    contract: SVI_SCORE_V9_CONTRACT.inference.contract,
  }));
  process.stdout.write(`${JSON.stringify({
    developmentSmoke: true,
    businessId: design.scenario.id,
    candidates: prepared.candidates.length,
    compiledCandidate: candidate.row.candidateId,
    fingerprintLength: fingerprint.length,
    hiddenTruthLabelsIncluded: false,
    auditCohortGenerated: false,
  }, null, 2)}\n`);
}

if (process.argv.includes("--development-smoke")) {
  await developmentSmoke();
} else if (process.argv.includes("--worker")) {
  await worker(
    integerArgument(process.argv, "shard", 0, 0),
    integerArgument(process.argv, "shards", 1),
  );
} else {
  await orchestrate();
}
