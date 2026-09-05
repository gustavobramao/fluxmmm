import { spawn } from "node:child_process";
import { cpus } from "node:os";
import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import type { SamplingResult } from "../../lib/mmm/sampling";
import type { ValidationOptions } from "../../lib/mmm/validation";
import type { ValidationModelSpec } from "../../lib/mmm/validation/adapter";
import { posteriorDecisionLabelFromDraws } from "../mcmc_score_v3/posterior";
import { prepareMcmcResearchBusiness } from "../mcmc_score_v3/cohort";
import {
  evaluateAlignedPosteriorDecisionTruth,
  evaluateAlignedPosteriorDecisionTruthExact,
} from "../score_v2/evaluate";
import { syntheticIndustryPriorOverrides } from "../score_v6/evidence";
import type { SviScoreV3Record } from "../svi_score_v3/types";
import { sviDesignBusinesses, type SviDesignBusiness } from "../svi_score_v3/design";
import { predictV5DCrossFitted } from "../svi_score_v5d/crossfit";
import type { V5DCrossFittedModel } from "../svi_score_v5d/types";
import {
  type V5DSviRecordMap,
  v5dSviFeatureProvider,
  v5dSviKey,
} from "../svi_score_v5d_svi/posterior";
import {
  temporalEvidenceForCandidate,
  wholeFlightGeneralization,
} from "../svi_score_v4/temporal";
import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";
import {
  SVI_SCORE_V8_CONTRACT,
  SVI_SCORE_V8_VERSION,
  V8_DANGEROUS_EXCESS_LOSS,
  V8_DECISION_SCENARIOS,
  V8_RISK_PREFERENCE,
} from "./contract";
import type { V8DatasetRow } from "./types";

const ROOT = resolve(".flux-artifacts/svi-score-v8");
const PARTS = resolve(ROOT, "sealed-audit-parts");
const RECORD_PATH = resolve(".flux-artifacts/svi-score-v3/design-pilot-records.json");
const FREEZE_PATH = resolve(ROOT, "sealed-audit-protocol-freeze.json");
const OPEN_PATH = resolve(ROOT, "sealed-audit-open-receipt.json");
const MANIFEST_PATH = resolve(ROOT, "sealed-audit-manifest.json");
const OUTPUT_PATH = resolve(ROOT, "sealed-audit-dataset.json");
const COMPARATOR_PATH = resolve(ROOT, "frozen-v5d-svi-comparator.json");

interface AuditManifest {
  version: string;
  sourceSha256: string;
  protocolFreezeSha256: string;
  records: SviScoreV3Record[];
}

interface AuditDatasetRow extends V8DatasetRow {
  v5dSviPredictedRisk: number;
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function average(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) /
    Math.max(values.length, 1);
}

function quantile(values: readonly number[], probability: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  if (!sorted.length) return 0;
  const position = Math.min(1, Math.max(0, probability)) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.tmp`;
  await writeFile(temporary, JSON.stringify(value));
  await rename(temporary, path);
}

async function frozenComparator(): Promise<{
  source: Buffer;
  model: V5DCrossFittedModel;
}> {
  const source = await readFile(COMPARATOR_PATH);
  const artifact = JSON.parse(source.toString("utf8")) as {
    version: string;
    activation: string;
    provenance: { auditAccessed: boolean; nutsUsed: boolean };
    model: V5DCrossFittedModel;
  };
  if (
    artifact.version !== SVI_SCORE_V8_VERSION ||
    artifact.activation !== "development-frozen" ||
    artifact.provenance.auditAccessed ||
    artifact.provenance.nutsUsed
  ) throw new Error("The V5D-SVI comparator was not frozen before audit opening.");
  return { source, model: artifact.model };
}

function validationOptions(
  design: SviDesignBusiness,
  activeChannels: string[],
  evidenceArm: SviScoreV3Record["evidenceArm"],
  business: Awaited<ReturnType<typeof prepareMcmcResearchBusiness>>["business"],
): ValidationOptions {
  void design;
  return {
    anchorIndependenceConfirmed: true,
    industryPriorChannels: activeChannels,
    industryPriorOverrides: syntheticIndustryPriorOverrides(business),
    industryBenchmarkScreeningEnabled: evidenceArm === "benchmark-gap-fill",
    materialSpendShareThreshold: 0.02,
  };
}

async function processBusiness(
  design: SviDesignBusiness,
  records: SviScoreV3Record[],
  comparator: V5DCrossFittedModel,
): Promise<{ rows: AuditDatasetRow[]; featureNames: string[] }> {
  if (records.length !== SVI_SCORE_V8_CONTRACT.cohort.candidatesPerBusiness) {
    throw new Error(`${design.scenario.id} has ${records.length}/48 sealed-audit records.`);
  }
  const prepared = await prepareMcmcResearchBusiness(
    design.scenario,
    design.family.id,
    design.family.split,
    undefined,
    { includeHiddenTruthLabels: false },
  );
  const recordByCandidate = new Map(records.map((record) => [record.candidateId, record]));
  const posterior: V5DSviRecordMap = new Map();
  records.forEach((record) => {
    if (
      (record.status !== "labelled" && record.status !== "review") ||
      !record.posteriorRoi || !record.diagnosticsReceipt || !record.seedDiagnostics
    ) throw new Error(`Incomplete audit SVI posterior for ${record.candidateId}.`);
    posterior.set(v5dSviKey(record), {
      status: record.status,
      posteriorRoi: structuredClone(record.posteriorRoi),
      diagnostics: structuredClone(record.diagnosticsReceipt),
      seedDiagnostics: structuredClone(record.seedDiagnostics),
    });
  });
  const provider = v5dSviFeatureProvider(posterior);

  // Construct every observable token before dereferencing hidden economic
  // labels. The generated business contains truth internally, but these
  // functions consume only advertiser-visible data, candidate residuals,
  // declared specifications, and external-evidence metadata.
  const observable = prepared.candidates.map((candidate) => {
    const record = recordByCandidate.get(candidate.row.candidateId);
    if (!record) throw new Error(`Missing audit record ${candidate.row.candidateId}.`);
    const baseTemporal = temporalEvidenceForCandidate(
      prepared.dataset,
      candidate.run.model!,
      candidate.run.spec.config,
    );
    const options = validationOptions(
      design,
      candidate.industryPriorChannels,
      record.evidenceArm,
      prepared.business,
    );
    const flight = wholeFlightGeneralization(prepared.dataset, {
      kind: candidate.run.spec.family,
      config: candidate.run.spec.config,
      advancedConfig: candidate.run.spec.advancedConfig,
      experiments: prepared.experiments,
      validationOptions: options,
    } satisfies ValidationModelSpec);
    const row: SviScoreV4CandidateRow = {
      ...candidate.row,
      temporal: {
        ...baseTemporal,
        wholeFlightGeneralization: flight.score,
        temporalIdentification: 100 * Math.sqrt(
          Math.max(0.01, baseTemporal.carryoverSupport / 100) *
            Math.max(0.01, flight.score / 100),
        ),
        detail: `${baseTemporal.detail} ${flight.folds} forward whole-flight holdout fold${flight.folds === 1 ? "" : "s"}.`,
      },
    };
    const features = provider.vector(row);
    const names = provider.names(row);
    if (
      names.length !== SVI_SCORE_V8_CONTRACT.tokens.count ||
      features.length !== names.length ||
      features.some((value) => !Number.isFinite(value))
    ) throw new Error(`Invalid sealed-audit token contract for ${row.candidateId}.`);
    return {
      row,
      features,
      names,
      baseline: predictV5DCrossFitted(row, comparator, provider).risk,
      record,
    };
  });
  const featureNames = observable[0].names;
  if (observable.some((item) => item.names.some((name, index) => name !== featureNames[index]))) {
    throw new Error(`${design.scenario.id} changed the V8 token ordering.`);
  }

  const labelled = [] as Array<{
    item: (typeof observable)[number];
    scenarioLoss: Record<(typeof V8_DECISION_SCENARIOS)[number], number>;
  }>;
  for (const item of observable) {
    const resultPath = resolve(
      ".flux-artifacts/svi-score-v3",
      `${item.record.inferenceFingerprint}.result.json.gz`,
    );
    const result = JSON.parse(
      gunzipSync(await readFile(resultPath)).toString("utf8"),
    ) as SamplingResult;
    let evaluated = evaluateAlignedPosteriorDecisionTruth(
      prepared.business,
      result.decisionDraws ?? [],
      64,
    );
    if (Math.abs(evaluated.decisionLoss - (item.record.sviDecisionLoss ?? NaN)) > 1e-10) {
      evaluated = evaluateAlignedPosteriorDecisionTruthExact(
        prepared.business,
        result.decisionDraws ?? [],
        64,
      );
    }
    if (Math.abs(evaluated.decisionLoss - (item.record.sviDecisionLoss ?? NaN)) > 1e-10) {
      throw new Error(`Frozen audit SVI label changed for ${item.row.candidateId}.`);
    }
    labelled.push({ item, scenarioLoss: evaluated.scenarioDecisionLoss });
  }
  const scenarioMinimum = Object.fromEntries(
    V8_DECISION_SCENARIOS.map((scenario) => [
      scenario,
      Math.min(...labelled.map(({ scenarioLoss }) => scenarioLoss[scenario])),
    ]),
  ) as Record<(typeof V8_DECISION_SCENARIOS)[number], number>;
  const summaries = labelled.map(({ item, scenarioLoss }) => {
    const values = V8_DECISION_SCENARIOS.map((scenario) => scenarioLoss[scenario]);
    const mean = average(values);
    const p90 = quantile(values, 0.9);
    return {
      item,
      scenarioLoss,
      mean,
      p90,
      risk: V8_RISK_PREFERENCE.mean * mean + V8_RISK_PREFERENCE.p90 * p90,
    };
  });
  const minimumMean = Math.min(...summaries.map((summary) => summary.mean));
  const minimumP90 = Math.min(...summaries.map((summary) => summary.p90));
  const minimumRisk = Math.min(...summaries.map((summary) => summary.risk));
  return {
    featureNames,
    rows: summaries.map((summary): AuditDatasetRow => {
      const excess = Math.max(0, summary.risk - minimumRisk);
      return {
        businessId: design.scenario.id,
        family: design.family.id,
        candidateId: summary.item.row.candidateId,
        fold: -1,
        features: summary.item.features,
        valid: summary.item.record.diagnosticsReceipt?.finite === true,
        validityReasons: summary.item.record.diagnosticsReceipt?.finite
          ? []
          : ["non-finite-svi-posterior"],
        scenarioLoss: structuredClone(summary.scenarioLoss),
        scenarioExcessLoss: Object.fromEntries(
          V8_DECISION_SCENARIOS.map((scenario) => [
            scenario,
            Math.max(0, summary.scenarioLoss[scenario] - scenarioMinimum[scenario]),
          ]),
        ) as V8DatasetRow["scenarioExcessLoss"],
        meanExcessLoss: Math.max(0, summary.mean - minimumMean),
        p90ExcessLoss: Math.max(0, summary.p90 - minimumP90),
        economicRisk: summary.risk,
        excessEconomicRisk: excess,
        normalizedExcessEconomicRisk: Math.min(excess, 1),
        dangerous: excess >= V8_DANGEROUS_EXCESS_LOSS,
        roiError: summary.item.record.sviRoiError ?? Number.NaN,
        contributionError: summary.item.record.sviContributionError ?? Number.NaN,
        v5dSviPredictedRisk: summary.item.baseline,
      };
    }),
  };
}

async function worker(shard: number, shards: number): Promise<void> {
  const manifest = JSON.parse(await readFile(MANIFEST_PATH, "utf8")) as AuditManifest;
  const comparator = await frozenComparator();
  const recordsByBusiness = new Map<string, SviScoreV3Record[]>();
  manifest.records.forEach((record) => recordsByBusiness.set(record.businessId, [
    ...(recordsByBusiness.get(record.businessId) ?? []),
    record,
  ]));
  const design = sviDesignBusinesses()
    .filter(({ family }) => family.split === "audit")
    .filter((_, index) => index % shards === shard);
  const rows: AuditDatasetRow[] = [];
  let featureNames: string[] = [];
  for (let index = 0; index < design.length; index += 1) {
    const item = design[index];
    const output = await processBusiness(
      item,
      recordsByBusiness.get(item.scenario.id) ?? [],
      comparator.model,
    );
    featureNames = output.featureNames;
    rows.push(...output.rows);
    process.stdout.write(
      `V8 sealed audit ${shard + 1}/${shards}: ${index + 1}/${design.length} businesses\n`,
    );
  }
  await mkdir(PARTS, { recursive: true });
  await atomicJson(resolve(PARTS, `part-${shard}-of-${shards}.json`), {
    version: SVI_SCORE_V8_VERSION,
    protocolFreezeSha256: manifest.protocolFreezeSha256,
    shard,
    shards,
    featureNames,
    rows,
  });
}

async function runChild(shard: number, shards: number): Promise<void> {
  await new Promise<void>((done, reject) => {
    const child = spawn(process.execPath, [
      "--import", "tsx", resolve("research/svi_score_v8/build-sealed-audit-dataset.ts"),
      "--worker", `--shard=${shard}`, `--shards=${shards}`,
    ], { cwd: process.cwd(), env: process.env, stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code) => code === 0 ? done() : reject(
      new Error(`V8 sealed-audit worker ${shard} exited ${code}.`),
    ));
  });
}

function integerArgument(name: string, fallback: number): number {
  const value = process.argv.find((argument) => argument.startsWith(`--${name}=`));
  return value ? Number.parseInt(value.slice(name.length + 3), 10) : fallback;
}

async function developmentSmoke(): Promise<void> {
  const records = JSON.parse(await readFile(RECORD_PATH, "utf8")) as SviScoreV3Record[];
  const design = sviDesignBusinesses().find(({ family }) => family.split !== "audit")!;
  const comparator = await frozenComparator();
  const output = await processBusiness(
    design,
    records.filter((record) => record.businessId === design.scenario.id),
    comparator.model,
  );
  process.stdout.write(`${JSON.stringify({
    developmentSmoke: true,
    businessId: design.scenario.id,
    rows: output.rows.length,
    tokens: output.featureNames.length,
    auditAccessed: false,
  }, null, 2)}\n`);
}

async function orchestrate(): Promise<void> {
  const freezeSource = await readFile(FREEZE_PATH);
  const freeze = JSON.parse(freezeSource.toString("utf8")) as {
    version: string;
    status: string;
  };
  if (freeze.version !== SVI_SCORE_V8_VERSION || freeze.status !== "frozen-sealed") {
    throw new Error("The V8 audit protocol is not frozen.");
  }
  const protocolFreezeSha256 = sha256(freezeSource);
  try {
    await writeFile(OPEN_PATH, JSON.stringify({
      artifactId: "flux-svi-score-v8-sealed-audit-open-receipt-v1",
      version: SVI_SCORE_V8_VERSION,
      openedAt: new Date().toISOString(),
      protocolFreezeSha256,
      oneTimeOpening: true,
    }), { flag: "wx" });
  } catch (error) {
    if (!(error instanceof Error && "code" in error &&
      (error as NodeJS.ErrnoException).code === "EEXIST")) throw error;
    const existing = JSON.parse(await readFile(OPEN_PATH, "utf8")) as {
      protocolFreezeSha256: string;
    };
    if (existing.protocolFreezeSha256 !== protocolFreezeSha256) {
      throw new Error("The sealed audit was already opened under another protocol.");
    }
  }

  // Audit access starts only after the immutable opening receipt exists.
  const recordSource = await readFile(RECORD_PATH);
  const records = (JSON.parse(recordSource.toString("utf8")) as SviScoreV3Record[])
    .filter((record) => record.split === "audit");
  if (
    records.length !== SVI_SCORE_V8_CONTRACT.cohort.sealedAuditBusinesses *
      SVI_SCORE_V8_CONTRACT.cohort.candidatesPerBusiness ||
    new Set(records.map((record) => record.businessId)).size !==
      SVI_SCORE_V8_CONTRACT.cohort.sealedAuditBusinesses
  ) throw new Error("The sealed-audit cohort contract changed.");
  await atomicJson(MANIFEST_PATH, {
    version: SVI_SCORE_V8_VERSION,
    sourceSha256: sha256(recordSource),
    protocolFreezeSha256,
    records,
  } satisfies AuditManifest);
  const workers = Math.max(1, Math.min(integerArgument("workers", Math.min(8, cpus().length)), 16));
  await mkdir(PARTS, { recursive: true });
  await Promise.all(Array.from({ length: workers }, (_, shard) => runChild(shard, workers)));
  const rows: AuditDatasetRow[] = [];
  let featureNames: string[] = [];
  for (let shard = 0; shard < workers; shard += 1) {
    const part = JSON.parse(
      await readFile(resolve(PARTS, `part-${shard}-of-${workers}.json`), "utf8"),
    ) as { version: string; protocolFreezeSha256: string; featureNames: string[]; rows: AuditDatasetRow[] };
    if (
      part.version !== SVI_SCORE_V8_VERSION ||
      part.protocolFreezeSha256 !== protocolFreezeSha256
    ) throw new Error(`Stale sealed-audit part ${shard}.`);
    if (!featureNames.length) featureNames = part.featureNames;
    if (part.featureNames.some((name, index) => name !== featureNames[index])) {
      throw new Error(`Sealed-audit token order changed in part ${shard}.`);
    }
    rows.push(...part.rows);
  }
  if (
    rows.length !== 3_840 || featureNames.length !== 136 ||
    new Set(rows.map((row) => `${row.businessId}\u0000${row.candidateId}`)).size !== rows.length
  ) throw new Error("The sealed-audit dataset is incomplete.");
  rows.sort((left, right) => left.businessId.localeCompare(right.businessId) ||
    left.candidateId.localeCompare(right.candidateId));
  const comparator = await frozenComparator();
  await atomicJson(OUTPUT_PATH, {
    artifactId: "flux-svi-score-v8-sealed-audit-dataset-v1",
    version: SVI_SCORE_V8_VERSION,
    generatedAt: new Date().toISOString(),
    activation: "sealed-audit-opened",
    provenance: {
      protocolFreezeSha256,
      openReceiptSha256: sha256(await readFile(OPEN_PATH)),
      auditManifestSha256: sha256(await readFile(MANIFEST_PATH)),
      comparatorSha256: sha256(comparator.source),
      sviEngine: SVI_SCORE_V8_CONTRACT.inference.engine,
      nutsUsed: false,
    },
    featureNames,
    summary: { businesses: 80, rows: rows.length, validRows: rows.filter((row) => row.valid).length },
    rows,
  });
  process.stdout.write(`${JSON.stringify({
    path: OUTPUT_PATH,
    businesses: 80,
    rows: rows.length,
    tokens: featureNames.length,
    auditOpened: true,
    nutsUsed: false,
  }, null, 2)}\n`);
}

if (process.argv.includes("--development-smoke")) {
  await developmentSmoke();
} else if (process.argv.includes("--worker")) {
  await worker(integerArgument("shard", 0), integerArgument("shards", 1));
} else {
  await orchestrate();
}
