import { spawn } from "node:child_process";
import { cpus } from "node:os";
import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import type { ValidationOptions } from "../../lib/mmm/validation";
import type { ValidationModelSpec } from "../../lib/mmm/validation/adapter";
import { posteriorDecisionLabelFromDraws } from "../mcmc_score_v3/posterior";
import { prepareMcmcResearchBusiness } from "../mcmc_score_v3/cohort";
import { evaluateAlignedPosteriorDecisionTruthExact } from "../score_v2/evaluate";
import { syntheticIndustryPriorOverrides } from "../score_v6/evidence";
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
  V8_DECISION_SCENARIOS,
  V8_DANGEROUS_EXCESS_LOSS,
  V8_RISK_PREFERENCE,
} from "../svi_score_v8/contract";
import { SVI_SCORE_V9_AUDIT_CONTRACT } from "./audit-contract";
import { v9SealedAuditBusinesses, type V9AuditBusiness } from "./audit-design";
import {
  v9PosteriorDecisionTokens,
  type V9SviResult,
} from "./audit-tokens";
import {
  SVI_SCORE_V9_CONTRACT,
  SVI_SCORE_V9_VERSION,
  V9_POSTERIOR_TOKEN_NAMES,
} from "./contract";
import type { V9AuditInferenceItem } from "./prepare-sealed-audit";

const ROOT = resolve(".flux-artifacts/svi-score-v9/audit");
const RESULTS = resolve(ROOT, "results");
const PARTS = resolve(ROOT, "dataset-parts");
const FREEZE_PATH = resolve(ROOT, "sealed-audit-protocol-freeze.json");
const OPEN_PATH = resolve(ROOT, "sealed-audit-open-receipt.json");
const MANIFEST_PATH = resolve(ROOT, "sealed-audit-inference-manifest.json");
const OUTPUT_PATH = resolve(ROOT, "sealed-audit-dataset.json");
const V8_DEVELOPMENT_DATASET = resolve(
  ".flux-artifacts/svi-score-v8/development-dataset.json",
);

interface AuditRow {
  businessId: string;
  family: string;
  candidateId: string;
  fold: number;
  features: number[];
  valid: boolean;
  validityReasons: string[];
  scenarioLoss: Record<string, number>;
  scenarioExcessLoss: Record<string, number>;
  meanExcessLoss: number;
  p90ExcessLoss: number;
  economicRisk: number;
  excessEconomicRisk: number;
  normalizedExcessEconomicRisk: number;
  dangerous: boolean;
  roiError: number;
  contributionError: number;
}

interface AuditManifest {
  version: string;
  protocolFreezeSha256: string;
  hiddenTruthDereferenced: boolean;
  nutsUsed: boolean;
  items: V9AuditInferenceItem[];
}

const average = (values: readonly number[]) =>
  values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);

function quantile(values: readonly number[], probability: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  if (!sorted.length) return 0;
  const position = Math.min(1, Math.max(0, probability)) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function digest(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function integerArgument(name: string, fallback: number): number {
  const prefix = `--${name}=`;
  const value = process.argv.find((argument) => argument.startsWith(prefix));
  return value ? Math.max(1, Number.parseInt(value.slice(prefix.length), 10)) : fallback;
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.tmp`;
  await writeFile(temporary, JSON.stringify(value));
  await rename(temporary, path);
}

function validationOptions(
  activeChannels: string[],
  evidenceArm: V9AuditInferenceItem["evidenceArm"],
  business: Awaited<ReturnType<typeof prepareMcmcResearchBusiness>>["business"],
): ValidationOptions {
  return {
    anchorIndependenceConfirmed: true,
    industryPriorChannels: activeChannels,
    industryPriorOverrides: syntheticIndustryPriorOverrides(business),
    industryBenchmarkScreeningEnabled: evidenceArm === "benchmark-gap-fill",
    materialSpendShareThreshold: 0.02,
  };
}

async function readResult(record: V9AuditInferenceItem): Promise<V9SviResult> {
  const path = resolve(RESULTS, `${record.inferenceFingerprint}.result.json.gz`);
  const result = JSON.parse(
    gunzipSync(await readFile(path)).toString("utf8"),
  ) as V9SviResult;
  if (
    result.fingerprint !== record.inferenceFingerprint ||
    (result.status !== "labelled" && result.status !== "review") ||
    JSON.stringify(result.contract) !==
      JSON.stringify(SVI_SCORE_V9_CONTRACT.inference.contract) ||
    !result.channels.length || !result.decisionDraws.length
  ) throw new Error(`Invalid V9 audit SVI result: ${record.inferenceFingerprint}.`);
  return result;
}

async function processBusiness(
  design: V9AuditBusiness,
  records: V9AuditInferenceItem[],
  expectedV8FeatureNames: string[],
): Promise<{ rows: AuditRow[]; featureNames: string[] }> {
  if (records.length !== SVI_SCORE_V9_AUDIT_CONTRACT.cohort.candidatesPerBusiness) {
    throw new Error(`${design.scenario.id} has ${records.length}/48 audit results.`);
  }
  const prepared = await prepareMcmcResearchBusiness(
    design.scenario,
    design.family.id,
    "audit",
    undefined,
    { includeHiddenTruthLabels: false },
  );
  const recordByCandidate = new Map(records.map((record) => [record.candidateId, record]));
  const resultByCandidate = new Map<string, V9SviResult>();
  const posterior: V5DSviRecordMap = new Map();
  for (const candidate of prepared.candidates) {
    const record = recordByCandidate.get(candidate.row.candidateId);
    if (!record) throw new Error(`Missing inference record for ${candidate.row.candidateId}.`);
    const result = await readResult(record);
    resultByCandidate.set(candidate.row.candidateId, result);
    posterior.set(v5dSviKey(candidate.row), {
      status: result.status,
      posteriorRoi: Object.fromEntries(
        result.channels.map((channel) => [channel.channel, channel.posteriorMedian]),
      ),
      diagnostics: structuredClone(result.diagnostics),
      seedDiagnostics: structuredClone(result.seeds),
    });
  }
  const provider = v5dSviFeatureProvider(posterior);
  const observable = prepared.candidates.map((candidate) => {
    const record = recordByCandidate.get(candidate.row.candidateId)!;
    const result = resultByCandidate.get(candidate.row.candidateId)!;
    const baseTemporal = temporalEvidenceForCandidate(
      prepared.dataset,
      candidate.run.model!,
      candidate.run.spec.config,
    );
    const options = validationOptions(
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
        detail: `${baseTemporal.detail} ${flight.folds} forward whole-flight holdout fold${
          flight.folds === 1 ? "" : "s"
        }.`,
      },
    };
    const v8Features = provider.vector(row);
    const v8Names = provider.names(row);
    if (
      v8Names.length !== SVI_SCORE_V9_CONTRACT.tokens.baseCount ||
      v8Names.some((name, index) => name !== expectedV8FeatureNames[index]) ||
      v8Features.some((value) => !Number.isFinite(value))
    ) throw new Error(`V8 token contract changed for ${row.candidateId}.`);
    const features = [...v8Features, ...v9PosteriorDecisionTokens(result)];
    return { row, candidate, result, features };
  });
  const featureNames = [...expectedV8FeatureNames, ...V9_POSTERIOR_TOKEN_NAMES];
  if (
    featureNames.length !== SVI_SCORE_V9_AUDIT_CONTRACT.cohort.tokens ||
    observable.some(({ features }) =>
      features.length !== featureNames.length ||
      features.some((value) => !Number.isFinite(value))
    )
  ) throw new Error(`V9 token contract changed for ${design.scenario.id}.`);

  // Hidden economic truth is dereferenced only here, after every posterior fit
  // has completed under the frozen protocol.
  const labelled = observable.map((item) => {
    const exact = evaluateAlignedPosteriorDecisionTruthExact(
      prepared.business,
      item.result.decisionDraws,
      64,
    );
    const posteriorRoi = Object.fromEntries(
      item.result.channels.map((channel) => [channel.channel, channel.posteriorMedian]),
    );
    const diagnosticLabel = posteriorDecisionLabelFromDraws(
      prepared.business,
      item.candidate.run.model!,
      item.candidate.run.spec.config,
      item.result.decisionDraws,
      posteriorRoi,
      item.result.diagnostics.finite,
    );
    if (
      !Number.isFinite(diagnosticLabel.roiError) ||
      !Number.isFinite(diagnosticLabel.contributionError)
    ) throw new Error(`Non-finite audit truth diagnostic for ${item.row.candidateId}.`);
    return {
      ...item,
      scenarioLoss: exact.scenarioDecisionLoss,
      roiError: diagnosticLabel.roiError!,
      contributionError: diagnosticLabel.contributionError!,
    };
  });
  const scenarioMinimum = Object.fromEntries(V8_DECISION_SCENARIOS.map((scenario) => [
    scenario,
    Math.min(...labelled.map(({ scenarioLoss }) => scenarioLoss[scenario])),
  ])) as Record<(typeof V8_DECISION_SCENARIOS)[number], number>;
  const summaries = labelled.map((item) => {
    const values = V8_DECISION_SCENARIOS.map((scenario) => item.scenarioLoss[scenario]);
    const mean = average(values);
    const p90 = quantile(values, 0.9);
    return {
      ...item,
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
    rows: summaries.map((summary): AuditRow => {
      const excess = Math.max(0, summary.risk - minimumRisk);
      return {
        businessId: design.scenario.id,
        family: design.family.id,
        candidateId: summary.row.candidateId,
        fold: -1,
        features: summary.features,
        valid: summary.result.diagnostics.finite,
        validityReasons: summary.result.diagnostics.finite
          ? []
          : ["non-finite-svi-posterior"],
        scenarioLoss: structuredClone(summary.scenarioLoss),
        scenarioExcessLoss: Object.fromEntries(V8_DECISION_SCENARIOS.map((scenario) => [
          scenario,
          Math.max(0, summary.scenarioLoss[scenario] - scenarioMinimum[scenario]),
        ])),
        meanExcessLoss: Math.max(0, summary.mean - minimumMean),
        p90ExcessLoss: Math.max(0, summary.p90 - minimumP90),
        economicRisk: summary.risk,
        excessEconomicRisk: excess,
        normalizedExcessEconomicRisk: Math.min(excess, 1),
        dangerous: excess >= V8_DANGEROUS_EXCESS_LOSS,
        roiError: summary.roiError,
        contributionError: summary.contributionError,
      };
    }),
  };
}

async function worker(shard: number, shards: number): Promise<void> {
  const manifest = JSON.parse(await readFile(MANIFEST_PATH, "utf8")) as AuditManifest;
  const development = JSON.parse(await readFile(V8_DEVELOPMENT_DATASET, "utf8")) as {
    featureNames: string[];
  };
  const recordsByBusiness = new Map<string, V9AuditInferenceItem[]>();
  manifest.items.forEach((record) => recordsByBusiness.set(record.businessId, [
    ...(recordsByBusiness.get(record.businessId) ?? []),
    record,
  ]));
  const design = v9SealedAuditBusinesses().filter((_, index) => index % shards === shard);
  const rows: AuditRow[] = [];
  let featureNames: string[] = [];
  for (let index = 0; index < design.length; index += 1) {
    const item = design[index];
    const output = await processBusiness(
      item,
      recordsByBusiness.get(item.scenario.id) ?? [],
      development.featureNames,
    );
    featureNames = output.featureNames;
    rows.push(...output.rows);
    process.stdout.write(
      `V9 audit labels ${shard + 1}/${shards}: ${index + 1}/${design.length} businesses\n`,
    );
  }
  await mkdir(PARTS, { recursive: true });
  await atomicJson(resolve(PARTS, `part-${shard}-of-${shards}.json`), {
    version: SVI_SCORE_V9_VERSION,
    protocolFreezeSha256: manifest.protocolFreezeSha256,
    shard,
    shards,
    featureNames,
    hiddenTruthDereferenced: true,
    rows,
  });
}

async function runChild(shard: number, shards: number): Promise<void> {
  await new Promise<void>((done, reject) => {
    const child = spawn(process.execPath, [
      "--import",
      "tsx",
      resolve("research/svi_score_v9/build-sealed-audit-dataset.ts"),
      "--worker",
      `--shard=${shard}`,
      `--shards=${shards}`,
    ], { cwd: process.cwd(), env: process.env, stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code) => code === 0 ? done() : reject(
      new Error(`V9 audit dataset worker ${shard} exited ${code}.`),
    ));
  });
}

async function orchestrate(): Promise<void> {
  const [freezeSource, openSource, manifestSource] = await Promise.all([
    readFile(FREEZE_PATH),
    readFile(OPEN_PATH),
    readFile(MANIFEST_PATH),
  ]);
  const freeze = JSON.parse(freezeSource.toString("utf8")) as {
    version: string;
    status: string;
  };
  const open = JSON.parse(openSource.toString("utf8")) as {
    protocolFreezeSha256: string;
    oneTimeOpening: boolean;
  };
  const manifest = JSON.parse(manifestSource.toString("utf8")) as AuditManifest;
  if (
    freeze.version !== SVI_SCORE_V9_VERSION || freeze.status !== "frozen-sealed" ||
    !open.oneTimeOpening || open.protocolFreezeSha256 !== digest(freezeSource) ||
    manifest.protocolFreezeSha256 !== digest(freezeSource) ||
    manifest.hiddenTruthDereferenced || manifest.nutsUsed
  ) throw new Error("V9 audit opening or inference manifest is invalid.");
  const missing: string[] = [];
  for (const record of manifest.items) {
    try {
      await readFile(resolve(RESULTS, `${record.inferenceFingerprint}.result.json.gz`));
    } catch {
      missing.push(record.inferenceFingerprint);
    }
  }
  if (missing.length) {
    throw new Error(`V9 audit inference is incomplete: ${missing.length} results are missing.`);
  }
  const workers = Math.min(16, integerArgument("workers", Math.min(8, cpus().length)));
  await Promise.all(Array.from({ length: workers }, (_, shard) => runChild(shard, workers)));
  const rows: AuditRow[] = [];
  let featureNames: string[] = [];
  for (let shard = 0; shard < workers; shard += 1) {
    const part = JSON.parse(
      await readFile(resolve(PARTS, `part-${shard}-of-${workers}.json`), "utf8"),
    ) as {
      version: string;
      protocolFreezeSha256: string;
      featureNames: string[];
      hiddenTruthDereferenced: boolean;
      rows: AuditRow[];
    };
    if (
      part.version !== SVI_SCORE_V9_VERSION ||
      part.protocolFreezeSha256 !== digest(freezeSource) ||
      !part.hiddenTruthDereferenced
    ) throw new Error(`Stale or invalid V9 audit dataset part ${shard}.`);
    if (!featureNames.length) featureNames = part.featureNames;
    if (part.featureNames.some((name, index) => name !== featureNames[index])) {
      throw new Error(`V9 audit token order changed in part ${shard}.`);
    }
    rows.push(...part.rows);
  }
  rows.sort((left, right) => left.businessId.localeCompare(right.businessId) ||
    left.candidateId.localeCompare(right.candidateId));
  if (
    rows.length !== SVI_SCORE_V9_AUDIT_CONTRACT.cohort.candidateRows ||
    featureNames.length !== SVI_SCORE_V9_AUDIT_CONTRACT.cohort.tokens ||
    new Set(rows.map((row) => `${row.businessId}\u0000${row.candidateId}`)).size !== rows.length
  ) throw new Error("V9 audit dataset is incomplete or duplicated.");
  await atomicJson(OUTPUT_PATH, {
    artifactId: "flux-svi-score-v9-sealed-audit-dataset-v1",
    version: SVI_SCORE_V9_VERSION,
    generatedAt: new Date().toISOString(),
    activation: "sealed-audit-opened-unscored",
    provenance: {
      protocolFreezeSha256: digest(freezeSource),
      openReceiptSha256: digest(openSource),
      inferenceManifestSha256: digest(manifestSource),
      v8DevelopmentTokenRegistrySha256: digest(await readFile(V8_DEVELOPMENT_DATASET)),
      hiddenTruthDereferencedAfterInference: true,
      nutsUsed: false,
    },
    featureNames,
    summary: {
      businesses: new Set(rows.map((row) => row.businessId)).size,
      rows: rows.length,
      validRows: rows.filter((row) => row.valid).length,
      families: Object.fromEntries(
        [...new Set(rows.map((row) => row.family))].sort().map((family) => [
          family,
          new Set(rows.filter((row) => row.family === family).map((row) => row.businessId)).size,
        ]),
      ),
    },
    rows,
  });
  process.stdout.write(`${JSON.stringify({
    path: OUTPUT_PATH,
    businesses: SVI_SCORE_V9_AUDIT_CONTRACT.cohort.businesses,
    rows: rows.length,
    tokens: featureNames.length,
    validRows: rows.filter((row) => row.valid).length,
    hiddenTruthDereferencedAfterInference: true,
    nutsUsed: false,
  }, null, 2)}\n`);
}

if (process.argv.includes("--worker")) {
  await worker(integerArgument("shard", 0), integerArgument("shards", 1));
} else {
  await orchestrate();
}
