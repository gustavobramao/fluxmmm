import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { cpus } from "node:os";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { gzipSync } from "node:zlib";
import {
  advancedModelFingerprint,
  runAdvancedModel,
} from "../../lib/mmm/advanced";
import type { AgenticCandidateSpec, AgenticCandidateRun } from "../../lib/mmm/agentic";
import { parseCsv, sha256 as modelSha256, toNumber } from "../../lib/mmm/csv";
import { modelFingerprint, runModel } from "../../lib/mmm/models";
import { compileSamplingModel } from "../../lib/mmm/sampling";
import { validationDiagnosticValues } from "../../lib/mmm/score-diagnostics";
import { createDataset, validateDataset } from "../../lib/mmm/schema";
import type { Dataset, Experiment, ModelResult } from "../../lib/mmm/types";
import {
  runModelValidation,
  validationFingerprint,
  type ValidationOptions,
} from "../../lib/mmm/validation";
import type { ValidationModelSpec } from "../../lib/mmm/validation/adapter";
import type { ScoreV6CandidateRow, ScoreV6EvidenceArm } from "../score_v6/types";
import type { SviInferenceContract } from "../svi_score_v3/contract";
import {
  temporalEvidenceForCandidate,
  wholeFlightGeneralization,
} from "../svi_score_v4/temporal";
import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";
import { V11_AMSS_CONFIRMATORY_CHANNELS, V11_AMSS_CONFIRMATORY_CONTRACT, V11_AMSS_CONFIRMATORY_VERSION } from "./contract";

const RESEARCH_ROOT = resolve("research/svi_score_v11_amss_confirmatory");
const COHORT_ROOT = resolve(".flux-artifacts/svi-score-v11-amss-confirmatory/sealed-cohort");
const INFERENCE_ROOT = resolve(".flux-artifacts/svi-score-v11-amss-confirmatory/inference");
const PAYLOAD_ROOT = resolve(INFERENCE_ROOT, "payloads");
const PART_ROOT = resolve(INFERENCE_ROOT, "preparation-parts");
const MANIFEST_PATH = resolve(INFERENCE_ROOT, "manifest.json");
const CLOUD_MANIFEST_PATH = resolve(INFERENCE_ROOT, "cloud-manifest.json");
const REMOTE_PREFIX = "v11-amss-confirmatory-001";

export interface V11ConfirmatoryInferenceItem {
  businessId: string;
  evidenceGroup: string;
  candidateId: string;
  baseCandidateId: string;
  evidenceArm: ScoreV6EvidenceArm;
  modelFamily: ScoreV6CandidateRow["modelFamily"];
  inferenceFingerprint: string;
  payloadPath: string;
  payloadObject: string;
  payloadSha256: string;
  outputObject: string;
  industryPriorChannels: string[];
  observableRow: SviScoreV4CandidateRow;
}

interface BusinessMetadata {
  businessId: string;
  evidenceGroup: string;
  grossMargin: number;
  baselineAnnualSpend: Record<string, number>;
  budgetUtilization: Record<string, number>;
}

interface V11ConfirmatoryInferenceManifest {
  artifactId: string;
  version: string;
  protocolFreezeSha256: string;
  cohortManifestSha256: string;
  inferenceContract: SviInferenceContract;
  hiddenTruthDereferenced: false;
  businesses: BusinessMetadata[];
  items: V11ConfirmatoryInferenceItem[];
}

function digest(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function argument(name: string, fallback: number): number {
  const prefix = `--${name}=`;
  const value = process.argv.find((item) => item.startsWith(prefix));
  return value ? Math.max(0, Number.parseInt(value.slice(prefix.length), 10)) : fallback;
}

function stringValue(row: Record<string, string | number>, key: string): string {
  return String(row[key] ?? "");
}

async function writeImmutable(path: string, bytes: Buffer): Promise<void> {
  try {
    await writeFile(path, bytes, { flag: "wx" });
  } catch (error) {
    if (!(
      error instanceof Error && "code" in error &&
      (error as NodeJS.ErrnoException).code === "EEXIST"
    )) throw error;
    if (digest(await readFile(path)) !== digest(bytes)) {
      throw new Error(`Immutable V11 confirmatory SVI payload changed: ${path}.`);
    }
  }
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.tmp`;
  await writeFile(temporary, JSON.stringify(value));
  await rename(temporary, path);
}

async function loadProtocol() {
  const [manifestSource, candidateSource, scientificSource] = await Promise.all([
    readFile(resolve(RESEARCH_ROOT, "frozen-protocol/manifest.json")),
    readFile(resolve(RESEARCH_ROOT, "frozen-protocol/candidate-specifications.json")),
    readFile(resolve(RESEARCH_ROOT, "frozen-protocol/scientific-contract.json")),
  ]);
  const protocol = JSON.parse(manifestSource.toString("utf8")) as {
    version: string;
    status: string;
    candidateSpecificationsFileSha256: string;
    frozenV11SelectorSha256: string;
    truthOpened: boolean;
  };
  if (
    protocol.version !== V11_AMSS_CONFIRMATORY_VERSION ||
    protocol.status !== "frozen-before-cohort-generation" ||
    protocol.truthOpened ||
    digest(candidateSource) !== protocol.candidateSpecificationsFileSha256 ||
    digest(await readFile(resolve(RESEARCH_ROOT, "frozen-selectors/v11-selector.json"))) !==
      protocol.frozenV11SelectorSha256
  ) throw new Error("The V11 confirmatory protocol or frozen V11 selector is invalid.");
  const candidates = JSON.parse(candidateSource.toString("utf8")) as AgenticCandidateSpec[];
  const scientific = JSON.parse(scientificSource.toString("utf8")) as {
    inference: SviInferenceContract;
  };
  if (candidates.length !== 24) throw new Error("V11 confirmatory requires exactly 24 frozen specifications.");
  return {
    protocolFreezeSha256: digest(manifestSource),
    candidates,
    inference: scientific.inference,
  };
}

async function loadCohort() {
  const manifestSource = await readFile(resolve(COHORT_ROOT, "manifest.json"));
  const cohort = JSON.parse(manifestSource.toString("utf8")) as {
    version: string;
    status: string;
    businesses: number;
    experiments: number;
    hiddenTruthDereferencedByCandidatePipeline: boolean;
    truthOpened: boolean;
  };
  if (
    cohort.version !== V11_AMSS_CONFIRMATORY_VERSION ||
    cohort.status !== "sealed-observed-inputs-visible-hidden-future-closed" ||
    cohort.businesses !== 100 || cohort.experiments !== 75 ||
    cohort.hiddenTruthDereferencedByCandidatePipeline || cohort.truthOpened
  ) throw new Error("The V11 confirmatory cohort is incomplete or its truth firewall is open.");
  const metadataRows = parseCsv(
    await readFile(resolve(COHORT_ROOT, "business-metadata.csv"), "utf8"),
  ).rows;
  const experimentRows = parseCsv(
    await readFile(resolve(COHORT_ROOT, "experiments.csv"), "utf8"),
  ).rows;
  const metadata = metadataRows.map((row): BusinessMetadata => ({
    businessId: stringValue(row, "business_id"),
    evidenceGroup: stringValue(row, "evidence_group"),
    grossMargin: toNumber(row.gross_margin),
    baselineAnnualSpend: {
      meta_acquisition_spend: toNumber(row.year4_paid_social_spend),
      google_search_nonbrand_spend: toNumber(row.year4_search_spend),
      ctv_spend: toNumber(row.year4_tv_spend),
    },
    budgetUtilization: {
      meta_acquisition_spend: toNumber(row.paid_social_utilization),
      google_search_nonbrand_spend: toNumber(row.search_utilization),
      ctv_spend: toNumber(row.tv_utilization),
    },
  }));
  const experiments = new Map<string, Experiment[]>();
  experimentRows.forEach((row) => {
    const businessId = stringValue(row, "business_id");
    const experiment: Experiment = {
      channel: stringValue(row, "channel"),
      startDate: stringValue(row, "start_date"),
      endDate: stringValue(row, "end_date"),
      outcomeEndDate: stringValue(row, "outcome_end_date"),
      incrementalOutcome: toNumber(row.incremental_outcome),
      incrementalSpend: toNumber(row.incremental_spend),
      standardError: toNumber(row.standard_error),
      confidence: toNumber(row.confidence),
      scope: stringValue(row, "scope") === "immediate" ? "immediate" : "total",
      source: stringValue(row, "source"),
    };
    experiments.set(businessId, [...(experiments.get(businessId) ?? []), experiment]);
  });
  return { cohortManifestSha256: digest(manifestSource), metadata, experiments };
}

async function fitCandidate(
  dataset: Dataset,
  spec: AgenticCandidateSpec,
  experiments: Experiment[],
  options: ValidationOptions,
): Promise<ModelResult> {
  if (spec.family === "advanced") {
    const fingerprint = await advancedModelFingerprint(
      dataset,
      spec.config,
      spec.advancedConfig,
      experiments,
      options.industryPriorChannels,
    );
    return runAdvancedModel(
      dataset,
      spec.config,
      spec.advancedConfig,
      experiments,
      fingerprint,
      options.industryPriorChannels,
    );
  }
  const fingerprint = await modelFingerprint(
    dataset,
    spec.config,
    experiments,
    spec.family,
    options.industryPriorChannels,
  );
  return runModel(
    dataset,
    spec.config,
    experiments,
    spec.family,
    fingerprint,
    options.industryPriorChannels,
  );
}

async function prepareBusiness(
  metadata: BusinessMetadata,
  experiments: Experiment[],
  candidates: AgenticCandidateSpec[],
  inference: SviInferenceContract,
  protocolFreezeSha256: string,
): Promise<V11ConfirmatoryInferenceItem[]> {
  const csv = await readFile(resolve(COHORT_ROOT, "observed", `${metadata.businessId}.csv`), "utf8");
  const parsed = parseCsv(csv);
  const dataset = await createDataset(
    `${metadata.businessId}.csv`,
    csv,
    parsed.columns,
    parsed.rows,
  );
  if (validateDataset(dataset).status === "blocked" || dataset.rows.length !== 156) {
    throw new Error(`${metadata.businessId} failed the frozen observed-data contract.`);
  }
  const experimentChannels = new Set(experiments.map((item) => item.channel.toLowerCase()));
  const items: V11ConfirmatoryInferenceItem[] = [];
  for (const evidenceArm of ["experiments-only", "benchmark-gap-fill"] as const) {
    const industryPriorChannels = evidenceArm === "benchmark-gap-fill"
      ? V11_AMSS_CONFIRMATORY_CHANNELS.filter((channel) => !experimentChannels.has(channel.toLowerCase()))
      : [];
    const options: ValidationOptions = {
      anchorIndependenceConfirmed: experiments.length > 0,
      industryPriorChannels,
      industryBenchmarkScreeningEnabled: evidenceArm === "benchmark-gap-fill",
      materialSpendShareThreshold: 0.02,
    };
    for (const base of candidates) {
      const candidateId = `${base.id} · ${evidenceArm}`;
      const spec: AgenticCandidateSpec = {
        ...structuredClone(base),
        id: candidateId,
        evidencePriorChannels: industryPriorChannels,
      };
      const model = await fitCandidate(dataset, spec, experiments, options);
      const validationId = await validationFingerprint(
        dataset,
        model,
        spec.config,
        spec.advancedConfig,
        experiments,
        options,
      );
      const validation = await runModelValidation(
        dataset,
        model,
        spec.config,
        spec.advancedConfig,
        experiments,
        validationId,
        undefined,
        options,
      );
      const row: ScoreV6CandidateRow = {
        businessId: metadata.businessId,
        family: "amss-external-transport",
        split: "audit",
        candidateId,
        evidenceArm,
        modelFamily: spec.family,
        eligible: validation.eligible,
        reviewEligible: validation.gates.every((gate) => !gate.applicable || gate.passed),
        eligibilityTier: validation.eligible ? "decision-grade" : "review",
        failedGateCount: validation.gates.filter((gate) => gate.applicable && !gate.passed).length,
        layerScores: Object.fromEntries(Object.entries(validation.layers).map(([id, layer]) => [
          id,
          layer.score,
        ])) as ScoreV6CandidateRow["layerScores"],
        diagnostics: validationDiagnosticValues(validation.layers),
        heuristicScore: validation.heuristicScore ?? 0,
        decisionLoss: 0,
        cappedRegret: 0,
        roiError: 0,
        contributionError: 0,
      };
      const baseTemporal = temporalEvidenceForCandidate(dataset, model, spec.config);
      const flight = wholeFlightGeneralization(dataset, {
        kind: model.kind,
        config: spec.config,
        advancedConfig: spec.advancedConfig,
        experiments,
        validationOptions: options,
      } satisfies ValidationModelSpec);
      const observableRow: SviScoreV4CandidateRow = {
        ...row,
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
      const run: AgenticCandidateRun = { spec, state: "complete", model, validation };
      const compiled = compileSamplingModel(
        dataset,
        run,
        experiments,
        industryPriorChannels,
      );
      const inferenceFingerprint = await modelSha256(JSON.stringify({
        version: V11_AMSS_CONFIRMATORY_VERSION,
        protocolFreezeSha256,
        model: compiled,
        contract: inference,
      }));
      const payload = gzipSync(JSON.stringify({
        fingerprint: inferenceFingerprint,
        model: compiled,
        contract: inference,
      }));
      const payloadPath = resolve(PAYLOAD_ROOT, `${inferenceFingerprint}.payload.json.gz`);
      await writeImmutable(payloadPath, payload);
      items.push({
        businessId: metadata.businessId,
        evidenceGroup: metadata.evidenceGroup,
        candidateId,
        baseCandidateId: base.id,
        evidenceArm,
        modelFamily: spec.family,
        inferenceFingerprint,
        payloadPath,
        payloadObject: `${REMOTE_PREFIX}/payloads/${inferenceFingerprint}.payload.json.gz`,
        payloadSha256: digest(payload),
        outputObject: `${REMOTE_PREFIX}/results/${inferenceFingerprint}.result.json.gz`,
        industryPriorChannels: [...industryPriorChannels],
        observableRow,
      });
    }
  }
  return items;
}

async function worker(shard: number, shards: number): Promise<void> {
  const [protocol, cohort] = await Promise.all([loadProtocol(), loadCohort()]);
  const selected = cohort.metadata.filter((_, index) => index % shards === shard);
  const items: V11ConfirmatoryInferenceItem[] = [];
  await mkdir(PAYLOAD_ROOT, { recursive: true });
  for (let index = 0; index < selected.length; index += 1) {
    const metadata = selected[index];
    items.push(...await prepareBusiness(
      metadata,
      cohort.experiments.get(metadata.businessId) ?? [],
      protocol.candidates,
      protocol.inference,
      protocol.protocolFreezeSha256,
    ));
    process.stdout.write(
      `V11 confirmatory inference preparation ${shard + 1}/${shards}: ${index + 1}/${selected.length} businesses\n`,
    );
  }
  await mkdir(PART_ROOT, { recursive: true });
  await atomicJson(resolve(PART_ROOT, `part-${shard}-of-${shards}.json`), {
    version: V11_AMSS_CONFIRMATORY_VERSION,
    protocolFreezeSha256: protocol.protocolFreezeSha256,
    cohortManifestSha256: cohort.cohortManifestSha256,
    hiddenTruthDereferenced: false,
    shard,
    shards,
    items,
  });
}

async function runChild(shard: number, shards: number): Promise<void> {
  await new Promise<void>((done, reject) => {
    const child = spawn(process.execPath, [
      "--import",
      "tsx",
      resolve("research/svi_score_v11_amss_confirmatory/prepare-inference.ts"),
      "--worker",
      `--shard=${shard}`,
      `--shards=${shards}`,
    ], { cwd: process.cwd(), env: process.env, stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code) => code === 0 ? done() : reject(
      new Error(`V11 confirmatory inference preparation worker ${shard} exited ${code}.`),
    ));
  });
}

async function orchestrate(): Promise<void> {
  const [protocol, cohort] = await Promise.all([loadProtocol(), loadCohort()]);
  const workers = Math.max(1, Math.min(argument("workers", Math.min(8, cpus().length)), 16));
  await mkdir(PART_ROOT, { recursive: true });
  await Promise.all(Array.from({ length: workers }, (_, shard) => runChild(shard, workers)));
  const items: V11ConfirmatoryInferenceItem[] = [];
  for (let shard = 0; shard < workers; shard += 1) {
    const part = JSON.parse(
      await readFile(resolve(PART_ROOT, `part-${shard}-of-${workers}.json`), "utf8"),
    ) as {
      version: string;
      protocolFreezeSha256: string;
      cohortManifestSha256: string;
      hiddenTruthDereferenced: boolean;
      items: V11ConfirmatoryInferenceItem[];
    };
    if (
      part.version !== V11_AMSS_CONFIRMATORY_VERSION ||
      part.protocolFreezeSha256 !== protocol.protocolFreezeSha256 ||
      part.cohortManifestSha256 !== cohort.cohortManifestSha256 ||
      part.hiddenTruthDereferenced
    ) throw new Error(`V11 confirmatory preparation part ${shard} is stale or invalid.`);
    items.push(...part.items);
  }
  items.sort((left, right) =>
    left.businessId.localeCompare(right.businessId) ||
    left.candidateId.localeCompare(right.candidateId)
  );
  if (
    items.length !== V11_AMSS_CONFIRMATORY_CONTRACT.cohort.candidateFits ||
    new Set(items.map((item) => item.inferenceFingerprint)).size !== items.length ||
    new Set(items.map((item) => item.businessId)).size !== 100
  ) throw new Error("The V11 confirmatory inference manifest is incomplete or duplicated.");
  const manifest: V11ConfirmatoryInferenceManifest = {
    artifactId: "flux-v11-amss-confirmatory-inference-manifest-v1",
    version: V11_AMSS_CONFIRMATORY_VERSION,
    protocolFreezeSha256: protocol.protocolFreezeSha256,
    cohortManifestSha256: cohort.cohortManifestSha256,
    inferenceContract: protocol.inference,
    hiddenTruthDereferenced: false,
    businesses: cohort.metadata,
    items,
  };
  await mkdir(INFERENCE_ROOT, { recursive: true });
  await atomicJson(MANIFEST_PATH, manifest);
  await atomicJson(CLOUD_MANIFEST_PATH, {
    version: "flux-svi-cvm-retry-v1",
    sourceCheckpoint: "v11-amss-confirmatory-inference-manifest.json",
    failureCount: items.length,
    remotePrefix: REMOTE_PREFIX,
    protocolFreezeSha256: protocol.protocolFreezeSha256,
    auditVersion: V11_AMSS_CONFIRMATORY_VERSION,
    items: items.map((item) => ({
      fingerprint: item.inferenceFingerprint,
      businessId: item.businessId,
      candidateId: item.candidateId,
      payloadObject: item.payloadObject,
      payloadSha256: item.payloadSha256,
      outputObject: item.outputObject,
    })),
  });
  console.log(JSON.stringify({
    prepared: true,
    businesses: cohort.metadata.length,
    payloads: items.length,
    uniqueFingerprints: new Set(items.map((item) => item.inferenceFingerprint)).size,
    cloudManifest: CLOUD_MANIFEST_PATH,
    hiddenTruthDereferenced: false,
  }, null, 2));
}

if (process.argv.includes("--worker")) {
  await worker(argument("shard", 0), argument("shards", 1));
} else {
  await orchestrate();
}
