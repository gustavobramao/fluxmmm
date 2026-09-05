import { spawn } from "node:child_process";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { gzipSync, gunzipSync } from "node:zlib";
import { compileSamplingModel } from "../../lib/mmm/sampling";
import { sha256 } from "../../lib/mmm/csv";
import { posteriorDecisionLabelFromDraws } from "../mcmc_score_v3/posterior";
import { prepareMcmcResearchBusiness } from "../mcmc_score_v3/cohort";
import { syntheticIndustryPriorOverrides } from "../score_v6/evidence";
import learnedScoreV6Artifact from "../score_v6/artifacts/learned-score-v6-pilot.json";
import type { ScoreV6CandidateRow } from "../score_v6/types";
import { selectSviGoldAudit } from "./audit-selection";
import {
  SVI_SCORE_V3_CONTRACT,
  SVI_SCORE_V3_VERSION,
  type SviInferenceContract,
} from "./contract";
import { stratifiedSviDesignSubset } from "./design";
import type {
  SviApproximationResult,
  SviScoreV3Artifact,
  SviScoreV3Record,
} from "./types";

const artifactDirectory = resolve("research/svi_score_v3/artifacts");
const checkpointDirectory = resolve(".flux-artifacts/svi-score-v3");
const compilerCacheDirectory = resolve(
  ".flux-artifacts/pytensor-svi-score-v3",
);

function integerArgument(name: string, fallback: number): number {
  const prefix = `--${name}=`;
  const value = process.argv.find((argument) => argument.startsWith(prefix));
  return value ? Math.max(1, Number.parseInt(value.slice(prefix.length), 10)) : fallback;
}

function stringArgument(name: string, fallback: string): string {
  const prefix = `--${name}=`;
  const value = process.argv.find((argument) => argument.startsWith(prefix));
  return (value?.slice(prefix.length) || fallback).replace(/[^a-z0-9-]/gi, "-");
}

function optionalArgument(name: string): string | undefined {
  const prefix = `--${name}=`;
  return process.argv.find((argument) => argument.startsWith(prefix))?.slice(prefix.length);
}

function validatedGcsUri(value: string | undefined): string | undefined {
  if (value === undefined) return undefined;
  const normalized = value.replace(/\/+$/, "");
  if (!/^gs:\/\/[a-z0-9][a-z0-9._-]+(?:\/[a-zA-Z0-9._/-]+)?$/.test(normalized)) {
    throw new Error(`Invalid --gcs-uri: ${value}`);
  }
  return normalized;
}

function quantile(values: number[], probability: number): number | undefined {
  if (!values.length) return undefined;
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.round((sorted.length - 1) * probability)),
  );
  return sorted[index];
}

async function inferenceFingerprint(
  model: ReturnType<typeof compileSamplingModel>,
  contract: SviInferenceContract,
): Promise<string> {
  return sha256(
    JSON.stringify({ version: SVI_SCORE_V3_VERSION, model, contract }),
  );
}

async function runSvi(
  fingerprint: string,
  model: ReturnType<typeof compileSamplingModel>,
  contract: SviInferenceContract,
  backend?: "c",
): Promise<SviApproximationResult> {
  const payloadPath = resolve(checkpointDirectory, `${fingerprint}.payload.json.gz`);
  const resultPath = resolve(checkpointDirectory, `${fingerprint}.result.json.gz`);
  try {
    return JSON.parse(
      gunzipSync(await readFile(resultPath)).toString("utf8"),
    ) as SviApproximationResult;
  } catch {
    // A missing result is a cache miss; the immutable payload is written below.
  }
  await writeFile(
    payloadPath,
    gzipSync(JSON.stringify({ fingerprint, model, contract })),
  );
  await new Promise<void>((resolveProcess, rejectProcess) => {
    const child = spawn(
      resolve(".venv/bin/python"),
      [
        resolve("scripts/svi_batch.py"),
        "--payload",
        payloadPath,
        "--output",
        resultPath,
        ...(backend === undefined ? [] : ["--backend", backend]),
      ],
      {
        cwd: process.cwd(),
        env: {
          ...process.env,
          // Each failed-only retry runs in its own process. Restrict native
          // numerical libraries to one thread so worker-level parallelism
          // does not multiply BLAS threads and saturate memory bandwidth.
          OMP_NUM_THREADS: "1",
          OPENBLAS_NUM_THREADS: "1",
          VECLIB_MAXIMUM_THREADS: "1",
          NUMEXPR_NUM_THREADS: "1",
          PYTENSOR_FLAGS: `base_compiledir=${resolve(
            backend === "c"
              ? `.flux-artifacts/pytensor-svi-score-v3/retry-${fingerprint}`
              : ".flux-artifacts/pytensor-svi-score-v3",
          )}`,
        },
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
    let standardError = "";
    child.stdout.on("data", (chunk) => process.stdout.write(String(chunk)));
    child.stderr.on("data", (chunk) => {
      standardError += String(chunk);
      process.stderr.write(String(chunk));
    });
    child.on("error", rejectProcess);
    child.on("close", (code) =>
      code === 0
        ? resolveProcess()
        : rejectProcess(new Error(standardError || `SVI batch exited ${code}.`)),
    );
  });
  return JSON.parse(
    gunzipSync(await readFile(resultPath)).toString("utf8"),
  ) as SviApproximationResult;
}

async function runGcloud(arguments_: string[]): Promise<void> {
  await new Promise<void>((resolveProcess, rejectProcess) => {
    const child = spawn("gcloud", arguments_, {
      cwd: process.cwd(),
      env: process.env,
      stdio: "inherit",
    });
    child.on("error", rejectProcess);
    child.on("close", (code) =>
      code === 0
        ? resolveProcess()
        : rejectProcess(new Error(`gcloud exited ${code}.`)),
    );
  });
}

async function syncResearchToGcs(
  gcsUri: string,
  artifactPath?: string,
): Promise<void> {
  try {
    await runGcloud([
      "storage",
      "rsync",
      checkpointDirectory,
      `${gcsUri}/cache`,
      "--recursive",
    ]);
    if (artifactPath !== undefined) {
      await runGcloud([
        "storage",
        "cp",
        artifactPath,
        `${gcsUri}/final-artifacts/`,
      ]);
    }
  } catch (error) {
    process.stderr.write(
      `GCS backup deferred: ${error instanceof Error ? error.message : String(error)}\n`,
    );
  }
}

function buildArtifact(
  records: SviScoreV3Record[],
  requestedBusinesses: number,
  requestedCandidates: number,
): SviScoreV3Artifact {
  const terminal = records.filter((record) => record.status !== "prepared");
  const businessCounts = new Map<string, number>();
  terminal.forEach((record) =>
    businessCounts.set(record.businessId, (businessCounts.get(record.businessId) ?? 0) + 1),
  );
  const runtimes = terminal.flatMap((record) =>
    record.runtimeSeconds === undefined ? [] : [record.runtimeSeconds],
  );
  const medianRuntime = quantile(runtimes, 0.5);
  const p90Runtime = quantile(runtimes, 0.9);
  const requestedFits = requestedBusinesses * requestedCandidates;
  const allRows = records.map(({ status: _status, ...row }) => row);
  const complete =
    requestedBusinesses === SVI_SCORE_V3_CONTRACT.designBusinesses &&
    requestedCandidates === SVI_SCORE_V3_CONTRACT.candidatesPerBusiness &&
    terminal.length === SVI_SCORE_V3_CONTRACT.targetSviFits;
  const posteriorRows = records.flatMap((record): ScoreV6CandidateRow[] => {
    if (record.sviDecisionLoss === undefined || !Number.isFinite(record.sviDecisionLoss)) {
      return [];
    }
    return [{
      businessId: record.businessId,
      family: record.family,
      split: record.split,
      candidateId: record.candidateId,
      evidenceArm: record.evidenceArm,
      modelFamily: record.modelFamily,
      eligible: record.eligible,
      reviewEligible: record.reviewEligible,
      eligibilityTier: record.eligibilityTier,
      failedGateCount: record.failedGateCount,
      layerScores: record.layerScores,
      diagnostics: record.diagnostics,
      heuristicScore: record.heuristicScore,
      decisionLoss: record.sviDecisionLoss,
      cappedRegret: record.sviProfitRegret ?? record.cappedRegret,
      roiError: record.sviRoiError ?? record.roiError,
      contributionError:
        record.sviContributionError ?? record.contributionError,
    }];
  });
  const labelledCoverage =
    posteriorRows.length / Math.max(SVI_SCORE_V3_CONTRACT.targetSviFits, 1);
  const trainingBusinesses = new Set(
    posteriorRows.filter((row) => row.split === "train").map((row) => row.businessId),
  ).size;
  const learning: SviScoreV3Artifact["learning"] = {
    status: "awaiting-complete-cohort",
    labelledCoverage,
    trainingBusinesses,
    baselineWeights: learnedScoreV6Artifact.model.weights,
    note:
      "Weights remain frozen until every planned fit has terminated and at least 98% of candidates have finite posterior decision labels; validation and audit outcomes are not inspected early.",
  };
  if (complete && labelledCoverage >= 0.98 && trainingBusinesses === 300) {
    learning.status = "ready-for-cross-validation";
    learning.note =
      "The posterior labels are complete and ready for the separate train-only grouped cross-validation stage. This assembly artifact does not learn weights or evaluate validation and audit outcomes.";
  }
  return {
    artifactId: "flux-svi-score-v3-unbiased-design-pilot",
    version: SVI_SCORE_V3_VERSION,
    generatedAt: new Date().toISOString(),
    activation: "research-only",
    stage: complete
      ? "complete"
      : terminal.length
        ? requestedFits < SVI_SCORE_V3_CONTRACT.targetSviFits
          ? "smoke"
          : "running"
        : "design",
    contract: {
      businesses: SVI_SCORE_V3_CONTRACT.designBusinesses,
      candidatesPerBusiness: SVI_SCORE_V3_CONTRACT.candidatesPerBusiness,
      targetSviFits: SVI_SCORE_V3_CONTRACT.targetSviFits,
      splits: { ...SVI_SCORE_V3_CONTRACT.splits },
      inference: records[0]?.inferenceContract ?? SVI_SCORE_V3_CONTRACT.inference,
      targetMcmcAuditFits: SVI_SCORE_V3_CONTRACT.goldAudit.targetMcmcFits,
    },
    progress: {
      businessesStarted: businessCounts.size,
      businessesComplete: [...businessCounts.values()].filter(
        (count) => count >= requestedCandidates,
      ).length,
      candidateAttempts: terminal.length,
      labelled: records.filter((record) => record.status === "labelled").length,
      review: records.filter((record) => record.status === "review").length,
      errors: records.filter((record) => record.status === "error").length,
      completionShare:
        terminal.length / Math.max(SVI_SCORE_V3_CONTRACT.targetSviFits, 1),
    },
    feasibility: {
      medianRuntimeSeconds: medianRuntime,
      p90RuntimeSeconds: p90Runtime,
      projectedSerialHours:
        medianRuntime === undefined
          ? undefined
          : (medianRuntime * SVI_SCORE_V3_CONTRACT.targetSviFits) / 3_600,
      projectedEightWorkerHours:
        medianRuntime === undefined
          ? undefined
          : (medianRuntime * SVI_SCORE_V3_CONTRACT.targetSviFits) / (3_600 * 8),
      observedFits: runtimes.length,
    },
    learning,
    safeguards: [
      "All 48 candidate/evidence contracts receive SVI; MAP score, gates, and hidden truth cannot shortlist the training labels.",
      "MAP values are permitted only as numerical initializers and as comparison features.",
      "Two independent full-rank fits are required; a third predeclared seed adjudicates unstable ELBO or ROI agreement.",
      "Finite review-labelled fits are retained rather than silently discarded, so approximation difficulty can be modeled instead of creating convergence-selection bias.",
      "The gold NUTS audit is sampled independently within estimator-family and evidence-arm strata with recorded inclusion probabilities.",
      "Businesses—not candidate rows—define train, mechanism-held-out validation, and sealed audit splits.",
      "Score tuning is a separate train-only command using five business-grouped folds; this assembly step never evaluates held-out performance.",
    ],
    goldAuditSelection: selectSviGoldAudit(allRows),
    records,
  };
}

await Promise.all([
  mkdir(artifactDirectory, { recursive: true }),
  mkdir(checkpointDirectory, { recursive: true }),
]);

const runId = stringArgument("run-id", "design-pilot");
const requestedBusinesses = integerArgument(
  "businesses",
  SVI_SCORE_V3_CONTRACT.designBusinesses,
);
const requestedCandidates = Math.min(
  SVI_SCORE_V3_CONTRACT.candidatesPerBusiness,
  integerArgument("candidates", SVI_SCORE_V3_CONTRACT.candidatesPerBusiness),
);
const workers = Math.min(8, integerArgument("workers", 1));
const contract: SviInferenceContract = {
  ...SVI_SCORE_V3_CONTRACT.inference,
  iterations: integerArgument(
    "iterations",
    SVI_SCORE_V3_CONTRACT.inference.iterations,
  ),
  draws: integerArgument("draws", SVI_SCORE_V3_CONTRACT.inference.draws),
};
const prepareOnly = process.argv.includes("--prepare-only");
const reuse = process.argv.includes("--reuse");
const resume = process.argv.includes("--resume");
const retryErrorsOnly = process.argv.includes("--retry-errors-only");
const gcsUri = validatedGcsUri(optionalArgument("gcs-uri"));
const gcsEvery = integerArgument("gcs-every", 10);
if (
  resume &&
  runId === "design-pilot" &&
  requestedBusinesses !== SVI_SCORE_V3_CONTRACT.designBusinesses
) {
  throw new Error(
    "The canonical design-pilot must resume with all 500 businesses. " +
      "A smaller count selects a stratified smoke-test cohort rather than the frozen full-run prefix.",
  );
}
if (retryErrorsOnly && !resume) {
  throw new Error("--retry-errors-only requires --resume and an existing checkpoint.");
}
const checkpointPath = resolve(checkpointDirectory, `${runId}-records.json`);
const artifactPath = resolve(
  artifactDirectory,
  `svi-score-v3-${runId}.json`,
);

let records: SviScoreV3Record[] = [];
let checkpointWrite = Promise.resolve();
function saveCheckpoint() {
  checkpointWrite = checkpointWrite.then(async () => {
    const temporaryPath = `${checkpointPath}.tmp`;
    await writeFile(temporaryPath, JSON.stringify(records, null, 2));
    await rename(temporaryPath, checkpointPath);
  });
  return checkpointWrite;
}
if (reuse || resume) {
  try {
    records = JSON.parse(await readFile(checkpointPath, "utf8"));
  } catch (error) {
    if (reuse) throw error;
    const isMissingCheckpoint =
      error instanceof Error &&
      "code" in error &&
      (error as NodeJS.ErrnoException).code === "ENOENT";
    if (!isMissingCheckpoint) {
      const quarantinePath = `${checkpointPath}.corrupt-${Date.now()}`;
      await rename(checkpointPath, quarantinePath);
      process.stderr.write(
        `Checkpoint was not valid JSON and was quarantined at ${quarantinePath}. ` +
          "Cached SVI results will be reused to reconstruct it.\n",
      );
    }
  }
}

if (!reuse) {
  const fullDesign = stratifiedSviDesignSubset(requestedBusinesses);
  const failedBusinessIds = new Set(
    records
      .filter((record) => record.status === "error")
      .map((record) => record.businessId),
  );
  const design = retryErrorsOnly
    ? fullDesign.filter(({ scenario }) => failedBusinessIds.has(scenario.id))
    : fullDesign;
  if (retryErrorsOnly && !design.length) {
    process.stdout.write("No failed SVI labels remain.\n");
  }
  for (let businessIndex = 0; businessIndex < design.length; businessIndex += 1) {
    const { family, scenario } = design[businessIndex];
    const prepared = await prepareMcmcResearchBusiness(
      scenario,
      family.id,
      family.split,
      (detail) =>
        process.stdout.write(
          `\rPreparing ${businessIndex + 1}/${design.length} · ${detail}`.padEnd(120),
        ),
    );
    process.stdout.write("\n");
    const candidates = prepared.candidates.slice(0, requestedCandidates);
    let nextCandidateIndex = 0;
    const runNextCandidate = async () => {
      while (nextCandidateIndex < candidates.length) {
        const candidateIndex = nextCandidateIndex;
        nextCandidateIndex += 1;
      const candidate = candidates[candidateIndex];
      const existing = records.find(
        (record) =>
          record.businessId === candidate.row.businessId &&
          record.candidateId === candidate.row.candidateId,
      );
      if (existing && existing.status !== "error") continue;
      const retryBackend = existing?.status === "error" ? "c" : undefined;
      if (existing) records = records.filter((record) => record !== existing);
      const compiled = compileSamplingModel(
        prepared.dataset,
        candidate.run,
        prepared.experiments,
        candidate.industryPriorChannels,
        syntheticIndustryPriorOverrides(prepared.business),
      );
      const fingerprint = await inferenceFingerprint(compiled, contract);
      const base: SviScoreV3Record = {
        ...candidate.row,
        status: "prepared",
        inferenceFingerprint: fingerprint,
        inferenceContract: contract,
      };
      if (prepareOnly) {
        records.push(base);
        continue;
      }
      process.stdout.write(
        `${scenario.id} · ${candidateIndex + 1}/${candidates.length} · ${candidate.row.candidateId}\n`,
      );
      try {
        const result = await runSvi(fingerprint, compiled, contract, retryBackend);
        const posteriorRoi = Object.fromEntries(
          result.channels.map((channel) => [channel.channel, channel.posteriorMedian]),
        );
        const label = posteriorDecisionLabelFromDraws(
          prepared.business,
          candidate.run.model!,
          candidate.run.spec.config,
          result.decisionDraws,
          posteriorRoi,
          result.diagnostics.finite,
        );
        records.push({
          ...base,
          status: result.status,
          sviDecisionLoss: label.decisionLoss,
          sviProfitRegret: label.profitRegret,
          sviRoiError: label.roiError,
          sviContributionError: label.contributionError,
          mapToSviLossShift:
            label.decisionLoss === undefined
              ? undefined
              : label.decisionLoss - candidate.row.decisionLoss,
          posteriorRoi: label.posteriorRoi,
          diagnosticsReceipt: result.diagnostics,
          seedDiagnostics: result.seeds,
          runtimeSeconds: result.runtimeSeconds,
          executionBackend: result.executionBackend,
        });
      } catch (error) {
        records.push({
          ...base,
          status: "error",
          error: error instanceof Error ? error.message : String(error),
        });
      }
      }
    };
    await Promise.all(
      Array.from(
        { length: Math.min(workers, candidates.length) },
        () => runNextCandidate(),
      ),
    );
    // Candidate inference results are immutable and persisted independently.
    // Checkpoint the assembled research rows once per business to avoid
    // rewriting an increasingly large JSON document after every candidate.
    // A mid-business interruption therefore replays only cached result files.
    await saveCheckpoint();
    if (gcsUri !== undefined && (businessIndex + 1) % gcsEvery === 0) {
      await syncResearchToGcs(gcsUri);
    }
    // PyTensor specializes compiled functions to each business's constant
    // data. Those modules are safe to regenerate and have no value once all
    // candidates for the business have terminated. Retaining them across 500
    // businesses would exhaust a typical laptop disk.
    await rm(compilerCacheDirectory, { recursive: true, force: true });
    await mkdir(compilerCacheDirectory, { recursive: true });
  }
  await saveCheckpoint();
}

const artifact = buildArtifact(records, requestedBusinesses, requestedCandidates);
await writeFile(artifactPath, JSON.stringify(artifact, null, 2));
if (gcsUri !== undefined) await syncResearchToGcs(gcsUri, artifactPath);
process.stdout.write(
  `${JSON.stringify(
    {
      stage: artifact.stage,
      progress: artifact.progress,
      feasibility: artifact.feasibility,
      goldAuditFitsSelected: artifact.goldAuditSelection.length,
    },
    null,
    2,
  )}\n`,
);
