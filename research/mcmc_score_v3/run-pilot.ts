import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import learnedScoreV6Artifact from "../score_v6/artifacts/learned-score-v6-pilot.json";
import {
  compileSamplingModel,
  samplingFingerprint,
  type CompiledSamplingModel,
  type SamplingContract,
  type SamplingResult,
} from "../../lib/mmm/sampling";
import {
  getSamplingJob,
  samplingServiceHealth,
  startSamplingJob,
} from "../../lib/mmm/sampling-api";
import {
  evaluateScoreV6,
  groupWeights,
  learnScoreV6,
  scoreV6,
} from "../score_v6/learn";
import type { ScoreV6CandidateRow } from "../score_v6/types";
import {
  GENERATOR_FAMILIES,
  randomizedAuditScenario,
} from "../score_v3/population";
import { syntheticIndustryPriorOverrides } from "../score_v6/evidence";
import { MCMC_SCORE_V3_CONTRACT, MCMC_SCORE_V3_VERSION } from "./contract";
import { prepareMcmcResearchBusiness } from "./cohort";
import { posteriorDecisionLabel, samplingConverged } from "./posterior";
import { selectMcmcCandidates } from "./selection";
import {
  summarizePairedEvidence,
  summarizeSamplerEvidence,
} from "./summary";
import type {
  McmcScoreV3Artifact,
  McmcScoreV3SampleRecord,
} from "./types";

const artifactDirectory = resolve("research/mcmc_score_v3/artifacts");
const checkpointDirectory = resolve(".flux-artifacts/mcmc-score-v3");
const MIN_EXPLORATORY_LABELS = 40;
const MIN_EXPLORATORY_BUSINESSES = 5;

function stringArgument(name: string, fallback: string): string {
  const prefix = `--${name}=`;
  const value = process.argv.find((argument) => argument.startsWith(prefix));
  return (value?.slice(prefix.length) || fallback).replace(/[^a-z0-9-]/gi, "-");
}

function integerArgument(name: string, fallback: number): number {
  const prefix = `--${name}=`;
  const value = process.argv.find((argument) => argument.startsWith(prefix));
  return value ? Math.max(1, Number.parseInt(value.slice(prefix.length), 10)) : fallback;
}

function delay(milliseconds: number) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

function labelledRow(record: McmcScoreV3SampleRecord): ScoreV6CandidateRow {
  return {
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
    heuristicScore: record.v6Score,
    decisionLoss: record.mcmcDecisionLoss!,
    cappedRegret: record.mcmcProfitRegret!,
    roiError: record.mcmcRoiError!,
    contributionError: record.mcmcContributionError!,
  };
}

async function runSampling(
  compiled: CompiledSamplingModel,
  initialContract: SamplingContract,
  useHttpSampler: boolean,
): Promise<{ result: SamplingResult; fingerprint: string; attempts: number }> {
  let contract = initialContract;
  let attempts = 0;
  let totalRuntimeSeconds = 0;
  while (attempts < 2) {
    attempts += 1;
    const fingerprint = await samplingFingerprint(compiled, contract);
    let result: SamplingResult;
    if (useHttpSampler) {
      const started = await startSamplingJob(fingerprint, compiled, contract);
      while (true) {
        const job = await getSamplingJob(started.id);
        process.stdout.write(
          `\r${compiled.promotedId} · ${job.progress.stage} · ${job.progress.completed}/${job.progress.total}`.padEnd(110),
        );
        if (job.status === "error") throw new Error(job.error ?? "Sampling failed.");
        if (job.status === "complete" && job.result) {
          process.stdout.write("\n");
          result = job.result;
          break;
        }
        await delay(750);
      }
    } else {
      const payloadPath = resolve(checkpointDirectory, `${fingerprint}.payload.json`);
      const resultPath = resolve(checkpointDirectory, `${fingerprint}.result.json`);
      await writeFile(
        payloadPath,
        JSON.stringify({ fingerprint, model: compiled, contract }),
      );
      process.stdout.write(`${compiled.promotedId} · direct PyMC batch · ${contract.chains} chains\n`);
      await new Promise<void>((resolveProcess, rejectProcess) => {
        const child = spawn(
          resolve(".venv/bin/python"),
          [
            resolve("scripts/mcmc_batch.py"),
            "--payload",
            payloadPath,
            "--output",
            resultPath,
          ],
          {
            cwd: process.cwd(),
            env: {
              ...process.env,
              PYTENSOR_FLAGS: `base_compiledir=${resolve(".flux-artifacts/pytensor-mcmc-score-v3")}`,
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
            : rejectProcess(new Error(standardError || `PyMC batch exited ${code}.`))
        );
      });
      result = JSON.parse(await readFile(resultPath, "utf8")) as SamplingResult;
    }
    totalRuntimeSeconds += result.runtimeSeconds;
    if (
      !samplingConverged(result) &&
      attempts < 2 &&
      result.retryRecommendation
    ) {
      contract = result.retryRecommendation.contract;
      continue;
    }
    return {
      result: { ...result, runtimeSeconds: totalRuntimeSeconds },
      fingerprint,
      attempts,
    };
  }
  throw new Error("Sampling exhausted its retry contract without a result.");
}

function buildArtifact(
  records: McmcScoreV3SampleRecord[],
  pilotBusinesses: number,
  pilotCandidatesPerBusiness: number,
): McmcScoreV3Artifact {
  const labelled = records.filter(
    (record) => record.status === "labelled" && record.mcmcDecisionLoss !== undefined,
  );
  const uniqueBusinesses = new Set(records.map((record) => record.businessId)).size;
  const comparison: McmcScoreV3Artifact["comparison"] = {
    baselineVersion: learnedScoreV6Artifact.version,
    baselineWeights: learnedScoreV6Artifact.model.weights,
    note:
      "V6 remains the runtime score. V3 weights are an offline research result until a family-held-out validation and sealed audit are complete.",
  };
  const labelledBusinesses = new Set(labelled.map((record) => record.businessId));
  if (
    labelled.length >= MIN_EXPLORATORY_LABELS &&
    labelledBusinesses.size >= MIN_EXPLORATORY_BUSINESSES
  ) {
    const rows = labelled.map(labelledRow);
    const learned = learnScoreV6(rows);
    comparison.learnedWeights = learned.weights;
    comparison.weightDelta = Object.fromEntries(
      Object.entries(learned.weights).map(([feature, weight]) => [
        feature,
        weight -
          learnedScoreV6Artifact.model.weights[
            feature as keyof typeof learnedScoreV6Artifact.model.weights
          ],
      ]),
    );
    comparison.groupWeights = groupWeights(learned.weights);
    comparison.trainBusinesses = labelledBusinesses.size;
    const v6 = evaluateScoreV6(rows, (row) =>
      scoreV6(row, learnedScoreV6Artifact.model.weights)
    );
    const v3 = evaluateScoreV6(rows, (row) => scoreV6(row, learned.weights));
    comparison.note =
      `Exploratory in-sample pilot only: mean posterior decision loss ${v6.meanLoss.toFixed(2)} (V6 weights) versus ${v3.meanLoss.toFixed(2)} (pilot V3 weights). No activation claim is permitted before held-out evaluation.`;
  }
  const converged = records.filter((record) => record.convergencePassed).length;
  const failed = records.filter((record) =>
    record.status === "error" || record.status === "non-converged"
  ).length;
  return {
    artifactId: "flux-mcmc-score-v3-local-pilot",
    version: MCMC_SCORE_V3_VERSION,
    generatedAt: new Date().toISOString(),
    activation: "research-only",
    stage:
      labelled.length >= MIN_EXPLORATORY_LABELS &&
        labelledBusinesses.size >= MIN_EXPLORATORY_BUSINESSES
        ? "pilot-labelled"
        : labelled.length
          ? "pilot-insufficient"
          : records.length
            ? "prepared"
            : "design",
    contract: {
      targetBusinesses: MCMC_SCORE_V3_CONTRACT.targetBusinesses,
      targetCandidatesPerBusiness:
        MCMC_SCORE_V3_CONTRACT.targetCandidatesPerBusiness,
      targetMcmcFits: MCMC_SCORE_V3_CONTRACT.targetMcmcFits,
      pilotBusinesses,
      pilotCandidatesPerBusiness,
      candidatePoolPerBusiness: MCMC_SCORE_V3_CONTRACT.candidatePoolPerBusiness,
      label: MCMC_SCORE_V3_CONTRACT.label,
      sampler: MCMC_SCORE_V3_CONTRACT.sampler,
      goldSampler: MCMC_SCORE_V3_CONTRACT.goldSampler,
      splits: { ...MCMC_SCORE_V3_CONTRACT.targetSplits },
    },
    progress: {
      businessesPrepared: uniqueBusinesses,
      candidatesPrepared: records.length,
      candidatesLabelled: labelled.length,
      convergedCandidates: converged,
      failedCandidates: failed,
      completionShare:
        labelled.length / Math.max(pilotBusinesses * pilotCandidatesPerBusiness, 1),
    },
    pairedEvidence: summarizePairedEvidence(records),
    samplerEvidence: summarizeSamplerEvidence(records),
    comparison,
    safeguards: [
      "Hidden simulator truth is unavailable to candidate fitting, validation, and MCMC shortlisting.",
      "MAP is a search accelerator and a visible baseline; it never supplies the V3 training target.",
      "A label is accepted only after the core NUTS convergence gates pass.",
      "The posterior action maximizes expected value across aligned joint draws; each draw cannot choose its own action.",
      "Random audit candidates and recorded selection probabilities protect against learning only from V6's favorite region.",
      "Businesses are split by generator family before learning; V3 cannot activate from this local pilot.",
    ],
    sampleRecords: records,
  };
}

await Promise.all([
  mkdir(artifactDirectory, { recursive: true }),
  mkdir(checkpointDirectory, { recursive: true }),
]);

const prepareOnly = process.argv.includes("--prepare-only");
const reuseSamples = process.argv.includes("--reuse-samples");
const resume = process.argv.includes("--resume");
const useHttpSampler = process.argv.includes("--http-sampler");
const runId = stringArgument("run-id", "pilot");
const artifactPath = resolve(artifactDirectory, `mcmc-score-v3-${runId}.json`);
const checkpointPath = resolve(checkpointDirectory, `${runId}-records.json`);
const pilotBusinesses = integerArgument(
  "businesses",
  MCMC_SCORE_V3_CONTRACT.pilotBusinesses,
);
const pilotCandidatesPerBusiness = integerArgument(
  "candidates",
  MCMC_SCORE_V3_CONTRACT.pilotCandidatesPerBusiness,
);

let records: McmcScoreV3SampleRecord[] = [];
if (reuseSamples || resume) {
  try {
    records = JSON.parse(await readFile(checkpointPath, "utf8"));
  } catch (error) {
    if (reuseSamples) throw error;
  }
}

if (!reuseSamples) {
  if (!prepareOnly && useHttpSampler) {
    const health = await samplingServiceHealth();
    if (!health.ready) {
      throw new Error(
        `${health.detail ?? "The local MCMC service is not ready"} Run pnpm mcmc:setup and pnpm mcmc:serve first.`,
      );
    }
  }
  const families = GENERATOR_FAMILIES.filter((family) => family.split === "train");
  for (let businessIndex = 0; businessIndex < pilotBusinesses; businessIndex += 1) {
    const family = families[businessIndex % families.length];
    const scenario = randomizedAuditScenario(family, 8_100 + businessIndex);
    const prepared = await prepareMcmcResearchBusiness(
      scenario,
      family.id,
      family.split,
      (detail) => process.stdout.write(`\rPreparing ${detail}`.padEnd(110)),
    );
    process.stdout.write("\n");
    const selected = selectMcmcCandidates(
      prepared.candidates,
      pilotCandidatesPerBusiness,
    );
    for (const item of selected) {
      const { candidate, reason, probability } = item;
      const existing = records.find(
        (record) =>
          record.businessId === candidate.row.businessId &&
          record.candidateId === candidate.row.candidateId,
      );
      if (existing && existing.status !== "error") continue;
      if (existing) records = records.filter((record) => record !== existing);
      const overrides = syntheticIndustryPriorOverrides(prepared.business);
      const compiled = compileSamplingModel(
        prepared.dataset,
        candidate.run,
        prepared.experiments,
        candidate.industryPriorChannels,
        overrides,
      );
      const fingerprint = await samplingFingerprint(
        compiled,
        MCMC_SCORE_V3_CONTRACT.sampler,
      );
      const baseRecord: McmcScoreV3SampleRecord = {
        ...candidate.row,
        v6Score: scoreV6(candidate.row, learnedScoreV6Artifact.model.weights),
        mapDecisionLoss: candidate.row.decisionLoss,
        selectionReason: reason,
        selectionProbability: probability,
        samplingFingerprint: fingerprint,
        samplingContract: MCMC_SCORE_V3_CONTRACT.sampler,
        status: "prepared",
      };
      if (prepareOnly) {
        records.push(baseRecord);
        continue;
      }
      try {
        const sampled = await runSampling(
          compiled,
          MCMC_SCORE_V3_CONTRACT.sampler,
          useHttpSampler,
        );
        const label = posteriorDecisionLabel(
          prepared.business,
          candidate.run.model!,
          candidate.run.spec.config,
          sampled.result,
        );
        records.push({
          ...baseRecord,
          samplingFingerprint: sampled.fingerprint,
          samplingContract: sampled.result.contract,
          status: label.convergencePassed ? "labelled" : "non-converged",
          samplerStatus: sampled.result.status,
          convergencePassed: label.convergencePassed,
          mcmcDecisionLoss: label.decisionLoss,
          mcmcProfitRegret: label.profitRegret,
          mcmcRoiError: label.roiError,
          mcmcContributionError: label.contributionError,
          mapToMcmcLossShift:
            label.decisionLoss === undefined
              ? undefined
              : label.decisionLoss - candidate.row.decisionLoss,
          posteriorRoi: label.posteriorRoi,
          diagnosticsReceipt: sampled.result.diagnostics,
          runtimeSeconds: sampled.result.runtimeSeconds,
          attempts: sampled.attempts,
        });
      } catch (error) {
        records.push({
          ...baseRecord,
          status: "error",
          error: error instanceof Error ? error.message : String(error),
        });
      }
      await writeFile(checkpointPath, JSON.stringify(records, null, 2));
    }
  }
  await writeFile(checkpointPath, JSON.stringify(records, null, 2));
}

const artifact = buildArtifact(
  records,
  pilotBusinesses,
  pilotCandidatesPerBusiness,
);
await writeFile(artifactPath, JSON.stringify(artifact, null, 2));
process.stdout.write(`${JSON.stringify({
  stage: artifact.stage,
  progress: artifact.progress,
  comparison: artifact.comparison.note,
}, null, 2)}\n`);
