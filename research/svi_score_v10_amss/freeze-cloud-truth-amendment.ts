import { createHash } from "node:crypto";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { V10_AMSS_VERSION } from "./contract";

const ROOT = resolve(".flux-artifacts/svi-score-v10-amss");
const OUTPUT = resolve(ROOT, "opened-truth/technical-amendment-002.json");
const paths = {
  candidateActions: resolve(ROOT, "observable-freeze/candidate-actions.csv"),
  frozenSelections: resolve(ROOT, "observable-freeze/frozen-selections.json"),
  truthOpenReceipt: resolve(ROOT, "opened-truth/truth-open-receipt.json"),
  cohortManifest: resolve(ROOT, "sealed-cohort/manifest.json"),
  technicalAmendment001: resolve(ROOT, "opened-truth/technical-amendment-001.json"),
  evaluator: resolve("research/svi_score_v10_amss/open-truth.R"),
  cloudTask: resolve("research/svi_score_v10_amss/cloud/truth_task.py"),
  dockerfile: resolve("research/svi_score_v10_amss/cloud/Dockerfile.truth"),
  cloudRun: resolve("research/svi_score_v10_amss/cloud/truth-cloud-run.yaml"),
  amssArchive: resolve("research/svi_score_v10_amss/cloud/amss-1.0.1-cbf5e7f6.tar.gz"),
};

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

const source = Object.fromEntries(await Promise.all(
  Object.entries(paths).map(async ([key, path]) => [key, await readFile(path)] as const),
));
const receipt = JSON.parse(source.truthOpenReceipt.toString("utf8")) as {
  version: string;
  workers: number;
  candidateActionsSha256: string;
  frozenSelectionsSha256: string;
};
if (
  receipt.version !== V10_AMSS_VERSION || receipt.workers !== 8 ||
  receipt.candidateActionsSha256 !== sha256(source.candidateActions) ||
  receipt.frozenSelectionsSha256 !== sha256(source.frozenSelections)
) throw new Error("The original truth-open receipt no longer matches the frozen inputs.");

const existingParts = (await readdir(resolve(ROOT, "opened-truth/parts")))
  .filter((name) => name.endsWith(".csv"));
if (existingParts.length !== 0) {
  throw new Error(`Cloud partition amendment requires zero retained truth parts; found ${existingParts.length}.`);
}

const amendment = {
  artifactId: "flux-v10-amss-truth-evaluator-technical-amendment-002",
  version: V10_AMSS_VERSION,
  status: "frozen-cloud-execution-only-amendment",
  amendedAt: new Date().toISOString(),
  reason:
    "Observed local AMSS counterfactual throughput projected 20-30 hours. The local run was stopped with zero retained parts; each sealed business is reassigned to one independent Cloud Run task.",
  scientificContractChanged: false,
  frozenCandidatesChanged: false,
  frozenActionsChanged: false,
  frozenSelectionsChanged: false,
  randomSeedsChanged: false,
  truthReplicatesChanged: false,
  estimandChanged: false,
  retrainingPermitted: false,
  reselectionPermitted: false,
  completedTruthPartsBeforeAmendment: 0,
  localExecution: { workers: 8, retainedParts: 0 },
  cloudExecution: {
    platform: "Google Cloud Run Jobs",
    region: "europe-west1",
    tasks: 100,
    businessesPerTask: 1,
    parallelism: 100,
    cpuPerTask: 1,
    memoryGiBPerTask: 2,
    maximumTaskSeconds: 14_400,
    maximumRetries: 0,
    projectedWallHours: [1.5, 3.5],
    projectedIncrementalCostUsd: [12, 18],
    requiresExplicitSpendApproval: true,
  },
  amssSource: {
    repository: "https://github.com/google/amss",
    version: "1.0.1",
    commit: "cbf5e7f6c668de493077ce1d08a2ab963891f0cc",
    archiveSha256: sha256(source.amssArchive),
    localFunctionBodiesCompared: 49,
    localFunctionBodyMismatches: 0,
  },
  immutableInputs: {
    candidateActionsSha256: sha256(source.candidateActions),
    frozenSelectionsSha256: sha256(source.frozenSelections),
    truthOpenReceiptSha256: sha256(source.truthOpenReceipt),
    cohortManifestSha256: sha256(source.cohortManifest),
    technicalAmendment001Sha256: sha256(source.technicalAmendment001),
  },
  executionSources: {
    evaluatorSha256: sha256(source.evaluator),
    cloudTaskSha256: sha256(source.cloudTask),
    dockerfileSha256: sha256(source.dockerfile),
    cloudRunTemplateSha256: sha256(source.cloudRun),
  },
};
const serialized = `${JSON.stringify(amendment, null, 2)}\n`;
try {
  await writeFile(OUTPUT, serialized, { flag: "wx" });
} catch (error) {
  if (!(
    error instanceof Error && "code" in error &&
    (error as NodeJS.ErrnoException).code === "EEXIST"
  )) throw error;
  if ((await readFile(OUTPUT, "utf8")) !== serialized) {
    throw new Error("The cloud truth amendment is already frozen with different content.");
  }
}
console.log(JSON.stringify({
  frozen: true,
  amendment: OUTPUT,
  tasks: amendment.cloudExecution.tasks,
  incrementalCostUsd: amendment.cloudExecution.projectedIncrementalCostUsd,
  spendAuthorized: false,
}, null, 2));
