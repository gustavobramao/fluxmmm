import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { cpus } from "node:os";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { V10_AMSS_VERSION } from "./contract";

const ROOT = resolve(".flux-artifacts/svi-score-v10-amss");
const COHORT_MANIFEST = resolve(ROOT, "sealed-cohort/manifest.json");
const OBSERVABLE_MANIFEST = resolve(ROOT, "observable-freeze/manifest.json");
const ACTIONS = resolve(ROOT, "observable-freeze/candidate-actions.csv");
const SELECTIONS = resolve(ROOT, "observable-freeze/frozen-selections.json");
const TRUTH_ROOT = resolve(ROOT, "opened-truth");
const RECEIPT = resolve(TRUTH_ROOT, "truth-open-receipt.json");
const PART_ROOT = resolve(TRUTH_ROOT, "parts");

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function argument(name: string, fallback: number): number {
  const prefix = `--${name}=`;
  const match = process.argv.find((item) => item.startsWith(prefix));
  return match ? Number.parseInt(match.slice(prefix.length), 10) : fallback;
}

async function runWorker(index: number, workers: number): Promise<void> {
  await new Promise<void>((done, reject) => {
    const child = spawn("Rscript", [
      resolve("research/svi_score_v10_amss/open-truth.R"),
      `--worker=${index}`,
      `--workers=${workers}`,
    ], { cwd: process.cwd(), env: process.env, stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code) => code === 0 ? done() : reject(
      new Error(`V10 AMSS truth worker ${index} exited ${code}.`),
    ));
  });
}

if (!process.argv.includes("--confirm-open-truth")) {
  throw new Error("Opening AMSS truth is irreversible. Pass --confirm-open-truth after actions and selections are frozen.");
}

const [cohortSource, observableSource, actionsSource, selectionsSource] = await Promise.all([
  readFile(COHORT_MANIFEST),
  readFile(OBSERVABLE_MANIFEST),
  readFile(ACTIONS),
  readFile(SELECTIONS),
]);
const cohort = JSON.parse(cohortSource.toString("utf8")) as {
  version: string;
  status: string;
  businesses: number;
  truthOpened: boolean;
};
const observable = JSON.parse(observableSource.toString("utf8")) as {
  version: string;
  status: string;
  actions: number;
  candidateActionsSha256: string;
  truthOpened: boolean;
};
const selections = JSON.parse(selectionsSource.toString("utf8")) as {
  version: string;
  status: string;
  businesses: number;
  provenance: { observableManifestSha256: string; hiddenTruthDereferenced: boolean };
};
if (
  cohort.version !== V10_AMSS_VERSION || observable.version !== V10_AMSS_VERSION ||
  selections.version !== V10_AMSS_VERSION || cohort.businesses !== 100 ||
  observable.actions !== 19_200 || selections.businesses !== 100 ||
  observable.candidateActionsSha256 !== sha256(actionsSource) ||
  selections.provenance.observableManifestSha256 !== sha256(observableSource) ||
  cohort.truthOpened || observable.truthOpened || selections.provenance.hiddenTruthDereferenced ||
  selections.status !== "frozen-before-amss-truth-open"
) throw new Error("The V10 truth firewall cannot open: a frozen input is incomplete or inconsistent.");

const workers = Math.max(1, Math.min(argument("workers", Math.min(8, cpus().length)), 16));
const receipt = {
  artifactId: "flux-v10-amss-truth-open-receipt-v1",
  version: V10_AMSS_VERSION,
  status: "truth-opened-for-frozen-action-evaluation",
  openedAt: new Date().toISOString(),
  cohortManifestSha256: sha256(cohortSource),
  observableManifestSha256: sha256(observableSource),
  candidateActionsSha256: sha256(actionsSource),
  frozenSelectionsSha256: sha256(selectionsSource),
  workers,
  permittedOperation: "evaluate already-frozen candidate actions only",
  retrainingPermitted: false,
  reselectionPermitted: false,
};
await mkdir(PART_ROOT, { recursive: true });
try {
  await writeFile(RECEIPT, `${JSON.stringify(receipt, null, 2)}\n`, { flag: "wx" });
} catch (error) {
  if (!(
    error instanceof Error && "code" in error &&
    (error as NodeJS.ErrnoException).code === "EEXIST"
  )) throw error;
  const existing = JSON.parse(await readFile(RECEIPT, "utf8")) as typeof receipt;
  for (const key of [
    "version", "status", "cohortManifestSha256", "observableManifestSha256",
    "candidateActionsSha256", "frozenSelectionsSha256", "workers",
  ] as const) {
    if (existing[key] !== receipt[key]) {
      throw new Error(`Existing truth-open receipt conflicts on ${key}; a new truth run is forbidden.`);
    }
  }
}

await Promise.all(Array.from({ length: workers }, (_, index) => runWorker(index, workers)));
const parts = (await readdir(PART_ROOT)).filter((name) =>
  name.endsWith(`-of-${String(workers).padStart(3, "0")}.csv`)
);
if (parts.length !== workers) throw new Error(`Truth evaluation produced ${parts.length}/${workers} parts.`);

console.log(JSON.stringify({
  opened: true,
  businesses: 100,
  workers,
  parts: parts.length,
  candidateActionsWereFrozen: true,
  retrainingPermitted: false,
}, null, 2));
