import { createHash } from "node:crypto";
import { readdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseCsv } from "../../lib/mmm/csv";
import { createDataset, validateDataset } from "../../lib/mmm/schema";
import { V11_AMSS_CONFIRMATORY_CONTRACT, V11_AMSS_CONFIRMATORY_VERSION } from "./contract";

const ROOT = resolve(".flux-artifacts/svi-score-v11-amss-confirmatory");
const STAGING = resolve(ROOT, "cohort-staging");
const SEALED = resolve(ROOT, "sealed-cohort");
const PROTOCOL = resolve("research/svi_score_v11_amss_confirmatory/frozen-protocol/manifest.json");

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

async function fileReceipt(root: string, name: string) {
  const source = await readFile(resolve(root, name));
  return { path: name, bytes: source.length, sha256: sha256(source) };
}

const observed = (await readdir(resolve(STAGING, "observed")))
  .filter((name) => name.endsWith(".csv"))
  .sort();
const hidden = (await readdir(resolve(STAGING, "hidden")))
  .filter((name) => name.endsWith(".rds"))
  .sort();
if (observed.length !== 100 || hidden.length !== 100) {
  throw new Error(`V11 confirmatory cohort is incomplete: ${observed.length} observed and ${hidden.length} hidden.`);
}
if (observed.some((name, index) => name.replace(/\.csv$/, "") !== hidden[index].replace(/\.rds$/, ""))) {
  throw new Error("Observed and sealed hidden V11 confirmatory business identities do not match.");
}

for (const name of observed) {
  const source = await readFile(resolve(STAGING, "observed", name), "utf8");
  const parsed = parseCsv(source);
  const dataset = await createDataset(name, source, parsed.columns, parsed.rows);
  const validation = validateDataset(dataset);
  if (
    validation.status === "blocked" ||
    dataset.rows.length !== 156 ||
    dataset.mediaColumns.length !== 3 ||
    dataset.modelCadence !== "weekly"
  ) throw new Error(`${name} violates the frozen Flux data contract.`);
}

const metadata = parseCsv(await readFile(resolve(STAGING, "business-metadata.csv"), "utf8"));
const experiments = parseCsv(await readFile(resolve(STAGING, "experiments.csv"), "utf8"));
if (metadata.rows.length !== 100 || experiments.rows.length !== 75) {
  throw new Error(`V11 confirmatory expected 100 metadata rows and 75 experiments; found ${metadata.rows.length}/${experiments.rows.length}.`);
}
const experimentCounts = new Map<string, number>();
experiments.rows.forEach((row) => {
  const id = String(row.business_id);
  experimentCounts.set(id, (experimentCounts.get(id) ?? 0) + 1);
});
if ([...experimentCounts.values()].some((count) => count !== 1)) {
  throw new Error("A V11 confirmatory business has more than one model-visible experiment.");
}

const protocolSource = await readFile(PROTOCOL);
const publicFiles = await Promise.all([
  fileReceipt(STAGING, "business-metadata.csv"),
  fileReceipt(STAGING, "experiments.csv"),
  fileReceipt(STAGING, "generation-receipt.txt"),
  ...await Promise.all(observed.map((name) => fileReceipt(resolve(STAGING, "observed"), name))),
]);
const hiddenFiles = await Promise.all(
  hidden.map((name) => fileReceipt(resolve(STAGING, "hidden"), name)),
);
const manifest = {
  artifactId: "flux-v11-amss-confirmatory-sealed-cohort-manifest-v1",
  version: V11_AMSS_CONFIRMATORY_VERSION,
  status: "sealed-observed-inputs-visible-hidden-future-closed",
  protocolFreezeSha256: sha256(protocolSource),
  businesses: observed.length,
  visibleWeeksPerBusiness: V11_AMSS_CONFIRMATORY_CONTRACT.time.visibleWeeks,
  experiments: experiments.rows.length,
  candidateFitsPlanned: V11_AMSS_CONFIRMATORY_CONTRACT.cohort.candidateFits,
  publicFiles,
  hiddenFiles,
  hiddenFileAggregateSha256: sha256(hiddenFiles.map((file) => file.sha256).join("\n")),
  hiddenTruthDereferencedByCandidatePipeline: false,
  truthOpened: false,
};
await writeFile(resolve(STAGING, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, {
  flag: "wx",
});
await rename(STAGING, SEALED);

console.log(JSON.stringify({
  sealed: true,
  path: SEALED,
  businesses: manifest.businesses,
  experiments: manifest.experiments,
  hiddenTruthDereferencedByCandidatePipeline: false,
  truthOpened: false,
}, null, 2));

