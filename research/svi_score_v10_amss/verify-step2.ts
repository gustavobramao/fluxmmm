import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const ROOT = resolve("research/svi_score_v10_amss");
const ARTIFACT_ROOT = resolve(ROOT, "step2-artifacts");

function invariant(condition: unknown, detail: string): asserts condition {
  if (!condition) throw new Error(`V10 AMSS Step 2 verification failed: ${detail}.`);
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

const manifest = JSON.parse(
  await readFile(resolve(ARTIFACT_ROOT, "manifest.json"), "utf8"),
) as {
  status: string;
  researchLabels: boolean;
  upstreamRevision: string;
  amssVersion: string;
  scenario: {
    timePoints: number;
    burnInTimePoints: number;
    observedTimePoints: number;
    nativeChannels: string[];
    hiddenConsumerStatesPerTimePoint: number;
  };
  files: Record<string, string>;
  inputs: Record<string, string>;
};
const lock = JSON.parse(
  await readFile(resolve(ROOT, "amss-upstream.lock.json"), "utf8"),
) as { revision: string; packageVersion: string };

invariant(manifest.status === "passed", "smoke status is not passed");
invariant(!manifest.researchLabels, "interface smoke was mislabeled as research truth");
invariant(manifest.upstreamRevision === lock.revision, "upstream revision drifted");
invariant(manifest.amssVersion === lock.packageVersion, "AMSS version drifted");
invariant(manifest.scenario.timePoints === 208, "time-point count changed");
invariant(manifest.scenario.burnInTimePoints === 52, "burn-in changed");
invariant(manifest.scenario.observedTimePoints === 156, "observed row count changed");
invariant(
  manifest.scenario.hiddenConsumerStatesPerTimePoint === 198,
  "hidden consumer-state count changed",
);
invariant(
  JSON.stringify(manifest.scenario.nativeChannels) === JSON.stringify(["tv", "search"]),
  "native channel interface changed",
);

for (const [path, expected] of Object.entries(manifest.files)) {
  const value = await readFile(resolve(ARTIFACT_ROOT, path));
  invariant(sha256(value) === expected, `${path} checksum changed`);
}
for (const [path, expected] of Object.entries(manifest.inputs)) {
  const value = await readFile(resolve(ROOT, path));
  invariant(sha256(value) === expected, `${path} checksum changed`);
}

const observed = await readFile(resolve(ARTIFACT_ROOT, "default-observed.csv"), "utf8");
const truth = await readFile(resolve(ARTIFACT_ROOT, "truth-interface-smoke.csv"), "utf8");
invariant(observed.trim().split("\n").length === 157, "observed CSV is incomplete");
invariant(
  truth.trim().split("\n").slice(1).every((row) => row.endsWith(",FALSE")),
  "truth smoke rows are not all excluded from research labels",
);

console.log(JSON.stringify({
  verified: true,
  upstreamRevision: manifest.upstreamRevision,
  amssVersion: manifest.amssVersion,
  observedRows: manifest.scenario.observedTimePoints,
  nativeChannels: manifest.scenario.nativeChannels,
  truthInterfaceRowsAdmissibleAsResearchLabels: false,
}, null, 2));
