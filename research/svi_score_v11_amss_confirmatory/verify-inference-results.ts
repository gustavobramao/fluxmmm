import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import type { V9SviResult } from "../svi_score_v9/audit-tokens";
import { V11_AMSS_CONFIRMATORY_VERSION } from "./contract";
import type { V11ConfirmatoryInferenceItem } from "./prepare-inference";

const ROOT = resolve(".flux-artifacts/svi-score-v11-amss-confirmatory/inference");
const RESULTS = resolve(ROOT, "results");
const manifest = JSON.parse(await readFile(resolve(ROOT, "manifest.json"), "utf8")) as {
  version: string;
  hiddenTruthDereferenced: boolean;
  items: V11ConfirmatoryInferenceItem[];
};
if (
  manifest.version !== V11_AMSS_CONFIRMATORY_VERSION || manifest.hiddenTruthDereferenced ||
  manifest.items.length !== 4_800
) throw new Error("V11 confirmatory inference manifest is invalid.");
const names = new Set((await readdir(RESULTS)).filter((name) => name.endsWith(".result.json.gz")));
const status = { labelled: 0, review: 0 };
let runtimeSeconds = 0;
for (const item of manifest.items) {
  const name = `${item.inferenceFingerprint}.result.json.gz`;
  if (!names.has(name)) throw new Error(`Missing V11 confirmatory SVI result: ${name}.`);
  const result = JSON.parse(
    gunzipSync(await readFile(resolve(RESULTS, name))).toString("utf8"),
  ) as V9SviResult;
  if (
    result.fingerprint !== item.inferenceFingerprint ||
    (result.status !== "labelled" && result.status !== "review") ||
    !result.channels.length || !result.decisionDraws.length ||
    !result.diagnostics.finite
  ) throw new Error(`Invalid V11 confirmatory SVI result: ${name}.`);
  status[result.status] += 1;
  runtimeSeconds += result.seeds.reduce((total, seed) => total + seed.runtimeSeconds, 0);
}

console.log(JSON.stringify({
  verified: true,
  results: manifest.items.length,
  uniqueFiles: names.size,
  status,
  aggregateRuntimeHours: runtimeSeconds / 3_600,
  hiddenTruthDereferenced: false,
}, null, 2));
