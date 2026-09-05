import { spawn } from "node:child_process";
import { mkdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { V10InferenceItem } from "./prepare-inference";

const ROOT = resolve(".flux-artifacts/svi-score-v10-amss/inference");
const manifest = JSON.parse(await readFile(resolve(ROOT, "manifest.json"), "utf8")) as {
  items: V10InferenceItem[];
};
if (manifest.items.length !== 4_800) throw new Error("V10 inference manifest is incomplete.");
const requested = Math.max(1, Math.min(
  Number.parseInt(process.argv.find((item) => item.startsWith("--count="))?.split("=")[1] ?? "3", 10),
  6,
));
const selected: V10InferenceItem[] = [];
for (const item of manifest.items) {
  if (
    !selected.some((value) => value.modelFamily === item.modelFamily && value.evidenceArm === item.evidenceArm)
  ) selected.push(item);
  if (selected.length >= requested) break;
}
while (selected.length < requested) selected.push(manifest.items[selected.length]);
await mkdir(resolve(ROOT, "smoke-results"), { recursive: true });

for (const item of selected) {
  const output = resolve(ROOT, "smoke-results", `${item.inferenceFingerprint}.result.json.gz`);
  await new Promise<void>((done, reject) => {
    const child = spawn(resolve(".venv/bin/python"), [
      resolve("scripts/svi_batch.py"),
      "--payload",
      item.payloadPath,
      "--output",
      output,
      "--backend",
      "c",
    ], { cwd: process.cwd(), env: process.env, stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code) => code === 0 ? done() : reject(
      new Error(`V10 SVI smoke failed for ${item.candidateId} with exit ${code}.`),
    ));
  });
}

console.log(JSON.stringify({
  passed: true,
  candidates: selected.map((item) => ({
    businessId: item.businessId,
    candidateId: item.candidateId,
    fingerprint: item.inferenceFingerprint,
  })),
  cloudCostStarted: false,
}, null, 2));
