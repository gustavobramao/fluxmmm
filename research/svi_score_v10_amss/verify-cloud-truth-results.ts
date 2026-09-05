import { createHash } from "node:crypto";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseCsv, toNumber } from "../../lib/mmm/csv";
import { V10_AMSS_DECISION_SCENARIOS, V10_AMSS_VERSION } from "./contract";

const ROOT = resolve(".flux-artifacts/svi-score-v10-amss");
const PART_ROOT = resolve(ROOT, "opened-truth/parts");
const OUTPUT = resolve(ROOT, "opened-truth/cloud-truth-retrieval-receipt.json");

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function expectedBusiness(index: number): string {
  return `amss-v10-${String(index + 1).padStart(3, "0")}`;
}

const partNames = (await readdir(PART_ROOT)).filter((name) => name.endsWith(".csv")).sort();
if (partNames.length !== 100) throw new Error(`V10 cloud truth has ${partNames.length}/100 parts.`);

const files: Array<{ path: string; bytes: number; sha256: string; rows: number }> = [];
const seen = new Set<string>();
for (let index = 0; index < partNames.length; index += 1) {
  const expectedName = `part-${String(index).padStart(3, "0")}-of-100.csv`;
  if (partNames[index] !== expectedName) {
    throw new Error(`Expected ${expectedName}; found ${partNames[index]}.`);
  }
  const source = await readFile(resolve(PART_ROOT, partNames[index]));
  const parsed = parseCsv(source.toString("utf8"));
  const businessId = expectedBusiness(index);
  if (parsed.rows.length !== 192) {
    throw new Error(`${businessId} has ${parsed.rows.length}/192 truth rows.`);
  }
  const scenarios = new Map<string, number>();
  const seeds = new Map<string, Set<number>>();
  for (const row of parsed.rows) {
    if (row.business_id !== businessId) throw new Error(`${partNames[index]} contains ${row.business_id}.`);
    const scenario = String(row.scenario);
    scenarios.set(scenario, (scenarios.get(scenario) ?? 0) + 1);
    if (!seeds.has(scenario)) seeds.set(scenario, new Set());
    seeds.get(scenario)!.add(toNumber(row.common_random_seed));
    const rowKey = `${row.business_id}\u0000${row.candidate_id}\u0000${scenario}`;
    if (seen.has(rowKey)) throw new Error(`Duplicate V10 truth row: ${rowKey}.`);
    seen.add(rowKey);
    for (const field of [
      "realized_profit", "realized_profit_se", "realized_revenue", "realized_spend",
      "baseline_profit", "baseline_profit_se", "baseline_revenue", "baseline_realized_spend",
    ]) {
      if (!Number.isFinite(toNumber(row[field]))) {
        throw new Error(`${businessId}/${scenario}/${row.candidate_id} has non-finite ${field}.`);
      }
    }
    if (toNumber(row.truth_replicates) !== 4) {
      throw new Error(`${businessId}/${scenario}/${row.candidate_id} changed truth replicates.`);
    }
  }
  for (const scenario of V10_AMSS_DECISION_SCENARIOS) {
    if (scenarios.get(scenario.id) !== 48 || seeds.get(scenario.id)?.size !== 1) {
      throw new Error(`${businessId}/${scenario.id} violates the 48-row common-seed contract.`);
    }
  }
  files.push({ path: partNames[index], bytes: source.length, sha256: sha256(source), rows: 192 });
}
if (seen.size !== 19_200) throw new Error(`V10 cloud truth has ${seen.size}/19200 unique rows.`);

const aggregate = sha256(files.map((file) => `${file.path}\u0000${file.sha256}`).join("\n"));
const receipt = {
  artifactId: "flux-v10-amss-cloud-truth-retrieval-receipt-v1",
  version: V10_AMSS_VERSION,
  status: "all-cloud-truth-parts-retrieved-and-verified",
  cloudRunExecution: "flux-v10-amss-truth-001-5lkmj",
  businesses: 100,
  parts: 100,
  rows: 19_200,
  rowsPerBusiness: 192,
  scenariosPerBusiness: 4,
  candidatesPerScenario: 48,
  truthReplicates: 4,
  commonRandomSeedPerBusinessScenario: true,
  uniqueCandidateScenarioRows: seen.size,
  aggregateSha256: aggregate,
  files,
};
const serialized = `${JSON.stringify(receipt, null, 2)}\n`;
try {
  await writeFile(OUTPUT, serialized, { flag: "wx" });
} catch (error) {
  if (!(
    error instanceof Error && "code" in error &&
    (error as NodeJS.ErrnoException).code === "EEXIST"
  )) throw error;
  if ((await readFile(OUTPUT, "utf8")) !== serialized) {
    throw new Error("Existing V10 cloud truth retrieval receipt differs.");
  }
}
console.log(JSON.stringify({ verified: true, businesses: 100, rows: seen.size, aggregateSha256: aggregate }, null, 2));
