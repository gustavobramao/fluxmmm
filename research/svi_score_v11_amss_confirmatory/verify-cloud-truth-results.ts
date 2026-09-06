import { createHash } from "node:crypto";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseCsv } from "../../lib/mmm/csv";
import { V11_AMSS_CONFIRMATORY_VERSION } from "./contract";

const ROOT = resolve(".flux-artifacts/svi-score-v11-amss-confirmatory");
const PART_ROOT = resolve(ROOT, "opened-truth/parts");
const RECEIPT_PATH = resolve(ROOT, "opened-truth/cloud-truth-retrieval-receipt.json");
const execution = process.argv.find((value) => value.startsWith("--execution="))?.slice(12);
if (!execution) throw new Error("Pass the completed Cloud Run execution as --execution=<name>.");
const sha256 = (source: Buffer | string) => createHash("sha256").update(source).digest("hex");
const names = (await readdir(PART_ROOT)).filter((name) => name.endsWith(".csv")).sort();
if (names.length !== 100) throw new Error(`Retrieved ${names.length}/100 truth parts.`);
let rows = 0;
const keys = new Set<string>();
const files = [];
for (let index = 0; index < names.length; index += 1) {
  const expected = `part-${String(index).padStart(3, "0")}-of-100.csv`;
  if (names[index] !== expected) throw new Error(`Unexpected truth part: ${names[index]}; expected ${expected}.`);
  const source = await readFile(resolve(PART_ROOT, names[index]));
  const parsed = parseCsv(source.toString("utf8"));
  if (parsed.rows.length !== 192) throw new Error(`${names[index]} has ${parsed.rows.length}/192 rows.`);
  for (const row of parsed.rows) {
    const key = `${row.business_id}\u0000${row.candidate_id}\u0000${row.scenario}`;
    if (keys.has(key)) throw new Error(`Duplicate truth action: ${key}.`);
    keys.add(key);
  }
  rows += parsed.rows.length;
  files.push({ path: names[index], rows: parsed.rows.length, sha256: sha256(source) });
}
if (rows !== 19_200 || keys.size !== 19_200) throw new Error("Cloud truth output is incomplete.");
const receipt = {
  artifactId: "flux-v11-amss-confirmatory-cloud-truth-retrieval-v1",
  version: V11_AMSS_CONFIRMATORY_VERSION,
  status: "all-cloud-truth-parts-retrieved-and-verified",
  verifiedAt: new Date().toISOString(),
  cloudRunExecution: execution,
  businesses: 100,
  parts: names.length,
  rows,
  uniqueCandidateScenarioRows: keys.size,
  files,
};
await writeFile(RECEIPT_PATH, `${JSON.stringify(receipt, null, 2)}\n`, { flag: "wx" });
console.log(JSON.stringify({ verified: true, execution, businesses: 100, rows }, null, 2));
