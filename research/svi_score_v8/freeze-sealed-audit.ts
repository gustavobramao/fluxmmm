import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { SVI_SCORE_V8_AUDIT_CONTRACT } from "./audit-contract";
import { SVI_SCORE_V8_VERSION } from "./contract";

const ROOT = resolve(".flux-artifacts/svi-score-v8");
const OUTPUT = resolve(ROOT, "sealed-audit-protocol-freeze.json");
const OPEN = resolve(ROOT, "sealed-audit-open-receipt.json");
const AUDIT_RESULT = resolve("research/svi_score_v8/artifacts/svi-score-v8-sealed-audit.json");
const DEVELOPMENT = resolve("research/svi_score_v8/artifacts/svi-score-v8-development.json");
const INPUTS = {
  developmentArtifact: DEVELOPMENT,
  developmentDataset: resolve(ROOT, "development-dataset.json"),
  frozenSelector: resolve(ROOT, "frozen-development-selector.json"),
  frozenComparator: resolve(ROOT, "frozen-v5d-svi-comparator.json"),
  v8Contract: resolve("research/svi_score_v8/contract.ts"),
  auditContract: resolve("research/svi_score_v8/audit-contract.ts"),
  auditDatasetBuilder: resolve("research/svi_score_v8/build-sealed-audit-dataset.ts"),
  auditScorer: resolve("research/svi_score_v8/score-sealed-audit.py"),
  learnerImplementation: resolve("research/svi_score_v8/train_selector.py"),
} as const;

async function exists(path: string): Promise<boolean> {
  try {
    await readFile(path);
    return true;
  } catch (error) {
    if (error instanceof Error && "code" in error &&
      (error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

if (await exists(OPEN) || await exists(AUDIT_RESULT)) {
  throw new Error("Cannot freeze: the V8 sealed audit has already been opened or scored.");
}
const development = JSON.parse(await readFile(DEVELOPMENT, "utf8")) as {
  version: string;
  checks: Record<string, boolean>;
  conclusion: { sealedAuditMayOpen: boolean; sealedAuditAccessed: boolean };
};
if (
  development.version !== SVI_SCORE_V8_VERSION ||
  !Object.values(development.checks).every(Boolean) ||
  !development.conclusion.sealedAuditMayOpen ||
  development.conclusion.sealedAuditAccessed
) throw new Error("V8 did not earn sealed-audit eligibility.");
const hashes = Object.fromEntries(await Promise.all(
  Object.entries(INPUTS).map(async ([id, path]) => [
    id,
    createHash("sha256").update(await readFile(path)).digest("hex"),
  ]),
));
const payload = {
  artifactId: "flux-svi-score-v8-sealed-audit-protocol-freeze-v1",
  version: SVI_SCORE_V8_VERSION,
  frozenAt: new Date().toISOString(),
  status: "frozen-sealed",
  contract: SVI_SCORE_V8_AUDIT_CONTRACT,
  inputSha256: hashes,
  developmentChecks: structuredClone(development.checks),
  auditAccessed: false,
  nutsUsed: false,
};
await writeFile(OUTPUT, JSON.stringify(payload, null, 2), { flag: "wx" });
process.stdout.write(`${JSON.stringify({
  path: OUTPUT,
  status: payload.status,
  inputsFrozen: Object.keys(hashes).length,
  auditAccessed: false,
  nutsUsed: false,
}, null, 2)}\n`);
