import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const ROOT = resolve("research/svi_score_v11_amss_confirmatory/frozen-selectors");
const SOURCES = {
  v11Selector: resolve(".flux-artifacts/svi-score-v11-evidence-adaptive/development-selector.json"),
  v11TrainingSource: resolve("research/svi_score_v11_evidence_adaptive/train_selector.py"),
  v11ContractSource: resolve("research/svi_score_v11_evidence_adaptive/contract.ts"),
  v9Selector: resolve("research/svi_score_v10_amss/frozen-v9/selector.json"),
  v9TrainingSource: resolve(
    "research/svi_score_v10_amss/frozen-v9/source/research/svi_score_v9/train_selector.py",
  ),
} as const;

function sha256(source: Buffer | string): string {
  return createHash("sha256").update(source).digest("hex");
}

async function immutable(path: string, source: Buffer): Promise<void> {
  try {
    await writeFile(path, source, { flag: "wx" });
  } catch (error) {
    if (!(
      error instanceof Error && "code" in error &&
      (error as NodeJS.ErrnoException).code === "EEXIST"
    )) throw error;
    if (sha256(await readFile(path)) !== sha256(source)) {
      throw new Error(`Frozen selector artifact changed: ${path}.`);
    }
  }
}

const entries = await Promise.all(Object.entries(SOURCES).map(async ([name, path]) => {
  const source = await readFile(path);
  return { name, path, source, sha256: sha256(source) };
}));
const v11 = JSON.parse(
  entries.find((entry) => entry.name === "v11Selector")!.source.toString("utf8"),
) as {
  activation: string;
  featureNames: string[];
  contextNames: string[];
  models: unknown[];
  policy: { danger_penalty: number; uncertainty_penalty: number };
  provenance: { posteriorRefits: number; amssConfirmatoryCohortAccessed: boolean };
};
if (
  v11.activation !== "development-only" || v11.featureNames.length !== 232 ||
  v11.contextNames.length !== 63 || v11.models.length !== 5 ||
  v11.policy.danger_penalty !== 0.5 || v11.policy.uncertainty_penalty !== 0.25 ||
  v11.provenance.posteriorRefits !== 0 || v11.provenance.amssConfirmatoryCohortAccessed
) throw new Error("The V11 selector does not satisfy the pre-audit freeze contract.");

await mkdir(ROOT, { recursive: true });
const targets: Record<string, string> = {
  v11Selector: "v11-selector.json",
  v11TrainingSource: "v11-train-selector.py",
  v11ContractSource: "v11-contract.ts",
  v9Selector: "v9-selector.json",
  v9TrainingSource: "v9-train-selector.py",
};
for (const entry of entries) {
  await immutable(resolve(ROOT, targets[entry.name]), entry.source);
}
const manifest = {
  artifactId: "flux-v11-amss-confirmatory-selector-freeze-v1",
  status: "selectors-frozen-before-cohort-generation",
  frozenAt: new Date().toISOString(),
  truthOpened: false,
  v11: {
    featureTokens: v11.featureNames.length,
    contextTokens: v11.contextNames.length,
    ensembleMembers: v11.models.length,
    policy: v11.policy,
  },
  files: Object.fromEntries(entries.map((entry) => [targets[entry.name], entry.sha256])),
};
await immutable(resolve(ROOT, "manifest.json"), Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`));
console.log(JSON.stringify({ frozen: true, ...manifest.v11, truthOpened: false }, null, 2));
