import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { V11_AMSS_CONFIRMATORY_VERSION } from "./contract";

const ROOT = resolve("research/svi_score_v11_amss_confirmatory");
const FREEZE = resolve(ROOT, "frozen-protocol");
const sha256 = (source: Buffer | string) => createHash("sha256").update(source).digest("hex");
const invariant: (value: unknown, message: string) => asserts value = (value, message) => {
  if (!value) throw new Error(`V11 AMSS protocol verification failed: ${message}.`);
};
const manifest = JSON.parse(await readFile(resolve(FREEZE, "manifest.json"), "utf8")) as {
  version: string; status: string; truthOpened: boolean; selectorFreezeSha256: string;
  frozenV11SelectorSha256: string; frozenV9SelectorSha256: string; contractSha256: string;
  candidateSpecificationsFileSha256: string; cohortDesignSha256: string; businesses: number;
  evidenceGroupCounts: Record<string, number>; moduleOrderCounts: Record<string, number>;
  sourceSha256: Record<string, string>;
  cloudImageDigest: string; cloudTruthImageDigest: string;
  discardedPreCohortTruthImageDigest: string;
};
invariant(manifest.version === V11_AMSS_CONFIRMATORY_VERSION, "version changed");
invariant(manifest.status === "frozen-before-cohort-generation", "status changed");
invariant(!manifest.truthOpened, "truth was open at protocol freeze");
invariant(manifest.businesses === 100, "business count changed");
invariant(Object.values(manifest.evidenceGroupCounts).every((count) => count === 25), "evidence groups are unbalanced");
invariant(Object.values(manifest.moduleOrderCounts).reduce((a, b) => a + b, 0) === 100, "module orders are incomplete");
invariant(sha256(await readFile(resolve(FREEZE, "scientific-contract.json"))) === manifest.contractSha256, "contract changed");
invariant(sha256(await readFile(resolve(FREEZE, "candidate-specifications.json"))) === manifest.candidateSpecificationsFileSha256, "candidate library changed");
invariant(sha256(await readFile(resolve(FREEZE, "cohort-design.csv"))) === manifest.cohortDesignSha256, "cohort design changed");
invariant(sha256(await readFile(resolve(ROOT, "frozen-selectors/manifest.json"))) === manifest.selectorFreezeSha256, "selector manifest changed");
invariant(sha256(await readFile(resolve(ROOT, "frozen-selectors/v11-selector.json"))) === manifest.frozenV11SelectorSha256, "V11 selector changed");
invariant(sha256(await readFile(resolve(ROOT, "frozen-selectors/v9-selector.json"))) === manifest.frozenV9SelectorSha256, "V9 comparator changed");
invariant(manifest.cloudImageDigest === "sha256:3276faca285998efdf40db81ec8b881d4102ab5b3eb528e4dba529e53c9e893e", "SVI image digest changed");
invariant(manifest.cloudTruthImageDigest === "sha256:ab78317233dd42d0391acd6b1611f451ef2616a1ed5b33fa08cf865bdcdec8a7", "truth image digest changed");
invariant(manifest.discardedPreCohortTruthImageDigest === "sha256:91ba93811793272189513139f98c52da0f87962e0d15bac3c9976743843b4dc6", "discarded pre-cohort image record changed");
for (const [path, expected] of Object.entries(manifest.sourceSha256)) {
  invariant(sha256(await readFile(resolve(path))) === expected, `${path} changed after freeze`);
}
console.log(JSON.stringify({ verified: true, businesses: 100, candidateFits: 4_800,
  evidenceGroupCounts: manifest.evidenceGroupCounts, moduleOrderCounts: manifest.moduleOrderCounts,
  truthFirewallAtFreeze: "closed" }, null, 2));
