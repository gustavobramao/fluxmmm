import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { SVI_SCORE_V9_AUDIT_CONTRACT } from "./audit-contract";
import { SVI_SCORE_V9_VERSION } from "./contract";

function invariant(condition: unknown, detail: string): asserts condition {
  if (!condition) throw new Error(`V9 sealed-audit verification failed: ${detail}.`);
}

const digest = (value: Buffer) => createHash("sha256").update(value).digest("hex");
const root = resolve(".flux-artifacts/svi-score-v9/audit");
const paths = {
  freeze: resolve(root, "sealed-audit-protocol-freeze.json"),
  open: resolve(root, "sealed-audit-open-receipt.json"),
  manifest: resolve(root, "sealed-audit-inference-manifest.json"),
  dataset: resolve(root, "sealed-audit-dataset.json"),
  v9: resolve(root, "frozen-v9-selector.json"),
  v8: resolve(root, "frozen-v8-selector.json"),
  result: resolve("research/svi_score_v9/artifacts/svi-score-v9-sealed-audit.json"),
};
const sources = Object.fromEntries(await Promise.all(
  Object.entries(paths).map(async ([key, path]) => [key, await readFile(path)]),
)) as Record<keyof typeof paths, Buffer>;
const freeze = JSON.parse(sources.freeze.toString("utf8")) as {
  version: string;
  status: string;
  contract: typeof SVI_SCORE_V9_AUDIT_CONTRACT;
  frozenV9SelectorSha256: string;
  frozenV8SelectorSha256: string;
};
const open = JSON.parse(sources.open.toString("utf8")) as {
  protocolFreezeSha256: string;
  oneTimeOpening: boolean;
  hiddenTruthDereferenced: boolean;
  nutsUsed: boolean;
};
const manifest = JSON.parse(sources.manifest.toString("utf8")) as {
  protocolFreezeSha256: string;
  hiddenTruthDereferenced: boolean;
  nutsUsed: boolean;
  items: unknown[];
};
const dataset = JSON.parse(sources.dataset.toString("utf8")) as {
  version: string;
  activation: string;
  featureNames: string[];
  summary: { businesses: number; rows: number; validRows: number };
  provenance: { protocolFreezeSha256: string; nutsUsed: boolean };
};
const result = JSON.parse(sources.result.toString("utf8")) as {
  version: string;
  activation: string;
  provenance: {
    auditDatasetSha256: string;
    frozenV9SelectorSha256: string;
    frozenV8SelectorSha256: string;
    protocolFreezeSha256: string;
    auditOpenedExactlyOnce: boolean;
    postAuditTuningPermitted: boolean;
    nutsUsed: boolean;
  };
  primaryInference: { resamples: number; seed: number };
  v9: { at100PercentCoverage: { businesses: number; theoremViolations: number } };
  checks: Record<string, boolean>;
  conclusion: {
    allConfirmatoryChecksPassed: boolean;
    externalRealWorldGeneralizationEstablished: boolean;
    sotaClaimPermitted: boolean;
    productionActivationPermitted: boolean;
    furtherTuningRequiresNewSealedCohort: boolean;
  };
};

invariant(freeze.version === SVI_SCORE_V9_VERSION, "freeze version changed");
invariant(freeze.status === "frozen-sealed", "protocol was not frozen");
invariant(
  JSON.stringify(freeze.contract) === JSON.stringify(SVI_SCORE_V9_AUDIT_CONTRACT),
  "audit contract changed",
);
invariant(open.oneTimeOpening, "audit was not opened exactly once");
invariant(!open.hiddenTruthDereferenced && !open.nutsUsed, "opening accessed truth or NUTS");
invariant(open.protocolFreezeSha256 === digest(sources.freeze), "opening freeze hash changed");
invariant(manifest.protocolFreezeSha256 === digest(sources.freeze), "manifest freeze hash changed");
invariant(!manifest.hiddenTruthDereferenced && !manifest.nutsUsed, "manifest accessed truth or NUTS");
invariant(
  manifest.items.length === SVI_SCORE_V9_AUDIT_CONTRACT.cohort.candidateRows,
  "inference manifest is incomplete",
);
invariant(dataset.version === SVI_SCORE_V9_VERSION, "dataset version changed");
invariant(dataset.activation === "sealed-audit-opened-unscored", "dataset stage changed");
invariant(dataset.summary.businesses === 140 && dataset.summary.rows === 6_720, "cohort changed");
invariant(dataset.summary.validRows > 0, "audit has no valid candidates");
invariant(dataset.featureNames.length === 232, "token contract changed");
invariant(!dataset.provenance.nutsUsed, "NUTS entered audit dataset");
invariant(result.version === SVI_SCORE_V9_VERSION, "result version changed");
invariant(result.activation === "sealed-audit-final", "audit is not final");
invariant(result.provenance.auditDatasetSha256 === digest(sources.dataset), "dataset hash changed");
invariant(result.provenance.frozenV9SelectorSha256 === digest(sources.v9), "V9 model changed");
invariant(result.provenance.frozenV8SelectorSha256 === digest(sources.v8), "V8 model changed");
invariant(result.provenance.protocolFreezeSha256 === digest(sources.freeze), "freeze hash changed");
invariant(result.provenance.auditOpenedExactlyOnce, "audit opening receipt changed");
invariant(!result.provenance.postAuditTuningPermitted, "post-audit tuning was enabled");
invariant(!result.provenance.nutsUsed, "NUTS entered audit scoring");
invariant(result.primaryInference.resamples === 10_000, "bootstrap count changed");
invariant(result.primaryInference.seed === 9_031_771, "bootstrap seed changed");
invariant(result.v9.at100PercentCoverage.businesses === 140, "not every business was scored");
invariant(result.v9.at100PercentCoverage.theoremViolations === 0, "theorem bound failed");
invariant(
  result.conclusion.allConfirmatoryChecksPassed === Object.values(result.checks).every(Boolean),
  "audit conclusion does not match checks",
);
invariant(!result.conclusion.externalRealWorldGeneralizationEstablished, "real-world claim escaped audit");
invariant(!result.conclusion.sotaClaimPermitted, "SOTA claim escaped audit");
invariant(!result.conclusion.productionActivationPermitted, "production activation escaped audit");
invariant(result.conclusion.furtherTuningRequiresNewSealedCohort, "new-cohort rule removed");

process.stdout.write(`${JSON.stringify({
  verified: true,
  version: result.version,
  businesses: dataset.summary.businesses,
  candidateFits: manifest.items.length,
  tokens: dataset.featureNames.length,
  allConfirmatoryChecksPassed: result.conclusion.allConfirmatoryChecksPassed,
  syntheticGeneralizationConfirmed: Object.values(result.checks).every(Boolean),
  furtherTuningRequiresNewSealedCohort: true,
  nutsUsed: false,
}, null, 2)}\n`);
