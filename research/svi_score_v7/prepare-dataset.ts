import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { loadSviScoreV4DevelopmentRows } from "../svi_score_v4/development-rows";
import { passesV4Safety, v4SelectionPool } from "../svi_score_v4/ranker";
import { V5A_COMMON_POOL_CONFIGURATION } from "../svi_score_v5a/contract";
import type { SviScoreV5ACrossValidationReceipt } from "../svi_score_v5a/types";
import { v5bCandidateUniverse } from "../svi_score_v5b/embedding";
import { v5cCandidateRegions } from "../svi_score_v5c/crossfit";
import { v5dFeatureNames, v5dFeatureVector } from "../svi_score_v5d/features";
import { SVI_SCORE_V7_CONTRACT, SVI_SCORE_V7_VERSION } from "./contract";

interface V5AArtifact {
  crossValidation: SviScoreV5ACrossValidationReceipt;
  provenance: { v3SealedAuditAccessed: boolean };
}

const v5aPath = resolve("research/svi_score_v5a/artifacts/svi-score-v5a-development.json");
const v5aSource = await readFile(v5aPath);
const v5a = JSON.parse(v5aSource.toString("utf8")) as V5AArtifact;
if (v5a.provenance.v3SealedAuditAccessed) {
  throw new Error("V7 cannot prepare data after sealed-audit access.");
}
const rows = await loadSviScoreV4DevelopmentRows();
if (rows.some((row) => row.split === "audit")) {
  throw new Error("V7 dataset contains a sealed-audit row.");
}
const featureNames = v5dFeatureNames(rows[0]);
rows.forEach((row) => {
  const names = v5dFeatureNames(row);
  if (
    names.length !== featureNames.length ||
    names.some((name, index) => name !== featureNames[index])
  ) {
    throw new Error(`V7 feature contract changed for ${row.candidateId}.`);
  }
});
const foldByBusiness = new Map(
  v5a.crossValidation.assignments.map((row) => [row.businessId, row.fold]),
);
const universe = v5bCandidateUniverse(rows.map((row) => row.candidateId));
const regionByCandidate = v5cCandidateRegions(
  universe.map((row) => row.candidateId),
);
const grouped = new Map<string, typeof rows>();
rows.forEach((row) => {
  grouped.set(row.businessId, [...(grouped.get(row.businessId) ?? []), row]);
});
const oracleByBusiness = new Map<string, number>();
const poolByBusiness = new Map<string, Set<string>>();
[...grouped.entries()].forEach(([businessId, business]) => {
  const pool = v4SelectionPool(business, V5A_COMMON_POOL_CONFIGURATION).rows;
  oracleByBusiness.set(
    businessId,
    Math.min(...pool.map((row) => row.decisionLoss)),
  );
  poolByBusiness.set(businessId, new Set(pool.map((row) => row.candidateId)));
});
const dataset = {
  version: SVI_SCORE_V7_VERSION,
  contract: SVI_SCORE_V7_CONTRACT,
  provenance: {
    v5aSha256: createHash("sha256").update(v5aSource).digest("hex"),
    freshValidationAccessed: false,
    auditAccessed: false,
  },
  featureNames,
  rows: rows.map((row) => ({
    businessId: row.businessId,
    family: row.family,
    split: row.split,
    candidateId: row.candidateId,
    fold: foldByBusiness.get(row.businessId),
    region: regionByCandidate[row.candidateId],
    features: v5dFeatureVector(row),
    targetExcessLoss: Math.max(
      0,
      row.decisionLoss - (oracleByBusiness.get(row.businessId) ?? 0),
    ),
    roiError: row.roiError,
    contributionError: row.contributionError,
    inSelectionPool: poolByBusiness.get(row.businessId)?.has(row.candidateId) ?? false,
    safetyAccepted: passesV4Safety(row, V5A_COMMON_POOL_CONFIGURATION),
  })),
};
const directory = resolve(".flux-artifacts/svi-score-v7");
const path = resolve(directory, "dataset.json");
await mkdir(directory, { recursive: true });
const temporary = `${path}.tmp`;
await writeFile(temporary, JSON.stringify(dataset));
await rename(temporary, path);
process.stdout.write(`${JSON.stringify({
  path,
  businesses: grouped.size,
  rows: dataset.rows.length,
  features: featureNames.length,
  folds: [...new Set(dataset.rows.map((row) => row.fold))].sort(),
  regions: [...new Set(dataset.rows.map((row) => row.region))].sort(),
  freshValidationAccessed: false,
  auditAccessed: false,
}, null, 2)}\n`);
