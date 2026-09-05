import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { loadSviScoreV4DevelopmentRows } from "../svi_score_v4/development-rows";
import type { SviScoreV5ACrossValidationReceipt } from "../svi_score_v5a/types";
import {
  fitV5DCrossFittedModel,
  predictV5DCrossFitted,
  v5dCrossFitHash,
} from "../svi_score_v5d/crossfit";
import { v5cCandidateRegions } from "../svi_score_v5c/crossfit";
import { v5bCandidateUniverse } from "../svi_score_v5b/embedding";
import {
  loadV5DSviPosteriorRecords,
  v5dSviFeatureProvider,
  v5dSviSelectionPolicy,
} from "../svi_score_v5d_svi/posterior";
import { SVI_SCORE_V8_VERSION } from "./contract";

interface V5AArtifact {
  provenance: { v3SealedAuditAccessed: boolean };
  crossValidation: SviScoreV5ACrossValidationReceipt;
}

const v5aPath = resolve(
  "research/svi_score_v5a/artifacts/svi-score-v5a-development.json",
);
const v5aSource = await readFile(v5aPath);
const v5a = JSON.parse(v5aSource.toString("utf8")) as V5AArtifact;
if (v5a.provenance.v3SealedAuditAccessed) {
  throw new Error("V8 baseline cannot run after sealed-audit access.");
}
const rows = await loadSviScoreV4DevelopmentRows();
const records = await loadV5DSviPosteriorRecords(rows);
const provider = v5dSviFeatureProvider(records);
const selectionPolicy = v5dSviSelectionPolicy(records);
const foldByBusiness = new Map(
  v5a.crossValidation.assignments.map((row) => [row.businessId, row.fold]),
);
const universe = v5bCandidateUniverse(rows.map((row) => row.candidateId));
const regionByCandidate = v5cCandidateRegions(
  universe.map((row) => row.candidateId),
);
const predictions: Array<{
  businessId: string;
  candidateId: string;
  fold: number;
  mean: number;
  p90: number;
  risk: number;
}> = [];
const modelHashes: Record<string, string> = {};
for (const fold of [...new Set(foldByBusiness.values())].sort()) {
  const training = rows.filter((row) => foldByBusiness.get(row.businessId) !== fold);
  const model = fitV5DCrossFittedModel(
    training,
    regionByCandidate,
    provider,
    selectionPolicy,
  );
  modelHashes[String(fold)] = v5dCrossFitHash(model);
  rows
    .filter((row) => foldByBusiness.get(row.businessId) === fold)
    .forEach((row) => {
      const prediction = predictV5DCrossFitted(row, model, provider);
      predictions.push({
        businessId: row.businessId,
        candidateId: row.candidateId,
        fold,
        mean: prediction.mean,
        p90: prediction.p90,
        risk: prediction.risk,
      });
    });
}
if (
  predictions.length !== 20_160 ||
  new Set(predictions.map((row) => `${row.businessId}\u0000${row.candidateId}`)).size !==
    predictions.length
) {
  throw new Error("V8 V5D-SVI baseline predictions are incomplete or duplicated.");
}
const artifact = {
  artifactId: "flux-svi-score-v8-v5d-svi-baseline-predictions-v1",
  version: SVI_SCORE_V8_VERSION,
  generatedAt: new Date().toISOString(),
  activation: "development-only",
  provenance: {
    v5aSha256: createHash("sha256").update(v5aSource).digest("hex"),
    modelHashes,
    auditAccessed: false,
    nutsUsed: false,
  },
  predictions,
};
const output = resolve(
  ".flux-artifacts/svi-score-v8/v5d-svi-baseline-predictions.json",
);
await mkdir(resolve(".flux-artifacts/svi-score-v8"), { recursive: true });
const temporary = `${output}.tmp`;
await writeFile(temporary, JSON.stringify(artifact));
await rename(temporary, output);
process.stdout.write(`${JSON.stringify({
  path: output,
  predictions: predictions.length,
  folds: Object.keys(modelHashes).length,
  auditAccessed: false,
}, null, 2)}\n`);
