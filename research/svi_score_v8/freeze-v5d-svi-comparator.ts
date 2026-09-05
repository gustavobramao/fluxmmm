import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fitV5DCrossFittedModel, v5dCrossFitHash } from "../svi_score_v5d/crossfit";
import { v5cCandidateRegions } from "../svi_score_v5c/crossfit";
import { loadSviScoreV4DevelopmentRows } from "../svi_score_v4/development-rows";
import {
  loadV5DSviPosteriorRecords,
  v5dSviFeatureProvider,
  v5dSviSelectionPolicy,
} from "../svi_score_v5d_svi/posterior";
import { SVI_SCORE_V8_VERSION } from "./contract";

const ROOT = resolve(".flux-artifacts/svi-score-v8");
const OUTPUT = resolve(ROOT, "frozen-v5d-svi-comparator.json");
const DEVELOPMENT = resolve(ROOT, "development-dataset.json");

const developmentSource = await readFile(DEVELOPMENT);
const development = JSON.parse(developmentSource.toString("utf8")) as {
  activation: string;
  provenance: { auditAccessed: boolean; nutsUsed: boolean };
};
if (
  development.activation !== "development-only" ||
  development.provenance.auditAccessed ||
  development.provenance.nutsUsed
) {
  throw new Error("V8 comparator freeze requires the sealed SVI development dataset.");
}
const rows = await loadSviScoreV4DevelopmentRows();
const posterior = await loadV5DSviPosteriorRecords(rows);
const provider = v5dSviFeatureProvider(posterior);
const policy = v5dSviSelectionPolicy(posterior);
const regions = v5cCandidateRegions(rows.map((row) => row.candidateId));
const model = fitV5DCrossFittedModel(rows, regions, provider, policy);
const payload = {
  artifactId: "flux-svi-score-v8-frozen-v5d-svi-comparator-v1",
  version: SVI_SCORE_V8_VERSION,
  generatedAt: new Date().toISOString(),
  activation: "development-frozen",
  provenance: {
    developmentDatasetSha256: createHash("sha256").update(developmentSource).digest("hex"),
    auditAccessed: false,
    nutsUsed: false,
  },
  modelSha256: v5dCrossFitHash(model),
  model,
};
await mkdir(ROOT, { recursive: true });
const temporary = `${OUTPUT}.tmp`;
await writeFile(temporary, JSON.stringify(payload));
await rename(temporary, OUTPUT);
process.stdout.write(`${JSON.stringify({
  path: OUTPUT,
  modelSha256: payload.modelSha256,
  trainingBusinesses: new Set(rows.map((row) => row.businessId)).size,
  auditAccessed: false,
}, null, 2)}\n`);
