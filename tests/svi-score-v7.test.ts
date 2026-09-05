import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { resolve } from "node:path";
import { SVI_SCORE_V7_CONTRACT } from "../research/svi_score_v7/contract";

test("V7 freezes a multi-task architecture without a SOTA claim", () => {
  assert.deepEqual(SVI_SCORE_V7_CONTRACT.model.hiddenLayers, [48, 24]);
  assert.equal(SVI_SCORE_V7_CONTRACT.model.outputs.length, 4);
  assert.equal(SVI_SCORE_V7_CONTRACT.crossFitting.models, 20);
  assert.equal(SVI_SCORE_V7_CONTRACT.evaluation.sotaClaimPermitted, false);
  assert.equal(SVI_SCORE_V7_CONTRACT.governance.auditMayOpen, false);
  assert.equal(SVI_SCORE_V7_CONTRACT.governance.freshValidationMayOpen, false);
});

test("V7 predictions are complete, unique, and cross-fitted", async () => {
  const [datasetSource, predictionSource] = await Promise.all([
    readFile(resolve(".flux-artifacts/svi-score-v7/dataset.json")),
    readFile(resolve(".flux-artifacts/svi-score-v7/predictions.json")),
  ]);
  const dataset = JSON.parse(datasetSource.toString("utf8")) as {
    rows: Array<{ businessId: string; candidateId: string; fold: number; region: number }>;
  };
  const predictions = JSON.parse(predictionSource.toString("utf8")) as {
    provenance: { freshValidationAccessed: boolean; auditAccessed: boolean };
    modelReceipts: Array<{ fold: number; region: number }>;
    predictions: Array<{
      businessId: string;
      candidateId: string;
      fold: number;
      region: number;
      mean: number;
      p90: number;
    }>;
  };
  assert.equal(predictions.provenance.freshValidationAccessed, false);
  assert.equal(predictions.provenance.auditAccessed, false);
  assert.equal(predictions.modelReceipts.length, 20);
  assert.equal(predictions.predictions.length, 20_160);
  const byKey = new Map(predictions.predictions.map((row) => [
    `${row.businessId}\u0000${row.candidateId}`,
    row,
  ]));
  assert.equal(byKey.size, 20_160);
  dataset.rows.forEach((row) => {
    const prediction = byKey.get(`${row.businessId}\u0000${row.candidateId}`);
    assert.ok(prediction);
    assert.equal(prediction.fold, row.fold);
    assert.equal(prediction.region, row.region);
    assert.ok(Number.isFinite(prediction.mean));
    assert.ok(prediction.p90 >= prediction.mean);
  });
});

test("V7 development result keeps the gate closed and exhaustive selection agrees", async () => {
  const artifact = JSON.parse(
    await readFile(
      resolve("research/svi_score_v7/artifacts/svi-score-v7-development.json"),
      "utf8",
    ),
  ) as {
    provenance: { freshValidationAccessed: boolean; auditAccessed: boolean };
    checks: Record<string, boolean>;
    conclusion: {
      allDevelopmentChecksPassed: boolean;
      publicationReady: boolean;
      sotaClaimPermitted: boolean;
    };
  };
  assert.equal(artifact.provenance.freshValidationAccessed, false);
  assert.equal(artifact.provenance.auditAccessed, false);
  assert.equal(artifact.checks.fullPoolAlgorithmAgreement, true);
  assert.equal(artifact.conclusion.allDevelopmentChecksPassed, false);
  assert.equal(artifact.conclusion.publicationReady, false);
  assert.equal(artifact.conclusion.sotaClaimPermitted, false);
});
