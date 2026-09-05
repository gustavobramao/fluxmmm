import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { scoreV6, selectScoreV6Candidates } from "../score_v6/learn";
import type { ScoreV6FeatureWeights } from "../score_v6/types";
import { SVI_SCORE_V4_CONTRACT, SVI_SCORE_V4_VERSION } from "./contract";
import { crossValidateScoreV4 } from "./cross-validation";
import {
  loadSviScoreV4DevelopmentRows,
  V4_FLIGHT_HOLDOUT_SIDECAR_PATH,
  V4_TEMPORAL_SIDECAR_PATH,
} from "./development-rows";
import { metricsFromV4Selections } from "./ranker";
import type { SviScoreV4Selection } from "./types";

const artifactDirectory = resolve("research/svi_score_v4/artifacts");
const artifactPath = resolve(
  artifactDirectory,
  "svi-score-v4.2-development.json",
);
const rows = await loadSviScoreV4DevelopmentRows();
const crossValidation = crossValidateScoreV4(rows);
const [temporalSource, flightSource, v3Source, activeV6Source] = await Promise.all([
  readFile(V4_TEMPORAL_SIDECAR_PATH),
  readFile(V4_FLIGHT_HOLDOUT_SIDECAR_PATH),
  readFile(resolve("research/svi_score_v3/artifacts/svi-score-v3-grouped-training.json")),
  readFile(resolve("research/score_v6/artifacts/learned-score-v6-pilot.json")),
]);
const v3Weights = (JSON.parse(v3Source.toString("utf8")) as {
  crossValidation: { finalFit: { weights: ScoreV6FeatureWeights } };
}).crossValidation.finalFit.weights;
const activeV6Weights = (JSON.parse(activeV6Source.toString("utf8")) as {
  model: { weights: ScoreV6FeatureWeights };
}).model.weights;
function baselineMetrics(weights: ScoreV6FeatureWeights) {
  const selections = selectScoreV6Candidates(
    rows,
    (row) => scoreV6(row, weights),
  ).map((selection): SviScoreV4Selection => ({
    businessId: selection.businessId,
    candidateId: selection.candidateId,
    family: selection.family,
    loss: selection.loss,
    excessLoss: selection.excess,
    normalizedExcessRegret: Math.min(1, selection.excess),
    oracle: selection.lowest,
    decisionGrade: selection.decisionGrade,
    safetyAccepted: false,
    reviewOnlyFallback: false,
  }));
  return metricsFromV4Selections(selections);
}
const artifact = {
  artifactId: "flux-svi-score-v4-development-v2",
  version: SVI_SCORE_V4_VERSION,
  generatedAt: new Date().toISOString(),
  activation: "research-only",
  stage: "development-complete-validation-not-generated",
  contract: SVI_SCORE_V4_CONTRACT,
  provenance: {
    v3TrainingAndRetiredValidationUsed: true,
    v3SealedAuditAccessed: false,
    temporalFeatureSha256: createHash("sha256")
      .update(temporalSource)
      .digest("hex"),
    flightHoldoutFeatureSha256: createHash("sha256")
      .update(flightSource)
      .digest("hex"),
    v3TrainingArtifactSha256: createHash("sha256").update(v3Source).digest("hex"),
    activeV6ArtifactSha256: createHash("sha256")
      .update(activeV6Source)
      .digest("hex"),
  },
  crossValidation,
  descriptiveComparators: {
    failedV3: baselineMetrics(v3Weights),
    activeV6: baselineMetrics(activeV6Weights),
  },
  nextGate:
    "Freeze this method, then generate a fresh independent validation cohort. Do not open the V3 audit.",
};
await mkdir(artifactDirectory, { recursive: true });
const temporary = `${artifactPath}.tmp`;
await writeFile(temporary, JSON.stringify(artifact, null, 2));
await rename(temporary, artifactPath);
process.stdout.write(`${JSON.stringify({
  artifactPath,
  stage: artifact.stage,
  selectedConfiguration: crossValidation.selectedConfiguration,
  outOfFold: crossValidation.configurations.find(
    (receipt) =>
      receipt.configuration.id === crossValidation.selectedConfiguration.id,
  )?.outOfFoldMetrics,
  theorem: crossValidation.finalFit.theorem,
  v3AuditAccessed: false,
}, null, 2)}\n`);
