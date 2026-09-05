import { createHash } from "node:crypto";
import { access, mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { crossValidateScoreV3 } from "./cross-validation";
import { SVI_SCORE_V3_CONTRACT, SVI_SCORE_V3_VERSION } from "./contract";
import { sviTrainingRows } from "./training-rows";
import type { SviScoreV3TrainingFeatureRow } from "./training-rows";
import type { SviScoreV3Record } from "./types";

const checkpointPath = resolve(
  ".flux-artifacts/svi-score-v3/design-pilot-records.json",
);
const artifactDirectory = resolve("research/svi_score_v3/artifacts");
const artifactPath = resolve(
  artifactDirectory,
  "svi-score-v3-grouped-training.json",
);
const freezePath = resolve(
  artifactDirectory,
  "svi-score-v3-training-freeze.json",
);

try {
  await access(freezePath);
  if (!process.argv.includes("--replace-frozen-training-release")) {
    throw new Error(
      "The grouped training release is frozen. Refusing to overwrite it. " +
        "Create a new version, or pass --replace-frozen-training-release and regenerate the freeze receipt explicitly.",
    );
  }
} catch (error) {
  if (
    !(error instanceof Error) ||
    !("code" in error) ||
    (error as NodeJS.ErrnoException).code !== "ENOENT"
  ) {
    throw error;
  }
}

const checkpointSource = await readFile(checkpointPath);
const featureCheckpointPath = resolve(
  ".flux-artifacts/svi-score-v3/train-features-v2.json",
);
const featureCheckpointSource = await readFile(featureCheckpointPath);
const refreshedFeatures = JSON.parse(
  featureCheckpointSource.toString("utf8"),
) as SviScoreV3TrainingFeatureRow[];
const records = JSON.parse(checkpointSource.toString("utf8")) as SviScoreV3Record[];
if (records.length !== SVI_SCORE_V3_CONTRACT.targetSviFits) {
  throw new Error(
    `The frozen SVI cohort is incomplete: ${records.length}/${SVI_SCORE_V3_CONTRACT.targetSviFits}.`,
  );
}
const trainingRows = sviTrainingRows(records, refreshedFeatures);
const trainingBusinesses = new Set(trainingRows.map((row) => row.businessId)).size;
if (trainingBusinesses !== SVI_SCORE_V3_CONTRACT.splits.train) {
  throw new Error(
    `Expected ${SVI_SCORE_V3_CONTRACT.splits.train} labelled training businesses; found ${trainingBusinesses}.`,
  );
}

const crossValidation = crossValidateScoreV3(trainingRows);
const artifact = {
  artifactId: "flux-svi-score-v3-grouped-training",
  version: SVI_SCORE_V3_VERSION,
  generatedAt: new Date().toISOString(),
  activation: "research-only" as const,
  source: {
    checkpoint: "design-pilot-records.json",
    checkpointSha256: createHash("sha256").update(checkpointSource).digest("hex"),
    featureCheckpoint: "train-features-v2.json",
    featureCheckpointSha256: createHash("sha256")
      .update(featureCheckpointSource)
      .digest("hex"),
    terminalFits: records.length,
    labelledTrainingRows: trainingRows.length,
    trainingBusinesses,
    refreshedDiagnosticRows: refreshedFeatures.length,
  },
  crossValidation,
  safeguards: [
    "Only rows declared as training are converted into score-learning labels.",
    "All candidates from a synthetic business remain in one grouped fold.",
    "Generator families are balanced across five deterministic folds.",
    "Tuning uses out-of-fold economic decision loss; validation and audit outcomes are not evaluated.",
    "After configuration selection, score weights are refit once on all 300 training businesses.",
  ],
};

await mkdir(artifactDirectory, { recursive: true });
const temporaryPath = `${artifactPath}.tmp`;
await writeFile(temporaryPath, JSON.stringify(artifact, null, 2));
await rename(temporaryPath, artifactPath);
process.stdout.write(`${JSON.stringify({
  artifactPath,
  source: artifact.source,
  contract: crossValidation.contract,
  selectedConfiguration: crossValidation.selectedConfiguration,
  foldBusinesses: crossValidation.data.foldBusinesses,
}, null, 2)}\n`);
