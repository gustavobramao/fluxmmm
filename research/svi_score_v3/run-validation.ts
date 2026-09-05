import { createHash } from "node:crypto";
import {
  access,
  mkdir,
  readFile,
  rename,
  writeFile,
} from "node:fs/promises";
import { resolve } from "node:path";
import {
  evaluateRegretBound,
  scoreV6,
  selectScoreV6Candidates,
} from "../score_v6/learn";
import type { ScoreV6CandidateRow, ScoreV6FeatureWeights } from "../score_v6/types";
import { evaluateSviScoreV3Validation } from "./validation-protocol";
import { sviValidationRows } from "./validation-rows";
import { verifyValidationProtocol } from "./verify-validation-protocol";
import type { SviScoreV3Record } from "./types";
import type { SviScoreV3TrainingFeatureRow } from "./training-rows";

const checkpointPath = resolve(".flux-artifacts/svi-score-v3/design-pilot-records.json");
const featurePath = resolve(".flux-artifacts/svi-score-v3/validation-features-v2.json");
const artifactDirectory = resolve("research/svi_score_v3/artifacts");
const protocolFreezePath = resolve(
  artifactDirectory,
  "svi-score-v3-validation-protocol-freeze.json",
);
const challengerPath = resolve(
  artifactDirectory,
  "svi-score-v3-grouped-training.json",
);
const comparatorPath = resolve(
  "research/score_v6/artifacts/learned-score-v6-pilot.json",
);
const resultPath = resolve(artifactDirectory, "svi-score-v3-validation-result.json");
const resultFreezePath = resolve(
  artifactDirectory,
  "svi-score-v3-validation-result-freeze.json",
);
const evaluationLockPath = resolve(
  ".flux-artifacts/svi-score-v3/validation-evaluation.lock.json",
);

async function sha256(path: string): Promise<string> {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

async function assertAbsent(path: string, description: string): Promise<void> {
  try {
    await access(path);
    throw new Error(`${description} already exists; validation cannot be executed twice.`);
  } catch (error) {
    if (
      error instanceof Error &&
      "code" in error &&
      (error as NodeJS.ErrnoException).code === "ENOENT"
    ) return;
    throw error;
  }
}

function average(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) /
    Math.max(values.length, 1);
}

function selectedSecondary(
  rows: ScoreV6CandidateRow[],
  weights: ScoreV6FeatureWeights,
) {
  const byCandidate = new Map(
    rows.map((row) => [`${row.businessId}\u0000${row.candidateId}`, row]),
  );
  const selections = selectScoreV6Candidates(rows, (row) => scoreV6(row, weights));
  const selectedRows = selections.map((selection) =>
    byCandidate.get(`${selection.businessId}\u0000${selection.candidateId}`)!
  );
  return {
    meanCappedRegret: average(selectedRows.map((row) => row.cappedRegret)),
    meanRoiError: average(selectedRows.map((row) => row.roiError)),
    meanContributionError: average(
      selectedRows.map((row) => row.contributionError),
    ),
    theoremAudit: evaluateRegretBound(rows, (row) => scoreV6(row, weights)),
    selections,
  };
}

await mkdir(artifactDirectory, { recursive: true });
await assertAbsent(evaluationLockPath, "Validation evaluation lock");
await assertAbsent(resultPath, "Validation result");
await assertAbsent(resultFreezePath, "Validation result freeze");
await verifyValidationProtocol();

const openedAt = new Date().toISOString();
await writeFile(
  evaluationLockPath,
  JSON.stringify({
    protocolId: "flux-svi-score-v3-validation-v2-powered-screen",
    status: "evaluation-started",
    openedAt,
    auditAccessed: false,
  }, null, 2),
  { flag: "wx" },
);

try {
  const [checkpointSource, featureSource, challengerSource, comparatorSource] =
    await Promise.all([
      readFile(checkpointPath),
      readFile(featurePath),
      readFile(challengerPath),
      readFile(comparatorPath),
    ]);
  const records = JSON.parse(checkpointSource.toString("utf8")) as SviScoreV3Record[];
  const features = JSON.parse(
    featureSource.toString("utf8"),
  ) as SviScoreV3TrainingFeatureRow[];
  const rows = sviValidationRows(records, features);
  const challengerArtifact = JSON.parse(challengerSource.toString("utf8")) as {
    crossValidation: { finalFit: { weights: ScoreV6FeatureWeights } };
  };
  const comparatorArtifact = JSON.parse(comparatorSource.toString("utf8")) as {
    model: { weights: ScoreV6FeatureWeights };
  };
  const challengerWeights = challengerArtifact.crossValidation.finalFit.weights;
  const comparatorWeights = comparatorArtifact.model.weights;
  const protocolReceipt = evaluateSviScoreV3Validation(
    rows,
    challengerWeights,
    comparatorWeights,
  );
  const challengerSecondary = selectedSecondary(rows, challengerWeights);
  const comparatorSecondary = selectedSecondary(rows, comparatorWeights);
  const comparatorByBusiness = new Map(
    comparatorSecondary.selections.map((selection) => [selection.businessId, selection]),
  );
  const businessReceipts = challengerSecondary.selections.map((challenger) => {
    const comparator = comparatorByBusiness.get(challenger.businessId)!;
    return {
      businessId: challenger.businessId,
      family: challenger.family,
      challengerCandidateId: challenger.candidateId,
      comparatorCandidateId: comparator.candidateId,
      challengerExcessLoss: challenger.excess,
      comparatorExcessLoss: comparator.excess,
      pairedDifference: challenger.excess - comparator.excess,
    };
  });
  const completedAt = new Date().toISOString();
  const result = {
    artifactId: "flux-svi-score-v3-validation-result-v1",
    protocolId: protocolReceipt.protocolId,
    status: protocolReceipt.proceedToSealedAudit ? "passed" : "did-not-pass",
    openedAt,
    completedAt,
    auditAccessed: false,
    provenance: {
      checkpointSha256: createHash("sha256").update(checkpointSource).digest("hex"),
      featureSha256: createHash("sha256").update(featureSource).digest("hex"),
      challengerSha256: createHash("sha256").update(challengerSource).digest("hex"),
      comparatorSha256: createHash("sha256").update(comparatorSource).digest("hex"),
      protocolFreezeSha256: await sha256(protocolFreezePath),
    },
    protocolReceipt,
    secondary: {
      challenger: {
        meanCappedRegret: challengerSecondary.meanCappedRegret,
        meanRoiError: challengerSecondary.meanRoiError,
        meanContributionError: challengerSecondary.meanContributionError,
        theoremAudit: challengerSecondary.theoremAudit,
      },
      comparator: {
        meanCappedRegret: comparatorSecondary.meanCappedRegret,
        meanRoiError: comparatorSecondary.meanRoiError,
        meanContributionError: comparatorSecondary.meanContributionError,
        theoremAudit: comparatorSecondary.theoremAudit,
      },
    },
    businessReceipts,
    governance: {
      validationExecutedOnce: true,
      weightsChangedAfterValidation: false,
      protocolChangedAfterValidation: false,
      auditMayOpen: protocolReceipt.proceedToSealedAudit,
    },
  };
  const temporaryResultPath = `${resultPath}.tmp`;
  await writeFile(temporaryResultPath, JSON.stringify(result, null, 2));
  await rename(temporaryResultPath, resultPath);
  const resultSha256 = await sha256(resultPath);
  const freeze = {
    schemaVersion: "flux-svi-score-v3-validation-result-freeze-v1",
    frozenAt: completedAt,
    protocolId: protocolReceipt.protocolId,
    resultPath: "research/svi_score_v3/artifacts/svi-score-v3-validation-result.json",
    resultSha256,
    status: result.status,
    auditAccessed: false,
    sourceHashes: result.provenance,
  };
  const temporaryFreezePath = `${resultFreezePath}.tmp`;
  await writeFile(temporaryFreezePath, JSON.stringify(freeze, null, 2));
  await rename(temporaryFreezePath, resultFreezePath);
  await writeFile(evaluationLockPath, JSON.stringify({
    protocolId: protocolReceipt.protocolId,
    status: "evaluation-complete",
    openedAt,
    completedAt,
    resultSha256,
    auditAccessed: false,
  }, null, 2));
  process.stdout.write(`${JSON.stringify({
    resultPath,
    resultFreezePath,
    status: result.status,
    auditMayOpen: result.governance.auditMayOpen,
    primary: protocolReceipt.pairedPrimary,
    gates: protocolReceipt.gates,
  }, null, 2)}\n`);
} catch (error) {
  await writeFile(evaluationLockPath, JSON.stringify({
    protocolId: "flux-svi-score-v3-validation-v2-powered-screen",
    status: "evaluation-failed-after-opening",
    openedAt,
    failedAt: new Date().toISOString(),
    auditAccessed: false,
    error: error instanceof Error ? error.message : String(error),
  }, null, 2));
  throw error;
}
