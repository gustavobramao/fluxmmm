import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

const MANIFEST_PATH =
  "research/svi_score_v3/artifacts/svi-score-v3-training-freeze.json";

interface FrozenFile {
  path: string;
  sha256: string;
}

interface TrainingFreezeManifest {
  schemaVersion: string;
  status: string;
  activation: string;
  cohort: {
    trainingBusinesses: number;
    validationBusinesses: number;
    auditBusinesses: number;
    candidateFits: number;
    trainingRows: number;
  };
  release: {
    artifactPath: string;
    artifactSha256: string;
    weightsSha256: string;
    selectedConfiguration: {
      id: string;
      regularization: number;
      epochs: number;
    };
    theoremViolations: number;
  };
  localSources: FrozenFile[];
  implementation: FrozenFile[];
}

interface FrozenTrainingArtifact {
  activation: string;
  source: {
    checkpointSha256: string;
    featureCheckpointSha256: string;
    terminalFits: number;
    labelledTrainingRows: number;
    trainingBusinesses: number;
  };
  crossValidation: {
    contract: { heldOutSplitsPermitted: boolean };
    selectedConfiguration: {
      id: string;
      regularization: number;
      epochs: number;
    };
    finalFit: {
      weights: Record<string, number>;
      theoremAudit: { violations: number };
    };
  };
}

async function digest(path: string): Promise<string> {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

function invariant(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`Training freeze verification failed: ${message}`);
}

export async function verifyTrainingFreeze(
  root = process.cwd(),
  options: { verifyLocalSources?: boolean } = {},
): Promise<{ checkedFiles: number; artifactSha256: string; weightsSha256: string }> {
  const manifest = JSON.parse(
    await readFile(resolve(root, MANIFEST_PATH), "utf8"),
  ) as TrainingFreezeManifest;
  invariant(
    manifest.schemaVersion === "flux-svi-score-v3-training-freeze-v1",
    "unsupported manifest schema",
  );
  invariant(
    manifest.status === "frozen-before-held-out-validation",
    "release is not in the pre-validation frozen state",
  );
  invariant(manifest.activation === "research-only", "release was activated prematurely");

  const releasePath = resolve(root, manifest.release.artifactPath);
  const artifactSha256 = await digest(releasePath);
  invariant(
    artifactSha256 === manifest.release.artifactSha256,
    `artifact hash changed (${artifactSha256})`,
  );
  const artifact = JSON.parse(
    await readFile(releasePath, "utf8"),
  ) as FrozenTrainingArtifact;
  const weightsSha256 = createHash("sha256")
    .update(JSON.stringify(artifact.crossValidation.finalFit.weights))
    .digest("hex");
  invariant(weightsSha256 === manifest.release.weightsSha256, "weight hash changed");
  const weightValues = Object.values(artifact.crossValidation.finalFit.weights);
  invariant(weightValues.length > 0, "no learned weights were stored");
  invariant(
    weightValues.every((weight) => Number.isFinite(weight) && weight >= 0),
    "weights must be finite and nonnegative",
  );
  invariant(
    Math.abs(weightValues.reduce((total, weight) => total + weight, 0) - 1) < 1e-10,
    "weights no longer sum to one",
  );
  invariant(artifact.activation === "research-only", "artifact was activated prematurely");
  invariant(
    artifact.crossValidation.contract.heldOutSplitsPermitted === false,
    "held-out splits were permitted during fitting",
  );
  invariant(
    artifact.source.terminalFits === manifest.cohort.candidateFits &&
      artifact.source.labelledTrainingRows === manifest.cohort.trainingRows &&
      artifact.source.trainingBusinesses === manifest.cohort.trainingBusinesses,
    "cohort counts changed",
  );
  invariant(
    JSON.stringify(artifact.crossValidation.selectedConfiguration) ===
      JSON.stringify(manifest.release.selectedConfiguration),
    "selected training configuration changed",
  );
  invariant(
    artifact.crossValidation.finalFit.theoremAudit.violations ===
      manifest.release.theoremViolations,
    "theorem audit changed",
  );
  invariant(
    artifact.source.checkpointSha256 === manifest.localSources[0]?.sha256 &&
      artifact.source.featureCheckpointSha256 === manifest.localSources[1]?.sha256,
    "source hashes embedded in the release do not match the freeze manifest",
  );

  const checked = [
    { path: manifest.release.artifactPath, sha256: manifest.release.artifactSha256 },
    ...manifest.implementation,
    ...(options.verifyLocalSources === false ? [] : manifest.localSources),
  ];
  for (const file of checked.slice(1)) {
    const actual = await digest(resolve(root, file.path));
    invariant(actual === file.sha256, `${file.path} hash changed (${actual})`);
  }
  return { checkedFiles: checked.length, artifactSha256, weightsSha256 };
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : "";
if (invokedPath === fileURLToPath(import.meta.url)) {
  const receipt = await verifyTrainingFreeze(process.cwd(), {
    verifyLocalSources: !process.argv.includes("--release-only"),
  });
  process.stdout.write(`${JSON.stringify({ status: "verified", ...receipt }, null, 2)}\n`);
}
