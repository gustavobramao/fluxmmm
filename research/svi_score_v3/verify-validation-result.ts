import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { verifyValidationProtocol } from "./verify-validation-protocol";

const artifactDirectory = "research/svi_score_v3/artifacts";
const freezePath = `${artifactDirectory}/svi-score-v3-validation-result-freeze.json`;
const lockPath = ".flux-artifacts/svi-score-v3/validation-evaluation.lock.json";

async function digest(path: string): Promise<string> {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

function invariant(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`Validation result verification failed: ${message}`);
}

export async function verifyValidationResult(root = process.cwd()) {
  await verifyValidationProtocol(root);
  const freeze = JSON.parse(await readFile(resolve(root, freezePath), "utf8")) as {
    schemaVersion: string;
    protocolId: string;
    resultPath: string;
    resultSha256: string;
    status: string;
    auditAccessed: boolean;
    sourceHashes: Record<string, string>;
  };
  invariant(
    freeze.schemaVersion === "flux-svi-score-v3-validation-result-freeze-v1",
    "unsupported freeze schema",
  );
  invariant(!freeze.auditAccessed, "freeze indicates audit access");
  const resultSha256 = await digest(resolve(root, freeze.resultPath));
  invariant(resultSha256 === freeze.resultSha256, "result hash changed");
  const result = JSON.parse(
    await readFile(resolve(root, freeze.resultPath), "utf8"),
  ) as {
    protocolId: string;
    status: string;
    auditAccessed: boolean;
    provenance: Record<string, string>;
    protocolReceipt: { proceedToSealedAudit: boolean };
    governance: {
      validationExecutedOnce: boolean;
      weightsChangedAfterValidation: boolean;
      protocolChangedAfterValidation: boolean;
      auditMayOpen: boolean;
    };
  };
  invariant(result.protocolId === freeze.protocolId, "protocol ID changed");
  invariant(result.status === freeze.status, "result status changed");
  invariant(!result.auditAccessed, "result indicates audit access");
  invariant(result.governance.validationExecutedOnce, "one-time execution was not recorded");
  invariant(!result.governance.weightsChangedAfterValidation, "weights changed after validation");
  invariant(!result.governance.protocolChangedAfterValidation, "protocol changed after validation");
  invariant(
    result.governance.auditMayOpen === result.protocolReceipt.proceedToSealedAudit,
    "audit decision disagrees with the frozen gates",
  );
  invariant(
    JSON.stringify(result.provenance) === JSON.stringify(freeze.sourceHashes),
    "embedded source hashes changed",
  );
  const sourceFiles: Record<string, string> = {
    checkpointSha256: ".flux-artifacts/svi-score-v3/design-pilot-records.json",
    featureSha256: ".flux-artifacts/svi-score-v3/validation-features-v2.json",
    challengerSha256:
      `${artifactDirectory}/svi-score-v3-grouped-training.json`,
    comparatorSha256:
      "research/score_v6/artifacts/learned-score-v6-pilot.json",
    protocolFreezeSha256:
      `${artifactDirectory}/svi-score-v3-validation-protocol-freeze.json`,
  };
  for (const [hashName, path] of Object.entries(sourceFiles)) {
    invariant(
      await digest(resolve(root, path)) === freeze.sourceHashes[hashName],
      `${hashName} source changed`,
    );
  }
  const lock = JSON.parse(await readFile(resolve(root, lockPath), "utf8")) as {
    status: string;
    resultSha256: string;
    auditAccessed: boolean;
  };
  invariant(lock.status === "evaluation-complete", "evaluation lock is incomplete");
  invariant(lock.resultSha256 === resultSha256, "evaluation lock hash changed");
  invariant(!lock.auditAccessed, "evaluation lock indicates audit access");
  return {
    status: result.status,
    resultSha256,
    auditMayOpen: result.governance.auditMayOpen,
    auditAccessed: false,
    checkedSources: Object.keys(sourceFiles).length,
  };
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : "";
if (invokedPath === fileURLToPath(import.meta.url)) {
  const receipt = await verifyValidationResult();
  process.stdout.write(`${JSON.stringify({ verified: true, ...receipt }, null, 2)}\n`);
}
