import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { SVI_SCORE_V3_VALIDATION_PROTOCOL } from "./validation-protocol";
import { verifyTrainingFreeze } from "./verify-training-freeze";

const MANIFEST_PATH =
  "research/svi_score_v3/artifacts/svi-score-v3-validation-protocol-freeze.json";

interface ProtocolFreezeManifest {
  schemaVersion: string;
  protocolId: string;
  status: string;
  validationOutcomesAccessed: boolean;
  auditOutcomesAccessed: boolean;
  cohort: {
    validationBusinesses: number;
    validationRows: number;
    sealedAuditBusinesses: number;
  };
  files: Array<{ role: string; path: string; sha256: string }>;
}

async function digest(path: string): Promise<string> {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

function invariant(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`Validation protocol verification failed: ${message}`);
}

export async function verifyValidationProtocol(
  root = process.cwd(),
): Promise<{ checkedFiles: number; protocolId: string }> {
  const manifest = JSON.parse(
    await readFile(resolve(root, MANIFEST_PATH), "utf8"),
  ) as ProtocolFreezeManifest;
  invariant(
    manifest.schemaVersion ===
      "flux-svi-score-v3-validation-protocol-freeze-v1",
    "unsupported manifest schema",
  );
  invariant(
    manifest.protocolId === SVI_SCORE_V3_VALIDATION_PROTOCOL.id,
    "protocol ID changed",
  );
  invariant(manifest.status === "predeclared-unopened", "protocol is not unopened");
  invariant(!manifest.validationOutcomesAccessed, "validation was already accessed");
  invariant(!manifest.auditOutcomesAccessed, "audit was already accessed");
  invariant(
    manifest.cohort.validationBusinesses ===
      SVI_SCORE_V3_VALIDATION_PROTOCOL.businesses &&
      manifest.cohort.validationRows === SVI_SCORE_V3_VALIDATION_PROTOCOL.rows,
    "validation cohort contract changed",
  );
  invariant(manifest.cohort.sealedAuditBusinesses === 80, "audit size changed");
  await verifyTrainingFreeze(root, { verifyLocalSources: false });
  for (const file of manifest.files) {
    const actual = await digest(resolve(root, file.path));
    invariant(actual === file.sha256, `${file.role} hash changed (${actual})`);
  }
  const comparatorFile = manifest.files.find((file) => file.role === "comparator");
  const challengerFile = manifest.files.find((file) => file.role === "challenger");
  invariant(comparatorFile && challengerFile, "comparator or challenger is missing");
  const comparator = JSON.parse(
    await readFile(resolve(root, comparatorFile.path), "utf8"),
  ) as { activation?: string };
  const challenger = JSON.parse(
    await readFile(resolve(root, challengerFile.path), "utf8"),
  ) as { activation?: string };
  invariant(comparator.activation === "active", "comparator is not the active V6 release");
  invariant(
    challenger.activation === "research-only",
    "challenger must remain research-only before validation",
  );
  return { checkedFiles: manifest.files.length, protocolId: manifest.protocolId };
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : "";
if (invokedPath === fileURLToPath(import.meta.url)) {
  const receipt = await verifyValidationProtocol();
  process.stdout.write(`${JSON.stringify({ status: "verified", ...receipt }, null, 2)}\n`);
}
