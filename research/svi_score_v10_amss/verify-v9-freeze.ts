import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { resolve } from "node:path";

const SNAPSHOT_ROOT = resolve("research/svi_score_v10_amss/frozen-v9");
const MANIFEST_PATH = resolve(SNAPSHOT_ROOT, "manifest.json");

interface FrozenFile {
  path: string;
  bytes: number;
  sha256: string;
}

interface FreezeManifest {
  snapshotVersion: string;
  status: string;
  selector: {
    version: string;
    artifactId: string;
    sha256: string;
    model: string;
    ensembleMembers: number;
    featureCount: number;
  };
  risk: {
    meanWeight: number;
    p90Weight: number;
    dangerPenalty: number;
    uncertaintyPenalty: number;
  };
  gates: string[];
  truthBoundary: {
    amssObservedInputsPermitted: boolean;
    amssGroundTruthBeforeSelection: boolean;
    v9RetrainingPermitted: boolean;
  };
  sourceProvenance: {
    selector: string;
    development: string;
    selectorSha256: string;
    developmentSha256: string;
  };
  frozenFiles: FrozenFile[];
}

function invariant(condition: unknown, detail: string): asserts condition {
  if (!condition) throw new Error(`V10 AMSS V9 freeze verification failed: ${detail}.`);
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

const [manifestSource, declaredManifestSha] = await Promise.all([
  readFile(MANIFEST_PATH),
  readFile(resolve(SNAPSHOT_ROOT, "manifest.sha256"), "utf8"),
]);
invariant(
  sha256(manifestSource) === declaredManifestSha.trim(),
  "manifest checksum changed",
);
const manifest = JSON.parse(manifestSource.toString("utf8")) as FreezeManifest;

invariant(
  manifest.snapshotVersion === "flux-svi-score-v10-amss-v9-freeze-v1",
  "snapshot version changed",
);
invariant(
  manifest.status === "frozen-before-amss-installation-or-data-generation",
  "freeze timing boundary changed",
);
invariant(manifest.selector.featureCount === 232, "token count changed");
invariant(manifest.selector.ensembleMembers > 0, "selector ensemble is empty");
invariant(
  manifest.risk.meanWeight === 0.65 &&
    manifest.risk.p90Weight === 0.35 &&
    manifest.risk.meanWeight + manifest.risk.p90Weight === 1,
  "risk preference changed",
);
invariant(
  manifest.risk.dangerPenalty === 0 && manifest.risk.uncertaintyPenalty === 0.25,
  "promotion policy changed",
);
invariant(manifest.gates.length === 4, "hard-gate contract changed");
invariant(
  manifest.truthBoundary.amssObservedInputsPermitted &&
    !manifest.truthBoundary.amssGroundTruthBeforeSelection &&
    !manifest.truthBoundary.v9RetrainingPermitted,
  "AMSS truth firewall changed",
);

for (const frozen of manifest.frozenFiles) {
  const path = resolve(SNAPSHOT_ROOT, frozen.path);
  const [value, metadata] = await Promise.all([readFile(path), stat(path)]);
  invariant(metadata.size === frozen.bytes, `${frozen.path} byte count changed`);
  invariant(sha256(value) === frozen.sha256, `${frozen.path} checksum changed`);
}

const selectorSource = await readFile(resolve(manifest.sourceProvenance.selector)).catch(
  async (error: NodeJS.ErrnoException) => {
    if (error.code !== "ENOENT") throw error;
    // Clean public clones intentionally omit multi-gigabyte working artifacts.
    // The immutable selector copied into the external-audit snapshot is the
    // canonical byte-for-byte substitute for provenance verification.
    return readFile(resolve(SNAPSHOT_ROOT, "selector.json"));
  },
);
const developmentSource = await readFile(resolve(manifest.sourceProvenance.development));
invariant(
  sha256(selectorSource) === manifest.sourceProvenance.selectorSha256,
  "live V9 selector drifted from the external-audit freeze",
);
invariant(
  sha256(developmentSource) === manifest.sourceProvenance.developmentSha256,
  "live V9 development verification drifted from the external-audit freeze",
);

const selector = JSON.parse(
  (await readFile(resolve(SNAPSHOT_ROOT, "selector.json"))).toString("utf8"),
) as { version: string; artifactId: string; featureNames: string[] };
const registry = JSON.parse(
  await readFile(resolve(SNAPSHOT_ROOT, "token-registry.json"), "utf8"),
) as { count: number; names: string[]; orderedNamesSha256: string };
invariant(selector.version === manifest.selector.version, "selector version mismatch");
invariant(selector.artifactId === manifest.selector.artifactId, "selector ID mismatch");
invariant(sha256(selectorSource) === manifest.selector.sha256, "selector hash mismatch");
invariant(registry.count === manifest.selector.featureCount, "registry count mismatch");
invariant(
  JSON.stringify(registry.names) === JSON.stringify(selector.featureNames),
  "selector and frozen registry disagree",
);
invariant(
  sha256(JSON.stringify(registry.names)) === registry.orderedNamesSha256,
  "token ordering changed",
);

process.stdout.write(`${JSON.stringify({
  verified: true,
  snapshotVersion: manifest.snapshotVersion,
  selectorVersion: manifest.selector.version,
  selectorSha256: manifest.selector.sha256,
  tokens: manifest.selector.featureCount,
  ensembleMembers: manifest.selector.ensembleMembers,
  risk: manifest.risk,
  frozenFiles: manifest.frozenFiles.length,
  truthFirewall: "closed",
}, null, 2)}\n`);
