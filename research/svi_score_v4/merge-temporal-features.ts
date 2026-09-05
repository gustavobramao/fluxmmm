import { readFile, rename, writeFile } from "node:fs/promises";
import { V4_TEMPORAL_SIDECAR_PATH } from "./development-rows";
import type { SviScoreV4TemporalSidecarRow } from "./development-rows";

const countArgument = process.argv.find((argument) => argument.startsWith("--shards="));
const shardCount = Number(countArgument?.slice("--shards=".length) ?? 8);
if (!Number.isInteger(shardCount) || shardCount < 1) {
  throw new Error("--shards must be a positive integer.");
}
const parts = await Promise.all(
  Array.from({ length: shardCount }, async (_, index) =>
    JSON.parse(
      (await readFile(
        `${V4_TEMPORAL_SIDECAR_PATH}.part-${index + 1}-of-${shardCount}`,
      )).toString("utf8"),
    ) as SviScoreV4TemporalSidecarRow[]
  ),
);
const rows = parts.flat().sort(
  (left, right) =>
    left.businessId.localeCompare(right.businessId) ||
    left.candidateId.localeCompare(right.candidateId),
);
const keys = new Set(
  rows.map((row) => `${row.businessId}\u0000${row.candidateId}`),
);
const businesses = new Set(rows.map((row) => row.businessId));
if (keys.size !== rows.length) throw new Error("Temporal shards overlap.");
if (businesses.size !== 420 || rows.length !== 20_160) {
  throw new Error(
    `Expected 420 businesses/20,160 rows; found ${businesses.size}/${rows.length}.`,
  );
}
const temporary = `${V4_TEMPORAL_SIDECAR_PATH}.tmp`;
await writeFile(temporary, JSON.stringify(rows, null, 2));
await rename(temporary, V4_TEMPORAL_SIDECAR_PATH);
process.stdout.write(`${JSON.stringify({
  outputPath: V4_TEMPORAL_SIDECAR_PATH,
  shards: shardCount,
  businesses: businesses.size,
  rows: rows.length,
}, null, 2)}\n`);
