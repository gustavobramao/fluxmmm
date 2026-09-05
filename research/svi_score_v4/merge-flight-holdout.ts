import { readFile, rename, writeFile } from "node:fs/promises";
import {
  V4_FLIGHT_HOLDOUT_SIDECAR_PATH,
  type SviScoreV4FlightHoldoutSidecarRow,
} from "./development-rows";

const countArgument = process.argv.find((argument) => argument.startsWith("--shards="));
const shardCount = Number(countArgument?.slice("--shards=".length) ?? 8);
const rows = (await Promise.all(
  Array.from({ length: shardCount }, async (_, index) =>
    JSON.parse((await readFile(
      `${V4_FLIGHT_HOLDOUT_SIDECAR_PATH}.part-${index + 1}-of-${shardCount}`,
    )).toString("utf8")) as SviScoreV4FlightHoldoutSidecarRow[]
  ),
)).flat().sort(
  (left, right) =>
    left.businessId.localeCompare(right.businessId) ||
    left.candidateId.localeCompare(right.candidateId),
);
const keys = new Set(rows.map((row) => `${row.businessId}\u0000${row.candidateId}`));
const businesses = new Set(rows.map((row) => row.businessId));
if (keys.size !== rows.length || businesses.size !== 420 || rows.length !== 20_160) {
  throw new Error(`Invalid flight sidecar: ${businesses.size} businesses/${rows.length} rows.`);
}
const temporary = `${V4_FLIGHT_HOLDOUT_SIDECAR_PATH}.tmp`;
await writeFile(temporary, JSON.stringify(rows, null, 2));
await rename(temporary, V4_FLIGHT_HOLDOUT_SIDECAR_PATH);
process.stdout.write(`${JSON.stringify({
  outputPath: V4_FLIGHT_HOLDOUT_SIDECAR_PATH,
  businesses: businesses.size,
  rows: rows.length,
}, null, 2)}\n`);
