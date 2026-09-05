import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { prepareMcmcResearchBusiness } from "../mcmc_score_v3/cohort";
import { sviDesignBusinesses } from "../svi_score_v3/design";
import { V4_TEMPORAL_SIDECAR_PATH } from "./development-rows";
import { temporalEvidenceForCandidate } from "./temporal";
import type { SviScoreV4TemporalSidecarRow } from "./development-rows";

const limitArgument = process.argv.find((argument) => argument.startsWith("--limit="));
const limit = limitArgument
  ? Number(limitArgument.slice("--limit=".length))
  : Number.POSITIVE_INFINITY;
const shardArgument = process.argv.find((argument) => argument.startsWith("--shard="));
const shardMatch = shardArgument?.slice("--shard=".length).match(/^(\d+)\/(\d+)$/);
const shardIndex = shardMatch ? Number(shardMatch[1]) - 1 : 0;
const shardCount = shardMatch ? Number(shardMatch[2]) : 1;
if (
  !Number.isInteger(shardIndex) || !Number.isInteger(shardCount) ||
  shardIndex < 0 || shardCount < 1 || shardIndex >= shardCount
) {
  throw new Error("--shard must use the form 1/8 with a valid one-based index.");
}
const baseOutputPath = process.argv.includes("--smoke")
  ? `${V4_TEMPORAL_SIDECAR_PATH}.smoke`
  : V4_TEMPORAL_SIDECAR_PATH;
const outputPath = shardCount > 1
  ? `${baseOutputPath}.part-${shardIndex + 1}-of-${shardCount}`
  : baseOutputPath;

async function existingRows(): Promise<SviScoreV4TemporalSidecarRow[]> {
  if (!process.argv.includes("--resume")) return [];
  try {
    return JSON.parse(
      (await readFile(outputPath)).toString("utf8"),
    ) as SviScoreV4TemporalSidecarRow[];
  } catch (error) {
    if (
      error instanceof Error &&
      "code" in error &&
      (error as NodeJS.ErrnoException).code === "ENOENT"
    ) return [];
    throw error;
  }
}

async function checkpoint(rows: SviScoreV4TemporalSidecarRow[]): Promise<void> {
  await mkdir(dirname(outputPath), { recursive: true });
  const temporary = `${outputPath}.tmp`;
  await writeFile(temporary, JSON.stringify(rows, null, 2));
  await rename(temporary, outputPath);
}

const rows = await existingRows();
const completed = new Set(rows.map((row) => row.businessId));
const development = sviDesignBusinesses()
  .filter((business) => business.family.split !== "audit")
  .filter((_business, index) => index % shardCount === shardIndex)
  .slice(0, Number.isFinite(limit) ? limit : undefined);

for (let index = 0; index < development.length; index += 1) {
  const design = development[index];
  if (completed.has(design.scenario.id)) continue;
  const prepared = await prepareMcmcResearchBusiness(
    design.scenario,
    design.family.id,
    design.family.split,
    (detail) => process.stdout.write(`\r${index + 1}/${development.length} · ${detail}`),
    { includeHiddenTruthLabels: false },
  );
  prepared.candidates.forEach((candidate) => {
    rows.push({
      businessId: design.scenario.id,
      candidateId: candidate.row.candidateId,
      temporal: temporalEvidenceForCandidate(
        prepared.dataset,
        candidate.run.model!,
        candidate.run.spec.config,
      ),
    });
  });
  completed.add(design.scenario.id);
  await checkpoint(rows);
  process.stdout.write(
    `\r${index + 1}/${development.length} · ${design.scenario.id} · ${rows.length} rows\n`,
  );
}

process.stdout.write(`${JSON.stringify({
  outputPath,
  businesses: completed.size,
  rows: rows.length,
  truthLabelsRead: false,
  shard: `${shardIndex + 1}/${shardCount}`,
}, null, 2)}\n`);
