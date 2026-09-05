import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { parseCsv } from "../../lib/mmm/csv";
import { createDataset, validateDataset } from "../../lib/mmm/schema";
import type { ValidationOptions } from "../../lib/mmm/validation";
import type { ValidationModelSpec } from "../../lib/mmm/validation/adapter";
import { generateSyntheticBusiness } from "../score_v2/simulator";
import { SCORE_V6_CANDIDATES } from "../score_v6/cohort";
import { syntheticIndustryPriorOverrides } from "../score_v6/evidence";
import type { ScoreV6EvidenceArm } from "../score_v6/types";
import { sviDesignBusinesses } from "../svi_score_v3/design";
import {
  V4_FLIGHT_HOLDOUT_SIDECAR_PATH,
  type SviScoreV4FlightHoldoutSidecarRow,
} from "./development-rows";
import { wholeFlightGeneralization } from "./temporal";

const shardArgument = process.argv.find((argument) => argument.startsWith("--shard="));
const shardMatch = shardArgument?.slice("--shard=".length).match(/^(\d+)\/(\d+)$/);
const shardIndex = shardMatch ? Number(shardMatch[1]) - 1 : 0;
const shardCount = shardMatch ? Number(shardMatch[2]) : 1;
if (
  !Number.isInteger(shardIndex) || !Number.isInteger(shardCount) ||
  shardIndex < 0 || shardCount < 1 || shardIndex >= shardCount
) throw new Error("--shard must use the form 1/8.");
const outputPath = shardCount > 1
  ? `${V4_FLIGHT_HOLDOUT_SIDECAR_PATH}.part-${shardIndex + 1}-of-${shardCount}`
  : V4_FLIGHT_HOLDOUT_SIDECAR_PATH;

async function existing(): Promise<SviScoreV4FlightHoldoutSidecarRow[]> {
  if (!process.argv.includes("--resume")) return [];
  try {
    return JSON.parse(
      (await readFile(outputPath)).toString("utf8"),
    ) as SviScoreV4FlightHoldoutSidecarRow[];
  } catch (error) {
    if (
      error instanceof Error && "code" in error &&
      (error as NodeJS.ErrnoException).code === "ENOENT"
    ) return [];
    throw error;
  }
}

async function checkpoint(rows: SviScoreV4FlightHoldoutSidecarRow[]) {
  await mkdir(dirname(outputPath), { recursive: true });
  const temporary = `${outputPath}.tmp`;
  await writeFile(temporary, JSON.stringify(rows, null, 2));
  await rename(temporary, outputPath);
}

function industryChannels(
  mediaColumns: string[],
  experimentChannels: string[],
  arm: ScoreV6EvidenceArm,
): string[] {
  if (arm === "experiments-only") return [];
  const experiments = new Set(experimentChannels.map((channel) => channel.toLowerCase()));
  return mediaColumns.filter((channel) => !experiments.has(channel.toLowerCase()));
}

const rows = await existing();
const completed = new Set(rows.map((row) => row.businessId));
const design = sviDesignBusinesses()
  .filter((business) => business.family.split !== "audit")
  .filter((_business, index) => index % shardCount === shardIndex);

for (let businessIndex = 0; businessIndex < design.length; businessIndex += 1) {
  const item = design[businessIndex];
  if (completed.has(item.scenario.id)) continue;
  const business = generateSyntheticBusiness(item.scenario);
  const parsed = parseCsv(business.observedCsv);
  const dataset = await createDataset(
    `${item.scenario.id}.csv`,
    business.observedCsv,
    parsed.columns,
    parsed.rows,
  );
  if (validateDataset(dataset).status === "blocked") {
    throw new Error(`${item.scenario.id} failed its data contract.`);
  }
  const experiments = business.truth.experiments.map((study) => study.experiment);
  const overrides = syntheticIndustryPriorOverrides(business);
  for (const arm of ["experiments-only", "benchmark-gap-fill"] as const) {
    const activeChannels = industryChannels(
      dataset.mediaColumns,
      experiments.map((experiment) => experiment.channel),
      arm,
    );
    const validationOptions: ValidationOptions = {
      anchorIndependenceConfirmed: true,
      industryPriorChannels: activeChannels,
      industryPriorOverrides: overrides,
      industryBenchmarkScreeningEnabled: arm === "benchmark-gap-fill",
      materialSpendShareThreshold: 0.02,
    };
    for (const baseSpec of SCORE_V6_CANDIDATES) {
      const candidateId = `${baseSpec.id} · ${arm}`;
      const receipt = wholeFlightGeneralization(dataset, {
        kind: baseSpec.family,
        config: baseSpec.config,
        advancedConfig: baseSpec.advancedConfig,
        experiments,
        validationOptions,
      } satisfies ValidationModelSpec);
      rows.push({
        businessId: item.scenario.id,
        candidateId,
        score: receipt.score,
        folds: receipt.folds,
        meanWape: Number.isFinite(receipt.meanWape) ? receipt.meanWape : null,
        meanSkill: Number.isFinite(receipt.meanSkill) ? receipt.meanSkill : null,
      });
    }
  }
  completed.add(item.scenario.id);
  await checkpoint(rows);
  process.stdout.write(
    `${shardIndex + 1}/${shardCount} · ${businessIndex + 1}/${design.length} · ${item.scenario.id}\n`,
  );
}
process.stdout.write(`${JSON.stringify({
  outputPath,
  businesses: completed.size,
  rows: rows.length,
  truthLabelsRead: false,
}, null, 2)}\n`);
