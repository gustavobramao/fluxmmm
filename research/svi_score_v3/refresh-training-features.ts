import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { prepareMcmcResearchBusiness } from "../mcmc_score_v3/cohort";
import { sviDesignBusinesses } from "./design";
import type { SviScoreV3TrainingFeatureRow } from "./training-rows";

const outputDirectory = resolve(".flux-artifacts/svi-score-v3");
const outputPath = resolve(outputDirectory, "train-features-v2.json");
const temporaryPath = `${outputPath}.tmp`;
const workerArgument = process.argv.find((argument) =>
  argument.startsWith("--workers=")
);
const workers = Math.min(
  8,
  Math.max(1, Number.parseInt(workerArgument?.split("=")[1] ?? "4", 10)),
);

await mkdir(outputDirectory, { recursive: true });

let rows: SviScoreV3TrainingFeatureRow[] = [];
try {
  rows = JSON.parse(await readFile(outputPath, "utf8"));
} catch {
  // A missing sidecar starts a new train-only feature refresh.
}

const completed = new Set(rows.map((row) => row.businessId));
const trainingBusinesses = sviDesignBusinesses().filter(
  (business) => business.family.split === "train",
);

const remaining = trainingBusinesses.filter(
  (business) => !completed.has(business.scenario.id),
);
for (let offset = 0; offset < remaining.length; offset += workers) {
  const batch = remaining.slice(offset, offset + workers);
  const results = await Promise.all(batch.map(async ({ family, scenario }, batchIndex) => {
    process.stdout.write(
      `${completed.size + offset + batchIndex + 1}/${trainingBusinesses.length} · ${family.id} · ${scenario.id}\n`,
    );
    const prepared = await prepareMcmcResearchBusiness(
      scenario,
      family.id,
      "train",
      undefined,
      { includeHiddenTruthLabels: false },
    );
    const refreshed = prepared.candidates.map(({ row }) => ({
      businessId: row.businessId,
      candidateId: row.candidateId,
      eligible: row.eligible,
      reviewEligible: row.reviewEligible,
      eligibilityTier: row.eligibilityTier,
      failedGateCount: row.failedGateCount,
      layerScores: row.layerScores,
      diagnostics: row.diagnostics,
      heuristicScore: row.heuristicScore,
    } satisfies SviScoreV3TrainingFeatureRow));
    if (refreshed.length !== 48) {
      throw new Error(`${scenario.id} produced ${refreshed.length}/48 feature rows.`);
    }
    return refreshed;
  }));
  rows.push(...results.flat());
  rows.sort((left, right) =>
    left.businessId.localeCompare(right.businessId) ||
    left.candidateId.localeCompare(right.candidateId)
  );
  await writeFile(temporaryPath, JSON.stringify(rows, null, 2));
  await rename(temporaryPath, outputPath);
}

if (rows.length !== trainingBusinesses.length * 48) {
  throw new Error(
    `Train-only feature sidecar is incomplete: ${rows.length}/${trainingBusinesses.length * 48}.`,
  );
}

process.stdout.write(`${JSON.stringify({
  outputPath,
  businesses: trainingBusinesses.length,
  rows: rows.length,
  hiddenTruthReadDuringRefresh: false,
  workers,
}, null, 2)}\n`);
