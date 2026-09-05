import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { prepareMcmcResearchBusiness } from "../mcmc_score_v3/cohort";
import { sviDesignBusinesses } from "./design";
import type { SviScoreV3TrainingFeatureRow } from "./training-rows";

const outputDirectory = resolve(".flux-artifacts/svi-score-v3");
const outputPath = resolve(outputDirectory, "validation-features-v2.json");
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
  // A missing sidecar starts a new truth-blind validation refresh.
}

const validationBusinesses = sviDesignBusinesses().filter(
  (business) => business.family.split === "validation",
);
const expectedIds = new Set(validationBusinesses.map(({ scenario }) => scenario.id));
if (rows.some((row) => !expectedIds.has(row.businessId))) {
  throw new Error("Validation feature sidecar contains a non-validation business.");
}
const rowCountByBusiness = new Map<string, number>();
rows.forEach((row) =>
  rowCountByBusiness.set(
    row.businessId,
    (rowCountByBusiness.get(row.businessId) ?? 0) + 1,
  )
);
const completed = new Set(
  [...rowCountByBusiness.entries()]
    .filter(([, count]) => count === 48)
    .map(([businessId]) => businessId),
);
if ([...rowCountByBusiness.values()].some((count) => count !== 48)) {
  throw new Error("Validation feature sidecar contains a partially written business.");
}

const remaining = validationBusinesses.filter(
  ({ scenario }) => !completed.has(scenario.id),
);
for (let offset = 0; offset < remaining.length; offset += workers) {
  const batch = remaining.slice(offset, offset + workers);
  const results = await Promise.all(batch.map(async ({ family, scenario }, batchIndex) => {
    process.stdout.write(
      `${completed.size + offset + batchIndex + 1}/${validationBusinesses.length} · ${family.id} · ${scenario.id}\n`,
    );
    const prepared = await prepareMcmcResearchBusiness(
      scenario,
      family.id,
      "validation",
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

if (rows.length !== validationBusinesses.length * 48) {
  throw new Error(
    `Validation feature sidecar is incomplete: ${rows.length}/${validationBusinesses.length * 48}.`,
  );
}
const source = await readFile(outputPath);
process.stdout.write(`${JSON.stringify({
  outputPath,
  sha256: createHash("sha256").update(source).digest("hex"),
  businesses: validationBusinesses.length,
  rows: rows.length,
  hiddenTruthReadDuringRefresh: false,
  workers,
}, null, 2)}\n`);
