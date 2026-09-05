import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  loadSviScoreV4DevelopmentRows,
  V4_FLIGHT_HOLDOUT_SIDECAR_PATH,
  V4_TEMPORAL_SIDECAR_PATH,
} from "../svi_score_v4/development-rows";
import type { SviScoreV4Metrics } from "../svi_score_v4/types";
import {
  SVI_SCORE_V5A_CONTRACT,
  SVI_SCORE_V5A_VERSION,
} from "./contract";
import { crossValidateScoreV5A } from "./cross-validation";
import { v5aSelectionObjective } from "./ranker";

interface V4DevelopmentArtifact {
  version: string;
  provenance?: { v3AuditAccessed?: boolean };
  crossValidation: {
    selectedConfiguration: { id: string };
    configurations: Array<{
      configuration: { id: string };
      outOfFoldMetrics: SviScoreV4Metrics;
    }>;
  };
  descriptiveComparators?: { activeV6?: SviScoreV4Metrics };
}

function selectedMetrics(artifact: V4DevelopmentArtifact): SviScoreV4Metrics {
  const id = artifact.crossValidation.selectedConfiguration.id;
  const metrics = artifact.crossValidation.configurations.find(
    (receipt) => receipt.configuration.id === id,
  )?.outOfFoldMetrics;
  if (!metrics) throw new Error(`Comparator ${artifact.version} has no selected OOF metrics.`);
  return metrics;
}

function relativeReduction(candidate: number, comparator: number): number {
  return (comparator - candidate) / Math.max(Math.abs(comparator), 1e-12);
}

const artifactDirectory = resolve("research/svi_score_v5a/artifacts");
const artifactPath = resolve(artifactDirectory, "svi-score-v5a-development.json");
const rows = await loadSviScoreV4DevelopmentRows();
const businesses = new Set(rows.map((row) => row.businessId));
const candidateCounts = new Map<string, number>();
rows.forEach((row) => {
  candidateCounts.set(row.businessId, (candidateCounts.get(row.businessId) ?? 0) + 1);
});
if (
  businesses.size !== SVI_SCORE_V5A_CONTRACT.development.businesses ||
  rows.length !== 20_160 ||
  [...candidateCounts.values()].some(
    (count) => count !== SVI_SCORE_V5A_CONTRACT.development.candidatesPerBusiness,
  )
) {
  throw new Error(
    `V5A common pool changed: found ${businesses.size} businesses/${rows.length} rows.`,
  );
}
if (rows.some((row) => row.split === "audit")) {
  throw new Error("The sealed V3 audit is forbidden in V5A development.");
}

const comparatorPaths = {
  v4_0: resolve("research/svi_score_v4/artifacts/svi-score-v4-development.json"),
  v4_1: resolve("research/svi_score_v4/artifacts/svi-score-v4.1-development.json"),
  v4_2: resolve("research/svi_score_v4/artifacts/svi-score-v4.2-development.json"),
};
const [temporalSource, flightSource, v40Source, v41Source, v42Source] =
  await Promise.all([
    readFile(V4_TEMPORAL_SIDECAR_PATH),
    readFile(V4_FLIGHT_HOLDOUT_SIDECAR_PATH),
    readFile(comparatorPaths.v4_0),
    readFile(comparatorPaths.v4_1),
    readFile(comparatorPaths.v4_2),
  ]);
const v40 = JSON.parse(v40Source.toString("utf8")) as V4DevelopmentArtifact;
const v41 = JSON.parse(v41Source.toString("utf8")) as V4DevelopmentArtifact;
const v42 = JSON.parse(v42Source.toString("utf8")) as V4DevelopmentArtifact;
if (v40.provenance?.v3AuditAccessed || v41.provenance?.v3AuditAccessed ||
  v42.provenance?.v3AuditAccessed) {
  throw new Error("A V5A comparator reports forbidden access to the sealed audit.");
}

const crossValidation = crossValidateScoreV5A(rows);
const selectedId = crossValidation.selectedConfiguration.id;
const v5aMetrics = crossValidation.configurations.find(
  (receipt) => receipt.configuration.id === selectedId,
)?.outOfFoldMetrics;
if (!v5aMetrics) throw new Error("V5A selected configuration has no OOF metrics.");
const v42Metrics = selectedMetrics(v42);
const activeV6 = v42.descriptiveComparators?.activeV6;
if (!activeV6) throw new Error("The locked active V6 comparator is missing.");
const familyDeltasVersusV42 = Object.fromEntries(
  Object.keys(v5aMetrics.familyMeanLoss).sort().map((family) => [
    family,
    {
      absolute: v5aMetrics.familyMeanLoss[family] -
        (v42Metrics.familyMeanLoss[family] ?? Number.NaN),
      relativeReduction: relativeReduction(
        v5aMetrics.familyMeanLoss[family],
        v42Metrics.familyMeanLoss[family] ?? Number.NaN,
      ),
    },
  ]),
);
const objectiveV5A = v5aSelectionObjective(v5aMetrics);
const objectiveV42 = v5aSelectionObjective(v42Metrics);
const selectorImprovedDevelopment =
  v5aMetrics.meanLoss < v42Metrics.meanLoss &&
  objectiveV5A < objectiveV42;

const artifact = {
  artifactId: "flux-svi-score-v5a-selector-development-v1",
  version: SVI_SCORE_V5A_VERSION,
  generatedAt: new Date().toISOString(),
  activation: "research-only",
  stage: "selector-development-complete-no-expanded-search",
  contract: SVI_SCORE_V5A_CONTRACT,
  provenance: {
    commonRows: rows.length,
    commonBusinesses: businesses.size,
    candidatesPerBusiness: 48,
    v3SealedAuditAccessed: false,
    temporalFeatureSha256: createHash("sha256").update(temporalSource).digest("hex"),
    flightHoldoutFeatureSha256: createHash("sha256").update(flightSource).digest("hex"),
    comparatorSha256: Object.fromEntries([
      ["v4.0", createHash("sha256").update(v40Source).digest("hex")],
      ["v4.1", createHash("sha256").update(v41Source).digest("hex")],
      ["v4.2", createHash("sha256").update(v42Source).digest("hex")],
    ]),
  },
  crossValidation,
  comparators: {
    note:
      "V4.2 is the primary selector comparator. V4.0/V4.1 preserve their historical temporal policies; active V6 is a locked descriptive comparator.",
    activeV6,
    v4_0: selectedMetrics(v40),
    v4_1: selectedMetrics(v41),
    v4_2: v42Metrics,
  },
  pairedDevelopmentComparisonVersusV42: {
    relativeMeanLossReduction: relativeReduction(
      v5aMetrics.meanLoss,
      v42Metrics.meanLoss,
    ),
    relativeCvar90LossReduction: relativeReduction(
      v5aMetrics.cvar90Loss,
      v42Metrics.cvar90Loss,
    ),
    relativeWorstFamilyMeanLossReduction: relativeReduction(
      v5aMetrics.worstFamilyMeanLoss,
      v42Metrics.worstFamilyMeanLoss,
    ),
    selectionObjective: {
      v5a: objectiveV5A,
      v4_2: objectiveV42,
      relativeReduction: relativeReduction(objectiveV5A, objectiveV42),
    },
    familyDeltas: familyDeltasVersusV42,
    selectorImprovedDevelopment,
  },
  nextGate: selectorImprovedDevelopment
    ? "Inspect stability and freeze V5A before starting an equal-compute expanded-search experiment. Do not generate validation yet."
    : "Do not expand search. Diagnose or reject V5A using development data only; keep the audit sealed.",
};

await mkdir(artifactDirectory, { recursive: true });
const temporary = `${artifactPath}.tmp`;
await writeFile(temporary, JSON.stringify(artifact, null, 2));
await rename(temporary, artifactPath);
process.stdout.write(`${JSON.stringify({
  artifactPath,
  stage: artifact.stage,
  selectedConfiguration: crossValidation.selectedConfiguration,
  outOfFold: v5aMetrics,
  versusV42: artifact.pairedDevelopmentComparisonVersusV42,
  theorem: crossValidation.finalFit.theorem,
  auditAccessed: false,
}, null, 2)}\n`);
