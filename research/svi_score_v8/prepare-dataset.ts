import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { loadSviScoreV4DevelopmentRows } from "../svi_score_v4/development-rows";
import type { SviScoreV5ACrossValidationReceipt } from "../svi_score_v5a/types";
import {
  loadV5DSviPosteriorRecords,
  v5dSviFeatureProvider,
  v5dSviKey,
} from "../svi_score_v5d_svi/posterior";
import {
  SVI_SCORE_V8_CONTRACT,
  SVI_SCORE_V8_VERSION,
  V8_DANGEROUS_EXCESS_LOSS,
  V8_DECISION_SCENARIOS,
  V8_RISK_PREFERENCE,
} from "./contract";
import type { V8DatasetRow, V8ScenarioTargetRow } from "./types";

const TARGET_PATH = resolve(
  ".flux-artifacts/svi-score-v8/development-scenario-targets.json",
);
const OUTPUT_PATH = resolve(".flux-artifacts/svi-score-v8/development-dataset.json");
const V5A_PATH = resolve(
  "research/svi_score_v5a/artifacts/svi-score-v5a-development.json",
);

interface ScenarioArtifact {
  version: string;
  provenance: {
    auditRowsDereferenced: number;
    nutsUsed: boolean;
  };
  rows: V8ScenarioTargetRow[];
}

interface V5AArtifact {
  provenance: { v3SealedAuditAccessed: boolean };
  crossValidation: SviScoreV5ACrossValidationReceipt;
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function average(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) /
    Math.max(values.length, 1);
}

function quantile(values: readonly number[], probability: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  if (!sorted.length) return 0;
  const position = Math.min(1, Math.max(0, probability)) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function finiteScenario(row: V8ScenarioTargetRow): boolean {
  return V8_DECISION_SCENARIOS.every((scenario) =>
    Number.isFinite(row.scenarioDecisionLoss[scenario]) &&
    row.scenarioDecisionLoss[scenario] >= 0
  );
}

const [targetSource, v5aSource] = await Promise.all([
  readFile(TARGET_PATH),
  readFile(V5A_PATH),
]);
const targets = JSON.parse(targetSource.toString("utf8")) as ScenarioArtifact;
const v5a = JSON.parse(v5aSource.toString("utf8")) as V5AArtifact;
if (
  targets.version !== SVI_SCORE_V8_VERSION ||
  targets.provenance.auditRowsDereferenced !== 0 ||
  targets.provenance.nutsUsed ||
  v5a.provenance.v3SealedAuditAccessed
) {
  throw new Error("V8 dataset preparation requires development-only SVI artifacts.");
}
const rows = await loadSviScoreV4DevelopmentRows();
const posteriorRecords = await loadV5DSviPosteriorRecords(rows);
const provider = v5dSviFeatureProvider(posteriorRecords);
const featureNames = provider.names(rows[0]);
if (featureNames.length !== SVI_SCORE_V8_CONTRACT.tokens.count) {
  throw new Error(`V8 expected 136 tokens; found ${featureNames.length}.`);
}
const targetByKey = new Map(
  targets.rows.map((row) => [`${row.businessId}\u0000${row.candidateId}`, row]),
);
if (targetByKey.size !== targets.rows.length) {
  throw new Error("V8 scenario target sidecar contains duplicate rows.");
}
const foldByBusiness = new Map(
  v5a.crossValidation.assignments.map((row) => [row.businessId, row.fold]),
);
const grouped = new Map<string, typeof rows>();
rows.forEach((row) => {
  if (row.split === "audit") {
    throw new Error("V8 development dataset received a sealed-audit row.");
  }
  grouped.set(row.businessId, [...(grouped.get(row.businessId) ?? []), row]);
});
const datasetRows: V8DatasetRow[] = [];
for (const [businessId, business] of [...grouped.entries()].sort()) {
  const prepared = business.map((row) => {
    const posterior = posteriorRecords.get(v5dSviKey(row));
    const target = targetByKey.get(`${businessId}\u0000${row.candidateId}`);
    const features = provider.vector(row);
    const validityReasons = [
      ...(!posterior ? ["missing-svi-posterior"] : []),
      ...(posterior && !posterior.diagnostics.finite
        ? ["non-finite-svi-posterior"]
        : []),
      ...(!target || !finiteScenario(target)
        ? ["missing-scenario-label"]
        : []),
      ...(features.length !== featureNames.length ||
          features.some((value) => !Number.isFinite(value))
        ? ["invalid-token-contract"]
        : []),
    ];
    return { row, posterior, target, features, validityReasons };
  });
  const valid = prepared.filter((item) => item.validityReasons.length === 0);
  if (!valid.length) throw new Error(`${businessId} has no valid V8 candidate.`);
  const scenarioMinimum = Object.fromEntries(
    V8_DECISION_SCENARIOS.map((scenario) => [
      scenario,
      Math.min(...valid.map((item) => item.target!.scenarioDecisionLoss[scenario])),
    ]),
  ) as Record<(typeof V8_DECISION_SCENARIOS)[number], number>;
  const summaries = valid.map((item) => {
    const losses = V8_DECISION_SCENARIOS.map(
      (scenario) => item.target!.scenarioDecisionLoss[scenario],
    );
    const mean = average(losses);
    const p90 = quantile(losses, 0.9);
    return {
      item,
      mean,
      p90,
      risk: V8_RISK_PREFERENCE.mean * mean + V8_RISK_PREFERENCE.p90 * p90,
    };
  });
  const minimumMean = Math.min(...summaries.map((summary) => summary.mean));
  const minimumP90 = Math.min(...summaries.map((summary) => summary.p90));
  const minimumRisk = Math.min(...summaries.map((summary) => summary.risk));
  const summaryByCandidate = new Map(
    summaries.map((summary) => [summary.item.row.candidateId, summary]),
  );
  for (const item of prepared) {
    if (!item.target) {
      throw new Error(`V8 cannot serialize ${businessId}/${item.row.candidateId} without a target.`);
    }
    const summary = summaryByCandidate.get(item.row.candidateId);
    const losses = item.target.scenarioDecisionLoss;
    const mean = summary?.mean ?? average(V8_DECISION_SCENARIOS.map((id) => losses[id]));
    const p90 = summary?.p90 ?? quantile(V8_DECISION_SCENARIOS.map((id) => losses[id]), 0.9);
    const risk = V8_RISK_PREFERENCE.mean * mean + V8_RISK_PREFERENCE.p90 * p90;
    const excessRisk = Math.max(0, risk - minimumRisk);
    datasetRows.push({
      businessId,
      family: item.row.family,
      candidateId: item.row.candidateId,
      fold: foldByBusiness.get(businessId) ?? -1,
      features: item.features,
      valid: item.validityReasons.length === 0,
      validityReasons: item.validityReasons,
      scenarioLoss: structuredClone(losses),
      scenarioExcessLoss: Object.fromEntries(
        V8_DECISION_SCENARIOS.map((scenario) => [
          scenario,
          Math.max(0, losses[scenario] - scenarioMinimum[scenario]),
        ]),
      ) as V8DatasetRow["scenarioExcessLoss"],
      meanExcessLoss: Math.max(0, mean - minimumMean),
      p90ExcessLoss: Math.max(0, p90 - minimumP90),
      economicRisk: risk,
      excessEconomicRisk: excessRisk,
      normalizedExcessEconomicRisk: Math.min(
        excessRisk / Math.max(item.row.economicScale ?? 1, 1e-12),
        1,
      ),
      dangerous: excessRisk >= V8_DANGEROUS_EXCESS_LOSS,
      roiError: item.row.roiError,
      contributionError: item.row.contributionError,
    });
  }
}
if (
  datasetRows.length !== SVI_SCORE_V8_CONTRACT.cohort.developmentRows ||
  new Set(datasetRows.map((row) => `${row.businessId}\u0000${row.candidateId}`)).size !==
    datasetRows.length ||
  datasetRows.some((row) => row.fold < 0)
) {
  throw new Error("V8 development dataset failed cohort or fold integrity checks.");
}
const artifact = {
  artifactId: "flux-svi-score-v8-development-dataset-v1",
  version: SVI_SCORE_V8_VERSION,
  generatedAt: new Date().toISOString(),
  activation: "development-only",
  provenance: {
    scenarioTargetsSha256: sha256(targetSource),
    v5aAssignmentsSha256: sha256(v5aSource),
    tokenNamesSha256: sha256(JSON.stringify(featureNames)),
    auditAccessed: false,
    nutsUsed: false,
  },
  contract: SVI_SCORE_V8_CONTRACT,
  featureNames,
  summary: {
    businesses: grouped.size,
    rows: datasetRows.length,
    validRows: datasetRows.filter((row) => row.valid).length,
    dangerousRows: datasetRows.filter((row) => row.dangerous).length,
    folds: [...new Set(datasetRows.map((row) => row.fold))].sort(),
    families: [...new Set(datasetRows.map((row) => row.family))].sort(),
  },
  rows: datasetRows,
};
await mkdir(resolve(".flux-artifacts/svi-score-v8"), { recursive: true });
const temporary = `${OUTPUT_PATH}.tmp`;
await writeFile(temporary, JSON.stringify(artifact));
await rename(temporary, OUTPUT_PATH);
process.stdout.write(`${JSON.stringify({
  path: OUTPUT_PATH,
  ...artifact.summary,
  auditAccessed: false,
  nutsUsed: false,
}, null, 2)}\n`);
