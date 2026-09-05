import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { SCORE_DIAGNOSTIC_GROUPS } from "../../lib/mmm/score-diagnostics";
import { loadSviScoreV4DevelopmentRows } from "../svi_score_v4/development-rows";
import type { SviScoreV5ACrossValidationReceipt } from "../svi_score_v5a/types";
import { v5bCandidateUniverse } from "../svi_score_v5b/embedding";
import { v5cCandidateRegions } from "../svi_score_v5c/crossfit";
import { fitV5DCrossFittedModel } from "./crossfit";

type ImportanceGroup =
  | "generalization"
  | "structure"
  | "causal-robustness"
  | "decision-coherence"
  | "temporal-evidence"
  | "model-specification-and-interactions";

interface CoefficientRow {
  feature: string;
  group: ImportanceGroup;
  mean: number;
  p90: number;
  risk: number;
}

interface V5AArtifact {
  crossValidation: SviScoreV5ACrossValidationReceipt;
  provenance: { v3SealedAuditAccessed: boolean };
}

const TEMPORAL_DIAGNOSTICS = new Set([
  "whole-flight-generalization",
  "carryover-support",
  "post-flight-residual-stability",
  "kernel-distinguishability",
]);

const DIAGNOSTIC_GROUP = new Map<string, ImportanceGroup>([
  ...SCORE_DIAGNOSTIC_GROUPS.generalization.map((name) => [name, "generalization"] as const),
  ...SCORE_DIAGNOSTIC_GROUPS.structure.map((name) => [name, "structure"] as const),
  ...SCORE_DIAGNOSTIC_GROUPS.causal.map((name) => [name, "causal-robustness"] as const),
  ...SCORE_DIAGNOSTIC_GROUPS.decision.map((name) => [name, "decision-coherence"] as const),
]);

function diagnosticName(feature: string): string | undefined {
  if (feature.startsWith("diagnostic:")) return feature.slice("diagnostic:".length);
  if (feature.startsWith("deficit:")) {
    return feature.slice("deficit:".length).split(":")[0];
  }
  return undefined;
}

function featureGroup(feature: string): ImportanceGroup {
  if (feature.startsWith("context:temporal:")) return "temporal-evidence";
  if (feature.startsWith("spec:") || feature.startsWith("interaction:")) {
    return "model-specification-and-interactions";
  }
  const diagnostic = diagnosticName(feature);
  if (diagnostic && TEMPORAL_DIAGNOSTICS.has(diagnostic)) return "temporal-evidence";
  return DIAGNOSTIC_GROUP.get(diagnostic ?? "") ??
    "model-specification-and-interactions";
}

function average(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) /
    Math.max(values.length, 1);
}

function summary(
  values: readonly CoefficientRow[],
  key: "mean" | "p90" | "risk",
): {
  groups: Array<{ group: ImportanceGroup; importance: number; meanCoefficient: number }>;
  features: Array<{
    feature: string;
    group: ImportanceGroup;
    importance: number;
    meanCoefficient: number;
    signStability: number;
    direction: "raises-predicted-loss" | "reduces-predicted-loss";
  }>;
} {
  const byFeature = new Map<string, CoefficientRow[]>();
  values.forEach((row) => {
    byFeature.set(row.feature, [...(byFeature.get(row.feature) ?? []), row]);
  });
  const rawFeatures = [...byFeature.entries()].map(([feature, rows]) => {
    const coefficients = rows.map((row) => row[key]);
    const meanCoefficient = average(coefficients);
    const positive = coefficients.filter((value) => value > 1e-12).length;
    const negative = coefficients.filter((value) => value < -1e-12).length;
    return {
      feature,
      group: rows[0].group,
      rawImportance: average(coefficients.map(Math.abs)),
      meanCoefficient,
      signStability: Math.max(positive, negative) / Math.max(coefficients.length, 1),
      direction: meanCoefficient >= 0
        ? "raises-predicted-loss" as const
        : "reduces-predicted-loss" as const,
    };
  });
  const total = rawFeatures.reduce((sum, row) => sum + row.rawImportance, 0);
  const features = rawFeatures
    .map(({ rawImportance, ...row }) => ({
      ...row,
      importance: rawImportance / Math.max(total, 1e-12),
    }))
    .sort((left, right) =>
      right.importance - left.importance || left.feature.localeCompare(right.feature)
    );
  const groups = [...new Set(features.map((row) => row.group))]
    .map((group) => {
      const rows = features.filter((row) => row.group === group);
      return {
        group,
        importance: rows.reduce((sum, row) => sum + row.importance, 0),
        meanCoefficient: average(rows.map((row) => row.meanCoefficient)),
      };
    })
    .sort((left, right) => right.importance - left.importance);
  return { groups, features };
}

const v5aPath = resolve("research/svi_score_v5a/artifacts/svi-score-v5a-development.json");
const v5dPath = resolve("research/svi_score_v5d/artifacts/svi-score-v5d-development.json");
const [v5aSource, v5dSource] = await Promise.all([
  readFile(v5aPath),
  readFile(v5dPath),
]);
const v5a = JSON.parse(v5aSource.toString("utf8")) as V5AArtifact;
if (v5a.provenance.v3SealedAuditAccessed) {
  throw new Error("V5D importance cannot access the sealed audit.");
}
const rows = await loadSviScoreV4DevelopmentRows();
if (rows.some((row) => row.split === "audit")) {
  throw new Error("V5D importance received a sealed audit row.");
}
const foldByBusiness = new Map(
  v5a.crossValidation.assignments.map((row) => [row.businessId, row.fold]),
);
const universe = v5bCandidateUniverse(rows.map((row) => row.candidateId));
const regions = v5cCandidateRegions(universe.map((row) => row.candidateId));
const coefficients: CoefficientRow[] = [];
const modelReceipts: Array<{
  fold: number;
  region: number;
  trainingRows: number;
  trainingBusinesses: number;
}> = [];
const folds = [...new Set(v5a.crossValidation.assignments.map((row) => row.fold))]
  .sort((left, right) => left - right);
folds.forEach((fold) => {
  const training = rows.filter((row) => foldByBusiness.get(row.businessId) !== fold);
  const fitted = fitV5DCrossFittedModel(training, regions);
  fitted.regions.forEach((receipt) => {
    const model = receipt.model;
    modelReceipts.push({
      fold,
      region: receipt.region,
      trainingRows: model.receipt.trainingRows,
      trainingBusinesses: model.receipt.trainingBusinesses,
    });
    model.featureNames.forEach((feature, index) => {
      const mean = model.meanWeights[index + 1] ?? 0;
      const p90 = model.p90Weights[index + 1] ?? 0;
      coefficients.push({
        feature,
        group: featureGroup(feature),
        mean,
        p90,
        risk: 0.65 * mean + 0.35 * p90,
      });
    });
  });
});
const meanLoss = summary(coefficients, "mean");
const p90Loss = summary(coefficients, "p90");
const risk = summary(coefficients, "risk");
const artifact = {
  artifactId: "flux-svi-score-v5d-standardized-coefficient-importance-v1",
  generatedAt: new Date().toISOString(),
  activation: "research-only",
  provenance: {
    v5aSha256: createHash("sha256").update(v5aSource).digest("hex"),
    v5dSha256: createHash("sha256").update(v5dSource).digest("hex"),
    freshValidationAccessed: false,
    auditAccessed: false,
  },
  method: {
    models: modelReceipts.length,
    folds: folds.length,
    candidateRegions: new Set(modelReceipts.map((row) => row.region)).size,
    importance:
      "mean absolute standardized coefficient across 20 cross-fitted models, normalized to sum to one",
    riskApproximation:
      "0.65 mean-loss coefficient + 0.35 P90-loss coefficient before output clamping",
    limitation:
      "correlated transformed features share or exchange coefficient importance; these are predictive, not causal, effects",
  },
  modelReceipts,
  meanLoss,
  p90Loss,
  risk,
};
const directory = resolve("research/svi_score_v5d/artifacts");
const artifactPath = resolve(directory, "svi-score-v5d-feature-importance.json");
await mkdir(directory, { recursive: true });
const temporary = `${artifactPath}.tmp`;
await writeFile(temporary, JSON.stringify(artifact, null, 2));
await rename(temporary, artifactPath);
process.stdout.write(`${JSON.stringify({
  artifactPath,
  modelCount: modelReceipts.length,
  riskGroups: risk.groups,
  topRiskFeatures: risk.features.slice(0, 15),
  topMeanFeatures: meanLoss.features.slice(0, 10),
  topP90Features: p90Loss.features.slice(0, 10),
  freshValidationAccessed: false,
  auditAccessed: false,
}, null, 2)}\n`);
