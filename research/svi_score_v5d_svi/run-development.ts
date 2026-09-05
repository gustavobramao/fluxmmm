import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { loadSviScoreV4DevelopmentRows } from "../svi_score_v4/development-rows";
import type { SviScoreV5ACrossValidationReceipt } from "../svi_score_v5a/types";
import { evaluateV5D, type V5DDevelopmentReceipt } from "../svi_score_v5d/evaluate";
import {
  V5D_MAP_FEATURE_PROVIDER,
  V5D_MAP_SELECTION_POLICY,
} from "../svi_score_v5d/model";
import {
  SVI_SCORE_V5D_SVI_CONTRACT,
  SVI_SCORE_V5D_SVI_VERSION,
} from "./contract";
import {
  loadV5DSviPosteriorRecords,
  v5dSviFeatureProvider,
  v5dSviSelectionPolicy,
} from "./posterior";

interface V5AArtifact {
  activation: string;
  provenance: { v3SealedAuditAccessed: boolean };
  crossValidation: SviScoreV5ACrossValidationReceipt;
}

interface V5DArtifact {
  activation: string;
  provenance: { auditAccessed: boolean; freshValidationAccessed: boolean };
  result: V5DDevelopmentReceipt;
}

function algorithm(
  result: V5DDevelopmentReceipt,
  id: string,
) {
  const receipt = result.algorithms.find((item) => item.algorithm === id);
  if (!receipt) throw new Error(`${result.version} has no ${id} receipt.`);
  return receipt;
}

function primarySummary(result: V5DDevelopmentReceipt) {
  const receipt = algorithm(result, result.primaryWinner);
  return {
    algorithm: receipt.algorithm,
    objective: receipt.primaryObjective,
    meanLoss: receipt.primary.meanLoss,
    p90Loss: receipt.primary.p90Loss,
    cvar90Loss: receipt.primary.cvar90Loss,
    worstFamilyMeanLoss: receipt.primary.worstFamilyMeanLoss,
    oracleRecall: receipt.primary.economicOracleRecall,
    reviewOnlyFallbackShare: receipt.primary.reviewOnlyFallbackShare,
  };
}

function fullPoolSummary(result: V5DDevelopmentReceipt) {
  const receipt = algorithm(result, result.primaryWinner);
  const checkpoint = receipt.checkpoints.find((item) => item.budget === 48);
  if (!checkpoint) throw new Error(`${result.version} has no full-pool checkpoint.`);
  return {
    objective: checkpoint.selectionObjective,
    meanLoss: checkpoint.meanLoss,
    p90Loss: checkpoint.p90Loss,
    cvar90Loss: checkpoint.cvar90Loss,
    worstFamilyMeanLoss: checkpoint.worstFamilyMeanLoss,
    oracleRecall: checkpoint.economicOracleRecall,
    reviewOnlyFallbackShare: checkpoint.reviewOnlyFallbackShare,
  };
}

function relativeChange(candidate: number, baseline: number): number {
  return (candidate - baseline) / Math.max(Math.abs(baseline), 1e-12);
}

function quantile(values: readonly number[], probability: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  if (!sorted.length) return 0;
  const position = Math.max(0, Math.min(1, probability)) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

const v5aPath = resolve("research/svi_score_v5a/artifacts/svi-score-v5a-development.json");
const v5dPath = resolve("research/svi_score_v5d/artifacts/svi-score-v5d-development.json");
const recordPath = resolve(".flux-artifacts/svi-score-v3/design-pilot-records.json");
const [v5aSource, v5dSource, recordSource] = await Promise.all([
  readFile(v5aPath),
  readFile(v5dPath),
  readFile(recordPath),
]);
const v5a = JSON.parse(v5aSource.toString("utf8")) as V5AArtifact;
const v5d = JSON.parse(v5dSource.toString("utf8")) as V5DArtifact;
if (
  v5a.activation !== "research-only" ||
  v5d.activation !== "research-only" ||
  v5a.provenance.v3SealedAuditAccessed ||
  v5d.provenance.auditAccessed ||
  v5d.provenance.freshValidationAccessed
) {
  throw new Error("V5D-SVI requires development-only inputs with fresh validation closed.");
}

const rows = await loadSviScoreV4DevelopmentRows();
const records = await loadV5DSviPosteriorRecords(rows);
const sviFeatures = v5dSviFeatureProvider(records);
const sviSafety = v5dSviSelectionPolicy(records);
const safetyCounts = new Map<string, number>();
let mapSafe = 0;
let sviSafe = 0;
let safeUnderBoth = 0;
rows.forEach((row) => {
  const map = V5D_MAP_SELECTION_POLICY.passes(row);
  const svi = sviSafety.passes(row);
  mapSafe += map ? 1 : 0;
  sviSafe += svi ? 1 : 0;
  safeUnderBoth += map && svi ? 1 : 0;
  safetyCounts.set(row.businessId, (safetyCounts.get(row.businessId) ?? 0) + (svi ? 1 : 0));
});
const safePerBusiness = [...safetyCounts.values()];
const safetyAudit = {
  candidateRows: rows.length,
  mapSafe,
  sviSafe,
  safeUnderBoth,
  mapOnly: mapSafe - safeUnderBoth,
  sviOnly: sviSafe - safeUnderBoth,
  businessesWithNoSviSafeCandidate: safePerBusiness.filter((count) => count === 0).length,
  sviSafeCandidatesPerBusiness: {
    minimum: Math.min(...safePerBusiness),
    median: quantile(safePerBusiness, 0.5),
    p90: quantile(safePerBusiness, 0.9),
    maximum: Math.max(...safePerBusiness),
  },
};

// A 2x2 ablation distinguishes posterior-feature value from the effect of
// replacing the MAP safety pool with a posterior-defined safety pool.
const featureOnly = evaluateV5D(
  rows,
  v5a.crossValidation.assignments,
  sviFeatures,
  V5D_MAP_SELECTION_POLICY,
);
const safetyOnly = evaluateV5D(
  rows,
  v5a.crossValidation.assignments,
  V5D_MAP_FEATURE_PROVIDER,
  sviSafety,
);
const full = evaluateV5D(
  rows,
  v5a.crossValidation.assignments,
  sviFeatures,
  sviSafety,
);

const baselinePrimary = primarySummary(v5d.result);
const fullPrimary = primarySummary(full);
const baselinePool = fullPoolSummary(v5d.result);
const fullPool = fullPoolSummary(full);
const comparison = {
  baseline: {
    primary: baselinePrimary,
    fullPool: baselinePool,
    proxy: v5d.result.proxyAssessment,
  },
  featureOnly: {
    primary: primarySummary(featureOnly),
    fullPool: fullPoolSummary(featureOnly),
    proxy: featureOnly.proxyAssessment,
  },
  safetyOnly: {
    primary: primarySummary(safetyOnly),
    fullPool: fullPoolSummary(safetyOnly),
    proxy: safetyOnly.proxyAssessment,
  },
  fullSvi: {
    primary: fullPrimary,
    fullPool,
    proxy: full.proxyAssessment,
  },
  relativeToV5D: {
    primaryObjective: relativeChange(fullPrimary.objective, baselinePrimary.objective),
    fullPoolObjective: relativeChange(fullPool.objective, baselinePool.objective),
    proxyMae: relativeChange(
      full.proxyAssessment.meanAbsoluteError,
      v5d.result.proxyAssessment.meanAbsoluteError,
    ),
    weightedPairAccuracy: relativeChange(
      full.proxyAssessment.economicallyWeightedPairwiseAccuracy,
      v5d.result.proxyAssessment.economicallyWeightedPairwiseAccuracy,
    ),
    championMeanExcessLoss: relativeChange(
      full.proxyAssessment.riskChampionMeanExcessLoss,
      v5d.result.proxyAssessment.riskChampionMeanExcessLoss,
    ),
    championP90ExcessLoss: relativeChange(
      full.proxyAssessment.riskChampionP90ExcessLoss,
      v5d.result.proxyAssessment.riskChampionP90ExcessLoss,
    ),
  },
};

const artifact = {
  artifactId: "flux-svi-score-v5d-svi-development-v1",
  version: SVI_SCORE_V5D_SVI_VERSION,
  generatedAt: new Date().toISOString(),
  activation: "public-research-release",
  stage: "cached-posterior-development-complete",
  contract: SVI_SCORE_V5D_SVI_CONTRACT,
  provenance: {
    v5aSha256: createHash("sha256").update(v5aSource).digest("hex"),
    v5dSha256: createHash("sha256").update(v5dSource).digest("hex"),
    sviRecordSha256: createHash("sha256").update(recordSource).digest("hex"),
    sviFitsReused: records.size,
    sviFitsRerun: 0,
    freshValidationAccessed: false,
    auditAccessed: false,
  },
  ablation: {
    safetyAudit,
    featureOnly,
    safetyOnly,
    full,
  },
  comparison,
  safetyAudit,
  conclusion: {
    posteriorFeaturesImprovedWeightedPairAccuracy:
      featureOnly.proxyAssessment.economicallyWeightedPairwiseAccuracy >
        v5d.result.proxyAssessment.economicallyWeightedPairwiseAccuracy,
    fullSviImprovedPrimaryObjective:
      fullPrimary.objective < baselinePrimary.objective,
    fullSviImprovedFullPoolObjective:
      fullPool.objective < baselinePool.objective,
    fullSviImprovedChampionMeanLoss:
      full.proxyAssessment.riskChampionMeanExcessLoss <
        v5d.result.proxyAssessment.riskChampionMeanExcessLoss,
    developmentComparisonComplete: true,
    freshGeneralizationEstablished: false,
    publicResearchReleasePermitted: true,
    sotaClaimPermitted: false,
    productionActivationPermitted: false,
  },
};

const directory = resolve("research/svi_score_v5d_svi/artifacts");
const artifactPath = resolve(directory, "svi-score-v5d-svi-development.json");
await mkdir(directory, { recursive: true });
const temporary = `${artifactPath}.tmp`;
await writeFile(temporary, JSON.stringify(artifact, null, 2));
await rename(temporary, artifactPath);
process.stdout.write(`${JSON.stringify({
  artifactPath,
  sviFitsReused: records.size,
  sviFitsRerun: 0,
  comparison,
  conclusion: artifact.conclusion,
}, null, 2)}\n`);
