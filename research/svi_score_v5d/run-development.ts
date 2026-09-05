import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { loadSviScoreV4DevelopmentRows } from "../svi_score_v4/development-rows";
import type { SviScoreV5ACrossValidationReceipt } from "../svi_score_v5a/types";
import { evaluateV5D } from "./evaluate";

interface V5AArtifact {
  activation: string;
  provenance: { v3SealedAuditAccessed: boolean };
  crossValidation: SviScoreV5ACrossValidationReceipt;
}

interface V5CArtifact {
  activation: string;
  provenance: {
    auditAccessed: boolean;
    freshValidationAccessed: boolean;
  };
  result: {
    algorithms: Array<{
      algorithm: string;
      primaryObjective: number;
      primary: {
        meanLoss: number;
        cvar90Loss: number;
        worstFamilyMeanLoss: number;
      };
    }>;
  };
}

function relativeReduction(candidate: number, comparator: number): number {
  return (comparator - candidate) / Math.max(Math.abs(comparator), 1e-12);
}

const v5aPath = resolve("research/svi_score_v5a/artifacts/svi-score-v5a-development.json");
const v5cPath = resolve("research/svi_score_v5c/artifacts/svi-score-v5c-development.json");
const [v5aSource, v5cSource] = await Promise.all([
  readFile(v5aPath),
  readFile(v5cPath),
]);
const v5a = JSON.parse(v5aSource.toString("utf8")) as V5AArtifact;
const v5c = JSON.parse(v5cSource.toString("utf8")) as V5CArtifact;
if (
  v5a.activation !== "research-only" ||
  v5c.activation !== "research-only" ||
  v5a.provenance.v3SealedAuditAccessed ||
  v5c.provenance.auditAccessed ||
  v5c.provenance.freshValidationAccessed
) {
  throw new Error("V5D requires frozen research artifacts with the sealed audit closed.");
}
const rows = await loadSviScoreV4DevelopmentRows();
const splitBusinesses = Object.fromEntries(
  [...new Set(rows.map((row) => row.businessId))]
    .map((businessId) => rows.find((row) => row.businessId === businessId)!)
    .reduce<Map<string, number>>((output, row) => {
      output.set(row.split, (output.get(row.split) ?? 0) + 1);
      return output;
    }, new Map()),
);
const result = evaluateV5D(rows, v5a.crossValidation.assignments);
const comparisonVersusV5C = Object.fromEntries(result.algorithms.map((algorithm) => {
  const baseline = v5c.result.algorithms.find(
    (item) => item.algorithm === algorithm.algorithm,
  );
  if (!baseline) throw new Error(`V5C is missing ${algorithm.algorithm}.`);
  return [algorithm.algorithm, {
    v5dObjective: algorithm.primaryObjective,
    v5cObjective: baseline.primaryObjective,
    relativeObjectiveReduction: relativeReduction(
      algorithm.primaryObjective,
      baseline.primaryObjective,
    ),
    relativeMeanLossReduction: relativeReduction(
      algorithm.primary.meanLoss,
      baseline.primary.meanLoss,
    ),
    relativeCvar90Reduction: relativeReduction(
      algorithm.primary.cvar90Loss,
      baseline.primary.cvar90Loss,
    ),
    relativeWorstFamilyReduction: relativeReduction(
      algorithm.primary.worstFamilyMeanLoss,
      baseline.primary.worstFamilyMeanLoss,
    ),
  }];
}));
const artifact = {
  artifactId: "flux-svi-score-v5d-direct-economic-loss-development-v1",
  generatedAt: new Date().toISOString(),
  activation: "research-only",
  stage: "direct-economic-loss-development-complete",
  provenance: {
    v5aSha256: createHash("sha256").update(v5aSource).digest("hex"),
    v5cSha256: createHash("sha256").update(v5cSource).digest("hex"),
    developmentComposition: splitBusinesses,
    priorOpenedValidationReusedAsDevelopment: true,
    freshValidationAccessed: false,
    auditAccessed: false,
  },
  result,
  comparisonVersusV5C,
  conclusion: {
    ...result.acceptance,
    globalOptimumClaimPermitted: false,
    safeToExpandCandidateSpace: result.acceptance.expandedSearchSafe,
  },
  nextGate: result.acceptance.expandedSearchSafe
    ? "Freeze V5D and predeclare a newly generated validation cohort before any candidate-space expansion."
    : "Keep V5D research-only. Diagnose direct-loss calibration and top-of-ranking errors using grouped development cross-validation only.",
};
const directory = resolve("research/svi_score_v5d/artifacts");
const artifactPath = resolve(directory, "svi-score-v5d-development.json");
await mkdir(directory, { recursive: true });
const temporary = `${artifactPath}.tmp`;
await writeFile(temporary, JSON.stringify(artifact, null, 2));
await rename(temporary, artifactPath);
process.stdout.write(`${JSON.stringify({
  artifactPath,
  primaryWinner: result.primaryWinner,
  acceptance: result.acceptance,
  proxyAssessment: result.proxyAssessment,
  primary: Object.fromEntries(result.algorithms.map((algorithm) => [
    algorithm.algorithm,
    {
      objective: algorithm.primaryObjective,
      meanLoss: algorithm.primary.meanLoss,
      cvar90Loss: algorithm.primary.cvar90Loss,
      worstFamilyMeanLoss: algorithm.primary.worstFamilyMeanLoss,
      riskChampionRecall: algorithm.primary.fullPoolChampionRecall,
      oracleRecall: algorithm.primary.economicOracleRecall,
      predictedMeanLoss: algorithm.primary.meanPredictedLoss,
      predictedP90Loss: algorithm.primary.meanPredictedP90Loss,
    },
  ])),
  comparisonVersusV5C,
  freshValidationAccessed: false,
  auditAccessed: false,
}, null, 2)}\n`);
