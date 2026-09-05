import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { loadSviScoreV4DevelopmentRows } from "../svi_score_v4/development-rows";
import type { SviScoreV5ACrossValidationReceipt } from "../svi_score_v5a/types";
import { evaluateV5C } from "./evaluate";

interface V5AArtifact {
  version: string;
  activation: string;
  provenance: { v3SealedAuditAccessed: boolean };
  crossValidation: SviScoreV5ACrossValidationReceipt;
}

interface V5BArtifact {
  activation: string;
  conclusion: {
    auditAccessed: boolean;
    optimizerPressureDetected: boolean;
  };
  primaryComparison: Record<string, {
    objective: number;
    meanLoss: number;
    cvar90Loss: number;
    worstFamilyMeanLoss: number;
  }>;
}

function relativeReduction(candidate: number, comparator: number): number {
  return (comparator - candidate) / Math.max(Math.abs(comparator), 1e-12);
}

const paths = {
  v5a: resolve("research/svi_score_v5a/artifacts/svi-score-v5a-development.json"),
  v5b: resolve("research/svi_score_v5b/artifacts/svi-score-v5b-development.json"),
};
const [v5aSource, v5bSource] = await Promise.all([
  readFile(paths.v5a),
  readFile(paths.v5b),
]);
const v5a = JSON.parse(v5aSource.toString("utf8")) as V5AArtifact;
const v5b = JSON.parse(v5bSource.toString("utf8")) as V5BArtifact;
if (
  v5a.activation !== "research-only" ||
  v5b.activation !== "research-only" ||
  v5a.provenance.v3SealedAuditAccessed ||
  v5b.conclusion.auditAccessed ||
  !v5b.conclusion.optimizerPressureDetected
) {
  throw new Error("V5C requires sealed V5A/V5B research artifacts and detected optimizer pressure.");
}
const rows = await loadSviScoreV4DevelopmentRows();
const result = evaluateV5C(rows, v5a.crossValidation.assignments);
const comparisonVersusV5B = Object.fromEntries(result.algorithms.map((algorithm) => {
  const baseline = v5b.primaryComparison[algorithm.algorithm];
  if (!baseline) throw new Error(`V5B is missing ${algorithm.algorithm}.`);
  return [
    algorithm.algorithm,
    {
      v5cObjective: algorithm.primaryObjective,
      v5bObjective: baseline.objective,
      relativeObjectiveReduction: relativeReduction(
        algorithm.primaryObjective,
        baseline.objective,
      ),
      relativeMeanLossReduction: relativeReduction(
        algorithm.primary.meanLoss,
        baseline.meanLoss,
      ),
      relativeCvar90Reduction: relativeReduction(
        algorithm.primary.cvar90Loss,
        baseline.cvar90Loss,
      ),
      relativeWorstFamilyReduction: relativeReduction(
        algorithm.primary.worstFamilyMeanLoss,
        baseline.worstFamilyMeanLoss,
      ),
    },
  ];
}));
const artifact = {
  artifactId: "flux-svi-score-v5c-search-robust-development-v1",
  generatedAt: new Date().toISOString(),
  activation: "research-only",
  stage: "search-robust-two-axis-development-complete",
  provenance: {
    v5aSha256: createHash("sha256").update(v5aSource).digest("hex"),
    v5bSha256: createHash("sha256").update(v5bSource).digest("hex"),
    auditAccessed: false,
    freshValidationAccessed: false,
  },
  result,
  comparisonVersusV5B,
  conclusion: {
    ...result.acceptance,
    globalOptimumClaimPermitted: false,
    safeToExpandCandidateSpace: result.acceptance.expandedSearchSafe,
  },
  nextGate: result.acceptance.expandedSearchSafe
    ? "Freeze V5C and predeclare a fresh, neutral expanded-universe experiment before fitting any new candidates. Keep the sealed audit closed."
    : "Do not expand candidate space. Keep all work in development and diagnose the remaining selector/search mismatch without opening validation or audit.",
};
const directory = resolve("research/svi_score_v5c/artifacts");
const artifactPath = resolve(directory, "svi-score-v5c-development.json");
await mkdir(directory, { recursive: true });
const temporary = `${artifactPath}.tmp`;
await writeFile(temporary, JSON.stringify(artifact, null, 2));
await rename(temporary, artifactPath);
process.stdout.write(`${JSON.stringify({
  artifactPath,
  primaryWinner: result.primaryWinner,
  acceptance: result.acceptance,
  primary: Object.fromEntries(result.algorithms.map((algorithm) => [
    algorithm.algorithm,
    {
      objective: algorithm.primaryObjective,
      meanLoss: algorithm.primary.meanLoss,
      cvar90Loss: algorithm.primary.cvar90Loss,
      worstFamilyMeanLoss: algorithm.primary.worstFamilyMeanLoss,
      robustChampionRecall: algorithm.primary.fullPoolChampionRecall,
      oracleRecall: algorithm.primary.economicOracleRecall,
      meanPromotions: algorithm.primary.meanPromotions,
    },
  ])),
  comparisonVersusV5B,
  auditAccessed: false,
}, null, 2)}\n`);
