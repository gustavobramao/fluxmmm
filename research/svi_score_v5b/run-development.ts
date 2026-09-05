import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { loadSviScoreV4DevelopmentRows } from "../svi_score_v4/development-rows";
import type { SviScoreV5ACrossValidationReceipt } from "../svi_score_v5a/types";
import { SVI_SCORE_V5B_CONTRACT } from "./contract";
import { evaluateV5BSearchAlgorithms } from "./evaluate";

interface V5AArtifact {
  version: string;
  stage: string;
  activation: string;
  provenance: { v3SealedAuditAccessed: boolean };
  crossValidation: SviScoreV5ACrossValidationReceipt;
}

function relativeReduction(candidate: number, comparator: number): number {
  return (comparator - candidate) / Math.max(Math.abs(comparator), 1e-12);
}

const v5aPath = resolve(
  "research/svi_score_v5a/artifacts/svi-score-v5a-development.json",
);
const source = await readFile(v5aPath);
const v5a = JSON.parse(source.toString("utf8")) as V5AArtifact;
if (
  v5a.version !== SVI_SCORE_V5B_CONTRACT.selector.version ||
  v5a.stage !== "selector-development-complete-no-expanded-search" ||
  v5a.activation !== "research-only" ||
  v5a.provenance.v3SealedAuditAccessed
) {
  throw new Error("V5B requires the frozen research-only V5A selector artifact.");
}
const rows = await loadSviScoreV4DevelopmentRows();
const result = evaluateV5BSearchAlgorithms(
  rows,
  v5a.crossValidation.assignments,
  v5a.crossValidation.selectedConfiguration,
);
const fixed = result.algorithms.find(
  (algorithm) => algorithm.algorithm === "fixed-coverage",
);
const winner = result.algorithms.find(
  (algorithm) => algorithm.algorithm === result.primaryWinner,
);
if (!fixed || !winner) throw new Error("V5B comparison is incomplete.");
const primaryComparison = Object.fromEntries(
  result.algorithms.map((algorithm) => [
    algorithm.algorithm,
    {
      objective: algorithm.primaryObjective,
      relativeObjectiveReductionVersusFixed: relativeReduction(
        algorithm.primaryObjective,
        fixed.primaryObjective,
      ),
      meanLoss: algorithm.primary.meanLoss,
      cvar90Loss: algorithm.primary.cvar90Loss,
      worstFamilyMeanLoss: algorithm.primary.worstFamilyMeanLoss,
      fullPoolChampionRecall: algorithm.primary.fullPoolChampionRecall,
      economicOracleRecall: algorithm.primary.economicOracleRecall,
    },
  ]),
);
const adaptive = result.algorithms.filter((algorithm) =>
  algorithm.algorithm === "gp-ucb" ||
  algorithm.algorithm === "hybrid-global-local"
);
const optimizerPressureDetected = adaptive.some((algorithm) =>
  algorithm.primary.fullPoolChampionRecall >
      fixed.primary.fullPoolChampionRecall + 0.2 &&
  algorithm.primaryObjective > fixed.primaryObjective
);
const fullPool = fixed.checkpoints.find((checkpoint) => checkpoint.budget === 48);
if (!fullPool) throw new Error("V5B requires its full-pool convergence checkpoint.");
const descriptiveBestCheckpoint = result.algorithms.flatMap((algorithm) =>
  algorithm.checkpoints.map((checkpoint) => ({
    algorithm: algorithm.algorithm,
    budget: checkpoint.budget,
    objective: checkpoint.selectionObjective,
  }))
).sort(
  (left, right) =>
    left.objective - right.objective ||
    left.budget - right.budget ||
    left.algorithm.localeCompare(right.algorithm),
)[0];
const artifact = {
  artifactId: "flux-svi-score-v5b-search-development-v1",
  generatedAt: new Date().toISOString(),
  activation: "research-only",
  stage: "equal-compute-search-development-complete",
  frozenSelector: {
    artifact: v5aPath,
    sha256: createHash("sha256").update(source).digest("hex"),
    version: v5a.version,
    selectedConfiguration: v5a.crossValidation.selectedConfiguration,
    refitDuringSearch: false,
  },
  result,
  primaryComparison,
  conclusion: {
    winner: result.primaryWinner,
    relativeObjectiveReductionVersusFixed: relativeReduction(
      winner.primaryObjective,
      fixed.primaryObjective,
    ),
    globalOptimumClaimPermitted: false,
    optimizerPressureDetected,
    adaptiveSearchSafeToPromote: !optimizerPressureDetected,
    fullPoolObjective: fullPool.selectionObjective,
    descriptiveBestCheckpoint,
    auditAccessed: false,
  },
  nextGate: optimizerPressureDetected
    ? "Do not expand the search space yet. Diagnose selector robustness under adaptive optimization pressure, including the proposal-phase features, using development data only; keep validation and audit sealed."
    : "Freeze the V5B search policy before generating an expanded candidate universe. The 48-candidate replay does not establish a continuous-space global optimum.",
};
const directory = resolve("research/svi_score_v5b/artifacts");
const artifactPath = resolve(directory, "svi-score-v5b-development.json");
await mkdir(directory, { recursive: true });
const temporary = `${artifactPath}.tmp`;
await writeFile(temporary, JSON.stringify(artifact, null, 2));
await rename(temporary, artifactPath);
process.stdout.write(`${JSON.stringify({
  artifactPath,
  primaryBudget: SVI_SCORE_V5B_CONTRACT.compute.primaryEvaluationsPerBusiness,
  primaryWinner: result.primaryWinner,
  primaryComparison,
  auditAccessed: false,
  globalOptimumClaimPermitted: false,
  optimizerPressureDetected,
  adaptiveSearchSafeToPromote: !optimizerPressureDetected,
}, null, 2)}\n`);
