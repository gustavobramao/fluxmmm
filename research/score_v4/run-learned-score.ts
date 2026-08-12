import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { SIMULATOR_VERSION } from "../score_v2/simulator";
import { GENERATOR_FAMILIES } from "../score_v3/population";
import { generateLearnedScoreCohort } from "./cohort";
import {
  evaluateWeights,
  HEURISTIC_WEIGHTS,
  learnWeights,
} from "./learn";
import type {
  LearnedScoreArtifact,
  LearnedScoreCandidateRow,
} from "./types";

function relativeReduction(baseline: number, learned: number): number {
  return (baseline - learned) / Math.max(baseline, 1e-9);
}

const directory = resolve("research/score_v4/artifacts");
await mkdir(directory, { recursive: true });
const cohortPath = resolve(directory, "cohort.json");
const cohort = process.argv.includes("--reuse-cohort")
  ? JSON.parse(await readFile(cohortPath, "utf8")) as LearnedScoreCandidateRow[]
  : await generateLearnedScoreCohort((detail) => {
      process.stdout.write(`\r${detail.padEnd(80)}`);
    });
if (!process.argv.includes("--reuse-cohort")) process.stdout.write("\n");

const bySplit = (split: LearnedScoreCandidateRow["split"]) =>
  cohort.filter((row) => row.split === split);
const fitted = learnWeights(bySplit("train"), bySplit("validation"));
const performance = Object.fromEntries(
  (["train", "validation", "audit"] as const).map((split) => {
    const rows = bySplit(split);
    const heuristic = evaluateWeights(rows, HEURISTIC_WEIGHTS);
    const learned = evaluateWeights(rows, fitted.weights);
    return [split, {
      heuristic,
      learned,
      relativeMeanRegretReduction: relativeReduction(
        heuristic.meanRegret,
        learned.meanRegret,
      ),
    }];
  }),
) as LearnedScoreArtifact["performance"];
const validationImproves =
  performance.validation.relativeMeanRegretReduction >= 0.02;
const auditNonInferior =
  performance.audit.learned.meanRegret <=
    performance.audit.heuristic.meanRegret * 1.02 &&
  performance.audit.learned.p90Regret <=
    performance.audit.heuristic.p90Regret * 1.02;
const validationTailNonInferior =
  performance.validation.learned.p90Regret <=
  performance.validation.heuristic.p90Regret * 1.000001;
const validationFamilyTolerance = Object.entries(
  performance.validation.heuristic.familyMeanRegret,
).every(
  ([family, regret]) =>
    (performance.validation.learned.familyMeanRegret[family] ?? Infinity) <=
    regret * 1.01,
);
const activation =
  validationImproves &&
  validationTailNonInferior &&
  auditNonInferior &&
  validationFamilyTolerance
    ? "active"
    : "fallback";
const artifact: LearnedScoreArtifact = {
  artifactId: "flux-learned-score-v4-2026.08.2-risk-aware",
  version: "flux-score-learner-v4.1.0",
  trainedAt: new Date().toISOString(),
  activation,
  activationReason:
    activation === "active"
      ? "At least 2% lower family-held-out mean regret, no held-out P90 regression, no validation-family regression above 1%, and no material untouched-audit regression."
      : "The learned model did not clear the mean-regret, P90 tail-risk, family-tolerance, and untouched-audit gates; Flux retains the heuristic score.",
  simulatorVersion: SIMULATOR_VERSION,
  cohort: {
    businesses: new Set(cohort.map((row) => row.businessId)).size,
    candidates: cohort.length,
    splits: {
      train: new Set(bySplit("train").map((row) => row.businessId)).size,
      validation: new Set(bySplit("validation").map((row) => row.businessId)).size,
      audit: new Set(bySplit("audit").map((row) => row.businessId)).size,
    },
    generatorFamilies: {
      train: GENERATOR_FAMILIES.filter((family) => family.split === "train").map((family) => family.id),
      validation: GENERATOR_FAMILIES.filter((family) => family.split === "validation").map((family) => family.id),
      audit: GENERATOR_FAMILIES.filter((family) => family.split === "audit").map((family) => family.id),
    },
  },
  model: {
    family: "constrained-geometric-ensemble",
    target: "mean-four-decision-profit-regret",
    weights: fitted.weights,
    heuristicWeights: HEURISTIC_WEIGHTS,
    minimumLayerWeight: 0.05,
    gridStep: 0.025,
    candidatesConsidered: fitted.gridSize,
    formula: "100 × G^wG × S^wS × C^wC × D^wD",
  },
  performance,
  guardrails: [
    "Validation, evidence-coherence, and ROI plausibility gates remain immutable.",
    "Candidate ranking is learned only within the zero-failed-gate pool; if no such candidate exists, the score ranks review candidates without changing their status.",
    "Truth labels never enter a real advertiser fit; they train this versioned offline artifact only.",
    "Generator families, rather than random candidate rows, define train, validation, and audit splits.",
    "Activation requires held-out P90 regret not to exceed the heuristic and allows at most 1% relative mean-regret tolerance in any validation family.",
    "The fixed heuristic remains the runtime fallback whenever the learned artifact is absent or fails activation.",
  ],
};

await Promise.all([
  writeFile(cohortPath, JSON.stringify(cohort, null, 2)),
  writeFile(resolve(directory, "learned-score-v4.json"), JSON.stringify(artifact, null, 2)),
]);
process.stdout.write(`${JSON.stringify(artifact, null, 2)}\n`);
