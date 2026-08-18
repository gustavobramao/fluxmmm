import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { SIMULATOR_VERSION } from "../score_v2/simulator";
import { generateScoreV6Cohort } from "./cohort";
import {
  evaluateScoreV6,
  evaluateV4Heuristic,
  groupWeights,
  learnScoreV6,
  scoreV6,
} from "./learn";
import type { ScoreV6Artifact, ScoreV6CandidateRow } from "./types";

const directory = resolve("research/score_v6/artifacts");
await mkdir(directory, { recursive: true });
const cohortPath = resolve(directory, "cohort-v6.json");
const cohort = process.argv.includes("--reuse-cohort")
  ? JSON.parse(await readFile(cohortPath, "utf8")) as ScoreV6CandidateRow[]
  : await generateScoreV6Cohort((detail) => process.stdout.write(`\r${detail.padEnd(90)}`));
if (!process.argv.includes("--reuse-cohort")) process.stdout.write("\n");
if (!process.argv.includes("--reuse-cohort")) {
  await writeFile(cohortPath, JSON.stringify(cohort, null, 2));
}

const bySplit = (split: ScoreV6CandidateRow["split"]) =>
  cohort.filter((row) => row.split === split);
const fitted = learnScoreV6(bySplit("train"));
const performance = Object.fromEntries(
  (["train", "validation", "audit"] as const).map((split) => {
    const rows = bySplit(split);
    const heuristic = evaluateV4Heuristic(rows);
    const learned = evaluateScoreV6(rows, (row) => scoreV6(row, fitted.weights));
    return [split, {
      heuristic,
      learned,
      relativeMeanLossReduction:
        (heuristic.meanLoss - learned.meanLoss) / Math.max(heuristic.meanLoss, 1e-9),
    }];
  }),
) as ScoreV6Artifact["performance"];

const validationImproves = performance.validation.relativeMeanLossReduction >= 0.05;
const validationTailSafe = performance.validation.learned.p90Loss <=
  performance.validation.heuristic.p90Loss * 1.000001;
const auditMeanSafe = performance.audit.learned.meanLoss <=
  performance.audit.heuristic.meanLoss * 1.02;
const auditTailSafe = performance.audit.learned.p90Loss <=
  performance.audit.heuristic.p90Loss * 1.02;
const familySafe = Object.entries(performance.validation.heuristic.familyMeanLoss).every(
  ([family, loss]) =>
    (performance.validation.learned.familyMeanLoss[family] ?? Infinity) <= loss * 1.02,
);
const requiredDecisionGradeShare = 0.5;
const decisionGradeBusinessShare =
  performance.validation.learned.decisionGradeBusinessShare;
const pipelineReady = decisionGradeBusinessShare >= requiredDecisionGradeShare;
const gates = {
  validationImproves,
  validationTailSafe,
  auditMeanSafe,
  auditTailSafe,
  validationFamilySafe: familySafe,
};
const activation = Object.values(gates).every(Boolean) ? "active" : "rejected";
const artifact: ScoreV6Artifact = {
  artifactId: "flux-learned-score-v6-pilot-2026.08.1",
  version: "flux-score-learner-v6.0.0-pilot",
  trainedAt: new Date().toISOString(),
  activation,
  activationReason: activation === "active"
    ? "V6 cleared the predeclared held-out ranking, tail, family, and sealed-audit gates and is active for ranking candidates within their immutable eligibility tier."
    : "V6 did not clear every predeclared ranker activation gate; the active runtime score remains unchanged.",
  runtimeContractChanged: activation === "active",
  simulatorVersion: SIMULATOR_VERSION,
  cohort: {
    businesses: new Set(cohort.map((row) => row.businessId)).size,
    candidates: cohort.length,
    pairs: fitted.pairCount,
    splits: {
      train: new Set(bySplit("train").map((row) => row.businessId)).size,
      validation: new Set(bySplit("validation").map((row) => row.businessId)).size,
      audit: new Set(bySplit("audit").map((row) => row.businessId)).size,
    },
  },
  model: {
    family: "monotonic-pairwise-logistic-ranker",
    target: "uncapped-economic-decision-loss",
    weights: fitted.weights,
    groupWeights: groupWeights(fitted.weights),
    regularization: 0.015,
    epochs: 180,
    formula: "100 × exp(Σ diagnostic_weight × log(diagnostic_score / 100))",
  },
  performance,
  gates,
  pipelineReadiness: {
    decisionGradeBusinessShare,
    requiredShare: requiredDecisionGradeShare,
    ready: pipelineReady,
    detail: pipelineReady
      ? "At least half of validation businesses produced a decision-grade candidate."
      : "Candidate generation produced a decision-grade model in fewer than half of validation businesses. This is reported separately because score weights cannot change eligibility.",
  },
  guardrails: [
    "V6 ranks only within the highest available immutable eligibility tier and cannot promote a failed-gate candidate into decision-grade status.",
    "All learned coefficients are non-negative; better diagnostics can never reduce the learned score.",
    "Each validation group retains 10–40% total weight and no diagnostic receives more than 16% before final normalization.",
    "Candidates are compared only within an immutable decision-grade or review eligibility tier.",
    "Decision-grade candidate coverage is a separate search-readiness diagnostic and is not an activation gate for the ranker.",
    "Scenario-specific benchmark evidence is injected into both fitting and validation; hidden ROI truth is never supplied to the estimator.",
    "Generator families, not candidate rows, define train, validation, and untouched audit partitions.",
  ],
};

await Promise.all([
  writeFile(resolve(directory, "learned-score-v6-pilot.json"), JSON.stringify(artifact, null, 2)),
]);
process.stdout.write(`${JSON.stringify(artifact, null, 2)}\n`);
