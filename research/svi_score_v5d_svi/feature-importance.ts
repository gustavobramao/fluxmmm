import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { SCORE_DIAGNOSTIC_GROUPS } from "../../lib/mmm/score-diagnostics";
import { loadSviScoreV4DevelopmentRows } from "../svi_score_v4/development-rows";
import type { SviScoreV5ACrossValidationReceipt } from "../svi_score_v5a/types";
import { v5bCandidateUniverse } from "../svi_score_v5b/embedding";
import { v5cCandidateRegions } from "../svi_score_v5c/crossfit";
import { fitV5DCrossFittedModel } from "../svi_score_v5d/crossfit";
import {
  loadV5DSviPosteriorRecords,
  v5dSviFeatureProvider,
  v5dSviSelectionPolicy,
} from "./posterior";

type ImportanceGroup =
  | "generalization"
  | "structure"
  | "causal-robustness"
  | "decision-coherence"
  | "temporal-evidence"
  | "posterior-reliability"
  | "posterior-predictive"
  | "posterior-decision-safety"
  | "posterior-roi"
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

const DIAGNOSTIC_DESCRIPTIONS: Record<string, string> = {
  "rolling-oos": "Accuracy on rolling future holdouts.",
  "predictive-coverage": "How often predictive intervals cover observed outcomes.",
  "fold-stability": "Consistency of performance across temporal folds.",
  "spend-regimes": "Generalization across low- and high-spend periods.",
  "residual-independence": "Absence of systematic time or regressor patterns in residuals.",
  "variance-structure": "Stability of residual variance across fitted values.",
  "likelihood-shape": "Agreement between residual behavior and the selected likelihood.",
  multicollinearity: "Whether media effects are separately identifiable under correlation.",
  "functional-form": "Adequacy of the response-curve and baseline functional form.",
  "influence-stability": "Sensitivity to influential observations.",
  "non-negative-boundary": "Reliance on the zero-effect boundary or coefficient clipping.",
  "anchor-recovery": "Recovery of held-out experimental evidence.",
  "new-data-stability": "Stability of estimates as new observations arrive.",
  "confounder-sensitivity": "Sensitivity to plausible unobserved demand confounding.",
  "future-media-placebo": "Whether future media spuriously explains current outcomes.",
  "roi-posterior-plausibility": "Posterior ROI mass inside evidence-informed plausible ranges.",
  "roi-decision-stability": "Stability of budget decisions under ROI uncertainty.",
  "roi-resolution": "Ability to distinguish economically different channel ROIs.",
  "roi-identification": "Strength of information identifying channel-level ROI.",
  "evidence-source-quality": "Quality and transportability of external evidence.",
  "evidence-compatibility": "Agreement between observational data and external evidence.",
  "evidence-decision-dependence": "How much decisions depend on a particular evidence source.",
  "roi-economic-consistency": "Consistency between ROI estimates and marginal-return economics.",
  "whole-flight-generalization": "Performance when complete media flights are held out.",
  "carryover-support": "Data support for the estimated persistence of media effects.",
  "post-flight-residual-stability": "Residual behavior after campaigns stop.",
  "kernel-distinguishability": "Ability to distinguish competing carryover kernels.",
};

const SVI_DESCRIPTIONS: Record<string, string> = {
  "svi:status:labelled": "Whether the posterior fit produced a usable economic-loss label.",
  "svi:convergence:finite": "Whether all posterior summaries are finite.",
  "svi:convergence:elbo-stable": "Whether the variational objective stabilized.",
  "svi:convergence:seed-agreement": "Whether independent SVI seeds agree on channel ROI.",
  "svi:convergence:adjudication-used": "Whether an additional fit was needed to resolve seed disagreement.",
  "svi:convergence:log-elbo-drift": "Log-scaled remaining drift in the variational objective.",
  "svi:convergence:seed-log-roi-difference": "Largest cross-seed difference in log ROI.",
  "svi:predictive:coverage": "Observed share covered by the SVI posterior predictive interval.",
  "svi:decision:implausible-probability": "Largest posterior probability of an implausible channel ROI.",
  "svi:decision:log-relative-roi-width": "Log-scaled largest ROI interval width relative to its center.",
  "svi:margin:predictive-coverage": "Distance above or below the posterior predictive coverage gate.",
  "svi:margin:roi-plausibility": "Distance inside or outside the posterior ROI plausibility gate.",
  "svi:margin:roi-precision": "Distance inside or outside the posterior ROI precision gate.",
  "svi:roi:log-center:paid-social": "Log posterior ROI center for paid social.",
  "svi:roi:log-center:nonbrand-search": "Log posterior ROI center for non-brand search.",
  "svi:roi:log-center:ctv": "Log posterior ROI center for connected TV.",
  "svi:roi:log-center:mean": "Mean log posterior ROI center across modeled channels.",
  "svi:roi:log-center:standard-deviation": "Dispersion of log posterior ROI centers across channels.",
  "svi:roi:log-center:minimum": "Lowest channel log posterior ROI center.",
  "svi:roi:log-center:maximum": "Highest channel log posterior ROI center.",
  "svi:roi:log-center:range": "Range between highest and lowest channel log posterior ROI centers.",
};

function diagnosticName(feature: string): string | undefined {
  if (feature.startsWith("diagnostic:")) return feature.slice("diagnostic:".length);
  if (feature.startsWith("deficit:")) return feature.slice("deficit:".length).split(":")[0];
  return undefined;
}

function featureGroup(feature: string): ImportanceGroup {
  if (feature.startsWith("svi:convergence:") || feature.startsWith("svi:status:")) {
    return "posterior-reliability";
  }
  if (feature.startsWith("svi:predictive:")) return "posterior-predictive";
  if (feature.startsWith("svi:decision:") || feature.startsWith("svi:margin:")) {
    return "posterior-decision-safety";
  }
  if (feature.startsWith("svi:roi:")) return "posterior-roi";
  if (feature.startsWith("context:temporal:")) return "temporal-evidence";
  if (feature.startsWith("spec:") || feature.startsWith("interaction:")) {
    return "model-specification-and-interactions";
  }
  const diagnostic = diagnosticName(feature);
  if (diagnostic && TEMPORAL_DIAGNOSTICS.has(diagnostic)) return "temporal-evidence";
  return DIAGNOSTIC_GROUP.get(diagnostic ?? "") ??
    "model-specification-and-interactions";
}

function sentence(value: string): string {
  return `${value.charAt(0).toUpperCase()}${value.slice(1).replaceAll("-", " ")}.`;
}

function featureDescription(feature: string): string {
  if (SVI_DESCRIPTIONS[feature]) return SVI_DESCRIPTIONS[feature];
  const diagnostic = diagnosticName(feature);
  if (feature.startsWith("diagnostic:") && diagnostic) {
    return DIAGNOSTIC_DESCRIPTIONS[diagnostic] ?? sentence(diagnostic);
  }
  if (feature.startsWith("deficit:") && diagnostic) {
    const threshold = feature.split(":").at(-1);
    const detail = DIAGNOSTIC_DESCRIPTIONS[diagnostic]?.replace(/\.$/, "").toLowerCase() ??
      diagnostic.replaceAll("-", " ");
    return `Shortfall below ${threshold}/100 for ${detail}.`;
  }
  if (feature.startsWith("context:temporal:spec:")) {
    return `Interaction between temporal identification and the ${feature.slice("context:temporal:spec:".length).replaceAll(":", " / ").replaceAll("-", " ")} specification.`;
  }
  if (feature.startsWith("context:temporal:")) {
    return `Temporal-identification interaction for ${feature.slice("context:temporal:".length).replaceAll(":", " / ").replaceAll("-", " ")}.`;
  }
  if (feature.startsWith("interaction:")) {
    return `Interaction between ${feature.slice("interaction:".length).replaceAll(":", " and ").replaceAll("-", " ")}.`;
  }
  if (feature.startsWith("spec:")) {
    return `Indicator for the ${feature.slice("spec:".length).replaceAll(":", " / ").replaceAll("-", " ")} model specification.`;
  }
  return sentence(feature.replaceAll(":", " "));
}

function average(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function summary(values: readonly CoefficientRow[], key: "mean" | "p90" | "risk") {
  const byFeature = new Map<string, CoefficientRow[]>();
  values.forEach((row) => byFeature.set(row.feature, [...(byFeature.get(row.feature) ?? []), row]));
  const rawFeatures = [...byFeature.entries()].map(([feature, rows]) => {
    const coefficients = rows.map((row) => row[key]);
    const meanCoefficient = average(coefficients);
    const positive = coefficients.filter((value) => value > 1e-12).length;
    const negative = coefficients.filter((value) => value < -1e-12).length;
    return {
      feature,
      description: featureDescription(feature),
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
    .map(({ rawImportance, ...row }) => ({ ...row, importance: rawImportance / Math.max(total, 1e-12) }))
    .sort((left, right) => right.importance - left.importance || left.feature.localeCompare(right.feature));
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
const releasePath = resolve("research/svi_score_v5d_svi/artifacts/svi-score-v5d-svi-development.json");
const [v5aSource, releaseSource] = await Promise.all([readFile(v5aPath), readFile(releasePath)]);
const v5a = JSON.parse(v5aSource.toString("utf8")) as V5AArtifact;
if (v5a.provenance.v3SealedAuditAccessed) throw new Error("V5D-SVI importance cannot access the sealed audit.");
const rows = await loadSviScoreV4DevelopmentRows();
if (rows.some((row) => row.split === "audit")) throw new Error("V5D-SVI importance received an audit row.");
const records = await loadV5DSviPosteriorRecords(rows);
const provider = v5dSviFeatureProvider(records);
const policy = v5dSviSelectionPolicy(records);
const foldByBusiness = new Map(v5a.crossValidation.assignments.map((row) => [row.businessId, row.fold]));
const universe = v5bCandidateUniverse(rows.map((row) => row.candidateId));
const regions = v5cCandidateRegions(universe.map((row) => row.candidateId));
const coefficients: CoefficientRow[] = [];
const modelReceipts: Array<{ fold: number; region: number; trainingRows: number; trainingBusinesses: number }> = [];
const folds = [...new Set(v5a.crossValidation.assignments.map((row) => row.fold))].sort((left, right) => left - right);
for (const fold of folds) {
  const training = rows.filter((row) => foldByBusiness.get(row.businessId) !== fold);
  const fitted = fitV5DCrossFittedModel(training, regions, provider, policy);
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
      coefficients.push({ feature, group: featureGroup(feature), mean, p90, risk: 0.65 * mean + 0.35 * p90 });
    });
  });
}
const meanLoss = summary(coefficients, "mean");
const p90Loss = summary(coefficients, "p90");
const risk = summary(coefficients, "risk");
const artifact = {
  artifactId: "flux-svi-score-v5d-svi-standardized-coefficient-importance-v1",
  releaseName: "V5D-SVI",
  generatedAt: new Date().toISOString(),
  activation: "public-research-release",
  provenance: {
    v5aSha256: createHash("sha256").update(v5aSource).digest("hex"),
    releaseSha256: createHash("sha256").update(releaseSource).digest("hex"),
    freshValidationAccessed: false,
    auditAccessed: false,
  },
  method: {
    models: modelReceipts.length,
    folds: folds.length,
    candidateRegions: new Set(modelReceipts.map((row) => row.region)).size,
    importance: "mean absolute standardized coefficient across 20 cross-fitted models, normalized to sum to one",
    riskApproximation: "0.65 mean-loss coefficient + 0.35 P90-loss coefficient before output clamping",
    interpretation: "positive coefficients raise predicted economic loss; negative coefficients reduce predicted economic loss",
    limitation: "correlated transformed features share or exchange coefficient importance; these are predictive associations, not causal effects",
  },
  modelReceipts,
  meanLoss,
  p90Loss,
  risk,
  top50RiskFeatures: risk.features.slice(0, 50).map((row, index) => ({ rank: index + 1, ...row })),
};
const directory = resolve("research/svi_score_v5d_svi/artifacts");
const artifactPath = resolve(directory, "svi-score-v5d-svi-feature-importance.json");
const reportPath = resolve("research/svi_score_v5d_svi/FEATURE_IMPORTANCE.md");
await mkdir(directory, { recursive: true });
const temporary = `${artifactPath}.tmp`;
await writeFile(temporary, JSON.stringify(artifact, null, 2));
await rename(temporary, artifactPath);
const report = [
  "# V5D-SVI learned feature importance",
  "",
  "This is the frozen public V5D-SVI selector. Importance is the mean absolute standardized risk coefficient across 20 cross-fitted models and is normalized across all features. The signed coefficient is conditional on every other transformed feature in the model: positive predicts more economic loss; negative predicts less.",
  "",
  "> These are predictive associations, not causal effects. Correlated features may share or exchange weight, so direction should not be read as a standalone intervention claim.",
  "",
  "## Feature-family importance",
  "",
  "| Rank | Feature family | Importance |",
  "|---:|---|---:|",
  ...risk.groups.map((row, index) =>
    `| ${index + 1} | ${row.group} | ${(row.importance * 100).toFixed(2)}% |`
  ),
  "",
  "## Top 50 learned features",
  "",
  "| Rank | Feature | Family | Importance | Coefficient | Stability | Description |",
  "|---:|---|---|---:|---:|---:|---|",
  ...artifact.top50RiskFeatures.map((row) =>
    `| ${row.rank} | \`${row.feature}\` | ${row.group} | ${(row.importance * 100).toFixed(2)}% | ${row.meanCoefficient >= 0 ? "+" : ""}${row.meanCoefficient.toFixed(3)} | ${(row.signStability * 100).toFixed(0)}% | ${row.description} |`
  ),
  "",
  "## Interpretation contract",
  "",
  "- Importance ranks how strongly a feature contributes to predictions after standardization; it is not a business-controlled weight.",
  "- Coefficient sign is conditional on the other features. A counterintuitive sign can arise from interactions or correlated features.",
  "- Stability is the share of the 20 cross-fitted models that agree with the majority sign.",
  "- V5D-SVI combines predicted mean and tail loss as `0.65 × mean loss + 0.35 × P90 loss`.",
  "- The model and results are frozen as evaluated; this report changes no fit, score, gate, or selection behavior.",
  "",
].join("\n");
await writeFile(reportPath, report);
process.stdout.write(`${JSON.stringify({
  artifactPath,
  reportPath,
  modelCount: modelReceipts.length,
  riskGroups: risk.groups,
  top50RiskFeatures: artifact.top50RiskFeatures,
  freshValidationAccessed: false,
  auditAccessed: false,
}, null, 2)}\n`);
