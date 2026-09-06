import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  SVI_SCORE_V11_EA_CONTRACT,
  SVI_SCORE_V11_EA_VERSION,
  V11_EA_CONTEXT_NAMES,
  V11_EA_EXPERTS,
  V11_EA_PAIRED_METRICS,
  V11_EA_REGIMES,
  V11_EA_SUMMARY_METRICS,
  type V11EaRegime,
  v11EaExpertForToken,
} from "./contract";

const INPUT_PATH = resolve(".flux-artifacts/svi-score-v9/development-dataset.json");
const OUTPUT_PATH = resolve(
  ".flux-artifacts/svi-score-v11-evidence-adaptive/development-dataset.json",
);

interface CandidateRow {
  businessId: string;
  family: string;
  candidateId: string;
  fold: number;
  features: number[];
  valid: boolean;
  validityReasons: string[];
  scenarioLoss: Record<string, number>;
  economicRisk: number;
  excessEconomicRisk: number;
  normalizedExcessEconomicRisk: number;
  dangerous: boolean;
}

interface V9Dataset {
  version: string;
  activation: string;
  provenance: { v8AuditDereferenced: boolean; nutsUsed: boolean };
  featureNames: string[];
  rows: CandidateRow[];
}

const sha256 = (source: Buffer | string) =>
  createHash("sha256").update(source).digest("hex");

const average = (values: readonly number[]) =>
  values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);

const standardDeviation = (values: readonly number[]) => {
  const center = average(values);
  return Math.sqrt(average(values.map((value) => (value - center) ** 2)));
};

const finite = (value: unknown) => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`Non-finite context value: ${value}`);
  return parsed;
};

function arm(candidateId: string): Exclude<V11EaRegime, "full-candidate-set"> {
  const value = candidateId.split(" · ").at(-1);
  if (value !== "experiments-only" && value !== "benchmark-gap-fill") {
    throw new Error(`Unknown cached evidence arm: ${candidateId}`);
  }
  return value;
}

function baseCandidate(candidateId: string): string {
  return candidateId.split(" · ")[0];
}

function summary(
  rows: readonly CandidateRow[],
  index: number,
): [number, number, number] {
  const values = rows.map((row) => finite(row.features[index]));
  return [average(values), standardDeviation(values), Math.max(...values)];
}

function contextFor(
  regime: V11EaRegime,
  activeRows: readonly CandidateRow[],
  allRows: readonly CandidateRow[],
  featureIndex: ReadonlyMap<string, number>,
): number[] {
  const experimentMean = featureIndex.get(
    "posterior:channel-set:evidence-experiment:mean",
  );
  const benchmarkMean = featureIndex.get(
    "posterior:channel-set:evidence-benchmark:mean",
  );
  if (experimentMean === undefined || benchmarkMean === undefined) {
    throw new Error("Evidence-source posterior tokens are missing.");
  }
  const experimentCoverage = Math.max(
    0,
    ...activeRows.map((row) => finite(row.features[experimentMean])),
  );
  const benchmarkCoverage = Math.max(
    0,
    ...activeRows.map((row) => finite(row.features[benchmarkMean])),
  );
  const experimentAvailable = experimentCoverage > 1e-12 ? 1 : 0;
  const benchmarkAvailable = benchmarkCoverage > 1e-12 ? 1 : 0;
  const externalAvailable = experimentAvailable || benchmarkAvailable ? 1 : 0;
  const values = [
    experimentAvailable,
    experimentCoverage,
    benchmarkAvailable,
    benchmarkCoverage,
    externalAvailable,
    Math.min(1, experimentCoverage + benchmarkCoverage),
    externalAvailable,
    externalAvailable,
    externalAvailable,
    regime === "full-candidate-set" ? 1 : 0,
    regime === "experiments-only" ? 1 : 0,
    regime === "benchmark-gap-fill" ? 1 : 0,
    activeRows.length / SVI_SCORE_V11_EA_CONTRACT.pairedRegimes.fullCandidates,
  ];

  for (const metric of V11_EA_SUMMARY_METRICS) {
    const index = featureIndex.get(metric);
    if (index === undefined) throw new Error(`Missing summary token ${metric}.`);
    const [mean, sd, best] = summary(activeRows, index);
    const isEvidenceMetric = metric.includes("evidence-");
    values.push(
      isEvidenceMetric && !externalAvailable ? 0 : mean,
      isEvidenceMetric && !externalAvailable ? 0 : sd,
      isEvidenceMetric && !externalAvailable ? 0 : best,
    );
  }

  const pairedAvailable = regime === "full-candidate-set" ? 1 : 0;
  values.push(pairedAvailable);
  const byBase = new Map<string, Partial<Record<"experiments-only" | "benchmark-gap-fill", CandidateRow>>>();
  for (const row of allRows) {
    const id = baseCandidate(row.candidateId);
    byBase.set(id, { ...(byBase.get(id) ?? {}), [arm(row.candidateId)]: row });
  }
  if (
    byBase.size !== 24 ||
    [...byBase.values()].some(
      (pair) => !pair["experiments-only"] || !pair["benchmark-gap-fill"],
    )
  ) {
    throw new Error(`Incomplete within-business evidence pairing for ${allRows[0]?.businessId}.`);
  }
  for (const metric of V11_EA_PAIRED_METRICS) {
    const index = featureIndex.get(metric);
    if (index === undefined) throw new Error(`Missing paired token ${metric}.`);
    const differences = [...byBase.values()].map((pair) =>
      finite(pair["benchmark-gap-fill"]!.features[index]) -
        finite(pair["experiments-only"]!.features[index])
    );
    values.push(
      pairedAvailable ? average(differences) : 0,
      pairedAvailable ? standardDeviation(differences) : 0,
    );
  }
  if (
    values.length !== V11_EA_CONTEXT_NAMES.length ||
    values.some((value) => !Number.isFinite(value))
  ) {
    throw new Error(`Context contract failed for ${allRows[0]?.businessId}/${regime}.`);
  }
  return values;
}

const source = await readFile(INPUT_PATH);
const input = JSON.parse(source.toString("utf8")) as V9Dataset;
if (
  input.activation !== "development-only" ||
  input.provenance.v8AuditDereferenced ||
  input.provenance.nutsUsed ||
  input.rows.length !== SVI_SCORE_V11_EA_CONTRACT.source.candidateMmmFits ||
  input.featureNames.length !== SVI_SCORE_V11_EA_CONTRACT.source.frozenV9CandidateTokens
) {
  throw new Error("Evidence-adaptive preparation requires frozen V9 development inputs only.");
}
const featureIndex = new Map(input.featureNames.map((name, index) => [name, index]));
const grouped = new Map<string, CandidateRow[]>();
for (const row of input.rows) {
  grouped.set(row.businessId, [...(grouped.get(row.businessId) ?? []), row]);
}

const sets = [];
for (const [businessId, rows] of [...grouped.entries()].sort()) {
  if (
    rows.length !== 48 ||
    new Set(rows.map((row) => row.fold)).size !== 1 ||
    new Set(rows.map((row) => row.family)).size !== 1
  ) {
    throw new Error(`Invalid candidate set for ${businessId}.`);
  }
  for (const regime of V11_EA_REGIMES) {
    const activeRows = regime === "full-candidate-set"
      ? rows
      : rows.filter((row) => arm(row.candidateId) === regime);
    const validRisks = activeRows
      .filter((row) => row.valid)
      .map((row) => row.economicRisk);
    if (
      activeRows.length !== (regime === "full-candidate-set" ? 48 : 24) ||
      !validRisks.length
    ) {
      throw new Error(`Incomplete ${regime} set for ${businessId}.`);
    }
    sets.push({
      setId: `${businessId}::${regime}`,
      businessId,
      family: rows[0].family,
      fold: rows[0].fold,
      regime,
      candidateIds: activeRows.map((row) => row.candidateId).sort(),
      context: contextFor(regime, activeRows, rows, featureIndex),
      withinRegimeMinimumEconomicRisk: Math.min(...validRisks),
    });
  }
}

const expertGroups = Object.fromEntries(
  V11_EA_EXPERTS.map((expert) => [
    expert,
    input.featureNames
      .map((name, index) => ({ name, index, expert: v11EaExpertForToken(name) }))
      .filter((token) => token.expert === expert)
      .map(({ name, index }) => ({ name, index })),
  ]),
);
const assigned = Object.values(expertGroups).flat();
if (
  grouped.size !== SVI_SCORE_V11_EA_CONTRACT.source.developmentBusinesses ||
  sets.length !== SVI_SCORE_V11_EA_CONTRACT.pairedRegimes.totalDevelopmentSets ||
  assigned.length !== input.featureNames.length ||
  new Set(assigned.map((item) => item.index)).size !== input.featureNames.length
) {
  throw new Error("Evidence-adaptive cohort or expert partition is incomplete.");
}

const artifact = {
  artifactId: "flux-svi-score-v11-evidence-adaptive-development-dataset-v1",
  version: SVI_SCORE_V11_EA_VERSION,
  generatedAt: new Date().toISOString(),
  activation: "development-only",
  contract: SVI_SCORE_V11_EA_CONTRACT,
  provenance: {
    v9DatasetSha256: sha256(source),
    posteriorRefits: 0,
    amssConfirmatoryCohortAccessed: false,
    previousAuditAccessed: false,
    nutsUsed: false,
  },
  featureNames: input.featureNames,
  contextNames: V11_EA_CONTEXT_NAMES,
  expertGroups,
  rows: input.rows,
  sets,
  summary: {
    businesses: grouped.size,
    candidateMmmFits: input.rows.length,
    developmentSets: sets.length,
    fullSets: sets.filter((set) => set.regime === "full-candidate-set").length,
    pairedArmSets: sets.filter((set) => set.regime !== "full-candidate-set").length,
    candidateTokens: input.featureNames.length,
    contextTokens: V11_EA_CONTEXT_NAMES.length,
    noExperimentBusinesses: sets.filter(
      (set) =>
        set.regime === "experiments-only" &&
        set.context[V11_EA_CONTEXT_NAMES.indexOf("evidence:experiment:available-mask")] === 0,
    ).length,
  },
};
await mkdir(resolve(".flux-artifacts/svi-score-v11-evidence-adaptive"), {
  recursive: true,
});
const temporary = `${OUTPUT_PATH}.tmp`;
await writeFile(temporary, JSON.stringify(artifact));
await rename(temporary, OUTPUT_PATH);
process.stdout.write(`${JSON.stringify({ path: OUTPUT_PATH, ...artifact.summary }, null, 2)}\n`);
