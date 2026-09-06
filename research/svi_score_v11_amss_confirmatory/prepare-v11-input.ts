import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  SVI_SCORE_V11_EA_VERSION,
  V11_EA_CONTEXT_NAMES,
  V11_EA_PAIRED_METRICS,
  V11_EA_SUMMARY_METRICS,
} from "../svi_score_v11_evidence_adaptive/contract";
import {
  V11_AMSS_CONFIRMATORY_CONTRACT,
  V11_AMSS_CONFIRMATORY_VERSION,
} from "./contract";

const AMSS_OBSERVABLE_PATH = resolve(
  ".flux-artifacts/svi-score-v11-amss-confirmatory/observable-freeze/observable-candidates.json",
);
const V11_SELECTOR_PATH = resolve(
  "research/svi_score_v11_amss_confirmatory/frozen-selectors/v11-selector.json",
);
const OUTPUT_PATH = resolve(
  ".flux-artifacts/svi-score-v11-amss-confirmatory/observable-freeze/v11-observable-input.json",
);

interface ObservableRow {
  businessId: string;
  evidenceGroup: string;
  candidateId: string;
  evidenceArm: "experiments-only" | "benchmark-gap-fill";
  modelFamily: string;
  valid: boolean;
  features: number[];
}

interface ObservableArtifact {
  activation: string;
  provenance: {
    hiddenTruthDereferenced: boolean;
    frozenV9Retrained: boolean;
  };
  featureNames: string[];
  rows: ObservableRow[];
}

interface SelectorArtifact {
  version: string;
  activation: string;
  provenance: {
    posteriorRefits: number;
    amssConfirmatoryCohortAccessed: boolean;
    previousAuditAccessed: boolean;
  };
  featureNames: string[];
  contextNames: string[];
  models: unknown[];
}

const sha256 = (source: Buffer | string) =>
  createHash("sha256").update(source).digest("hex");

const mean = (values: readonly number[]) =>
  values.reduce((total, value) => total + value, 0) / Math.max(values.length, 1);

const standardDeviation = (values: readonly number[]) => {
  const center = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - center) ** 2)));
};

const finite = (value: unknown): number => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`Non-finite AMSS token: ${value}`);
  return parsed;
};

const baseCandidate = (candidateId: string) => candidateId.split(" · ")[0];

function contextFor(
  rows: readonly ObservableRow[],
  featureIndex: ReadonlyMap<string, number>,
): number[] {
  const experimentIndex = featureIndex.get(
    "posterior:channel-set:evidence-experiment:mean",
  );
  const benchmarkIndex = featureIndex.get(
    "posterior:channel-set:evidence-benchmark:mean",
  );
  if (experimentIndex === undefined || benchmarkIndex === undefined) {
    throw new Error("AMSS observable candidates lack evidence-source tokens.");
  }
  const experimentCoverage = Math.max(
    0,
    ...rows.map((row) => finite(row.features[experimentIndex])),
  );
  const benchmarkCoverage = Math.max(
    0,
    ...rows.map((row) => finite(row.features[benchmarkIndex])),
  );
  const experimentAvailable = experimentCoverage > 1e-12 ? 1 : 0;
  const benchmarkAvailable = benchmarkCoverage > 1e-12 ? 1 : 0;
  const externalAvailable = experimentAvailable || benchmarkAvailable ? 1 : 0;
  const context = [
    experimentAvailable,
    experimentCoverage,
    benchmarkAvailable,
    benchmarkCoverage,
    externalAvailable,
    Math.min(1, experimentCoverage + benchmarkCoverage),
    externalAvailable,
    externalAvailable,
    externalAvailable,
    1,
    0,
    0,
    1,
  ];

  for (const metric of V11_EA_SUMMARY_METRICS) {
    const index = featureIndex.get(metric);
    if (index === undefined) throw new Error(`Missing AMSS summary token ${metric}.`);
    const values = rows.map((row) => finite(row.features[index]));
    const isEvidenceMetric = metric.includes("evidence-");
    context.push(
      isEvidenceMetric && !externalAvailable ? 0 : mean(values),
      isEvidenceMetric && !externalAvailable ? 0 : standardDeviation(values),
      isEvidenceMetric && !externalAvailable ? 0 : Math.max(...values),
    );
  }

  const pairs = new Map<
    string,
    Partial<Record<ObservableRow["evidenceArm"], ObservableRow>>
  >();
  rows.forEach((row) => {
    const id = baseCandidate(row.candidateId);
    pairs.set(id, { ...(pairs.get(id) ?? {}), [row.evidenceArm]: row });
  });
  if (
    pairs.size !== 24 ||
    [...pairs.values()].some(
      (pair) => !pair["experiments-only"] || !pair["benchmark-gap-fill"],
    )
  ) {
    throw new Error(`AMSS evidence arms are not paired for ${rows[0]?.businessId}.`);
  }
  context.push(1);
  for (const metric of V11_EA_PAIRED_METRICS) {
    const index = featureIndex.get(metric);
    if (index === undefined) throw new Error(`Missing AMSS paired token ${metric}.`);
    const differences = [...pairs.values()].map(
      (pair) =>
        finite(pair["benchmark-gap-fill"]!.features[index]) -
        finite(pair["experiments-only"]!.features[index]),
    );
    context.push(mean(differences), standardDeviation(differences));
  }
  if (
    context.length !== V11_EA_CONTEXT_NAMES.length ||
    context.some((value) => !Number.isFinite(value))
  ) {
    throw new Error(`AMSS context contract failed for ${rows[0]?.businessId}.`);
  }
  return context;
}

const [observableSource, selectorSource] = await Promise.all([
  readFile(AMSS_OBSERVABLE_PATH),
  readFile(V11_SELECTOR_PATH),
]);
const observable = JSON.parse(observableSource.toString("utf8")) as ObservableArtifact;
const selector = JSON.parse(selectorSource.toString("utf8")) as SelectorArtifact;
if (
  observable.activation !== "external-audit-truth-sealed" ||
  observable.provenance.hiddenTruthDereferenced ||
  observable.provenance.frozenV9Retrained ||
  observable.rows.length !== V11_AMSS_CONFIRMATORY_CONTRACT.cohort.candidateFits
) {
  throw new Error("The AMSS observable artifact is not the frozen truth-blind candidate set.");
}
if (
  selector.version !== SVI_SCORE_V11_EA_VERSION ||
  selector.activation !== "development-only" ||
  selector.provenance.posteriorRefits !== 0 ||
  selector.provenance.amssConfirmatoryCohortAccessed ||
  selector.provenance.previousAuditAccessed ||
  selector.models.length !== 5 ||
  JSON.stringify(selector.featureNames) !== JSON.stringify(observable.featureNames) ||
  JSON.stringify(selector.contextNames) !== JSON.stringify(V11_EA_CONTEXT_NAMES)
) {
  throw new Error("The frozen V11 selector is incompatible with the AMSS observable set.");
}

const featureIndex = new Map(
  observable.featureNames.map((name, index) => [name, index]),
);
const grouped = new Map<string, ObservableRow[]>();
observable.rows.forEach((row) => {
  grouped.set(row.businessId, [...(grouped.get(row.businessId) ?? []), row]);
});
const sets = [...grouped.entries()].sort().map(([businessId, rows]) => {
  if (
    rows.length !== V11_AMSS_CONFIRMATORY_CONTRACT.cohort.candidatesPerBusiness ||
    new Set(rows.map((row) => row.evidenceGroup)).size !== 1 ||
    rows.some((row) => row.features.length !== observable.featureNames.length)
  ) {
    throw new Error(`Incomplete AMSS candidate set for ${businessId}.`);
  }
  return {
    businessId,
    evidenceGroup: rows[0].evidenceGroup,
    candidateIds: rows.map((row) => row.candidateId).sort(),
    context: contextFor(rows, featureIndex),
  };
});
if (
  grouped.size !== V11_AMSS_CONFIRMATORY_CONTRACT.cohort.businesses ||
  sets.filter((set) => set.evidenceGroup === "no-experiment").length !== 25 ||
  sets.filter(
    (set) =>
      set.context[
        V11_EA_CONTEXT_NAMES.indexOf("evidence:experiment:available-mask")
      ] === 0,
  ).length !== 25
) {
  throw new Error("AMSS evidence-group and missingness-mask counts disagree.");
}

const artifact = {
  artifactId: "flux-v11-amss-confirmatory-observable-input-v1",
  version: V11_AMSS_CONFIRMATORY_VERSION,
  generatedAt: new Date().toISOString(),
  status: "truth-blind-input-prepared-for-confirmatory-selection",
  scientificLabel: "fresh confirmatory AMSS audit; truth sealed",
  contract: V11_AMSS_CONFIRMATORY_CONTRACT,
  provenance: {
    confirmatoryObservableSha256: sha256(observableSource),
    frozenV11SelectorSha256: sha256(selectorSource),
    hiddenTruthReadByPreparation: false,
    posteriorRefits: 0,
    cloudComputeUsed: true,
  },
  featureNames: observable.featureNames,
  contextNames: V11_EA_CONTEXT_NAMES,
  rows: observable.rows.map((row) => ({
    businessId: row.businessId,
    evidenceGroup: row.evidenceGroup,
    candidateId: row.candidateId,
    valid: row.valid,
    features: row.features,
  })),
  sets,
};
await mkdir(resolve(OUTPUT_PATH, ".."), { recursive: true });
const temporary = `${OUTPUT_PATH}.tmp`;
await writeFile(temporary, JSON.stringify(artifact));
await rename(temporary, OUTPUT_PATH);
process.stdout.write(
  `${JSON.stringify({
    path: OUTPUT_PATH,
    businesses: sets.length,
    candidates: observable.rows.length,
    noExperimentMasks: 25,
    posteriorRefits: 0,
    cloudComputeUsed: true,
  }, null, 2)}\n`,
);
