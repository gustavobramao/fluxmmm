import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  advancedModelFingerprint,
  runAdvancedModel,
} from "../../lib/mmm/advanced";
import type { AgenticCandidateSpec } from "../../lib/mmm/agentic";
import { parseCsv } from "../../lib/mmm/csv";
import { modelFingerprint, runModel } from "../../lib/mmm/models";
import { createDataset, validateDataset } from "../../lib/mmm/schema";
import type { Experiment } from "../../lib/mmm/types";
import { generateSyntheticBusiness } from "../score_v2/simulator";
import type { SyntheticBusiness } from "../score_v2/types";
import { SCORE_V6_CANDIDATES } from "../score_v6/cohort";
import { syntheticIndustryPriorOverrides } from "../score_v6/evidence";
import type { ScoreV6EvidenceArm } from "../score_v6/types";
import { sviDesignBusinesses } from "../svi_score_v3/design";

const predictionsPath = resolve(
  ".flux-artifacts/svi-score-v9/cross-fitted-predictions.json",
);
const outputPath = resolve(
  "research/svi_score_v9/artifacts/svi-score-v9-evidence-attribution.json",
);

const sourceOrder = [
  "observational",
  "experiment",
  "benchmark",
  "regularization",
] as const;
type Source = (typeof sourceOrder)[number];

interface Selection {
  businessId: string;
  family: string;
  candidateId: string;
}

interface AttributionRow {
  businessId: string;
  family: string;
  candidateId: string;
  evidenceArm: ScoreV6EvidenceArm;
  channel: string;
  spend: number;
  conditionalDataShare: number;
  attributionResidual: number;
  shares: Record<Source, number>;
}

function industryChannels(
  business: SyntheticBusiness,
  arm: ScoreV6EvidenceArm,
): string[] {
  if (arm === "experiments-only") return [];
  const experiments = new Set(
    business.truth.experiments.map((study) =>
      study.experiment.channel.toLowerCase()
    ),
  );
  return business.truth.channels
    .map((channel) => channel.spendColumn)
    .filter((channel) => !experiments.has(channel.toLowerCase()));
}

async function fitCandidate(
  dataset: Awaited<ReturnType<typeof createDataset>>,
  spec: AgenticCandidateSpec,
  experiments: Experiment[],
  industryPriorChannels: string[],
  overrides: ReturnType<typeof syntheticIndustryPriorOverrides>,
) {
  if (spec.family === "advanced") {
    const fingerprint = await advancedModelFingerprint(
      dataset,
      spec.config,
      spec.advancedConfig,
      experiments,
      industryPriorChannels,
      overrides,
    );
    return runAdvancedModel(
      dataset,
      spec.config,
      spec.advancedConfig,
      experiments,
      fingerprint,
      industryPriorChannels,
      overrides,
    );
  }
  const fingerprint = await modelFingerprint(
    dataset,
    spec.config,
    experiments,
    spec.family,
    industryPriorChannels,
    overrides,
  );
  return runModel(
    dataset,
    spec.config,
    experiments,
    spec.family,
    fingerprint,
    industryPriorChannels,
    overrides,
  );
}

function quantile(values: number[], probability: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const position = Math.max(0, Math.min(1, probability)) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const fraction = position - lower;
  return (sorted[lower] ?? 0) * (1 - fraction) + (sorted[upper] ?? 0) * fraction;
}

function aggregate(rows: AttributionRow[]) {
  const totalSpend = rows.reduce((sum, row) => sum + row.spend, 0);
  return {
    channels: rows.length,
    businesses: new Set(rows.map((row) => row.businessId)).size,
    equalChannelMean: Object.fromEntries(sourceOrder.map((source) => [
      source,
      rows.reduce((sum, row) => sum + row.shares[source], 0) / Math.max(rows.length, 1),
    ])),
    equalChannelMedian: Object.fromEntries(sourceOrder.map((source) => [
      source,
      quantile(rows.map((row) => row.shares[source]), 0.5),
    ])),
    spendWeightedMean: Object.fromEntries(sourceOrder.map((source) => [
      source,
      rows.reduce((sum, row) => sum + row.shares[source] * row.spend, 0) /
        Math.max(totalSpend, 1),
    ])),
    conditionalDataShare: {
      mean: rows.reduce((sum, row) => sum + row.conditionalDataShare, 0) /
        Math.max(rows.length, 1),
      median: quantile(rows.map((row) => row.conditionalDataShare), 0.5),
    },
    maximumAttributionResidual: Math.max(
      0,
      ...rows.map((row) => row.attributionResidual),
    ),
  };
}

async function analyze(selection: Selection): Promise<AttributionRow[]> {
  const design = sviDesignBusinesses().find(
    (item) => item.scenario.id === selection.businessId,
  );
  if (!design) throw new Error(`Missing scenario ${selection.businessId}`);
  const business = generateSyntheticBusiness(design.scenario);
  const parsed = parseCsv(business.observedCsv);
  const dataset = await createDataset(
    `${design.scenario.id}.csv`,
    business.observedCsv,
    parsed.columns,
    parsed.rows,
  );
  if (validateDataset(dataset).status === "blocked") {
    throw new Error(`${selection.businessId} failed its generated data contract`);
  }
  const [baseId, armValue] = selection.candidateId.split(" · ");
  const evidenceArm = armValue as ScoreV6EvidenceArm;
  if (evidenceArm !== "experiments-only" && evidenceArm !== "benchmark-gap-fill") {
    throw new Error(`Unknown evidence arm in ${selection.candidateId}`);
  }
  const base = SCORE_V6_CANDIDATES.find((candidate) => candidate.id === baseId);
  if (!base) throw new Error(`Unknown candidate ${baseId}`);
  const active = industryChannels(business, evidenceArm);
  const spec: AgenticCandidateSpec = {
    ...base,
    id: selection.candidateId,
    evidencePriorChannels: active,
  };
  const model = await fitCandidate(
    dataset,
    spec,
    business.truth.experiments.map((study) => study.experiment),
    active,
    syntheticIndustryPriorOverrides(business),
  );
  const attributions = model.numerical?.evidenceAttribution ?? [];
  if (attributions.length !== business.truth.channels.length) {
    throw new Error(`Incomplete attribution for ${selection.businessId}`);
  }
  const spendByColumn = new Map(
    business.truth.channels.map((channel) => [channel.spendColumn, channel.totalSpend]),
  );
  return attributions.map((attribution) => ({
    businessId: selection.businessId,
    family: selection.family,
    candidateId: selection.candidateId,
    evidenceArm,
    channel: attribution.channel,
    spend: spendByColumn.get(attribution.channel) ?? 0,
    conditionalDataShare: attribution.conditionalDataShare,
    attributionResidual: attribution.attributionResidual,
    shares: attribution.uncertaintyShare,
  }));
}

const predictions = JSON.parse(await readFile(predictionsPath, "utf8")) as {
  selections: Selection[];
};
const rows: AttributionRow[] = [];
const workers = 8;
for (let offset = 0; offset < predictions.selections.length; offset += workers) {
  const batch = predictions.selections.slice(offset, offset + workers);
  rows.push(...(await Promise.all(batch.map(analyze))).flat());
  process.stdout.write(`\r${Math.min(offset + workers, predictions.selections.length)}/${predictions.selections.length}`);
}
process.stdout.write("\n");

const group = <T extends string>(key: (row: AttributionRow) => T) =>
  Object.fromEntries(
    [...new Set(rows.map(key))].sort().map((value) => [
      value,
      aggregate(rows.filter((row) => key(row) === value)),
    ]),
  );

const artifact = {
  artifactId: "flux-svi-score-v9-development-evidence-attribution-v1",
  activation: "development-only",
  estimand: "local Laplace ROI-uncertainty curvature share among outer-fold selected candidates",
  selectedBusinesses: predictions.selections.length,
  channelRows: rows.length,
  overall: aggregate(rows),
  byChannel: group((row) => row.channel),
  byEvidenceArm: group((row) => row.evidenceArm),
  byGeneratorFamily: group((row) => row.family),
  interpretation: {
    influenceOnly: true,
    causalValidityImplied: false,
    note: "Shares describe which fitted source locally supplies ROI precision; they do not score source quality or prove exogeneity.",
  },
  provenance: {
    selections: "outer-fold cross-fitted development selections",
    sealedAuditDereferenced: false,
    hiddenCausalTruthUsed: false,
  },
};

await writeFile(outputPath, JSON.stringify(artifact, null, 2));
process.stdout.write(`${JSON.stringify({ outputPath, overall: artifact.overall }, null, 2)}\n`);
