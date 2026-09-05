import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import {
  SVI_SCORE_V9_CONTRACT,
  SVI_SCORE_V9_VERSION,
  V9_POSTERIOR_CHANNEL_METRICS,
  V9_POSTERIOR_TOKEN_NAMES,
} from "./contract";

const V8_DATASET_PATH = resolve(
  ".flux-artifacts/svi-score-v8/development-dataset.json",
);
const DEVELOPMENT_MANIFEST_PATH = resolve(
  ".flux-artifacts/svi-score-v8/development-record-manifest.json",
);
const OUTPUT_PATH = resolve(".flux-artifacts/svi-score-v9/development-dataset.json");

interface V8Row {
  businessId: string;
  family: string;
  candidateId: string;
  fold: number;
  features: number[];
  valid: boolean;
  validityReasons: string[];
  scenarioLoss: Record<string, number>;
  scenarioExcessLoss: Record<string, number>;
  meanExcessLoss: number;
  p90ExcessLoss: number;
  economicRisk: number;
  excessEconomicRisk: number;
  normalizedExcessEconomicRisk: number;
  dangerous: boolean;
  roiError: number;
  contributionError: number;
}

interface V8Dataset {
  activation: string;
  provenance: { auditAccessed: boolean; nutsUsed: boolean };
  featureNames: string[];
  rows: V8Row[];
}

interface ManifestRecord {
  businessId: string;
  candidateId: string;
  inferenceFingerprint: string;
  split: "train" | "validation";
}

interface DevelopmentManifest {
  records: ManifestRecord[];
}

interface PosteriorChannel {
  channel: string;
  posteriorMean: number;
  posteriorMedian: number;
  posteriorLow: number;
  posteriorHigh: number;
  posteriorSamples: number[];
  implausibleProbability?: number;
  evidenceSource?: string;
  explanation?: {
    locationShiftSd?: number;
    intervalOverlap?: number;
    posteriorSkewness?: number;
    nearZeroProbability?: number;
  };
}

interface DecisionDrawChannel {
  channel: string;
  roi: number;
  contribution: number;
  response: {
    adstockType: "geometric" | "weibull";
    adstock: number;
    weibullShape: number;
    weibullScale: number;
    saturation: number;
    halfSaturationQuantile: number;
    kernelNormalization: "peak" | "sum";
  };
}

interface SviResult {
  fingerprint: string;
  contract: unknown;
  channels: PosteriorChannel[];
  decisionDraws: Array<{ channels: DecisionDrawChannel[] }>;
}

const finite = (value: unknown, fallback = 0) => {
  const result = Number(value);
  return Number.isFinite(result) ? result : fallback;
};

const average = (values: readonly number[]) =>
  values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);

const standardDeviation = (values: readonly number[]) => {
  const center = average(values);
  return Math.sqrt(average(values.map((value) => (value - center) ** 2)));
};

const logPositive = (value: unknown) => Math.log(Math.max(finite(value), 1e-6));

function correlation(left: readonly number[], right: readonly number[]): number {
  const length = Math.min(left.length, right.length);
  if (length < 2) return 0;
  const x = left.slice(0, length);
  const y = right.slice(0, length);
  const xMean = average(x);
  const yMean = average(y);
  const numerator = x.reduce(
    (sum, value, index) => sum + (value - xMean) * (y[index] - yMean),
    0,
  );
  const denominator = Math.sqrt(
    x.reduce((sum, value) => sum + (value - xMean) ** 2, 0) *
      y.reduce((sum, value) => sum + (value - yMean) ** 2, 0),
  );
  return denominator > 1e-12 ? numerator / denominator : 0;
}

function aggregate(values: readonly number[]): number[] {
  const safe = values.map((value) => finite(value));
  return [
    average(safe),
    standardDeviation(safe),
    Math.min(...safe),
    Math.max(...safe),
  ];
}

function channelMetricVector(
  channel: PosteriorChannel,
  draws: DecisionDrawChannel[],
): number[] {
  const roiDraws = channel.posteriorSamples.map((value) => logPositive(value));
  const contributions = draws.map((draw) => Math.max(finite(draw.contribution), 0));
  const contributionMean = average(contributions);
  const source = (channel.evidenceSource ?? "").toLowerCase();
  const explanation = channel.explanation ?? {};
  const responses = draws.map((draw) => draw.response);
  const adstock = responses.map((response) => finite(response.adstock));
  const weibullScale = responses.map((response) => logPositive(response.weibullScale));
  const weibullShape = responses.map((response) => finite(response.weibullShape));
  const saturation = responses.map((response) => finite(response.saturation));
  const halfSaturation = responses.map((response) =>
    finite(response.halfSaturationQuantile)
  );
  return [
    logPositive(channel.posteriorMedian),
    logPositive(channel.posteriorMean),
    logPositive(channel.posteriorLow),
    logPositive(channel.posteriorHigh),
    Math.log1p(
      Math.max(0, finite(channel.posteriorHigh) - finite(channel.posteriorLow)) /
        Math.max(finite(channel.posteriorMedian), 1e-6),
    ),
    standardDeviation(roiDraws),
    finite(channel.implausibleProbability),
    finite(explanation.nearZeroProbability),
    finite(explanation.posteriorSkewness),
    finite(explanation.locationShiftSd),
    finite(explanation.intervalOverlap),
    source.includes("experiment") ? 1 : 0,
    source.includes("benchmark") || source.includes("industry") ? 1 : 0,
    average(adstock),
    standardDeviation(adstock),
    average(weibullScale),
    average(weibullShape),
    average(saturation),
    standardDeviation(saturation),
    average(halfSaturation),
    average(responses.map((response) => response.adstockType === "weibull" ? 1 : 0)),
    average(
      responses.map((response) => response.kernelNormalization === "sum" ? 1 : 0),
    ),
    Math.log1p(
      standardDeviation(contributions) / Math.max(Math.abs(contributionMean), 1e-6),
    ),
  ];
}

function pairwiseAbsoluteCorrelations(series: number[][]): [number, number] {
  const values: number[] = [];
  for (let left = 0; left < series.length; left += 1) {
    for (let right = left + 1; right < series.length; right += 1) {
      values.push(Math.abs(correlation(series[left], series[right])));
    }
  }
  return values.length ? [average(values), Math.max(...values)] : [0, 0];
}

function posteriorDecisionTokens(result: SviResult): number[] {
  if (!result.channels.length || !result.decisionDraws.length) {
    throw new Error(`V9 posterior ${result.fingerprint} has no channel draws.`);
  }
  const drawChannels = new Map<string, DecisionDrawChannel[]>();
  result.decisionDraws.forEach((draw) => {
    draw.channels.forEach((channel) => {
      drawChannels.set(channel.channel, [
        ...(drawChannels.get(channel.channel) ?? []),
        channel,
      ]);
    });
  });
  const channelVectors = result.channels.map((channel) =>
    channelMetricVector(channel, drawChannels.get(channel.channel) ?? [])
  );
  const aggregated: number[] = [];
  for (let metric = 0; metric < V9_POSTERIOR_CHANNEL_METRICS.length; metric += 1) {
    aggregated.push(...aggregate(channelVectors.map((vector) => vector[metric])));
  }
  const roiSeries = result.channels.map((channel) =>
    channel.posteriorSamples.map((value) => logPositive(value))
  );
  const contributionSeries = result.channels.map((channel) =>
    (drawChannels.get(channel.channel) ?? []).map((draw) =>
      Math.log1p(Math.max(finite(draw.contribution), 0))
    )
  );
  aggregated.push(
    ...pairwiseAbsoluteCorrelations(roiSeries),
    ...pairwiseAbsoluteCorrelations(contributionSeries),
  );
  if (
    aggregated.length !== V9_POSTERIOR_TOKEN_NAMES.length ||
    aggregated.some((value) => !Number.isFinite(value))
  ) {
    throw new Error(`V9 posterior token contract failed for ${result.fingerprint}.`);
  }
  return aggregated;
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

async function mapWithConcurrency<T, U>(
  values: readonly T[],
  concurrency: number,
  mapper: (value: T, index: number) => Promise<U>,
): Promise<U[]> {
  const output = new Array<U>(values.length);
  let cursor = 0;
  await Promise.all(Array.from({ length: concurrency }, async () => {
    while (cursor < values.length) {
      const index = cursor;
      cursor += 1;
      output[index] = await mapper(values[index], index);
    }
  }));
  return output;
}

const [v8Source, manifestSource] = await Promise.all([
  readFile(V8_DATASET_PATH),
  readFile(DEVELOPMENT_MANIFEST_PATH),
]);
const v8 = JSON.parse(v8Source.toString("utf8")) as V8Dataset;
const manifest = JSON.parse(manifestSource.toString("utf8")) as DevelopmentManifest;
if (
  v8.activation !== "development-only" ||
  v8.provenance.auditAccessed ||
  v8.provenance.nutsUsed ||
  v8.rows.length !== SVI_SCORE_V9_CONTRACT.cohort.developmentRows ||
  v8.featureNames.length !== SVI_SCORE_V9_CONTRACT.tokens.baseCount ||
  manifest.records.some((record) => record.split === ("audit" as string))
) {
  throw new Error("V9 preparation requires the frozen development-only V8 cohort.");
}
const fingerprintByKey = new Map(
  manifest.records.map((record) => [
    `${record.businessId}\u0000${record.candidateId}`,
    record.inferenceFingerprint,
  ]),
);
if (fingerprintByKey.size !== v8.rows.length) {
  throw new Error("V9 development manifest is incomplete or duplicated.");
}

let completed = 0;
const rows = await mapWithConcurrency(v8.rows, 32, async (row) => {
  const fingerprint = fingerprintByKey.get(`${row.businessId}\u0000${row.candidateId}`);
  if (!fingerprint) {
    throw new Error(`V9 has no posterior for ${row.businessId}/${row.candidateId}.`);
  }
  const path = resolve(
    ".flux-artifacts/svi-score-v3",
    `${fingerprint}.result.json.gz`,
  );
  const result = JSON.parse(
    gunzipSync(await readFile(path)).toString("utf8"),
  ) as SviResult;
  if (
    result.fingerprint !== fingerprint ||
    JSON.stringify(result.contract) !== JSON.stringify(SVI_SCORE_V9_CONTRACT.inference.contract)
  ) {
    throw new Error(`V9 SVI contract mismatch for ${row.businessId}/${row.candidateId}.`);
  }
  completed += 1;
  if (completed % 2_000 === 0) {
    process.stdout.write(`V9 posterior tokens: ${completed}/${v8.rows.length}\n`);
  }
  return {
    ...row,
    features: [...row.features, ...posteriorDecisionTokens(result)],
  };
});

const featureNames = [...v8.featureNames, ...V9_POSTERIOR_TOKEN_NAMES];
if (
  featureNames.length !== SVI_SCORE_V9_CONTRACT.tokens.count ||
  rows.some((row) =>
    row.features.length !== featureNames.length ||
    row.features.some((value) => !Number.isFinite(value))
  )
) {
  throw new Error("V9 assembled token matrix violates the declared contract.");
}
const artifact = {
  artifactId: "flux-svi-score-v9-development-dataset-v1",
  version: SVI_SCORE_V9_VERSION,
  generatedAt: new Date().toISOString(),
  activation: "development-only",
  contract: SVI_SCORE_V9_CONTRACT,
  provenance: {
    v8DatasetSha256: sha256(v8Source),
    developmentManifestSha256: sha256(manifestSource),
    v8AuditDereferenced: false,
    nutsUsed: false,
  },
  featureNames,
  summary: {
    businesses: new Set(rows.map((row) => row.businessId)).size,
    rows: rows.length,
    candidatesPerBusiness: SVI_SCORE_V9_CONTRACT.cohort.candidatesPerBusiness,
    tokens: featureNames.length,
    posteriorDecisionTokens: V9_POSTERIOR_TOKEN_NAMES.length,
    channelOrderInvariant: true,
  },
  rows,
};
await mkdir(resolve(".flux-artifacts/svi-score-v9"), { recursive: true });
const temporary = `${OUTPUT_PATH}.tmp`;
await writeFile(temporary, JSON.stringify(artifact));
await rename(temporary, OUTPUT_PATH);
process.stdout.write(`${JSON.stringify({
  path: OUTPUT_PATH,
  ...artifact.summary,
  v8AuditDereferenced: false,
  nutsUsed: false,
}, null, 2)}\n`);
