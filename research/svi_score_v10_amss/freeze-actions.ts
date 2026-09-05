import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import { parseCsv, toNumber } from "../../lib/mmm/csv";
import type { SamplingDecisionDraw } from "../../lib/mmm/sampling";
import {
  type V5DSviRecordMap,
  v5dSviFeatureProvider,
  v5dSviKey,
} from "../svi_score_v5d_svi/posterior";
import {
  v9PosteriorDecisionTokens,
  type V9SviResult,
} from "../svi_score_v9/audit-tokens";
import { recommendV10CandidateActions } from "./actions";
import { V10_AMSS_CHANNELS, V10_AMSS_VERSION } from "./contract";
import type { V10InferenceItem } from "./prepare-inference";

const ROOT = resolve(".flux-artifacts/svi-score-v10-amss");
const COHORT_ROOT = resolve(ROOT, "sealed-cohort");
const INFERENCE_ROOT = resolve(ROOT, "inference");
const RESULTS_ROOT = resolve(INFERENCE_ROOT, "results");
const OUTPUT_ROOT = resolve(ROOT, "observable-freeze");
const DATASET_PATH = resolve(OUTPUT_ROOT, "observable-candidates.json");
const ACTIONS_PATH = resolve(OUTPUT_ROOT, "candidate-actions.csv");
const MANIFEST_PATH = resolve(OUTPUT_ROOT, "manifest.json");

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function csvCell(value: string | number): string {
  const source = String(value);
  return /[",\n\r]/.test(source) ? `"${source.replaceAll('"', '""')}"` : source;
}

async function writeImmutable(path: string, value: string): Promise<void> {
  try {
    await writeFile(path, value, { flag: "wx" });
  } catch (error) {
    if (!(
      error instanceof Error && "code" in error &&
      (error as NodeJS.ErrnoException).code === "EEXIST"
    )) throw error;
    if ((await readFile(path, "utf8")) !== value) {
      throw new Error(`Frozen V10 observable artifact changed: ${path}.`);
    }
  }
}

const inferenceSource = await readFile(resolve(INFERENCE_ROOT, "manifest.json"));
const inference = JSON.parse(inferenceSource.toString("utf8")) as {
  version: string;
  protocolFreezeSha256: string;
  cohortManifestSha256: string;
  hiddenTruthDereferenced: boolean;
  businesses: Array<{
    businessId: string;
    evidenceGroup: string;
    grossMargin: number;
    baselineAnnualSpend: Record<string, number>;
    budgetUtilization: Record<string, number>;
  }>;
  items: V10InferenceItem[];
};
if (
  inference.version !== V10_AMSS_VERSION ||
  inference.hiddenTruthDereferenced ||
  inference.items.length !== 4_800
) throw new Error("V10 inference manifest is incomplete or opened hidden truth.");

const tokenRegistry = JSON.parse(
  await readFile(resolve("research/svi_score_v10_amss/frozen-v9/token-registry.json"), "utf8"),
) as { count: number; names: string[] };
const resultByFingerprint = new Map<string, V9SviResult>();
const posterior: V5DSviRecordMap = new Map();
for (const item of inference.items) {
  const result = JSON.parse(
    gunzipSync(await readFile(resolve(RESULTS_ROOT, `${item.inferenceFingerprint}.result.json.gz`))).toString("utf8"),
  ) as V9SviResult;
  if (
    result.fingerprint !== item.inferenceFingerprint ||
    (result.status !== "labelled" && result.status !== "review") ||
    !result.channels.length || !result.decisionDraws.length
  ) throw new Error(`Invalid V10 SVI result: ${item.inferenceFingerprint}.`);
  resultByFingerprint.set(item.inferenceFingerprint, result);
  posterior.set(v5dSviKey(item.observableRow), {
    status: result.status,
    posteriorRoi: Object.fromEntries(
      result.channels.map((channel) => [channel.channel, channel.posteriorMedian]),
    ),
    diagnostics: structuredClone(result.diagnostics),
    seedDiagnostics: structuredClone(result.seeds),
  });
}

const provider = v5dSviFeatureProvider(posterior);
const businessById = new Map(inference.businesses.map((business) => [business.businessId, business]));
const spendByBusiness = new Map<string, Record<string, number[]>>();
for (const business of inference.businesses) {
  const parsed = parseCsv(
    await readFile(resolve(COHORT_ROOT, "observed", `${business.businessId}.csv`), "utf8"),
  );
  spendByBusiness.set(business.businessId, Object.fromEntries(
    V10_AMSS_CHANNELS.map((channel) => [
      channel,
      parsed.rows.map((row) => toNumber(row[channel])),
    ]),
  ));
}

const rows = inference.items.map((item) => {
  const result = resultByFingerprint.get(item.inferenceFingerprint)!;
  const baseNames = provider.names(item.observableRow);
  const baseFeatures = provider.vector(item.observableRow);
  const featureNames = [...baseNames, ...tokenRegistry.names.slice(baseNames.length)];
  const features = [...baseFeatures, ...v9PosteriorDecisionTokens(result)];
  if (
    featureNames.length !== tokenRegistry.count ||
    featureNames.some((name, index) => name !== tokenRegistry.names[index]) ||
    features.length !== tokenRegistry.count ||
    features.some((value) => !Number.isFinite(value))
  ) throw new Error(`V10 token contract changed for ${item.businessId}/${item.candidateId}.`);
  const business = businessById.get(item.businessId)!;
  const actions = recommendV10CandidateActions({
    grossMargin: business.grossMargin,
    baselineAnnualSpend: business.baselineAnnualSpend,
    observedSpend: spendByBusiness.get(item.businessId)!,
  }, result.decisionDraws as SamplingDecisionDraw[]);
  return {
    businessId: item.businessId,
    evidenceGroup: item.evidenceGroup,
    candidateId: item.candidateId,
    evidenceArm: item.evidenceArm,
    modelFamily: item.modelFamily,
    valid: result.diagnostics.finite,
    status: result.status,
    heuristicScore: item.observableRow.heuristicScore,
    failedGateCount: item.observableRow.failedGateCount,
    features,
    actions,
  };
});
if (
  rows.length !== 4_800 ||
  new Set(rows.map((row) => `${row.businessId}\u0000${row.candidateId}`)).size !== rows.length ||
  rows.some((row) => row.actions.length !== 4)
) throw new Error("V10 observable candidate/action freeze is incomplete.");

const dataset = {
  artifactId: "flux-v10-amss-observable-candidate-set-v1",
  version: V10_AMSS_VERSION,
  activation: "external-audit-truth-sealed",
  provenance: {
    inferenceManifestSha256: sha256(inferenceSource),
    protocolFreezeSha256: inference.protocolFreezeSha256,
    cohortManifestSha256: inference.cohortManifestSha256,
    hiddenTruthDereferenced: false,
    frozenV9Retrained: false,
  },
  featureNames: tokenRegistry.names,
  rows,
};
const datasetSource = `${JSON.stringify(dataset)}\n`;
const actionSource = [
  [
    "business_id",
    "candidate_id",
    "scenario",
    "meta_acquisition_spend",
    "google_search_nonbrand_spend",
    "ctv_spend",
    "predicted_incremental_profit",
  ].join(","),
  ...rows.flatMap((row) => row.actions.map((action) => [
    row.businessId,
    row.candidateId,
    action.scenario,
    action.annualSpend.meta_acquisition_spend,
    action.annualSpend.google_search_nonbrand_spend,
    action.annualSpend.ctv_spend,
    action.predictedIncrementalProfit,
  ].map(csvCell).join(","))),
  "",
].join("\n");

await mkdir(OUTPUT_ROOT, { recursive: true });
await writeImmutable(DATASET_PATH, datasetSource);
await writeImmutable(ACTIONS_PATH, actionSource);
const manifest = {
  artifactId: "flux-v10-amss-observable-freeze-manifest-v1",
  version: V10_AMSS_VERSION,
  status: "all-candidate-actions-frozen-before-truth",
  businesses: inference.businesses.length,
  candidates: rows.length,
  actions: rows.length * 4,
  validCandidates: rows.filter((row) => row.valid).length,
  observableDatasetSha256: sha256(datasetSource),
  candidateActionsSha256: sha256(actionSource),
  inferenceManifestSha256: sha256(inferenceSource),
  hiddenTruthDereferenced: false,
  truthOpened: false,
};
await writeImmutable(MANIFEST_PATH, `${JSON.stringify(manifest, null, 2)}\n`);

console.log(JSON.stringify({
  frozen: true,
  businesses: manifest.businesses,
  candidates: manifest.candidates,
  actions: manifest.actions,
  validCandidates: manifest.validCandidates,
  truthOpened: false,
}, null, 2));
