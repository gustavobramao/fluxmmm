import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { SCORE_V6_CANDIDATES } from "../score_v6/cohort";
import {
  V10_AMSS_CONTRACT,
  V10_AMSS_EVIDENCE_GROUPS,
  V10_AMSS_MODULE_ORDERS,
  V10_AMSS_VERSION,
} from "./contract";

const ROOT = resolve("research/svi_score_v10_amss");
const FREEZE_ROOT = resolve(ROOT, "frozen-protocol");
const FREEZE_PATH = resolve(FREEZE_ROOT, "manifest.json");
const CANDIDATE_PATH = resolve(FREEZE_ROOT, "candidate-specifications.json");
const DESIGN_PATH = resolve(FREEZE_ROOT, "cohort-design.csv");
const CONTRACT_PATH = resolve(FREEZE_ROOT, "scientific-contract.json");

const SOURCE_PATHS = [
  "research/svi_score_v10_amss/freeze-protocol.ts",
  "research/svi_score_v10_amss/contract.ts",
  "research/svi_score_v10_amss/amss-cohort-scenario.R",
  "research/svi_score_v10_amss/generate-cohort.R",
  "research/svi_score_v10_amss/seal-cohort.ts",
  "research/svi_score_v10_amss/prepare-inference.ts",
  "research/svi_score_v10_amss/smoke-inference.ts",
  "research/svi_score_v10_amss/verify-inference-results.ts",
  "research/svi_score_v10_amss/actions.ts",
  "research/svi_score_v10_amss/freeze-actions.ts",
  "research/svi_score_v10_amss/open-truth.ts",
  "research/svi_score_v10_amss/open-truth.R",
  "research/svi_score_v10_amss/build-external-audit.ts",
  "research/svi_score_v10_amss/score-external-audit.py",
  "research/svi_score_v10_amss/cloud/inference-cloud-run.yaml",
  "tests/svi-score-v10-amss.test.ts",
  "scripts/build_cloud_svi_shards.py",
  "scripts/cloud_svi_shard_task.py",
  "research/score_v6/cohort.ts",
  "lib/mmm/agentic-search.ts",
  "lib/mmm/advanced.ts",
  "lib/mmm/models.ts",
  "lib/mmm/response.ts",
  "lib/mmm/sampling.ts",
  "lib/mmm/score-diagnostics.ts",
  "lib/mmm/schema.ts",
  "lib/mmm/validation/index.ts",
  "lib/mmm/validation/adapter.ts",
  "lib/mmm/validation/decision.ts",
  "lib/mmm/validation/evidence.ts",
  "lib/mmm/validation/evidence-dependence.ts",
  "scripts/mcmc_server.py",
  "scripts/svi_batch.py",
  "research/svi_score_v3/cloud/requirements.lock",
] as const;

interface DesignRow {
  business_id: string;
  business_seed: number;
  truth_seed: number;
  module_order: string;
  evidence_group: string;
  population: number;
  gross_margin: number;
  price: number;
  market_trend: number;
  season_amplitude: number;
  social_planning_correlation: number;
  social_budget: number;
  social_growth: number;
  social_unit_cost: number;
  social_hill_ec: number;
  social_hill_slope: number;
  search_budget: number;
  search_growth: number;
  search_ctr_scale: number;
  tv_budget: number;
  tv_growth: number;
  tv_hill_ec: number;
  tv_hill_slope: number;
  tv_lag_weeks: number;
  demand_scale: number;
  experiment_bias_share: number;
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function quantiles(dimension: string): number[] {
  return Array.from({ length: V10_AMSS_CONTRACT.cohort.businesses }, (_, index) => ({
    index,
    key: sha256(`${V10_AMSS_CONTRACT.cohort.businessSeedStart}:${dimension}:${index}`),
  }))
    .sort((left, right) => left.key.localeCompare(right.key))
    .reduce((values, item, rank) => {
      values[item.index] = (rank + 0.5) / V10_AMSS_CONTRACT.cohort.businesses;
      return values;
    }, Array<number>(V10_AMSS_CONTRACT.cohort.businesses));
}

function range(values: number[], low: number, high: number): number[] {
  return values.map((value) => low + value * (high - low));
}

function rounded(value: number, digits = 6): number {
  return Number(value.toFixed(digits));
}

function buildDesign(): DesignRow[] {
  const dimensions = {
    population: range(quantiles("population"), 80e6, 320e6),
    grossMargin: range(quantiles("gross-margin"), 0.35, 0.7),
    price: range(quantiles("price"), 55, 140),
    marketTrend: range(quantiles("market-trend"), 0.5, 0.82),
    seasonAmplitude: range(quantiles("season-amplitude"), 0.1, 0.38),
    socialCorrelation: range(quantiles("social-correlation"), 0.2, 0.75),
    socialBudget: range(quantiles("social-budget"), 10e6, 45e6),
    socialGrowth: range(quantiles("social-growth"), -0.05, 0.25),
    socialUnitCost: range(quantiles("social-unit-cost"), 0.004, 0.016),
    socialHillEc: range(quantiles("social-hill-ec"), 1.4, 4),
    socialHillSlope: range(quantiles("social-hill-slope"), 0.8, 2),
    searchBudget: range(quantiles("search-budget"), 5e6, 28e6),
    searchGrowth: range(quantiles("search-growth"), 0, 0.35),
    searchCtrScale: range(quantiles("search-ctr-scale"), 0.6, 1.4),
    tvBudget: range(quantiles("tv-budget"), 25e6, 80e6),
    tvGrowth: range(quantiles("tv-growth"), -0.1, 0.12),
    tvHillEc: range(quantiles("tv-hill-ec"), 0.8, 3),
    tvHillSlope: range(quantiles("tv-hill-slope"), 0.7, 1.8),
    tvLag: range(quantiles("tv-lag"), 0, 10.999),
    demandScale: range(quantiles("demand-scale"), 0.65, 1.35),
    experimentBias: range(quantiles("experiment-bias"), -0.1, 0.1),
  };
  return Array.from({ length: V10_AMSS_CONTRACT.cohort.businesses }, (_, index) => ({
    business_id: `amss-v10-${String(index + 1).padStart(3, "0")}`,
    business_seed: V10_AMSS_CONTRACT.cohort.businessSeedStart + index,
    truth_seed: V10_AMSS_CONTRACT.cohort.truthSeedStart + index,
    module_order: V10_AMSS_MODULE_ORDERS[
      (index + Math.floor(index / V10_AMSS_EVIDENCE_GROUPS.length)) % V10_AMSS_MODULE_ORDERS.length
    ],
    evidence_group: V10_AMSS_EVIDENCE_GROUPS[index % V10_AMSS_EVIDENCE_GROUPS.length],
    population: Math.round(dimensions.population[index]),
    gross_margin: rounded(dimensions.grossMargin[index]),
    price: rounded(dimensions.price[index], 2),
    market_trend: rounded(dimensions.marketTrend[index]),
    season_amplitude: rounded(dimensions.seasonAmplitude[index]),
    social_planning_correlation: rounded(dimensions.socialCorrelation[index]),
    social_budget: Math.round(dimensions.socialBudget[index]),
    social_growth: rounded(dimensions.socialGrowth[index]),
    social_unit_cost: rounded(dimensions.socialUnitCost[index]),
    social_hill_ec: rounded(dimensions.socialHillEc[index]),
    social_hill_slope: rounded(dimensions.socialHillSlope[index]),
    search_budget: Math.round(dimensions.searchBudget[index]),
    search_growth: rounded(dimensions.searchGrowth[index]),
    search_ctr_scale: rounded(dimensions.searchCtrScale[index]),
    tv_budget: Math.round(dimensions.tvBudget[index]),
    tv_growth: rounded(dimensions.tvGrowth[index]),
    tv_hill_ec: rounded(dimensions.tvHillEc[index]),
    tv_hill_slope: rounded(dimensions.tvHillSlope[index]),
    tv_lag_weeks: Math.floor(dimensions.tvLag[index]),
    demand_scale: rounded(dimensions.demandScale[index]),
    experiment_bias_share: rounded(dimensions.experimentBias[index]),
  }));
}

function csv(rows: DesignRow[]): string {
  const columns = Object.keys(rows[0]) as (keyof DesignRow)[];
  return [
    columns.join(","),
    ...rows.map((row) => columns.map((column) => String(row[column])).join(",")),
    "",
  ].join("\n");
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
      throw new Error(`Frozen V10 protocol artifact changed: ${path}.`);
    }
  }
}

const candidates = SCORE_V6_CANDIDATES.map((candidate) => structuredClone(candidate));
if (candidates.length !== 24) throw new Error("V10 requires exactly 24 candidate specifications.");
const candidateSource = `${canonical(candidates)}\n`;
const expectedCandidateSha256 =
  "0142e371fef155465ba3acf32d30564ff4008e1c8ab0905c51171504f73e2403";
if (sha256(candidateSource.trim()) !== expectedCandidateSha256) {
  throw new Error("The declared 24-candidate library changed before V10 freeze.");
}

const design = buildDesign();
const designSource = csv(design);
const sourceSha256: Record<string, string> = {};
for (const path of SOURCE_PATHS) sourceSha256[path] = sha256(await readFile(resolve(path)));
const frozenV9Manifest = await readFile(resolve(ROOT, "frozen-v9/manifest.json"));
const frozenV9Selector = await readFile(resolve(ROOT, "frozen-v9/selector.json"));
const contractSource = `${JSON.stringify(V10_AMSS_CONTRACT, null, 2)}\n`;

await mkdir(FREEZE_ROOT, { recursive: true });
await writeImmutable(CANDIDATE_PATH, `${JSON.stringify(candidates, null, 2)}\n`);
await writeImmutable(DESIGN_PATH, designSource);
await writeImmutable(CONTRACT_PATH, contractSource);
const manifest = {
  artifactId: "flux-v10-amss-external-transport-protocol-freeze-v1",
  version: V10_AMSS_VERSION,
  status: "frozen-before-cohort-generation",
  frozenAt: new Date().toISOString(),
  contractSha256: sha256(contractSource),
  candidateSpecificationsSha256: sha256(candidateSource.trim()),
  candidateSpecificationsFileSha256: sha256(await readFile(CANDIDATE_PATH)),
  cohortDesignSha256: sha256(designSource),
  businesses: design.length,
  evidenceGroupCounts: Object.fromEntries(V10_AMSS_EVIDENCE_GROUPS.map((group) => [
    group,
    design.filter((row) => row.evidence_group === group).length,
  ])),
  moduleOrderCounts: Object.fromEntries(V10_AMSS_MODULE_ORDERS.map((order) => [
    order,
    design.filter((row) => row.module_order === order).length,
  ])),
  evidenceByModuleOrderCounts: Object.fromEntries(V10_AMSS_EVIDENCE_GROUPS.map((group) => [
    group,
    Object.fromEntries(V10_AMSS_MODULE_ORDERS.map((order) => [
      order,
      design.filter((row) => row.evidence_group === group && row.module_order === order).length,
    ])),
  ])),
  frozenV9ManifestSha256: sha256(frozenV9Manifest),
  frozenV9SelectorSha256: sha256(frozenV9Selector),
  cloudImageDigest: V10_AMSS_CONTRACT.orchestration.containerImageDigest,
  sourceSha256,
  truthOpened: false,
};
await writeImmutable(FREEZE_PATH, `${JSON.stringify(manifest, null, 2)}\n`);

console.log(JSON.stringify({
  frozen: true,
  manifest: FREEZE_PATH,
  businesses: design.length,
  candidateSpecifications: candidates.length,
  candidateFits: design.length * candidates.length * 2,
  candidateSpecificationsSha256: manifest.candidateSpecificationsSha256,
  cohortDesignSha256: manifest.cohortDesignSha256,
  truthOpened: false,
}, null, 2));
