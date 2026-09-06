import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  V11_AMSS_CONFIRMATORY_CONTRACT,
  V11_AMSS_CONFIRMATORY_EVIDENCE_GROUPS,
  V11_AMSS_CONFIRMATORY_MODULE_ORDERS,
  V11_AMSS_CONFIRMATORY_VERSION,
} from "./contract";

const ROOT = resolve("research/svi_score_v11_amss_confirmatory");
const FREEZE_ROOT = resolve(ROOT, "frozen-protocol");
const SELECTOR_ROOT = resolve(ROOT, "frozen-selectors");
const CANDIDATE_SOURCE = resolve("research/svi_score_v10_amss/frozen-protocol/candidate-specifications.json");
const SOURCE_PATHS = [
  "research/svi_score_v11_amss_confirmatory/contract.ts",
  "research/svi_score_v11_amss_confirmatory/freeze-selectors.ts",
  "research/svi_score_v11_amss_confirmatory/freeze-protocol.ts",
  "research/svi_score_v11_amss_confirmatory/verify-protocol.ts",
  "research/svi_score_v11_amss_confirmatory/amss-cohort-scenario.R",
  "research/svi_score_v11_amss_confirmatory/generate-cohort.R",
  "research/svi_score_v11_amss_confirmatory/seal-cohort.ts",
  "research/svi_score_v11_amss_confirmatory/prepare-inference.ts",
  "research/svi_score_v11_amss_confirmatory/verify-inference-results.ts",
  "research/svi_score_v11_amss_confirmatory/actions.ts",
  "research/svi_score_v11_amss_confirmatory/freeze-actions.ts",
  "research/svi_score_v11_amss_confirmatory/prepare-v11-input.ts",
  "research/svi_score_v11_amss_confirmatory/freeze-selections.py",
  "research/svi_score_v11_amss_confirmatory/open-truth.ts",
  "research/svi_score_v11_amss_confirmatory/open-truth.R",
  "research/svi_score_v11_amss_confirmatory/build-audit.ts",
  "research/svi_score_v11_amss_confirmatory/economic-loss.ts",
  "research/svi_score_v11_amss_confirmatory/verify-cloud-truth-results.ts",
  "research/svi_score_v11_evidence_adaptive/train_selector.py",
  "research/svi_score_v10_amss/frozen-v9/source/research/svi_score_v9/train_selector.py",
  "research/svi_score_v11_amss_confirmatory/cloud/inference-cloud-run.yaml",
  "research/svi_score_v11_amss_confirmatory/cloud/truth-cloud-run.yaml",
  "research/svi_score_v11_amss_confirmatory/cloud/Dockerfile.truth",
  "research/svi_score_v11_amss_confirmatory/cloud/truth_task.py",
  "tests/svi-score-v11-amss-confirmatory.test.ts",
  "scripts/build_cloud_svi_shards.py",
  "scripts/cloud_svi_shard_task.py",
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
  business_id: string; business_seed: number; truth_seed: number;
  module_order: string; evidence_group: string; population: number;
  gross_margin: number; price: number; market_trend: number;
  season_amplitude: number; social_planning_correlation: number;
  social_budget: number; social_growth: number; social_unit_cost: number;
  social_hill_ec: number; social_hill_slope: number; search_budget: number;
  search_growth: number; search_ctr_scale: number; tv_budget: number;
  tv_growth: number; tv_hill_ec: number; tv_hill_slope: number;
  tv_lag_weeks: number; demand_scale: number; experiment_bias_share: number;
}
const sha256 = (source: Buffer | string) => createHash("sha256").update(source).digest("hex");
const quantiles = (dimension: string) => Array.from(
  { length: V11_AMSS_CONFIRMATORY_CONTRACT.cohort.businesses },
  (_, index) => ({
    index,
    key: sha256(`${V11_AMSS_CONFIRMATORY_CONTRACT.cohort.businessSeedStart}:${dimension}:${index}`),
  }),
).sort((a, b) => a.key.localeCompare(b.key)).reduce((values, item, rank) => {
  values[item.index] = (rank + 0.5) / V11_AMSS_CONFIRMATORY_CONTRACT.cohort.businesses;
  return values;
}, Array<number>(V11_AMSS_CONFIRMATORY_CONTRACT.cohort.businesses));
const ranged = (values: number[], low: number, high: number) => values.map((x) => low + x * (high - low));
const rounded = (value: number, digits = 6) => Number(value.toFixed(digits));

function design(): DesignRow[] {
  const d = {
    population: ranged(quantiles("population"), 80e6, 320e6),
    grossMargin: ranged(quantiles("gross-margin"), 0.35, 0.7),
    price: ranged(quantiles("price"), 55, 140),
    marketTrend: ranged(quantiles("market-trend"), 0.5, 0.82),
    seasonAmplitude: ranged(quantiles("season-amplitude"), 0.1, 0.38),
    socialCorrelation: ranged(quantiles("social-correlation"), 0.2, 0.75),
    socialBudget: ranged(quantiles("social-budget"), 10e6, 45e6),
    socialGrowth: ranged(quantiles("social-growth"), -0.05, 0.25),
    socialUnitCost: ranged(quantiles("social-unit-cost"), 0.004, 0.016),
    socialHillEc: ranged(quantiles("social-hill-ec"), 1.4, 4),
    socialHillSlope: ranged(quantiles("social-hill-slope"), 0.8, 2),
    searchBudget: ranged(quantiles("search-budget"), 5e6, 28e6),
    searchGrowth: ranged(quantiles("search-growth"), 0, 0.35),
    searchCtrScale: ranged(quantiles("search-ctr-scale"), 0.6, 1.4),
    tvBudget: ranged(quantiles("tv-budget"), 25e6, 80e6),
    tvGrowth: ranged(quantiles("tv-growth"), -0.1, 0.12),
    tvHillEc: ranged(quantiles("tv-hill-ec"), 0.8, 3),
    tvHillSlope: ranged(quantiles("tv-hill-slope"), 0.7, 1.8),
    tvLag: ranged(quantiles("tv-lag"), 0, 10.999),
    demandScale: ranged(quantiles("demand-scale"), 0.65, 1.35),
    experimentBias: ranged(quantiles("experiment-bias"), -0.1, 0.1),
  };
  return Array.from({ length: 100 }, (_, index) => ({
    business_id: `amss-v11c-${String(index + 1).padStart(3, "0")}`,
    business_seed: V11_AMSS_CONFIRMATORY_CONTRACT.cohort.businessSeedStart + index,
    truth_seed: V11_AMSS_CONFIRMATORY_CONTRACT.cohort.truthSeedStart + index,
    module_order: V11_AMSS_CONFIRMATORY_MODULE_ORDERS[
      (index + Math.floor(index / 4)) % V11_AMSS_CONFIRMATORY_MODULE_ORDERS.length
    ],
    evidence_group: V11_AMSS_CONFIRMATORY_EVIDENCE_GROUPS[index % 4],
    population: Math.round(d.population[index]), gross_margin: rounded(d.grossMargin[index]),
    price: rounded(d.price[index], 2), market_trend: rounded(d.marketTrend[index]),
    season_amplitude: rounded(d.seasonAmplitude[index]),
    social_planning_correlation: rounded(d.socialCorrelation[index]),
    social_budget: Math.round(d.socialBudget[index]), social_growth: rounded(d.socialGrowth[index]),
    social_unit_cost: rounded(d.socialUnitCost[index]), social_hill_ec: rounded(d.socialHillEc[index]),
    social_hill_slope: rounded(d.socialHillSlope[index]), search_budget: Math.round(d.searchBudget[index]),
    search_growth: rounded(d.searchGrowth[index]), search_ctr_scale: rounded(d.searchCtrScale[index]),
    tv_budget: Math.round(d.tvBudget[index]), tv_growth: rounded(d.tvGrowth[index]),
    tv_hill_ec: rounded(d.tvHillEc[index]), tv_hill_slope: rounded(d.tvHillSlope[index]),
    tv_lag_weeks: Math.floor(d.tvLag[index]), demand_scale: rounded(d.demandScale[index]),
    experiment_bias_share: rounded(d.experimentBias[index]),
  }));
}
function csv(rows: DesignRow[]): string {
  const columns = Object.keys(rows[0]) as (keyof DesignRow)[];
  return `${columns.join(",")}\n${rows.map((row) => columns.map((c) => row[c]).join(",")).join("\n")}\n`;
}
async function immutable(path: string, source: Buffer | string): Promise<void> {
  const bytes = Buffer.isBuffer(source) ? source : Buffer.from(source);
  try { await writeFile(path, bytes, { flag: "wx" }); }
  catch (error) {
    if (!(error instanceof Error && "code" in error && (error as NodeJS.ErrnoException).code === "EEXIST")) throw error;
    if (sha256(await readFile(path)) !== sha256(bytes)) throw new Error(`Frozen protocol changed: ${path}.`);
  }
}

const selectorManifest = await readFile(resolve(SELECTOR_ROOT, "manifest.json"));
const selectorFreeze = JSON.parse(selectorManifest.toString("utf8")) as { status: string; truthOpened: boolean };
if (selectorFreeze.status !== "selectors-frozen-before-cohort-generation" || selectorFreeze.truthOpened) {
  throw new Error("Selectors were not frozen before the AMSS cohort.");
}
const candidateBytes = await readFile(CANDIDATE_SOURCE);
const candidates = JSON.parse(candidateBytes.toString("utf8")) as unknown[];
if (candidates.length !== 24 || sha256(candidateBytes) !== "ec40ca53b0f94243694a16ae5e1b4d1779bad284e42aa8cc38f7d4d529ef6270") {
  throw new Error("The 24-candidate AMSS library is not the frozen V10 library used in V11 development.");
}
const rows = design();
const designSource = csv(rows);
const contractSource = `${JSON.stringify(V11_AMSS_CONFIRMATORY_CONTRACT, null, 2)}\n`;
const sourceSha256: Record<string, string> = {};
for (const path of SOURCE_PATHS) sourceSha256[path] = sha256(await readFile(resolve(path)));
await mkdir(FREEZE_ROOT, { recursive: true });
await immutable(resolve(FREEZE_ROOT, "candidate-specifications.json"), candidateBytes);
await immutable(resolve(FREEZE_ROOT, "cohort-design.csv"), designSource);
await immutable(resolve(FREEZE_ROOT, "scientific-contract.json"), contractSource);
const manifest = {
  artifactId: "flux-v11-amss-fresh-confirmatory-protocol-freeze-v1",
  version: V11_AMSS_CONFIRMATORY_VERSION,
  status: "frozen-before-cohort-generation",
  frozenAt: new Date().toISOString(), truthOpened: false,
  selectorFreezeSha256: sha256(selectorManifest),
  frozenV11SelectorSha256: sha256(await readFile(resolve(SELECTOR_ROOT, "v11-selector.json"))),
  frozenV9SelectorSha256: sha256(await readFile(resolve(SELECTOR_ROOT, "v9-selector.json"))),
  contractSha256: sha256(contractSource), candidateSpecificationsFileSha256: sha256(candidateBytes),
  cohortDesignSha256: sha256(designSource), businesses: rows.length,
  evidenceGroupCounts: Object.fromEntries(V11_AMSS_CONFIRMATORY_EVIDENCE_GROUPS.map((group) => [
    group, rows.filter((row) => row.evidence_group === group).length,
  ])),
  moduleOrderCounts: Object.fromEntries(V11_AMSS_CONFIRMATORY_MODULE_ORDERS.map((order) => [
    order, rows.filter((row) => row.module_order === order).length,
  ])),
  sourceSha256,
  cloudImageDigest: V11_AMSS_CONFIRMATORY_CONTRACT.orchestration.containerImageDigest,
  cloudTruthImageDigest: V11_AMSS_CONFIRMATORY_CONTRACT.orchestration.truthContainerImageDigest,
  discardedPreCohortTruthImageDigest:
    V11_AMSS_CONFIRMATORY_CONTRACT.orchestration.discardedPreCohortTruthImageDigest,
};
await immutable(resolve(FREEZE_ROOT, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify({ frozen: true, businesses: rows.length, candidateFits: 4_800,
  evidenceGroupCounts: manifest.evidenceGroupCounts, moduleOrderCounts: manifest.moduleOrderCounts,
  truthOpened: false }, null, 2));
