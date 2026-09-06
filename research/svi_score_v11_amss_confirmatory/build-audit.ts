import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseCsv, toNumber } from "../../lib/mmm/csv";
import {
  V11_AMSS_CONFIRMATORY_CONTRACT,
  V11_AMSS_CONFIRMATORY_DECISION_SCENARIOS,
  V11_AMSS_CONFIRMATORY_VERSION,
} from "./contract";
import { calculateV11ConfirmatoryEconomicLoss } from "./economic-loss";

const ROOT = resolve(".flux-artifacts/svi-score-v11-amss-confirmatory");
const OUTPUT_ROOT = resolve(ROOT, "external-audit");
const sha256 = (source: Buffer | string) => createHash("sha256").update(source).digest("hex");
const mean = (values: readonly number[]) => values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1);
function quantile(values: readonly number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const position = Math.min(1, Math.max(0, p)) * (sorted.length - 1);
  const low = Math.floor(position); const high = Math.ceil(position); const weight = position - low;
  return sorted[low] * (1 - weight) + sorted[high] * weight;
}
function rng(seed: number): () => number {
  let state = seed >>> 0;
  return () => { state += 0x6d2b79f5; let x = state;
    x = Math.imul(x ^ x >>> 15, x | 1); x ^= x + Math.imul(x ^ x >>> 7, x | 61);
    return ((x ^ x >>> 14) >>> 0) / 4_294_967_296; };
}
const endpoint = (rows: readonly CandidateLoss[]) => {
  const values = rows.map((row) => row.risk); const uncapped = rows.map((row) => row.uncappedRisk);
  const meanRisk = mean(values); const p90 = quantile(values, 0.9);
  return { businesses: rows.length, mean: meanRisk, p90, p95: quantile(values, 0.95),
    selectionObjective: 0.65 * meanRisk + 0.35 * p90,
    uncappedMean: mean(uncapped), uncappedP90: quantile(uncapped, 0.9),
    uncappedP95: quantile(uncapped, 0.95) };
};
interface CandidateLoss { businessId: string; evidenceGroup: string; candidateId: string; valid: boolean;
  scenarioLosses: Record<string, number>; scenarioUncappedLosses: Record<string, number>;
  risk: number; uncappedRisk: number; }
function bootstrap(left: readonly CandidateLoss[], right: readonly CandidateLoss[], seed: number) {
  const rightById = new Map(right.map((row) => [row.businessId, row]));
  const groups = new Map<string, Array<[CandidateLoss, CandidateLoss]>>();
  for (const row of left) groups.set(row.evidenceGroup, [
    ...(groups.get(row.evidenceGroup) ?? []), [row, rightById.get(row.businessId)!],
  ]);
  const generator = rng(seed); const objective = []; const meanDifference = [];
  for (let iteration = 0; iteration < 10_000; iteration += 1) {
    const l: CandidateLoss[] = []; const r: CandidateLoss[] = [];
    for (const pairs of groups.values()) for (let i = 0; i < pairs.length; i += 1) {
      const pair = pairs[Math.floor(generator() * pairs.length)]; l.push(pair[0]); r.push(pair[1]);
    }
    objective.push(endpoint(l).selectionObjective - endpoint(r).selectionObjective);
    meanDifference.push(mean(l.map((row, i) => row.risk - r[i].risk)));
  }
  return {
    estimand: "V11 minus comparator; negative favors V11",
    meanDifference: mean(left.map((row) => row.risk - rightById.get(row.businessId)!.risk)),
    meanDifferenceInterval95: [quantile(meanDifference, 0.025), quantile(meanDifference, 0.975)],
    selectionObjectiveDifference: endpoint(left).selectionObjective - endpoint(right).selectionObjective,
    selectionObjectiveDifferenceInterval95: [quantile(objective, 0.025), quantile(objective, 0.975)],
    oneSidedObjectiveUpper95: quantile(objective, 0.95), resamples: 10_000, seed,
    stratifiedByEvidenceGroup: true,
  };
}
function key(business: string, candidate: string, scenario: string) {
  return `${business}\u0000${candidate}\u0000${scenario}`;
}
async function immutable(path: string, value: string) {
  try { await writeFile(path, value, { flag: "wx" }); }
  catch (error) { if (!(error instanceof Error && "code" in error &&
    (error as NodeJS.ErrnoException).code === "EEXIST")) throw error;
    if ((await readFile(path, "utf8")) !== value) throw new Error(`Audit artifact changed: ${path}.`); }
}

const [observableSource, observableManifestSource, selectionsSource, truthReceiptSource,
  cloudReceiptSource, metadataSource] = await Promise.all([
  readFile(resolve(ROOT, "observable-freeze/observable-candidates.json")),
  readFile(resolve(ROOT, "observable-freeze/manifest.json")),
  readFile(resolve(ROOT, "observable-freeze/frozen-selections.json")),
  readFile(resolve(ROOT, "opened-truth/truth-open-receipt.json")),
  readFile(resolve(ROOT, "opened-truth/cloud-truth-retrieval-receipt.json")),
  readFile(resolve(ROOT, "sealed-cohort/business-metadata.csv")),
]);
const observable = JSON.parse(observableSource.toString("utf8")) as { version: string; rows: Array<{
  businessId: string; evidenceGroup: string; candidateId: string; valid: boolean }> };
const selections = JSON.parse(selectionsSource.toString("utf8")) as { version: string; status: string;
  businesses: number; selections: Array<{ businessId: string; evidenceGroup: string;
    v11CandidateId: string; v9CandidateId: string; predictionOnlyCandidateId: string;
    evidenceGate: Record<string, number> }> };
const truthReceipt = JSON.parse(truthReceiptSource.toString("utf8")) as { version: string; status: string;
  observableManifestSha256: string; frozenSelectionsSha256: string };
const cloudReceipt = JSON.parse(cloudReceiptSource.toString("utf8")) as { version: string; status: string;
  cloudRunExecution: string; files: Array<{ path: string; rows: number; sha256: string }> };
if (observable.version !== V11_AMSS_CONFIRMATORY_VERSION || selections.version !== V11_AMSS_CONFIRMATORY_VERSION ||
  truthReceipt.version !== V11_AMSS_CONFIRMATORY_VERSION || cloudReceipt.version !== V11_AMSS_CONFIRMATORY_VERSION ||
  selections.status !== "all-comparator-selections-frozen-before-amss-truth-open" || selections.businesses !== 100 ||
  truthReceipt.status !== "truth-opened-for-frozen-action-evaluation" ||
  cloudReceipt.status !== "all-cloud-truth-parts-retrieved-and-verified" ||
  truthReceipt.observableManifestSha256 !== sha256(observableManifestSource) ||
  truthReceipt.frozenSelectionsSha256 !== sha256(selectionsSource)) throw new Error("Audit freeze chain is invalid.");

const baselineBudget = new Map(parseCsv(metadataSource.toString("utf8")).rows.map((row) => [String(row.business_id),
  toNumber(row.year4_paid_social_spend) + toNumber(row.year4_search_spend) + toNumber(row.year4_tv_spend)]));
const realized = new Map<string, { profit: number; baselineProfit: number }>();
const partNames = (await readdir(resolve(ROOT, "opened-truth/parts"))).filter((name) => name.endsWith(".csv")).sort();
if (partNames.length !== 100) throw new Error("Truth result set is incomplete.");
for (const name of partNames) {
  const source = await readFile(resolve(ROOT, "opened-truth/parts", name));
  const receipt = cloudReceipt.files.find((item) => item.path === name);
  if (!receipt || receipt.rows !== 192 || receipt.sha256 !== sha256(source)) throw new Error(`Truth receipt mismatch: ${name}.`);
  for (const row of parseCsv(source.toString("utf8")).rows) {
    const id = key(String(row.business_id), String(row.candidate_id), String(row.scenario));
    if (realized.has(id)) throw new Error(`Duplicate realized action: ${id}.`);
    realized.set(id, { profit: toNumber(row.realized_profit), baselineProfit: toNumber(row.baseline_profit) });
  }
}
if (realized.size !== 19_200) throw new Error(`Truth contains ${realized.size}/19,200 actions.`);

const rowsByBusiness = new Map<string, typeof observable.rows>();
for (const row of observable.rows) rowsByBusiness.set(row.businessId, [...(rowsByBusiness.get(row.businessId) ?? []), row]);
const candidateLosses: CandidateLoss[] = [];
for (const [businessId, rows] of rowsByBusiness) {
  const validRows = rows.filter((row) => row.valid);
  if (rows.length !== 48 || validRows.length < 2) throw new Error(`${businessId} has an invalid candidate set.`);
  const oracleProfit = Object.fromEntries(
    V11_AMSS_CONFIRMATORY_DECISION_SCENARIOS.map((scenario) => [
      scenario.id,
      Math.max(...validRows.map((row) =>
        realized.get(key(businessId, row.candidateId, scenario.id))!.profit
      )),
    ]),
  );
  for (const row of rows) {
    const scenarioLosses: Record<string, number> = {}; const scenarioUncappedLosses: Record<string, number> = {};
    for (const scenario of V11_AMSS_CONFIRMATORY_DECISION_SCENARIOS) {
      const measured = realized.get(key(businessId, row.candidateId, scenario.id))!;
      const loss = calculateV11ConfirmatoryEconomicLoss({ oracleProfit: oracleProfit[scenario.id],
        realizedProfit: measured.profit, baselineProfit: measured.baselineProfit,
        baselineBudget: baselineBudget.get(businessId)!, normalizedLossCap: 1 });
      scenarioLosses[scenario.id] = loss.capped; scenarioUncappedLosses[scenario.id] = loss.uncapped;
    }
    const capped = V11_AMSS_CONFIRMATORY_DECISION_SCENARIOS.map((s) => scenarioLosses[s.id]);
    const uncapped = V11_AMSS_CONFIRMATORY_DECISION_SCENARIOS.map((s) => scenarioUncappedLosses[s.id]);
    candidateLosses.push({ businessId, evidenceGroup: row.evidenceGroup, candidateId: row.candidateId,
      valid: row.valid, scenarioLosses, scenarioUncappedLosses,
      risk: 0.65 * mean(capped) + 0.35 * quantile(capped, 0.9),
      uncappedRisk: 0.65 * mean(uncapped) + 0.35 * quantile(uncapped, 0.9) });
  }
}
const lossById = new Map(candidateLosses.map((row) => [`${row.businessId}\u0000${row.candidateId}`, row]));
const chosen = (business: string, candidate: string) => {
  const row = lossById.get(`${business}\u0000${candidate}`); if (!row) throw new Error(`Missing ${business}/${candidate}.`); return row;
};
const v11: CandidateLoss[] = []; const v9: CandidateLoss[] = []; const prediction: CandidateLoss[] = [];
const randomExpected: CandidateLoss[] = []; const oracle: CandidateLoss[] = [];
const details = selections.selections.map((selection) => {
  const all = candidateLosses.filter((row) => row.businessId === selection.businessId && row.valid);
  const a = chosen(selection.businessId, selection.v11CandidateId);
  const b = chosen(selection.businessId, selection.v9CandidateId);
  const c = chosen(selection.businessId, selection.predictionOnlyCandidateId);
  const d = [...all].sort((left, right) => left.risk - right.risk)[0];
  const e: CandidateLoss = { ...d, candidateId: "uniform-random-valid-expectation",
    risk: mean(all.map((row) => row.risk)), uncappedRisk: mean(all.map((row) => row.uncappedRisk)) };
  v11.push(a); v9.push(b); prediction.push(c); oracle.push(d); randomExpected.push(e);
  return { businessId: selection.businessId, evidenceGroup: selection.evidenceGroup,
    v11CandidateId: a.candidateId, v11Risk: a.risk, v9CandidateId: b.candidateId, v9Risk: b.risk,
    predictionOnlyCandidateId: c.candidateId, predictionOnlyRisk: c.risk,
    oracleCandidateId: d.candidateId, oracleRisk: d.risk, evidenceGate: selection.evidenceGate };
});
const endpoints = { evidenceAdaptiveV11: endpoint(v11), frozenV9: endpoint(v9),
  predictionOnly: endpoint(prediction), uniformRandomValidExpectation: endpoint(randomExpected), inPoolOracle: endpoint(oracle) };
const comparisons = { v11MinusFrozenV9: bootstrap(v11, v9, 11_210_031),
  v11MinusPredictionOnly: bootstrap(v11, prediction, 11_210_041) };
const betterComparatorP95 = Math.min(endpoints.frozenV9.p95, endpoints.predictionOnly.p95);
const checks = {
  lowerObjectiveThanFrozenV9: endpoints.evidenceAdaptiveV11.selectionObjective < endpoints.frozenV9.selectionObjective,
  noHigherObjectiveThanPredictionOnly: endpoints.evidenceAdaptiveV11.selectionObjective <= endpoints.predictionOnly.selectionObjective,
  p95TailSafety: endpoints.evidenceAdaptiveV11.p95 <= betterComparatorP95 + 0.02,
};
const groups = [...new Set(v11.map((row) => row.evidenceGroup))].sort();
const result = { artifactId: "flux-v11-amss-fresh-confirmatory-result-v1", version: V11_AMSS_CONFIRMATORY_VERSION,
  status: "fresh-confirmatory-audit-complete-no-post-truth-tuning",
  provenance: { observableDatasetSha256: sha256(observableSource), frozenSelectionsSha256: sha256(selectionsSource),
    truthOpenReceiptSha256: sha256(truthReceiptSource), cloudTruthRetrievalReceiptSha256: sha256(cloudReceiptSource),
    cloudRunExecution: cloudReceipt.cloudRunExecution, candidateActionsFrozenBeforeTruth: true,
    selectorRetrainedAfterTruth: false, reselectionAfterTruth: false },
  cohort: { businesses: 100, candidateModels: 4_800, decisionActions: 19_200, evidenceGroups: groups },
  primaryEndpoint: "65% cohort mean candidate risk plus 35% cohort P90 candidate risk",
  endpoints, pairedComparisons: comparisons,
  confirmatoryDecisionRule: { predeclared: V11_AMSS_CONFIRMATORY_CONTRACT.evaluation.confirmatoryDecisionRule,
    checks, passed: Object.values(checks).every(Boolean) },
  subgroupsDescriptiveOnly: Object.fromEntries(groups.map((group) => [group, {
    evidenceAdaptiveV11: endpoint(v11.filter((row) => row.evidenceGroup === group)),
    frozenV9: endpoint(v9.filter((row) => row.evidenceGroup === group)),
    predictionOnly: endpoint(prediction.filter((row) => row.evidenceGroup === group)),
  }])), selections: details,
  governance: { truthOpenedExactlyOnce: true, postTruthTuningPermitted: false,
    publicationUpdateBeforeAcceptancePermitted: false, externalRealAdvertiserClaimPermitted: false },
};
const columns = ["business_id", "evidence_group", "candidate_id", "valid", ...V11_AMSS_CONFIRMATORY_DECISION_SCENARIOS.map((s) => `${s.id}_loss`),
  ...V11_AMSS_CONFIRMATORY_DECISION_SCENARIOS.map((s) => `${s.id}_uncapped_loss`), "risk", "uncapped_risk"];
const csv = [columns.join(","), ...candidateLosses.map((row) => [row.businessId, row.evidenceGroup, JSON.stringify(row.candidateId), row.valid,
  ...V11_AMSS_CONFIRMATORY_DECISION_SCENARIOS.map((s) => row.scenarioLosses[s.id]),
  ...V11_AMSS_CONFIRMATORY_DECISION_SCENARIOS.map((s) => row.scenarioUncappedLosses[s.id]), row.risk, row.uncappedRisk].join(",")), ""].join("\n");
await mkdir(OUTPUT_ROOT, { recursive: true });
await immutable(resolve(OUTPUT_ROOT, "candidate-losses.csv"), csv);
await immutable(resolve(OUTPUT_ROOT, "result.json"), `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ endpoints, comparisons, decision: result.confirmatoryDecisionRule }, null, 2));
