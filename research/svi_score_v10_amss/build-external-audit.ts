import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseCsv, toNumber } from "../../lib/mmm/csv";
import {
  V10_AMSS_CONTRACT,
  V10_AMSS_DECISION_SCENARIOS,
  V10_AMSS_VERSION,
} from "./contract";
import { calculateV10EconomicLoss } from "./economic-loss";

const ROOT = resolve(".flux-artifacts/svi-score-v10-amss");
const OBSERVABLE_PATH = resolve(ROOT, "observable-freeze/observable-candidates.json");
const OBSERVABLE_MANIFEST_PATH = resolve(ROOT, "observable-freeze/manifest.json");
const SELECTION_PATH = resolve(ROOT, "observable-freeze/frozen-selections.json");
const RECEIPT_PATH = resolve(ROOT, "opened-truth/truth-open-receipt.json");
const AMENDMENT_001_PATH = resolve(ROOT, "opened-truth/technical-amendment-001.json");
const AMENDMENT_002_PATH = resolve(ROOT, "opened-truth/technical-amendment-002.json");
const AMENDMENT_003_PATH = resolve(ROOT, "opened-truth/technical-amendment-003.json");
const AMENDMENT_004_PATH = resolve(ROOT, "opened-truth/technical-amendment-004.json");
const CLOUD_IMAGE_RECEIPT_PATH = resolve(ROOT, "opened-truth/cloud-image-receipt.json");
const CLOUD_TRUTH_RECEIPT_PATH = resolve(ROOT, "opened-truth/cloud-truth-retrieval-receipt.json");
const BUSINESS_METADATA_PATH = resolve(ROOT, "sealed-cohort/business-metadata.csv");
const PART_ROOT = resolve(ROOT, "opened-truth/parts");
const OUTPUT_ROOT = resolve(ROOT, "external-audit");
const RESULT_PATH = resolve(OUTPUT_ROOT, "result.json");
const LOSS_PATH = resolve(OUTPUT_ROOT, "candidate-losses.csv");

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function key(businessId: string, candidateId: string, scenario: string): string {
  return `${businessId}\u0000${candidateId}\u0000${scenario}`;
}

function quantile(values: readonly number[], probability: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const position = Math.min(1, Math.max(0, probability)) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function mean(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0) / Math.max(values.length, 1);
}

function csvCell(value: string | number | boolean): string {
  const source = String(value);
  return /[",\n\r]/.test(source) ? `"${source.replaceAll('"', '""')}"` : source;
}

function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4_294_967_296;
  };
}

function pairedBootstrap(left: readonly number[], right: readonly number[]) {
  if (left.length !== 100 || right.length !== 100) {
    throw new Error("V10 paired bootstrap requires exactly 100 businesses.");
  }
  const generator = random(10_031_771);
  const differences = Array.from({ length: 10_000 }, () => {
    let total = 0;
    for (let index = 0; index < left.length; index += 1) {
      const sampled = Math.floor(generator() * left.length);
      total += left[sampled] - right[sampled];
    }
    return total / left.length;
  });
  const point = mean(left) - mean(right);
  return {
    pointDifference: point,
    confidenceInterval95: [quantile(differences, 0.025), quantile(differences, 0.975)],
    oneSidedUpper95: quantile(differences, 0.95),
    probabilityV9LowerLoss: mean(differences.map((value) => value < 0 ? 1 : 0)),
    resamples: differences.length,
    seed: 10_031_771,
  };
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
      throw new Error(`Frozen V10 external-audit artifact changed: ${path}.`);
    }
  }
}

const [
  observableSource, observableManifestSource, selectionSource, receiptSource,
  amendment001Source, amendment002Source, amendment003Source, amendment004Source,
  cloudImageReceiptSource,
  cloudTruthReceiptSource, metadataSource,
] =
  await Promise.all([
    readFile(OBSERVABLE_PATH),
    readFile(OBSERVABLE_MANIFEST_PATH),
    readFile(SELECTION_PATH),
    readFile(RECEIPT_PATH),
    readFile(AMENDMENT_001_PATH),
    readFile(AMENDMENT_002_PATH),
    readFile(AMENDMENT_003_PATH),
    readFile(AMENDMENT_004_PATH),
    readFile(CLOUD_IMAGE_RECEIPT_PATH),
    readFile(CLOUD_TRUTH_RECEIPT_PATH),
    readFile(BUSINESS_METADATA_PATH),
  ]);
const observable = JSON.parse(observableSource.toString("utf8")) as {
  version: string;
  rows: Array<{
    businessId: string;
    evidenceGroup: string;
    candidateId: string;
    valid: boolean;
    heuristicScore: number;
  }>;
};
const selections = JSON.parse(selectionSource.toString("utf8")) as {
  version: string;
  status: string;
  businesses: number;
  selections: Array<{
    businessId: string;
    evidenceGroup: string;
    v9CandidateId: string;
    heuristicCandidateId: string;
  }>;
};
const receipt = JSON.parse(receiptSource.toString("utf8")) as {
  version: string;
  status: string;
  observableManifestSha256: string;
  frozenSelectionsSha256: string;
  workers: number;
};
const amendment002 = JSON.parse(amendment002Source.toString("utf8")) as {
  version: string;
  status: string;
  scientificContractChanged: boolean;
  completedTruthPartsBeforeAmendment: number;
  cloudExecution: { tasks: number; businessesPerTask: number };
  immutableInputs: {
    candidateActionsSha256: string;
    frozenSelectionsSha256: string;
    truthOpenReceiptSha256: string;
    technicalAmendment001Sha256: string;
  };
};
const amendment003 = JSON.parse(amendment003Source.toString("utf8")) as {
  version: string;
  status: string;
  scientificContractChanged: boolean;
  failedBuild: { imageProduced: boolean; truthTasksStarted: number; truthPartsProduced: number };
  immutableInputs: {
    technicalAmendment002Sha256: string;
    candidateActionsSha256: string;
    frozenSelectionsSha256: string;
  };
};
const amendment004 = JSON.parse(amendment004Source.toString("utf8")) as {
  version: string;
  status: string;
  scientificContractChanged: boolean;
  frozenCandidatesChanged: boolean;
  frozenActionsChanged: boolean;
  frozenSelectionsChanged: boolean;
  amssTruthChanged: boolean;
  requiredCorrection: string;
  correctedSources: {
    auditAssemblerSha256: string;
    economicLossModuleSha256: string;
    contractTestSha256: string;
  };
  immutableInputs: {
    cloudTruthRetrievalReceiptSha256: string;
    candidateActionsSha256: string;
    frozenSelectionsSha256: string;
  };
};
const cloudImageReceipt = JSON.parse(cloudImageReceiptSource.toString("utf8")) as {
  version: string;
  status: string;
  truthTasksStartedAtFreeze: number;
  scientificContractChanged: boolean;
  immutableInputs: {
    technicalAmendment003Sha256: string;
    candidateActionsSha256: string;
    frozenSelectionsSha256: string;
  };
};
const cloudTruthReceipt = JSON.parse(cloudTruthReceiptSource.toString("utf8")) as {
  version: string;
  status: string;
  cloudRunExecution: string;
  businesses: number;
  parts: number;
  rows: number;
  uniqueCandidateScenarioRows: number;
  files: Array<{ path: string; sha256: string; rows: number }>;
};
const baselineBudgetByBusiness = new Map(
  parseCsv(metadataSource.toString("utf8")).rows.map((row) => [
    String(row.business_id),
    toNumber(row.year4_paid_social_spend) +
      toNumber(row.year4_search_spend) +
      toNumber(row.year4_tv_spend),
  ]),
);
if (
  observable.version !== V10_AMSS_VERSION || selections.version !== V10_AMSS_VERSION ||
  receipt.version !== V10_AMSS_VERSION || selections.businesses !== 100 ||
  selections.status !== "frozen-before-amss-truth-open" ||
  receipt.status !== "truth-opened-for-frozen-action-evaluation" ||
  receipt.observableManifestSha256 !== sha256(observableManifestSource) ||
  receipt.frozenSelectionsSha256 !== sha256(selectionSource)
) throw new Error("V10 frozen selections and truth-open receipt are inconsistent.");
if (
  amendment002.version !== V10_AMSS_VERSION ||
  amendment002.status !== "frozen-cloud-execution-only-amendment" ||
  amendment002.scientificContractChanged ||
  amendment002.completedTruthPartsBeforeAmendment !== 0 ||
  amendment002.cloudExecution.tasks !== 100 ||
  amendment002.cloudExecution.businessesPerTask !== 1 ||
  amendment002.immutableInputs.candidateActionsSha256 !== sha256(
    await readFile(resolve(ROOT, "observable-freeze/candidate-actions.csv")),
  ) ||
  amendment002.immutableInputs.frozenSelectionsSha256 !== sha256(selectionSource) ||
  amendment002.immutableInputs.truthOpenReceiptSha256 !== sha256(receiptSource) ||
  amendment002.immutableInputs.technicalAmendment001Sha256 !== sha256(amendment001Source)
) throw new Error("V10 cloud truth execution amendment is inconsistent.");
if (
  amendment003.version !== V10_AMSS_VERSION ||
  amendment003.status !== "frozen-container-dependency-only-amendment" ||
  amendment003.scientificContractChanged || amendment003.failedBuild.imageProduced ||
  amendment003.failedBuild.truthTasksStarted !== 0 || amendment003.failedBuild.truthPartsProduced !== 0 ||
  amendment003.immutableInputs.technicalAmendment002Sha256 !== sha256(amendment002Source) ||
  amendment003.immutableInputs.candidateActionsSha256 !==
    amendment002.immutableInputs.candidateActionsSha256 ||
  amendment003.immutableInputs.frozenSelectionsSha256 !== sha256(selectionSource)
) throw new Error("V10 container dependency amendment is inconsistent.");
if (
  amendment004.version !== V10_AMSS_VERSION ||
  amendment004.status !== "frozen-contract-enforcement-correction" ||
  amendment004.scientificContractChanged || amendment004.frozenCandidatesChanged ||
  amendment004.frozenActionsChanged || amendment004.frozenSelectionsChanged ||
  amendment004.amssTruthChanged ||
  amendment004.immutableInputs.cloudTruthRetrievalReceiptSha256 !== sha256(cloudTruthReceiptSource) ||
  amendment004.immutableInputs.candidateActionsSha256 !==
    amendment002.immutableInputs.candidateActionsSha256 ||
  amendment004.immutableInputs.frozenSelectionsSha256 !== sha256(selectionSource) ||
  amendment004.correctedSources.auditAssemblerSha256 !== sha256(
    await readFile(resolve("research/svi_score_v10_amss/build-external-audit.ts")),
  ) ||
  amendment004.correctedSources.economicLossModuleSha256 !== sha256(
    await readFile(resolve("research/svi_score_v10_amss/economic-loss.ts")),
  ) ||
  amendment004.correctedSources.contractTestSha256 !== sha256(
    await readFile(resolve("tests/svi-score-v10-amss.test.ts")),
  )
) throw new Error("V10 normalized-loss-cap correction amendment is inconsistent.");
if (
  cloudImageReceipt.version !== V10_AMSS_VERSION ||
  cloudImageReceipt.status !== "image-built-and-pinned-before-truth-tasks" ||
  cloudImageReceipt.truthTasksStartedAtFreeze !== 0 || cloudImageReceipt.scientificContractChanged ||
  cloudImageReceipt.immutableInputs.technicalAmendment003Sha256 !== sha256(amendment003Source) ||
  cloudImageReceipt.immutableInputs.candidateActionsSha256 !==
    amendment002.immutableInputs.candidateActionsSha256 ||
  cloudImageReceipt.immutableInputs.frozenSelectionsSha256 !== sha256(selectionSource)
) throw new Error("V10 cloud truth image receipt is inconsistent.");

const partNames = (await readdir(PART_ROOT)).filter((name) => name.endsWith(".csv")).sort();
const expectedParts = amendment002.cloudExecution.tasks;
if (partNames.length !== expectedParts) {
  throw new Error(`V10 truth evaluation has ${partNames.length}/${expectedParts} parts.`);
}
if (
  cloudTruthReceipt.version !== V10_AMSS_VERSION ||
  cloudTruthReceipt.status !== "all-cloud-truth-parts-retrieved-and-verified" ||
  cloudTruthReceipt.cloudRunExecution !== "flux-v10-amss-truth-001-5lkmj" ||
  cloudTruthReceipt.businesses !== 100 || cloudTruthReceipt.parts !== expectedParts ||
  cloudTruthReceipt.rows !== 19_200 || cloudTruthReceipt.uniqueCandidateScenarioRows !== 19_200 ||
  cloudTruthReceipt.files.length !== expectedParts
) throw new Error("V10 cloud truth retrieval receipt is incomplete.");
const realized = new Map<string, {
  profit: number;
  profitSe: number;
  baselineProfit: number;
  baselineProfitSe: number;
}>();
for (const name of partNames) {
  const partSource = await readFile(resolve(PART_ROOT, name));
  const receiptFile = cloudTruthReceipt.files.find((file) => file.path === name);
  if (!receiptFile || receiptFile.rows !== 192 || receiptFile.sha256 !== sha256(partSource)) {
    throw new Error(`V10 cloud truth part does not match its retrieval receipt: ${name}.`);
  }
  const parsed = parseCsv(partSource.toString("utf8"));
  for (const row of parsed.rows) {
    const itemKey = key(String(row.business_id), String(row.candidate_id), String(row.scenario));
    if (realized.has(itemKey)) throw new Error(`Duplicate V10 truth row: ${itemKey}.`);
    realized.set(itemKey, {
      profit: toNumber(row.realized_profit),
      profitSe: toNumber(row.realized_profit_se),
      baselineProfit: toNumber(row.baseline_profit),
      baselineProfitSe: toNumber(row.baseline_profit_se),
    });
  }
}
if (realized.size !== 19_200) throw new Error(`V10 truth contains ${realized.size}/19200 actions.`);

const byBusiness = new Map<string, typeof observable.rows>();
for (const row of observable.rows) {
  byBusiness.set(row.businessId, [...(byBusiness.get(row.businessId) ?? []), row]);
}
const candidateLosses: Array<{
  businessId: string;
  evidenceGroup: string;
  candidateId: string;
  valid: boolean;
  heuristicScore: number;
  scenarioLosses: Record<string, number>;
  scenarioUncappedLosses: Record<string, number>;
  meanLoss: number;
  p90Loss: number;
  risk: number;
  uncappedMeanLoss: number;
  uncappedP90Loss: number;
  uncappedRisk: number;
}> = [];
for (const [businessId, rows] of byBusiness) {
  if (rows.length !== 48) throw new Error(`${businessId} has ${rows.length}/48 candidates.`);
  const oracle = Object.fromEntries(V10_AMSS_DECISION_SCENARIOS.map((scenario) => [
    scenario.id,
    Math.max(...rows.map((row) => realized.get(key(businessId, row.candidateId, scenario.id))!.profit)),
  ]));
  for (const row of rows) {
    const scenarioLosses: Record<string, number> = {};
    const scenarioUncappedLosses: Record<string, number> = {};
    for (const scenario of V10_AMSS_DECISION_SCENARIOS) {
      const measured = realized.get(key(businessId, row.candidateId, scenario.id));
      if (!measured) throw new Error(`Missing V10 truth for ${businessId}/${row.candidateId}/${scenario.id}.`);
      const normalized = calculateV10EconomicLoss({
        oracleProfit: oracle[scenario.id],
        realizedProfit: measured.profit,
        baselineProfit: measured.baselineProfit,
        baselineBudget: baselineBudgetByBusiness.get(businessId) ?? 0,
        normalizedLossCap: V10_AMSS_CONTRACT.decisions.normalizedLossCap,
      });
      scenarioLosses[scenario.id] = normalized.capped;
      scenarioUncappedLosses[scenario.id] = normalized.uncapped;
    }
    const losses = V10_AMSS_DECISION_SCENARIOS.map((scenario) => scenarioLosses[scenario.id]);
    const uncappedLosses = V10_AMSS_DECISION_SCENARIOS.map(
      (scenario) => scenarioUncappedLosses[scenario.id],
    );
    const meanLoss = mean(losses);
    const p90Loss = quantile(losses, 0.9);
    const uncappedMeanLoss = mean(uncappedLosses);
    const uncappedP90Loss = quantile(uncappedLosses, 0.9);
    candidateLosses.push({
      businessId,
      evidenceGroup: row.evidenceGroup,
      candidateId: row.candidateId,
      valid: row.valid,
      heuristicScore: row.heuristicScore,
      scenarioLosses,
      scenarioUncappedLosses,
      meanLoss,
      p90Loss,
      risk: 0.65 * meanLoss + 0.35 * p90Loss,
      uncappedMeanLoss,
      uncappedP90Loss,
      uncappedRisk: 0.65 * uncappedMeanLoss + 0.35 * uncappedP90Loss,
    });
  }
}

const lossByCandidate = new Map(candidateLosses.map((row) => [
  `${row.businessId}\u0000${row.candidateId}`,
  row,
]));
const selectionRows = selections.selections.map((selection) => {
  const candidates = candidateLosses.filter((row) => row.businessId === selection.businessId);
  const valid = candidates.filter((row) => row.valid);
  const v9 = lossByCandidate.get(`${selection.businessId}\u0000${selection.v9CandidateId}`);
  const heuristic = lossByCandidate.get(`${selection.businessId}\u0000${selection.heuristicCandidateId}`);
  if (!v9 || !heuristic || !valid.length) throw new Error(`Frozen selection is missing for ${selection.businessId}.`);
  const oracle = [...candidates].sort((left, right) => left.risk - right.risk)[0];
  return {
    businessId: selection.businessId,
    evidenceGroup: selection.evidenceGroup,
    v9CandidateId: v9.candidateId,
    v9Risk: v9.risk,
    v9UncappedRisk: v9.uncappedRisk,
    heuristicCandidateId: heuristic.candidateId,
    heuristicRisk: heuristic.risk,
    heuristicUncappedRisk: heuristic.uncappedRisk,
    randomValidExpectedRisk: mean(valid.map((row) => row.risk)),
    randomValidExpectedUncappedRisk: mean(valid.map((row) => row.uncappedRisk)),
    oracleCandidateId: oracle.candidateId,
    oracleRisk: oracle.risk,
    oracleUncappedRisk: oracle.uncappedRisk,
    v9OracleExcessRisk: v9.risk - oracle.risk,
    heuristicOracleExcessRisk: heuristic.risk - oracle.risk,
    v9Won: v9.risk < heuristic.risk,
    tied: Math.abs(v9.risk - heuristic.risk) < 1e-12,
  };
});

const v9Risks = selectionRows.map((row) => row.v9Risk);
const heuristicRisks = selectionRows.map((row) => row.heuristicRisk);
const randomRisks = selectionRows.map((row) => row.randomValidExpectedRisk);
const oracleRisks = selectionRows.map((row) => row.oracleRisk);
const v9UncappedRisks = selectionRows.map((row) => row.v9UncappedRisk);
const heuristicUncappedRisks = selectionRows.map((row) => row.heuristicUncappedRisk);
const randomUncappedRisks = selectionRows.map((row) => row.randomValidExpectedUncappedRisk);
const oracleUncappedRisks = selectionRows.map((row) => row.oracleUncappedRisk);
const endpoint = (values: readonly number[]) => ({
  mean: mean(values),
  p90: quantile(values, 0.9),
  p95: quantile(values, 0.95),
});
const subgroup = Object.fromEntries(
  [...new Set(selectionRows.map((row) => row.evidenceGroup))].sort().map((group) => {
    const rows = selectionRows.filter((row) => row.evidenceGroup === group);
    return [group, {
      businesses: rows.length,
      v9: endpoint(rows.map((row) => row.v9Risk)),
      heuristic: endpoint(rows.map((row) => row.heuristicRisk)),
      meanPairedDifference: mean(rows.map((row) => row.v9Risk - row.heuristicRisk)),
    }];
  }),
);
const result = {
  artifactId: "flux-v10-amss-external-transport-result-v1",
  version: V10_AMSS_VERSION,
  status: "frozen-external-audit-complete-no-post-open-tuning",
  provenance: {
    observableDatasetSha256: sha256(observableSource),
    observableManifestSha256: sha256(observableManifestSource),
    frozenSelectionsSha256: sha256(selectionSource),
    truthOpenReceiptSha256: sha256(receiptSource),
    technicalAmendment001Sha256: sha256(amendment001Source),
    technicalAmendment002Sha256: sha256(amendment002Source),
    technicalAmendment003Sha256: sha256(amendment003Source),
    technicalAmendment004Sha256: sha256(amendment004Source),
    cloudImageReceiptSha256: sha256(cloudImageReceiptSource),
    cloudTruthRetrievalReceiptSha256: sha256(cloudTruthReceiptSource),
    truthParts: partNames,
    frozenV9Retrained: false,
    candidateActionsFrozenBeforeTruth: true,
  },
  cohort: { businesses: 100, candidateModels: 4_800, decisionActions: 19_200 },
  estimand: "realized in-pool excess economic loss of frozen model selections in AMSS",
  endpoints: {
    frozenV9: endpoint(v9Risks),
    originalFluxHeuristic: endpoint(heuristicRisks),
    uniformRandomValidExpectation: endpoint(randomRisks),
    inPoolOracle: endpoint(oracleRisks),
    pairedV9MinusHeuristic: pairedBootstrap(v9Risks, heuristicRisks),
    v9BusinessWins: selectionRows.filter((row) => row.v9Won).length,
    ties: selectionRows.filter((row) => row.tied).length,
    heuristicBusinessWins: selectionRows.filter((row) => !row.v9Won && !row.tied).length,
    uncappedSafety: {
      frozenV9: endpoint(v9UncappedRisks),
      originalFluxHeuristic: endpoint(heuristicUncappedRisks),
      uniformRandomValidExpectation: endpoint(randomUncappedRisks),
      riskAtCappedInPoolOracle: endpoint(oracleUncappedRisks),
    },
  },
  subgroupsDescriptiveOnly: subgroup,
  selections: selectionRows,
  governance: {
    truthOpenedExactlyOnce: true,
    retrainingAfterOpen: false,
    reselectionAfterOpen: false,
    subgroupInferencePermitted: false,
    externalRealAdvertiserClaimPermitted: false,
  },
};

const lossSource = [
  [
    "business_id", "evidence_group", "candidate_id", "valid", "heuristic_score",
    ...V10_AMSS_DECISION_SCENARIOS.map((scenario) => `${scenario.id}_loss`),
    ...V10_AMSS_DECISION_SCENARIOS.map((scenario) => `${scenario.id}_uncapped_loss`),
    "mean_loss", "p90_loss", "risk",
    "uncapped_mean_loss", "uncapped_p90_loss", "uncapped_risk",
  ].join(","),
  ...candidateLosses.map((row) => [
    row.businessId,
    row.evidenceGroup,
    row.candidateId,
    row.valid,
    row.heuristicScore,
    ...V10_AMSS_DECISION_SCENARIOS.map((scenario) => row.scenarioLosses[scenario.id]),
    ...V10_AMSS_DECISION_SCENARIOS.map((scenario) => row.scenarioUncappedLosses[scenario.id]),
    row.meanLoss,
    row.p90Loss,
    row.risk,
    row.uncappedMeanLoss,
    row.uncappedP90Loss,
    row.uncappedRisk,
  ].map(csvCell).join(",")),
  "",
].join("\n");
await mkdir(OUTPUT_ROOT, { recursive: true });
await writeImmutable(LOSS_PATH, lossSource);
await writeImmutable(RESULT_PATH, `${JSON.stringify(result, null, 2)}\n`);

console.log(JSON.stringify(result.endpoints, null, 2));
