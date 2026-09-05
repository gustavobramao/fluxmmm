import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";
import type { SviScoreV3Record } from "../svi_score_v3/types";
import {
  v5dFeatureNames,
  v5dFeatureVector,
} from "../svi_score_v5d/features";
import type {
  V5DFeatureProvider,
  V5DSelectionPolicy,
} from "../svi_score_v5d/model";
import { SVI_SCORE_V5D_SVI_CONTRACT } from "./contract";

const RECORD_PATH = resolve(
  ".flux-artifacts/svi-score-v3/design-pilot-records.json",
);

export interface V5DSviPosteriorRecord {
  status: "labelled" | "review";
  posteriorRoi: Record<string, number>;
  diagnostics: NonNullable<SviScoreV3Record["diagnosticsReceipt"]>;
  seedDiagnostics: NonNullable<SviScoreV3Record["seedDiagnostics"]>;
}

export type V5DSviRecordMap = Map<string, V5DSviPosteriorRecord>;

export function v5dSviKey(row: Pick<SviScoreV4CandidateRow, "businessId" | "candidateId">): string {
  return `${row.businessId}\u0000${row.candidateId}`;
}

function finite(value: unknown, fallback = 0): number {
  const output = Number(value);
  return Number.isFinite(output) ? output : fallback;
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) /
    Math.max(values.length, 1);
}

function standardDeviation(values: readonly number[]): number {
  const center = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - center) ** 2)));
}

function logRoi(value: number): number {
  return Math.log(Math.max(value, 1e-4));
}

const CHANNEL_FEATURES = [
  "meta_acquisition_spend",
  "google_search_nonbrand_spend",
  "ctv_spend",
] as const;

const POSTERIOR_FEATURE_NAMES = [
  "svi:status:labelled",
  "svi:convergence:finite",
  "svi:convergence:elbo-stable",
  "svi:convergence:seed-agreement",
  "svi:convergence:adjudication-used",
  "svi:convergence:log-elbo-drift",
  "svi:convergence:seed-log-roi-difference",
  "svi:predictive:coverage",
  "svi:decision:implausible-probability",
  "svi:decision:log-relative-roi-width",
  "svi:margin:predictive-coverage",
  "svi:margin:roi-plausibility",
  "svi:margin:roi-precision",
  "svi:roi:log-center:paid-social",
  "svi:roi:log-center:nonbrand-search",
  "svi:roi:log-center:ctv",
  "svi:roi:log-center:mean",
  "svi:roi:log-center:standard-deviation",
  "svi:roi:log-center:minimum",
  "svi:roi:log-center:maximum",
  "svi:roi:log-center:range",
] as const;

function recordFor(
  row: SviScoreV4CandidateRow,
  records: V5DSviRecordMap,
): V5DSviPosteriorRecord {
  const record = records.get(v5dSviKey(row));
  if (!record) {
    throw new Error(`V5D-SVI has no posterior record for ${row.businessId}/${row.candidateId}.`);
  }
  return record;
}

function posteriorVector(record: V5DSviPosteriorRecord): number[] {
  const diagnostics = record.diagnostics;
  const channelLogs = CHANNEL_FEATURES.map((channel) =>
    logRoi(finite(record.posteriorRoi[channel], 1e-4))
  );
  const minimum = Math.min(...channelLogs);
  const maximum = Math.max(...channelLogs);
  return [
    record.status === "labelled" ? 1 : 0,
    diagnostics.finite ? 1 : 0,
    diagnostics.elboStable ? 1 : 0,
    diagnostics.seedAgreement ? 1 : 0,
    diagnostics.adjudicationUsed ? 1 : 0,
    Math.log1p(Math.max(0, finite(diagnostics.maximumElboDrift))),
    finite(diagnostics.maximumSeedLogRoiDifference),
    finite(diagnostics.predictiveCoverage),
    finite(diagnostics.maximumImplausibleProbability),
    Math.log1p(Math.max(0, finite(diagnostics.maximumRelativeRoiWidth))),
    finite(diagnostics.predictiveCoverage) -
      SVI_SCORE_V5D_SVI_CONTRACT.safety.predictiveCoverageMinimum,
    SVI_SCORE_V5D_SVI_CONTRACT.safety.maximumImplausibleProbability -
      finite(diagnostics.maximumImplausibleProbability),
    SVI_SCORE_V5D_SVI_CONTRACT.safety.maximumRelativeRoiWidth -
      finite(diagnostics.maximumRelativeRoiWidth),
    ...channelLogs,
    mean(channelLogs),
    standardDeviation(channelLogs),
    minimum,
    maximum,
    maximum - minimum,
  ];
}

export async function loadV5DSviPosteriorRecords(
  rows: readonly SviScoreV4CandidateRow[],
): Promise<V5DSviRecordMap> {
  const permitted = new Set(rows.map(v5dSviKey));
  const source = JSON.parse(
    await readFile(RECORD_PATH, "utf8"),
  ) as SviScoreV3Record[];
  const selected = source.filter((record) => permitted.has(v5dSviKey(record)));
  if (
    selected.length !== rows.length ||
    selected.some((record) => record.split === "audit")
  ) {
    throw new Error("V5D-SVI posterior join accessed an incomplete or audit cohort.");
  }
  const output: V5DSviRecordMap = new Map();
  selected.forEach((record) => {
    if (
      (record.status !== "labelled" && record.status !== "review") ||
      !record.posteriorRoi ||
      !record.diagnosticsReceipt ||
      !record.seedDiagnostics
    ) {
      throw new Error(`V5D-SVI posterior summary is incomplete for ${record.candidateId}.`);
    }
    output.set(v5dSviKey(record), {
      status: record.status,
      posteriorRoi: structuredClone(record.posteriorRoi),
      diagnostics: structuredClone(record.diagnosticsReceipt),
      seedDiagnostics: structuredClone(record.seedDiagnostics),
    });
  });
  if (output.size !== rows.length) {
    throw new Error(`V5D-SVI expected ${rows.length} unique posterior records; found ${output.size}.`);
  }
  return output;
}

export function v5dSviFeatureProvider(records: V5DSviRecordMap): V5DFeatureProvider {
  return {
    names: (row) => [
      ...v5dFeatureNames(row),
      ...POSTERIOR_FEATURE_NAMES,
    ],
    vector: (row) => [
      ...v5dFeatureVector(row),
      ...posteriorVector(recordFor(row, records)),
    ],
  };
}

function passes(record: V5DSviPosteriorRecord): boolean {
  const diagnostics = record.diagnostics;
  return record.status === "labelled" &&
    diagnostics.finite &&
    diagnostics.elboStable &&
    diagnostics.seedAgreement &&
    diagnostics.predictiveCoverage >=
      SVI_SCORE_V5D_SVI_CONTRACT.safety.predictiveCoverageMinimum &&
    diagnostics.maximumImplausibleProbability <=
      SVI_SCORE_V5D_SVI_CONTRACT.safety.maximumImplausibleProbability &&
    diagnostics.maximumRelativeRoiWidth <=
      SVI_SCORE_V5D_SVI_CONTRACT.safety.maximumRelativeRoiWidth;
}

export function v5dSviSelectionPolicy(records: V5DSviRecordMap): V5DSelectionPolicy {
  const isSafe = (row: SviScoreV4CandidateRow) => passes(recordFor(row, records));
  return {
    passes: isSafe,
    pool: (rows) => {
      const safe = rows.filter(isSafe);
      if (safe.length) {
        return { rows: safe, hasSafetyAcceptedCandidate: true };
      }
      const converged = rows.filter((row) => {
        const record = recordFor(row, records);
        return record.status === "labelled" && record.diagnostics.finite &&
          record.diagnostics.elboStable && record.diagnostics.seedAgreement;
      });
      return {
        rows: converged.length ? converged : [...rows],
        hasSafetyAcceptedCandidate: false,
      };
    },
  };
}
