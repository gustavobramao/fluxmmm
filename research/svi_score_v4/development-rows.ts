import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { sviTrainingRows } from "../svi_score_v3/training-rows";
import type { SviScoreV3TrainingFeatureRow } from "../svi_score_v3/training-rows";
import type { SviScoreV3Record } from "../svi_score_v3/types";
import { sviValidationRows } from "../svi_score_v3/validation-rows";
import type {
  SviScoreV4CandidateRow,
  SviScoreV4TemporalReceipt,
} from "./types";

export const V4_TEMPORAL_SIDECAR_PATH = resolve(
  ".flux-artifacts/svi-score-v4/development-temporal-features.json",
);
export const V4_FLIGHT_HOLDOUT_SIDECAR_PATH = resolve(
  ".flux-artifacts/svi-score-v4/development-flight-holdout.json",
);

export interface SviScoreV4TemporalSidecarRow {
  businessId: string;
  candidateId: string;
  temporal: SviScoreV4TemporalReceipt;
}

export interface SviScoreV4FlightHoldoutSidecarRow {
  businessId: string;
  candidateId: string;
  score: number;
  folds: number;
  meanWape: number | null;
  meanSkill: number | null;
}

export async function loadSviScoreV4DevelopmentRows(): Promise<
  SviScoreV4CandidateRow[]
> {
  const [
    recordSource,
    trainFeatureSource,
    validationFeatureSource,
    temporalSource,
    flightSource,
  ] =
    await Promise.all([
      readFile(resolve(".flux-artifacts/svi-score-v3/design-pilot-records.json")),
      readFile(resolve(".flux-artifacts/svi-score-v3/train-features-v2.json")),
      readFile(resolve(".flux-artifacts/svi-score-v3/validation-features-v2.json")),
      readFile(V4_TEMPORAL_SIDECAR_PATH),
      readFile(V4_FLIGHT_HOLDOUT_SIDECAR_PATH),
    ]);
  const records = JSON.parse(recordSource.toString("utf8")) as SviScoreV3Record[];
  const trainFeatures = JSON.parse(
    trainFeatureSource.toString("utf8"),
  ) as SviScoreV3TrainingFeatureRow[];
  const validationFeatures = JSON.parse(
    validationFeatureSource.toString("utf8"),
  ) as SviScoreV3TrainingFeatureRow[];
  const temporalRows = JSON.parse(
    temporalSource.toString("utf8"),
  ) as SviScoreV4TemporalSidecarRow[];
  const flightRows = JSON.parse(
    flightSource.toString("utf8"),
  ) as SviScoreV4FlightHoldoutSidecarRow[];
  const flightByCandidate = new Map(
    flightRows.map((row) => [
      `${row.businessId}\u0000${row.candidateId}`,
      row,
    ]),
  );
  const temporalByCandidate = new Map(
    temporalRows.map((row) => [
      `${row.businessId}\u0000${row.candidateId}`,
      row.temporal,
    ]),
  );
  if (temporalByCandidate.size !== temporalRows.length) {
    throw new Error("V4 temporal sidecar contains duplicate candidates.");
  }
  const baseRows = [
    ...sviTrainingRows(records, trainFeatures),
    ...sviValidationRows(records, validationFeatures),
  ];
  const development = baseRows.map((row): SviScoreV4CandidateRow => {
    const temporal = temporalByCandidate.get(
      `${row.businessId}\u0000${row.candidateId}`,
    );
    if (!temporal) {
      throw new Error(
        `V4 temporal features missing for ${row.businessId}/${row.candidateId}.`,
      );
    }
    const flight = flightByCandidate.get(
      `${row.businessId}\u0000${row.candidateId}`,
    );
    if (!flight) {
      throw new Error(
        `V4 flight-holdout evidence missing for ${row.businessId}/${row.candidateId}.`,
      );
    }
    const temporalIdentification = 100 * Math.sqrt(
      Math.max(0.01, temporal.carryoverSupport / 100) *
        Math.max(0.01, flight.score / 100),
    );
    return {
      ...row,
      temporal: {
        ...temporal,
        wholeFlightGeneralization: flight.score,
        temporalIdentification,
        detail: `${temporal.detail} ${flight.folds} forward whole-flight holdout fold${flight.folds === 1 ? "" : "s"}.`,
      },
    };
  });
  const businesses = new Set(development.map((row) => row.businessId));
  if (businesses.size !== 420 || development.length !== 20_160) {
    throw new Error(
      `V4 development expects 420 businesses/20,160 rows; found ${businesses.size}/${development.length}.`,
    );
  }
  if (development.some((row) => row.split === "audit")) {
    throw new Error("The sealed V3 audit is forbidden in V4 development.");
  }
  return development;
}
