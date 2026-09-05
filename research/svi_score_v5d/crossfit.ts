import { createHash } from "node:crypto";
import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";
import { v5cCandidateRegions } from "../svi_score_v5c/crossfit";
import { V5D_TRAINING_CONFIGURATION } from "./contract";
import {
  learnV5DModel,
  predictV5DLoss,
  V5D_MAP_FEATURE_PROVIDER,
  V5D_MAP_SELECTION_POLICY,
  type V5DFeatureProvider,
  type V5DSelectionPolicy,
} from "./model";
import type {
  V5DCrossFittedModel,
  V5DLossPrediction,
} from "./types";

export function fitV5DCrossFittedModel(
  rows: readonly SviScoreV4CandidateRow[],
  regionByCandidateId = v5cCandidateRegions(rows.map((row) => row.candidateId)),
  provider: V5DFeatureProvider = V5D_MAP_FEATURE_PROVIDER,
  selectionPolicy: V5DSelectionPolicy = V5D_MAP_SELECTION_POLICY,
): V5DCrossFittedModel {
  if (!rows.length) throw new Error("V5D cross-fitting requires training rows.");
  const regionIds = [...new Set(Object.values(regionByCandidateId))].sort(
    (left, right) => left - right,
  );
  return {
    regionByCandidateId: structuredClone(regionByCandidateId),
    regions: regionIds.map((region) => {
      const excludedCandidateIds = Object.entries(regionByCandidateId)
        .filter(([, value]) => value === region)
        .map(([candidateId]) => candidateId)
        .sort();
      const training = rows.filter(
        (row) => regionByCandidateId[row.candidateId] !== region,
      );
      return {
        region,
        excludedCandidateIds,
        model: learnV5DModel(
          training,
          V5D_TRAINING_CONFIGURATION,
          provider,
          selectionPolicy,
        ),
      };
    }),
  };
}

export function predictV5DCrossFitted(
  row: SviScoreV4CandidateRow,
  crossFitted: V5DCrossFittedModel,
  provider: V5DFeatureProvider = V5D_MAP_FEATURE_PROVIDER,
): V5DLossPrediction {
  const region = crossFitted.regionByCandidateId[row.candidateId];
  const receipt = crossFitted.regions.find((item) => item.region === region);
  if (!receipt) throw new Error(`V5D has no excluded-region model for ${row.candidateId}.`);
  if (!receipt.excludedCandidateIds.includes(row.candidateId)) {
    throw new Error(`V5D attempted an in-region prediction for ${row.candidateId}.`);
  }
  return predictV5DLoss(row, receipt.model, provider);
}

export function v5dCrossFitHash(crossFitted: V5DCrossFittedModel): string {
  return createHash("sha256").update(JSON.stringify(crossFitted)).digest("hex");
}
