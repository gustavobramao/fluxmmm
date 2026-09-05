import { SCORE_V6_DIAGNOSTIC_GROUPS } from "../score_v6/types";
import type { ScoreV6Feature } from "../score_v6/types";
import {
  V4_TEMPORAL_FEATURES,
  type SviScoreV4CandidateRow,
  type SviScoreV4Feature,
} from "./types";

export const SVI_SCORE_V4_FEATURES = [
  ...(Object.values(SCORE_V6_DIAGNOSTIC_GROUPS).flat() as ScoreV6Feature[]),
  ...V4_TEMPORAL_FEATURES,
] as const satisfies readonly SviScoreV4Feature[];

export function v4FeatureValue(
  row: SviScoreV4CandidateRow,
  feature: SviScoreV4Feature,
): number {
  if (feature === "whole-flight-generalization") {
    return row.temporal.wholeFlightGeneralization;
  }
  if (feature === "carryover-support") return row.temporal.carryoverSupport;
  if (feature === "post-flight-residual-stability") {
    return row.temporal.postFlightResidualStability;
  }
  if (feature === "kernel-distinguishability") {
    return row.temporal.kernelDistinguishability;
  }
  return row.diagnostics[feature] ?? 50;
}

export function transformedV4Feature(
  row: SviScoreV4CandidateRow,
  feature: SviScoreV4Feature,
): number {
  const value = Math.min(100, Math.max(1, v4FeatureValue(row, feature)));
  return Math.log(value / 100);
}
