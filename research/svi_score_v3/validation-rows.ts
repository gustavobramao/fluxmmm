import type { ScoreV6CandidateRow } from "../score_v6/types";
import type { SviScoreV3TrainingFeatureRow } from "./training-rows";
import type { SviScoreV3Record } from "./types";

/**
 * Assemble the held-out validation rows only after truth-blind features have
 * been generated. Split filtering occurs before any posterior label is read.
 */
export function sviValidationRows(
  records: readonly SviScoreV3Record[],
  refreshedFeatures: readonly SviScoreV3TrainingFeatureRow[],
): ScoreV6CandidateRow[] {
  const featureByCandidate = new Map(
    refreshedFeatures.map((row) => [
      `${row.businessId}\u0000${row.candidateId}`,
      row,
    ]),
  );
  if (featureByCandidate.size !== refreshedFeatures.length) {
    throw new Error("Validation feature sidecar contains duplicate candidates.");
  }
  return records.flatMap((record): ScoreV6CandidateRow[] => {
    if (record.split !== "validation") return [];
    if (record.sviDecisionLoss === undefined || !Number.isFinite(record.sviDecisionLoss)) {
      throw new Error(
        `Validation label is not finite for ${record.businessId}/${record.candidateId}.`,
      );
    }
    const refreshed = featureByCandidate.get(
      `${record.businessId}\u0000${record.candidateId}`,
    );
    if (!refreshed) {
      throw new Error(
        `Truth-blind validation features are missing for ${record.businessId}/${record.candidateId}.`,
      );
    }
    return [{
      businessId: record.businessId,
      family: record.family,
      split: "validation",
      candidateId: record.candidateId,
      evidenceArm: record.evidenceArm,
      modelFamily: record.modelFamily,
      eligible: refreshed.eligible,
      reviewEligible: refreshed.reviewEligible,
      eligibilityTier: refreshed.eligibilityTier,
      failedGateCount: refreshed.failedGateCount,
      layerScores: refreshed.layerScores,
      diagnostics: refreshed.diagnostics,
      heuristicScore: refreshed.heuristicScore,
      economicScale: 1,
      decisionLoss: record.sviDecisionLoss,
      cappedRegret: record.sviProfitRegret ?? record.cappedRegret,
      roiError: record.sviRoiError ?? record.roiError,
      contributionError:
        record.sviContributionError ?? record.contributionError,
    }];
  });
}
