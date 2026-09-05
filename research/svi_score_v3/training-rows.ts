import type { ScoreV6CandidateRow } from "../score_v6/types";
import type { SviScoreV3Record } from "./types";

export interface SviScoreV3TrainingFeatureRow {
  businessId: string;
  candidateId: string;
  eligible: boolean;
  reviewEligible: boolean;
  eligibilityTier: ScoreV6CandidateRow["eligibilityTier"];
  failedGateCount: number;
  layerScores: ScoreV6CandidateRow["layerScores"];
  diagnostics: ScoreV6CandidateRow["diagnostics"];
  heuristicScore: number;
}

export function sviTrainingRows(
  records: readonly SviScoreV3Record[],
  refreshedFeatures: readonly SviScoreV3TrainingFeatureRow[] = [],
): ScoreV6CandidateRow[] {
  const featureByCandidate = new Map(
    refreshedFeatures.map((row) => [
      `${row.businessId}\u0000${row.candidateId}`,
      row,
    ]),
  );
  return records.flatMap((record): ScoreV6CandidateRow[] => {
    // Split filtering deliberately happens before any posterior label is read.
    // Validation and audit records therefore cannot influence training choices.
    if (record.split !== "train") return [];
    if (record.sviDecisionLoss === undefined || !Number.isFinite(record.sviDecisionLoss)) {
      return [];
    }
    const refreshed = featureByCandidate.get(
      `${record.businessId}\u0000${record.candidateId}`,
    );
    return [{
      businessId: record.businessId,
      family: record.family,
      split: "train",
      candidateId: record.candidateId,
      evidenceArm: record.evidenceArm,
      modelFamily: record.modelFamily,
      eligible: refreshed?.eligible ?? record.eligible,
      reviewEligible: refreshed?.reviewEligible ?? record.reviewEligible,
      eligibilityTier: refreshed?.eligibilityTier ?? record.eligibilityTier,
      failedGateCount: refreshed?.failedGateCount ?? record.failedGateCount,
      layerScores: refreshed?.layerScores ?? record.layerScores,
      diagnostics: refreshed?.diagnostics ?? record.diagnostics,
      heuristicScore: refreshed?.heuristicScore ?? record.heuristicScore,
      economicScale: 1,
      decisionLoss: record.sviDecisionLoss,
      cappedRegret: record.sviProfitRegret ?? record.cappedRegret,
      roiError: record.sviRoiError ?? record.roiError,
      contributionError:
        record.sviContributionError ?? record.contributionError,
    }];
  });
}
