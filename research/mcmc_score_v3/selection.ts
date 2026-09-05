import learnedScoreV6Artifact from "../score_v6/artifacts/learned-score-v6-pilot.json";
import { scoreV6 } from "../score_v6/learn";
import type { ScoreV6CandidateRow } from "../score_v6/types";
import { MCMC_SCORE_V3_CONTRACT } from "./contract";
import type { McmcSelectionReason } from "./types";

export interface SelectableMcmcCandidate {
  row: ScoreV6CandidateRow;
}

export interface SelectedMcmcCandidate<T extends SelectableMcmcCandidate> {
  candidate: T;
  reason: McmcSelectionReason;
  probability: number;
}

const V6_WEIGHTS = learnedScoreV6Artifact.model.weights;
const FEATURES = Object.keys(V6_WEIGHTS) as Array<keyof typeof V6_WEIGHTS>;

function seededUnit(value: string): number {
  let hash = 2166136261;
  for (const character of value) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 0x1_0000_0000;
}

function score(candidate: SelectableMcmcCandidate): number {
  return scoreV6(candidate.row, V6_WEIGHTS);
}

function featureDistance(
  left: SelectableMcmcCandidate,
  right: SelectableMcmcCandidate,
): number {
  return FEATURES.reduce((total, feature) => {
    const difference =
      ((left.row.diagnostics[feature] ?? 50) -
        (right.row.diagnostics[feature] ?? 50)) /
      100;
    return total + difference * difference;
  }, 0);
}

function highestTier<T extends SelectableMcmcCandidate>(candidates: T[]): T[] {
  const decisionGrade = candidates.filter((candidate) => candidate.row.eligible);
  if (decisionGrade.length) return decisionGrade;
  const review = candidates.filter((candidate) => candidate.row.reviewEligible);
  return review.length ? review : candidates;
}

/**
 * Predeclared shortlist using only fit-visible information. MAP decision loss
 * and simulator truth are deliberately unavailable to this function.
 */
export function selectMcmcCandidates<T extends SelectableMcmcCandidate>(
  candidates: T[],
  requested: number = MCMC_SCORE_V3_CONTRACT.targetCandidatesPerBusiness,
): SelectedMcmcCandidate<T>[] {
  const pool = highestTier(candidates);
  const ranked = [...pool].sort(
    (left, right) =>
      score(right) - score(left) ||
      left.row.candidateId.localeCompare(right.row.candidateId),
  );
  const selected: SelectedMcmcCandidate<T>[] = [];
  const ids = new Set<string>();
  const add = (
    candidate: T | undefined,
    reason: McmcSelectionReason,
    probability = 1,
  ) => {
    if (!candidate || ids.has(candidate.row.candidateId) || selected.length >= requested) return;
    ids.add(candidate.row.candidateId);
    selected.push({ candidate, reason, probability });
  };

  ranked.slice(0, MCMC_SCORE_V3_CONTRACT.selection.v6Leaders).forEach((candidate) =>
    add(candidate, "v6-leader")
  );

  const families = [...new Set(ranked.map((candidate) => candidate.row.modelFamily))];
  families
    .slice(0, MCMC_SCORE_V3_CONTRACT.selection.familyRepresentatives)
    .forEach((family) =>
      add(
        ranked.find((candidate) => candidate.row.modelFamily === family),
        "family-representative",
      )
    );

  for (
    let index = 0;
    index < MCMC_SCORE_V3_CONTRACT.selection.diagnosticDiversity;
    index += 1
  ) {
    const remaining = ranked.filter((candidate) => !ids.has(candidate.row.candidateId));
    const diverse = remaining.sort((left, right) => {
      const leftDistance = Math.min(
        ...selected.map((item) => featureDistance(left, item.candidate)),
      );
      const rightDistance = Math.min(
        ...selected.map((item) => featureDistance(right, item.candidate)),
      );
      return rightDistance - leftDistance || score(right) - score(left);
    })[0];
    add(diverse, "diagnostic-diversity");
  }

  const approximationRisk = ranked
    .filter((candidate) => !ids.has(candidate.row.candidateId))
    .sort((left, right) => {
      const risk = (candidate: T) =>
        200 -
        (candidate.row.diagnostics["roi-identification"] ?? 50) -
        (candidate.row.diagnostics["functional-form"] ?? 50);
      return risk(right) - risk(left) || score(right) - score(left);
    });
  approximationRisk
    .slice(0, MCMC_SCORE_V3_CONTRACT.selection.approximationRisk)
    .forEach((candidate) => add(candidate, "approximation-risk"));

  const randomPool = ranked.filter((candidate) => !ids.has(candidate.row.candidateId));
  const randomCount = Math.min(
    MCMC_SCORE_V3_CONTRACT.selection.randomAudit,
    Math.max(0, requested - selected.length),
  );
  const randomProbability = randomCount / Math.max(randomPool.length, 1);
  [...randomPool]
    .sort(
      (left, right) =>
        seededUnit(`${left.row.businessId}:${left.row.candidateId}`) -
        seededUnit(`${right.row.businessId}:${right.row.candidateId}`),
    )
    .slice(0, randomCount)
    .forEach((candidate) => add(candidate, "random-audit", randomProbability));

  ranked
    .filter((candidate) => !ids.has(candidate.row.candidateId))
    .forEach((candidate) => add(candidate, "diagnostic-diversity"));
  return selected.slice(0, requested);
}
