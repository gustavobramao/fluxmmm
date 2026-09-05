import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";
import { v5cFeatureTerms } from "../svi_score_v5c/features";

export interface V5DFeatureTerm {
  name: string;
  value: number;
}

function permitted(name: string): boolean {
  return !name.includes("search-phase") &&
    !name.includes("proposal") &&
    !name.includes("restart") &&
    !name.includes("candidate-id");
}

export function v5dFeatureTerms(row: SviScoreV4CandidateRow): V5DFeatureTerm[] {
  return v5cFeatureTerms(row)
    .filter((term) => permitted(term.name))
    .map((term) => ({ name: term.name, value: term.value }));
}

export function v5dFeatureNames(row: SviScoreV4CandidateRow): string[] {
  return v5dFeatureTerms(row).map((term) => term.name);
}

export function v5dFeatureVector(row: SviScoreV4CandidateRow): number[] {
  return v5dFeatureTerms(row).map((term) => term.value);
}
