import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";
import { v5aFeatureTerms } from "../svi_score_v5a/features";

export interface V5CFeatureTerm {
  name: string;
  value: number;
  base: boolean;
}

function permitted(name: string): boolean {
  return !name.includes("search-phase") &&
    !name.includes("proposal") &&
    !name.includes("restart") &&
    !name.includes("candidate-id");
}

export function v5cFeatureTerms(row: SviScoreV4CandidateRow): V5CFeatureTerm[] {
  return v5aFeatureTerms(row)
    .filter((term) => permitted(term.name))
    .map((term) => ({ ...term }));
}

export function v5cFeatureNames(row: SviScoreV4CandidateRow): string[] {
  return v5cFeatureTerms(row).map((term) => term.name);
}

export function v5cFeatureVector(row: SviScoreV4CandidateRow): number[] {
  return v5cFeatureTerms(row).map((term) => term.value);
}

export function v5cPriorWeights(row: SviScoreV4CandidateRow): number[] {
  const terms = v5cFeatureTerms(row);
  const baseCount = terms.filter((term) => term.base).length;
  const extensionCount = Math.max(1, terms.length - baseCount);
  return terms.map((term) =>
    term.base ? 0.72 / Math.max(baseCount, 1) : 0.28 / extensionCount
  );
}
