import type { ScoreV6CandidateRow } from "../score_v6/types";
import { SVI_SCORE_V3_CONTRACT } from "./contract";
import type { SviGoldAuditSelection } from "./types";

function seededUnit(value: string): number {
  let hash = 2166136261;
  for (const character of value) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 0x1_0000_0000;
}

function modelStratum(row: ScoreV6CandidateRow): "bayesian" | "advanced" {
  return row.modelFamily === "advanced" ? "advanced" : "bayesian";
}

/**
 * Selects the gold NUTS audit independently of score, truth and SVI quality.
 * Within each selected business exactly one candidate is sampled from each
 * estimator-family × evidence-arm stratum, and every inclusion probability is
 * recorded for inverse-probability analyses.
 */
export function selectSviGoldAudit(
  rows: ScoreV6CandidateRow[],
): SviGoldAuditSelection[] {
  const byBusiness = new Map<string, ScoreV6CandidateRow[]>();
  rows.forEach((row) => {
    const existing = byBusiness.get(row.businessId) ?? [];
    existing.push(row);
    byBusiness.set(row.businessId, existing);
  });
  const selectedBusinessIds = new Set<string>();
  for (const split of ["train", "validation", "audit"] as const) {
    const splitEntries = [...byBusiness.entries()].filter(
      ([, candidates]) => candidates[0]?.split === split,
    );
    const families = [...new Set(splitEntries.map(([, rows]) => rows[0].family))];
    const target = SVI_SCORE_V3_CONTRACT.goldAudit.businessAllocation[split];
    const basePerFamily = Math.floor(target / Math.max(families.length, 1));
    let remainder = target - basePerFamily * families.length;
    families.sort().forEach((family) => {
      const familyTarget = basePerFamily + (remainder-- > 0 ? 1 : 0);
      splitEntries
        .filter(([, rows]) => rows[0].family === family)
        .sort(
          ([left], [right]) =>
            seededUnit(`business:${left}`) - seededUnit(`business:${right}`),
        )
        .slice(0, familyTarget)
        .forEach(([businessId]) => selectedBusinessIds.add(businessId));
    });
  }

  const selection: SviGoldAuditSelection[] = [];
  for (const businessId of selectedBusinessIds) {
    const candidates = byBusiness.get(businessId) ?? [];
    if (!candidates.length) continue;
    const split = candidates[0].split;
    const familyBusinessCount = new Set(
      rows
        .filter((row) => row.split === split && row.family === candidates[0].family)
        .map((row) => row.businessId),
    ).size;
    const familyCount = new Set(
      rows.filter((row) => row.split === split).map((row) => row.family),
    ).size;
    const familyTarget = Math.floor(
      SVI_SCORE_V3_CONTRACT.goldAudit.businessAllocation[split] /
        Math.max(familyCount, 1),
    );
    const businessProbability = familyTarget / Math.max(familyBusinessCount, 1);
    for (const modelFamily of ["bayesian", "advanced"] as const) {
      for (const evidenceArm of [
        "experiments-only",
        "benchmark-gap-fill",
      ] as const) {
        const stratumRows = candidates.filter(
          (row) =>
            modelStratum(row) === modelFamily && row.evidenceArm === evidenceArm,
        );
        const chosen = [...stratumRows].sort(
          (left, right) =>
            seededUnit(`candidate:${businessId}:${left.candidateId}`) -
            seededUnit(`candidate:${businessId}:${right.candidateId}`),
        )[0];
        if (!chosen) continue;
        const candidateProbability = 1 / stratumRows.length;
        selection.push({
          businessId,
          family: chosen.family,
          split,
          candidateId: chosen.candidateId,
          evidenceArm,
          modelFamily: chosen.modelFamily,
          businessInclusionProbability: businessProbability,
          candidateInclusionProbability: candidateProbability,
          jointInclusionProbability: businessProbability * candidateProbability,
          stratum: `${modelFamily}:${evidenceArm}`,
        });
      }
    }
  }
  return selection;
}
