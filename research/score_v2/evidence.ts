import type { SeededRandom } from "./random";

export const EVIDENCE_REGISTRY_VERSION = "flux-score-v2-evidence-2026.08.2";

export interface RoiEvidenceEntry {
  id: string;
  tactic: string;
  objective: string;
  geography: string;
  outcome: string;
  median: number;
  low: number;
  high: number;
  effectiveSampleSize: number;
  sourceLabel: string;
  sourceUrl: string;
  evidenceType: "geo-experiment-panel" | "cross-brand-benchmark";
  limitation: string;
}

/**
 * Versioned, auditable proposal distributions—not answer keys. Entries are
 * deliberately wide because published DTC panels are heterogeneous and often
 * self-selected. Synthetic truth is drawn hierarchically around them.
 */
export const ROI_EVIDENCE_REGISTRY: RoiEvidenceEntry[] = [
  {
    id: "dtc-meta-acquisition",
    tactic: "Meta acquisition / prospecting",
    objective: "DTC ecommerce revenue",
    geography: "United States",
    outcome: "Incremental revenue / spend",
    median: 2.92,
    low: 1.05,
    high: 5.6,
    effectiveSampleSize: 45,
    sourceLabel: "Stella 2025 DTC incrementality panel",
    sourceUrl:
      "https://www.stellaheystella.com/blog/2025-dtc-digital-advertising-incrementality-benchmarks",
    evidenceType: "geo-experiment-panel",
    limitation:
      "Self-selected client panel; execution quality and attribution windows vary.",
  },
  {
    id: "dtc-search-nonbrand",
    tactic: "Paid search · non-brand",
    objective: "DTC ecommerce revenue",
    geography: "United States",
    outcome: "Incremental revenue / spend",
    median: 1.46,
    low: 0.45,
    high: 3.9,
    effectiveSampleSize: 35,
    sourceLabel: "Stella 2025 DTC incrementality panel",
    sourceUrl:
      "https://www.stellaheystella.com/blog/2025-dtc-digital-advertising-incrementality-benchmarks",
    evidenceType: "geo-experiment-panel",
    limitation:
      "Search incrementality depends strongly on query mix and demand harvesting.",
  },
  {
    id: "dtc-ctv",
    tactic: "CTV / streaming television",
    objective: "DTC ecommerce revenue",
    geography: "United States",
    outcome: "Incremental revenue / spend",
    median: 3.3,
    low: 0.8,
    high: 7.5,
    effectiveSampleSize: 18,
    sourceLabel: "Stella 2025 DTC incrementality panel",
    sourceUrl:
      "https://www.stellaheystella.com/blog/2025-dtc-digital-advertising-incrementality-benchmarks",
    evidenceType: "geo-experiment-panel",
    limitation:
      "Small channel sample with large creative, reach, and measurement heterogeneity.",
  },
];

export function roiEvidence(id: string): RoiEvidenceEntry {
  const entry = ROI_EVIDENCE_REGISTRY.find((candidate) => candidate.id === id);
  if (!entry) throw new Error(`Unknown ROI evidence entry: ${id}.`);
  return entry;
}

function logNormalSigma(entry: RoiEvidenceEntry): number {
  return Math.max(
    (Math.log(entry.high) - Math.log(entry.low)) / (2 * 1.281551565545),
    0.18,
  );
}

export function sampleHierarchicalRoi(
  id: string,
  random: SeededRandom,
  brandLogEffect: number,
): number {
  const entry = roiEvidence(id);
  const crossBusinessSigma = logNormalSigma(entry) * 0.7;
  const logRoi =
    Math.log(entry.median) +
    brandLogEffect +
    crossBusinessSigma * random.normal();
  // For the realism pilot, the declared evidence interval is the admissible
  // population envelope—not merely a plotting annotation. Dedicated stress
  // scenarios can still override truth explicitly when we want tail behavior.
  return Math.min(entry.high, Math.max(entry.low, Math.exp(logRoi)));
}
