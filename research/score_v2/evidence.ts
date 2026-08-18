import type { SeededRandom } from "./random";

export const EVIDENCE_REGISTRY_VERSION = "flux-score-v6-evidence-2026.08.1";

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

const LATENT_TRUTH_POPULATION: Record<
  string,
  { median: number; low: number; high: number; logSigma: number }
> = {
  "dtc-meta-acquisition": { median: 2.45, low: 0.45, high: 8.2, logSigma: 0.62 },
  "dtc-search-nonbrand": { median: 1.25, low: 0.18, high: 6.2, logSigma: 0.72 },
  "dtc-ctv": { median: 2.05, low: 0.25, high: 9.5, logSigma: 0.78 },
};

export function sampleHierarchicalRoi(
  id: string,
  random: SeededRandom,
  brandLogEffect: number,
): number {
  // The hidden population is versioned separately from model-visible evidence.
  // This avoids making benchmark agreement a disguised answer key.
  const entry = LATENT_TRUTH_POPULATION[id];
  if (!entry) throw new Error(`Unknown latent ROI population: ${id}.`);
  const logRoi =
    Math.log(entry.median) +
    brandLogEffect +
    entry.logSigma * random.normal();
  return Math.min(entry.high, Math.max(entry.low, Math.exp(logRoi)));
}
