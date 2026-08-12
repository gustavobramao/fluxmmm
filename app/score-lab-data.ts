export type ScoreLabLayer =
  | "generalization"
  | "structure"
  | "causal"
  | "decision";

export interface ScoreLabCandidate {
  id: string;
  label: string;
  score: number;
  layers: Record<ScoreLabLayer, number>;
  roi: Record<"paid_social" | "search" | "tv", number>;
  roiError: number;
  contributionError: number;
  budgetRegret: number;
}

export interface ScoreLabScenario {
  id: string;
  label: string;
  shortLabel: string;
  description: string;
  seed: number;
  benchmarkError: number;
  truth: Record<"paid_social" | "search" | "tv", number>;
  complication: string;
  candidates: ScoreLabCandidate[];
}

const candidateLabels: Record<string, string> = {
  "short-memory": "Geometric · short memory",
  balanced: "Geometric · balanced",
  "long-memory": "Geometric · long memory",
  "strong-saturation": "Geometric · strong saturation",
  weibull: "Weibull · delayed response",
};

function candidate(
  id: string,
  score: number,
  layers: [number, number, number, number],
  roi: [number, number, number],
  roiError: number,
  contributionError: number,
  budgetRegret: number,
): ScoreLabCandidate {
  return {
    id,
    label: candidateLabels[id],
    score,
    layers: {
      generalization: layers[0],
      structure: layers[1],
      causal: layers[2],
      decision: layers[3],
    },
    roi: { paid_social: roi[0], search: roi[1], tv: roi[2] },
    roiError,
    contributionError,
    budgetRegret,
  };
}

const sharedTruth = { paid_social: 2.5, search: 1.2, tv: 1.8 } as const;

export const SCORE_LAB_SCENARIOS: ScoreLabScenario[] = [
  {
    id: "clean-identification",
    label: "Clean identification",
    shortLabel: "Clean",
    description: "Low confounding, an observable demand proxy, and moderate noise.",
    complication: "The easy control: a well-specified model should recover ROI and choose the right allocation.",
    seed: 20260813,
    benchmarkError: 0.13,
    truth: sharedTruth,
    candidates: [
      candidate("short-memory", 61.6, [92.9, 90.7, 80.4, 36.7], [2.39, 1.55, 0.61], 0.32, 0.229, 0),
      candidate("balanced", 61.9, [93.8, 86.5, 83.5, 36.7], [2.25, 1.42, 0.8], 0.255, 0.205, 0),
      candidate("long-memory", 76.5, [92.3, 89.4, 75.7, 66.1], [2.26, 0.71, 1.67], 0.3, 0.194, 0),
      candidate("strong-saturation", 52.8, [92.9, 88.1, 81.3, 25.1], [1.45, 0.9, 0.6], 0.515, 0.407, 0),
      candidate("weibull", 66.1, [77.5, 48.8, 89.8, 56.4], [1.95, 0, 1.45], 6.737, 0.466, 0),
    ],
  },
  {
    id: "demand-confounded-search",
    label: "Demand-confounded search",
    shortLabel: "Search confounding",
    description: "Search spend rises with hidden consumer demand, which also raises baseline revenue.",
    complication: "A model can forecast sales well by assigning latent demand to search—even when the causal search ROI is only 1.20×.",
    seed: 20260814,
    benchmarkError: 0.144,
    truth: sharedTruth,
    candidates: [
      candidate("short-memory", 84.4, [93.5, 86.3, 86, 78.6], [2.65, 3.81, 1.26], 0.708, 0.932, 0.612),
      candidate("balanced", 82.3, [88.5, 87.3, 81.9, 77.8], [2.52, 3.15, 1.43], 0.569, 0.677, 0.534),
      candidate("long-memory", 74.9, [79, 78.6, 67.4, 76.4], [2.47, 2.93, 1.71], 0.506, 0.583, 0.461),
      candidate("strong-saturation", 81.7, [88.2, 84.1, 82.4, 77.5], [2.04, 2.02, 1.27], 0.398, 0.397, 0.023),
      candidate("weibull", 27.2, [47, 69.1, 8, 31.3], [2.43, 0.42, 1.85], 0.6, 0.276, 0),
    ],
  },
  {
    id: "saturated-paid-social",
    label: "Saturated paid social",
    shortLabel: "Saturation",
    description: "Paid social has strong diminishing returns inside the observed spend range.",
    complication: "Average ROI can look acceptable while the wrong response curve sends the next dollar to the wrong channel.",
    seed: 20260815,
    benchmarkError: 0.138,
    truth: sharedTruth,
    candidates: [
      candidate("short-memory", 60.3, [88.1, 84.6, 80.7, 36.6], [3.39, 4.31, 0.7], 0.908, 1.21, 0.633),
      candidate("balanced", 57.7, [85.5, 82.9, 70.6, 36.4], [3.15, 3.49, 1.08], 0.709, 0.881, 0.296),
      candidate("long-memory", 71, [82.3, 71.4, 57.8, 74.7], [2.78, 2.98, 1.78], 0.511, 0.596, 0.329),
      candidate("strong-saturation", 58.2, [85.7, 86.7, 71.5, 36.3], [2.47, 2.27, 0.74], 0.47, 0.424, 0.022),
      candidate("weibull", 61.6, [46.4, 68.4, 73.5, 61.1], [2.46, 0, 1.81], 7.297, 0.373, 0),
    ],
  },
  {
    id: "delayed-tv",
    label: "Delayed TV",
    shortLabel: "TV delay",
    description: "Long TV carryover makes contemporaneous attribution misleading.",
    complication: "The injected truth uses a Weibull delay; shorter-memory candidates forecast well but misallocate the incremental budget.",
    seed: 20260816,
    benchmarkError: 0.147,
    truth: sharedTruth,
    candidates: [
      candidate("short-memory", 61.2, [89, 86.4, 81.9, 37.3], [2.45, 4.13, 0.71], 0.835, 1.095, 0.623),
      candidate("balanced", 80.2, [87.8, 87, 79.2, 75], [2.49, 3.45, 1.14], 0.662, 0.821, 0.623),
      candidate("long-memory", 78, [83.9, 76.4, 77, 76.5], [2.54, 3.26, 1.74], 0.576, 0.713, 0.623),
      candidate("strong-saturation", 58.5, [81.4, 80, 76.7, 37.3], [2.12, 2.2, 0.94], 0.482, 0.479, 0.072),
      candidate("weibull", 58.9, [70.8, 64.3, 51.6, 56.5], [2.45, 1.24, 1.76], 0.028, 0.026, 0),
    ],
  },
  {
    id: "correlated-media",
    label: "Correlated media",
    shortLabel: "Correlation",
    description: "Channels share a strong latent planning process and become difficult to separate.",
    complication: "High fit quality cannot tell which synchronized channel caused the outcome; the hidden counterfactual can.",
    seed: 20260817,
    benchmarkError: 0.143,
    truth: sharedTruth,
    candidates: [
      candidate("short-memory", 50.5, [89.3, 84.5, 73.2, 24.8], [1.4, 3.42, 0.8], 0.875, 1.015, 0.573),
      candidate("balanced", 50.5, [88.8, 85.5, 73.5, 24.7], [1.55, 2.93, 0.97], 0.728, 0.812, 0.573),
      candidate("long-memory", 68.4, [82, 64.8, 63.1, 67], [2.44, 2.25, 1.69], 0.36, 0.364, 0.161),
      candidate("strong-saturation", 48.6, [82.5, 76.9, 71.3, 24.8], [1.31, 1.89, 0.92], 0.544, 0.516, 0.573),
      candidate("weibull", 19.5, [73.8, 56.3, 1.7, 30.4], [2.47, 0, 1.78], 7.639, 0.4, 0),
    ],
  },
  {
    id: "wrong-industry-benchmark",
    label: "Wrong industry benchmark",
    shortLabel: "Wrong benchmark",
    description: "External benchmarks intentionally disagree with causal truth.",
    complication: "This is the anti-circularity test: benchmark agreement is fallible evidence and never defines the answer key.",
    seed: 20260818,
    benchmarkError: 1.032,
    truth: sharedTruth,
    candidates: [
      candidate("short-memory", 60.1, [90.6, 85.5, 81.8, 35.4], [2.67, 3.9, 0.99], 0.752, 0.966, 0.631),
      candidate("balanced", 79.6, [90.7, 89.6, 75.3, 73.9], [2.62, 3.36, 1.31], 0.615, 0.75, 0.476),
      candidate("long-memory", 73, [83, 73, 64, 74.4], [2.5, 2.88, 1.78], 0.466, 0.531, 0.44),
      candidate("strong-saturation", 58.2, [89.9, 85.1, 73.8, 35.1], [2.19, 2.12, 0.92], 0.459, 0.435, 0),
      candidate("weibull", 69, [73, 67.6, 75.7, 63.8], [2.51, 0, 1.79], 7.418, 0.38, 0),
    ],
  },
];
