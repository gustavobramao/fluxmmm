import cleanReport from "../research/score_v2/artifacts/clean-identification/report.json";
import correlatedReport from "../research/score_v2/artifacts/correlated-media/report.json";
import delayedReport from "../research/score_v2/artifacts/delayed-tv/report.json";
import confoundedReport from "../research/score_v2/artifacts/demand-confounded-search/report.json";
import saturatedReport from "../research/score_v2/artifacts/saturated-paid-social/report.json";
import wrongBenchmarkReport from "../research/score_v2/artifacts/wrong-industry-benchmark/report.json";
import type { ModelConfig } from "../lib/mmm/types";

export type ScoreLabLayer =
  | "generalization"
  | "structure"
  | "causal"
  | "decision";
export type ScoreLabChannel = "paid_social" | "search" | "tv";
export type ScoreLabEvidenceArm =
  | "experiments-only"
  | "benchmark-gap-fill";

export interface ScoreLabCandidate {
  id: string;
  label: string;
  evidenceArm: ScoreLabEvidenceArm;
  score: number;
  config: ModelConfig;
  layers: Record<ScoreLabLayer, number>;
  roi: Record<ScoreLabChannel, number>;
  roiError: number;
  contributionError: number;
  profitRegret: number;
  revenueRegret: number;
  recommendedSpend: number;
}

export interface ScoreLabTruthChannel {
  channel: ScoreLabChannel;
  targetRoi: number;
  marginalRoiAtObserved: number;
  totalSpend: number;
  evidenceId: string;
  response: {
    family: "geometric" | "weibull";
    decay?: number;
    shape?: number;
    scale?: number;
    hillShape: number;
    halfSaturationQuantile: number;
    kernelNormalization: "sum";
  };
  delivery: {
    kind: "auction" | "reach" | "flighted-grp";
    priceLabel: "CPC" | "CPM" | "CPP";
  };
}

export interface ScoreLabScenario {
  id: string;
  label: string;
  shortLabel: string;
  description: string;
  seed: number;
  benchmarkError: number;
  truth: Record<ScoreLabChannel, number>;
  truthChannels: ScoreLabTruthChannel[];
  experiments: {
    channel: ScoreLabChannel;
    design: string;
    trueRoi: number;
    observedRoi: number;
    standardError: number;
  }[];
  oracle: {
    effectiveRevenueMargin: number;
    evaluatedBudgetShares: number[];
    optimalSpend: number;
    optimalIncrementalProfit: number;
  };
  complication: string;
  candidates: ScoreLabCandidate[];
}

const candidateLabels: Record<string, string> = {
  "short-memory": "Geometric · short memory",
  balanced: "Geometric · balanced",
  "long-memory": "Geometric · long memory",
  "strong-saturation": "Geometric · strong saturation",
  weibull: "Global Weibull",
  "channel-specific": "Channel-specific response",
  "channel-specific-long": "Channel-specific · longer carryover",
};

const scenarioPresentation: Record<
  string,
  { shortLabel: string; complication: string }
> = {
  "clean-identification": {
    shortLabel: "Clean",
    complication:
      "The easy control: observable demand should make ROI and the profit decision recoverable.",
  },
  "demand-confounded-search": {
    shortLabel: "Search confounding",
    complication:
      "Search harvests latent demand. Strong forecasts can still imply the wrong causal search ROI.",
  },
  "saturated-paid-social": {
    shortLabel: "Saturation",
    complication:
      "Frequency erodes social reach, so average ROI alone is insufficient for the next dollar.",
  },
  "delayed-tv": {
    shortLabel: "TV delay",
    complication:
      "Flighted TV uses a delayed Weibull response; a common short-memory curve is misspecified.",
  },
  "correlated-media": {
    shortLabel: "Correlation",
    complication:
      "An unobserved commercial plan moves media and baseline together, obscuring attribution.",
  },
  "wrong-industry-benchmark": {
    shortLabel: "Wrong benchmark",
    complication:
      "This anti-circularity test intentionally makes the observational benchmark disagree with randomized truth.",
  },
};

type RawReport = typeof cleanReport;

function scoreLabScenario(report: RawReport): ScoreLabScenario {
  const truthChannels = report.truth.channels as ScoreLabTruthChannel[];
  return {
    id: report.scenario.id,
    label: report.scenario.label,
    shortLabel: scenarioPresentation[report.scenario.id].shortLabel,
    description: report.scenario.description,
    seed: report.scenario.seed,
    benchmarkError: report.truth.weightedLogBenchmarkTruthError,
    truth: Object.fromEntries(
      truthChannels.map((channel) => [channel.channel, channel.targetRoi]),
    ) as Record<ScoreLabChannel, number>,
    truthChannels,
    experiments: report.truth.experiments as ScoreLabScenario["experiments"],
    oracle: report.truth.allocation,
    complication: scenarioPresentation[report.scenario.id].complication,
    candidates: report.candidates.map((candidate) => {
      const [baseId, evidenceArm] = candidate.id.split(" · ") as [
        string,
        ScoreLabEvidenceArm,
      ];
      return {
        id: candidate.id,
        label: candidateLabels[baseId] ?? baseId,
        evidenceArm,
        score: candidate.fluxScore ?? 0,
        config: candidate.config as ModelConfig,
        layers: candidate.layers as Record<ScoreLabLayer, number>,
        roi: candidate.roi as Record<ScoreLabChannel, number>,
        roiError: candidate.weightedLogRoiError,
        contributionError: candidate.contributionError,
        profitRegret: candidate.profitRegret,
        revenueRegret: candidate.revenueRegret,
        recommendedSpend: candidate.recommendedSpend,
      };
    }),
  };
}

export const SCORE_LAB_SCENARIOS: ScoreLabScenario[] = [
  cleanReport,
  confoundedReport,
  saturatedReport,
  delayedReport,
  correlatedReport,
  wrongBenchmarkReport,
].map((report) => scoreLabScenario(report as RawReport));
