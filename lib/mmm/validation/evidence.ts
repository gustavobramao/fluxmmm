import {
  channelExperiments,
  inferIndustryPrior,
  industryPriorPercentile,
  industryRoiTail,
} from "../benchmarks";
import { toNumber } from "../csv";
import type { Dataset, Experiment, ModelResult } from "../types";
import type {
  ChannelEvidenceCoherence,
  EvidenceCoherenceAssessment,
  ValidationOptions,
} from "./types";

const DEFAULT_MATERIAL_SPEND_SHARE = 0.02;

function experimentSummary(experiments: Experiment[]): {
  center: number;
  low: number;
  high: number;
  standardError: number;
  label: string;
} {
  const estimates = experiments.map((experiment) => ({
    roi:
      experiment.incrementalOutcome /
      Math.max(experiment.incrementalSpend, 1),
    standardError: Math.max(experiment.standardError, 0.01),
  }));
  const totalPrecision = estimates.reduce(
    (total, estimate) => total + 1 / estimate.standardError ** 2,
    0,
  );
  const center =
    estimates.reduce(
      (total, estimate) =>
        total + estimate.roi / estimate.standardError ** 2,
      0,
    ) / Math.max(totalPrecision, 1e-9);
  const standardError = Math.sqrt(1 / Math.max(totalPrecision, 1e-9));
  return {
    center,
    low: Math.max(0, center - 1.96 * standardError),
    high: center + 1.96 * standardError,
    standardError,
    label:
      experiments.length === 1
        ? experiments[0].source
        : `${experiments.length} pooled experiments`,
  };
}

function comparableExperimentEstimate(
  model: ModelResult,
  channel: string,
  experiments: Experiment[],
): {
  roi: number;
  low: number;
  high: number;
  basis: "experiment-window" | "full-history";
  label: string;
} | undefined {
  const receipts = model.advanced?.calibrationReceipts?.filter(
    (receipt) =>
      receipt.source === "experiment" &&
      receipt.channel.toLowerCase() === channel.toLowerCase(),
  ) ?? [];
  if (!receipts.length) return undefined;
  const weights = receipts.map((receipt) => {
    const experiment = experiments.find(
      (candidate) =>
        candidate.channel.toLowerCase() === channel.toLowerCase() &&
        candidate.source === receipt.evidenceLabel &&
        candidate.startDate === receipt.startDate &&
        candidate.endDate === receipt.endDate,
    );
    return 1 / Math.max(experiment?.standardError ?? 1, 0.01) ** 2;
  });
  const totalWeight = weights.reduce((total, weight) => total + weight, 0);
  const weighted = (field: "modelRoi" | "modelLow" | "modelHigh") =>
    receipts.reduce(
      (total, receipt, index) => total + receipt[field] * weights[index],
      0,
    ) / Math.max(totalWeight, 1e-9);
  return {
    roi: weighted("modelRoi"),
    low: weighted("modelLow"),
    high: weighted("modelHigh"),
    basis: "experiment-window",
    label:
      receipts.length === 1
        ? `${receipts[0].startDate}–${receipts[0].endDate}`
        : `${receipts.length} matched experiment windows`,
  };
}

function spendShares(dataset: Dataset): Map<string, number> {
  const spend = dataset.mediaColumns.map((channel) => ({
    channel,
    value: dataset.rows.reduce(
      (total, row) => total + Math.max(0, toNumber(row[channel])),
      0,
    ),
  }));
  const total = spend.reduce((sum, channel) => sum + channel.value, 0);
  return new Map(
    spend.map((channel) => [
      channel.channel.toLowerCase(),
      channel.value / Math.max(total, 1),
    ]),
  );
}

function materialInterval(
  roiLow: number,
  roiHigh: number,
  evidenceCenter: number,
  evidenceHigh: number,
): boolean {
  const width = Math.max(roiHigh - roiLow, 0);
  return (
    width > Math.max(evidenceHigh * 2, evidenceCenter * 6, 1) ||
    (roiLow <= 0.01 && roiHigh > Math.max(evidenceHigh * 1.5, 2))
  );
}

function priorIdentification(
  model: ModelResult,
  channel: string,
): ChannelEvidenceCoherence["identification"] {
  return (
    model.numerical?.priorInfluence.find(
      (item) => item.channel.toLowerCase() === channel.toLowerCase(),
    )?.classification ?? "unknown"
  );
}

function wasClipped(model: ModelResult, channel: string): boolean {
  return Boolean(
    model.numerical?.clipping.channels.some(
      (item) => item.channel.toLowerCase() === channel.toLowerCase(),
    ),
  );
}

export function assessEvidenceCoherence(
  dataset: Dataset,
  model: ModelResult,
  experiments: Experiment[],
  options: ValidationOptions,
): EvidenceCoherenceAssessment {
  const materialSpendShareThreshold =
    options.materialSpendShareThreshold ?? DEFAULT_MATERIAL_SPEND_SHARE;
  const benchmarkScreening =
    options.industryBenchmarkScreeningEnabled ?? true;
  const shares = spendShares(dataset);

  const channels = model.channels.map((estimate): ChannelEvidenceCoherence => {
    const spendShare = shares.get(estimate.channel.toLowerCase()) ?? 0;
    const material = spendShare >= materialSpendShareThreshold;
    const clipped = wasClipped(model, estimate.channel);
    const identification = priorIdentification(model, estimate.channel);
    const evidence = channelExperiments(experiments, estimate.channel);
    const priorUse = estimate.priorSource
      ? "calibration"
      : evidence.length || benchmarkScreening
        ? "screening-only"
        : "none";

    if (evidence.length) {
      const experiment = experimentSummary(evidence);
      const comparable = comparableExperimentEstimate(
        model,
        estimate.channel,
        evidence,
      );
      const modelRoi = comparable?.roi ?? estimate.roi;
      const modelLow = comparable?.low ?? estimate.roiLow;
      const modelHigh = comparable?.high ?? estimate.roiHigh;
      const modelStandardError =
        Math.max(modelHigh - modelLow, 0) / (2 * 1.96);
      const standardizedGap =
        Math.abs(modelRoi - experiment.center) /
        Math.max(
          Math.sqrt(
            modelStandardError ** 2 + experiment.standardError ** 2,
          ),
          1e-6,
        );
      const unidentified = materialInterval(
        modelLow,
        modelHigh,
        experiment.center,
        experiment.high,
      );
      const boundaryCollapse =
        material &&
        modelRoi <= 0.01 &&
        clipped &&
        unidentified;
      const conflict = material && standardizedGap > 2.5;
      const status = boundaryCollapse
        ? "boundary-collapse"
        : conflict
          ? "experiment-conflict"
          : unidentified && material
            ? "unidentified"
            : "aligned";
      const blocking =
        status === "boundary-collapse" ||
        status === "experiment-conflict";
      return {
        channel: estimate.channel,
        spendShare,
        material,
        roi: modelRoi,
        roiLow: modelLow,
        roiHigh: modelHigh,
        evidenceSource: "experiment",
        evidenceLabel: experiment.label,
        evidenceCenter: experiment.center,
        evidenceLow: experiment.low,
        evidenceHigh: experiment.high,
        comparisonBasis: comparable?.basis ?? "full-history",
        comparisonLabel: comparable
          ? `Experiment-window ROI · ${comparable.label}`
          : "Full-history ROI",
        standardizedGap,
        priorUse,
        identification,
        clipped,
        status,
        blocking,
        detail: boundaryCollapse
          ? "The material channel was clipped to zero while its interval still spans the experimental effect. This is non-identification, not evidence of zero."
          : conflict
            ? `The estimate is ${standardizedGap.toFixed(1)} combined standard errors from the experimental ROI.`
            : unidentified && material
              ? "The channel interval is too wide for a decision-grade experiment comparison."
              : priorUse === "calibration"
                ? `${comparable ? "The matched experiment-window ROI" : "The fitted ROI"} is coherent with experiment evidence used for calibration; no additional validation credit is awarded.`
                : "The fitted ROI is coherent with independently registered experiment evidence.",
      };
    }

    if (!benchmarkScreening) {
      return {
        channel: estimate.channel,
        spendShare,
        material,
        roi: estimate.roi,
        roiLow: estimate.roiLow,
        roiHigh: estimate.roiHigh,
        evidenceSource: "none",
        evidenceLabel: "No external evidence",
        comparisonBasis: "full-history",
        comparisonLabel: "Full-history ROI",
        priorUse: "none",
        identification,
        clipped,
        status: "unanchored",
        blocking: false,
        detail: "No experiment or enabled industry screen is available for this channel.",
      };
    }

    const benchmark = inferIndustryPrior(estimate.channel);
    const percentile = industryPriorPercentile(estimate.roi, benchmark);
    const tail = industryRoiTail(estimate.roi, benchmark);
    const credibleMatch =
      benchmark.matchQuality !== "Generic fallback" &&
      benchmark.confidence !== "Very low";
    const unidentified = materialInterval(
      estimate.roiLow,
      estimate.roiHigh,
      benchmark.median,
      benchmark.high,
    );
    const boundaryCollapse =
      material &&
      estimate.roi <= 0.01 &&
      (clipped || estimate.roiLow <= 0.01) &&
      unidentified;
    const blockingTension = material && Boolean(tail) && credibleMatch;
    const status = boundaryCollapse
      ? "boundary-collapse"
      : tail
        ? "benchmark-tension"
        : unidentified && material
          ? "unidentified"
          : "aligned";
    const blocking = boundaryCollapse || blockingTension;
    return {
      channel: estimate.channel,
      spendShare,
      material,
      roi: estimate.roi,
      roiLow: estimate.roiLow,
      roiHigh: estimate.roiHigh,
      evidenceSource: "industry",
      evidenceLabel: benchmark.label,
      evidenceCenter: benchmark.median,
      evidenceLow: benchmark.low,
      evidenceHigh: benchmark.high,
      comparisonBasis: "full-history",
      comparisonLabel: "Full-history ROI",
      percentile,
      priorUse,
      identification,
      clipped,
      status,
      blocking,
      detail: boundaryCollapse
        ? `The material channel is at the zero boundary with an interval spanning far beyond the ${benchmark.low.toFixed(2)}–${benchmark.high.toFixed(2)}× benchmark range.`
        : tail
          ? `${tail === "low" ? "Below" : "Above"} the two-sided industry plausibility screen (${benchmark.matchQuality.toLowerCase()}, ${benchmark.confidence.toLowerCase()} confidence).`
          : unidentified && material
            ? "The point estimate is plausible, but its interval remains too broad for a stable channel decision."
            : priorUse === "calibration"
              ? "The ROI satisfies an industry prior used for calibration; this is contract adherence, not independent causal validation."
              : "The ROI falls inside the broad industry plausibility range.",
    };
  });

  const blocking = channels.filter((channel) => channel.blocking);
  const review = channels.filter(
    (channel) =>
      channel.material &&
      !channel.blocking &&
      (channel.status === "benchmark-tension" ||
        channel.status === "unidentified"),
  );
  const hasExperimentConflict = blocking.some(
    (channel) => channel.status === "experiment-conflict",
  );
  const hasBoundaryCollapse = blocking.some(
    (channel) => channel.status === "boundary-collapse",
  );
  const causalScoreCap = hasExperimentConflict
    ? 49
    : hasBoundaryCollapse
      ? 59
      : blocking.length
        ? 64
        : review.length
          ? 74
          : 100;
  const rescueChannels = blocking
    .filter(
      (channel) =>
        channel.evidenceSource === "industry" &&
        channel.priorUse !== "calibration",
    )
    .map((channel) => channel.channel);
  const status = blocking.length ? "fail" : review.length ? "review" : "pass";
  return {
    status,
    causalScoreCap,
    materialSpendShareThreshold,
    blockingChannels: blocking.map((channel) => channel.channel),
    rescueChannels,
    summary: blocking.length
      ? `${blocking.length} material channel${blocking.length === 1 ? "" : "s"} conflict with the best available evidence or collapse at the non-negative boundary.`
      : review.length
        ? `${review.length} material channel${review.length === 1 ? "" : "s"} need evidence review; no hard contradiction was found.`
        : "Every material channel is coherent with its best available experiment or industry plausibility evidence.",
    channels,
  };
}

export function rescueIndustryPriorChannels(
  assessment: EvidenceCoherenceAssessment,
  alreadyActive: readonly string[] = [],
): string[] {
  const active = new Set(alreadyActive.map((channel) => channel.toLowerCase()));
  return assessment.rescueChannels.filter(
    (channel) => !active.has(channel.toLowerCase()),
  );
}
