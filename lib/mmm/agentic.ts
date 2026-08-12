import type {
  AdvancedModelConfig,
  Dataset,
  Experiment,
  ModelConfig,
  ModelResult,
} from "./types";
import {
  channelExperiments,
  inferIndustryPrior,
  industryPriorPercentile,
  industryRoiTail,
  isHighlyImprobableIndustryRoi,
} from "./benchmarks";
import type {
  ValidationModelKind,
  ValidationProgress,
  ValidationResult,
} from "./validation";

export const AGENTIC_SEARCH_VERSION = "flux-agentic-search-v5.0.0-global-channel-response";

export interface AgenticSearchContract {
  families: Record<ValidationModelKind, boolean>;
  candidateBudget: 96 | 192 | 384;
}

export interface AgenticProposalEvidence {
  method:
    | "space-filling"
    | "response-covering-array"
    | "covering-array"
    | "tpe"
    | "local-challenge"
    | "evidence-rescue";
  expectedScore?: number;
  eligibilityProbability?: number;
  acquisition?: number;
  reason: string;
}

export interface AgenticCandidateSpec {
  id: string;
  family: ValidationModelKind;
  label: string;
  summary: string;
  hypothesis: string;
  config: ModelConfig;
  advancedConfig: AdvancedModelConfig;
  searchPhase:
    | "seed"
    | "response-coverage"
    | "advanced-challenge"
    | "adaptive"
    | "local-challenge"
    | "rescue";
  proposal: AgenticProposalEvidence;
  restart: 1 | 2 | 3;
  rescueOf?: string;
  challengeOf?: string;
  evidencePriorChannels?: string[];
}

export type AgenticCandidateState =
  | "queued"
  | "running"
  | "complete"
  | "error";

export interface AgenticCandidateRun {
  spec: AgenticCandidateSpec;
  state: AgenticCandidateState;
  model?: ModelResult;
  validation?: ValidationResult;
  roiGuardrailViolations?: AgenticRoiGuardrailViolation[];
  progress?: ValidationProgress;
  restoredFromCache?: boolean;
  error?: string;
}

export interface AgenticRoiGuardrailViolation {
  channel: string;
  roi: number;
  benchmarkLabel: string;
  percentile: number;
  tail?: "low" | "high";
  spendShare?: number;
  reason?: string;
}

export interface AgenticForcePromotionAudit {
  type: "forced";
  forcedAt: string;
  reason: "No candidate passed every eligibility check";
  failedGates: {
    id: string;
    label: string;
  }[];
  roiGuardrailViolations: AgenticRoiGuardrailViolation[];
}

export function createAgenticForcePromotionAudit(
  run: AgenticCandidateRun,
  forcedAt = new Date().toISOString(),
): AgenticForcePromotionAudit {
  return {
    type: "forced",
    forcedAt,
    reason: "No candidate passed every eligibility check",
    failedGates: (run.validation?.gates ?? [])
      .filter((gate) => gate.applicable && !gate.passed)
      .map((gate) => ({ id: gate.id, label: gate.label })),
    roiGuardrailViolations: [...(run.roiGuardrailViolations ?? [])],
  };
}

export const DEFAULT_AGENTIC_SEARCH_CONTRACT: AgenticSearchContract = {
  families: {
    frequentist: true,
    bayesian: true,
    advanced: true,
  },
  candidateBudget: 192,
};

export function passesApplicableGates(
  validation: ValidationResult,
): boolean {
  return validation.gates
    .filter((gate) => gate.applicable)
    .every((gate) => gate.passed);
}

export function findAgenticRoiGuardrailViolations(
  model: ModelResult,
  experiments: Experiment[],
  enabled = true,
  dataset?: Dataset,
): AgenticRoiGuardrailViolation[] {
  if (!enabled) return [];
  const spendByChannel = new Map(
    (dataset?.mediaColumns ?? []).map((channel) => [
      channel.toLowerCase(),
      dataset?.rows.reduce(
        (total, row) => total + Math.max(0, Number(row[channel]) || 0),
        0,
      ) ?? 0,
    ]),
  );
  const totalSpend = [...spendByChannel.values()].reduce(
    (total, spend) => total + spend,
    0,
  );
  return model.channels.flatMap((channel) => {
    if (channelExperiments(experiments, channel.channel).length) return [];
    const spendShare = dataset
      ? (spendByChannel.get(channel.channel.toLowerCase()) ?? 0) /
        Math.max(totalSpend, 1)
      : undefined;
    if (spendShare !== undefined && spendShare < 0.02) return [];
    const benchmark = inferIndustryPrior(channel.channel);
    if (!isHighlyImprobableIndustryRoi(channel.roi, benchmark)) return [];
    const tail = industryRoiTail(channel.roi, benchmark);
    const clipped = Boolean(
      model.numerical?.clipping.channels.some(
        (item) => item.channel.toLowerCase() === channel.channel.toLowerCase(),
      ),
    );
    const boundaryCollapse =
      channel.roi <= 0.01 &&
      clipped &&
      (channel.roiHigh ?? 0) > Math.max(benchmark.high * 1.5, 2);
    const credibleMatch =
      benchmark.matchQuality !== "Generic fallback" &&
      benchmark.confidence !== "Very low";
    if (!credibleMatch && !boundaryCollapse) return [];
    return [
      {
        channel: channel.channel,
        roi: channel.roi,
        benchmarkLabel: benchmark.label,
        percentile: industryPriorPercentile(channel.roi, benchmark),
        tail,
        spendShare,
        reason:
          tail === "low"
            ? "ROI is below the P05 industry plausibility tail."
            : "ROI is above the P95 industry plausibility tail.",
      },
    ];
  });
}

export function passesAgenticEligibility(
  run: AgenticCandidateRun,
): boolean {
  return Boolean(
    run.validation &&
      (run.validation.finalScore ?? 0) >= 75 &&
      passesApplicableGates(run.validation) &&
      !run.roiGuardrailViolations?.length,
  );
}

export function rankAgenticCandidates(
  runs: AgenticCandidateRun[],
): AgenticCandidateRun[] {
  return runs
    .filter(
      (run) =>
        run.state === "complete" &&
        run.validation?.finalScore !== null,
    )
    .sort((left, right) => {
      const leftPasses = passesAgenticEligibility(left);
      const rightPasses = passesAgenticEligibility(right);
      if (leftPasses !== rightPasses) return leftPasses ? -1 : 1;
      return (
        (right.validation?.finalScore ?? -1) -
        (left.validation?.finalScore ?? -1)
      );
    });
}

export function selectAgenticWinner(
  runs: AgenticCandidateRun[],
): AgenticCandidateRun | undefined {
  return rankAgenticCandidates(runs).find(passesAgenticEligibility);
}

export function agenticSearchStage(
  completed: number,
  total: number,
  seedCount = Math.min(12, total),
  advancedChallengeCount = 0,
  localChallengeCount = 0,
  responseChallengeCount = 0,
): "seed" | "response" | "advanced" | "adaptive" | "local" | "select" {
  if (total <= 0 || completed < seedCount) return "seed";
  if (completed < seedCount + responseChallengeCount) return "response";
  if (completed < seedCount + responseChallengeCount + advancedChallengeCount) return "advanced";
  if (completed < Math.max(seedCount + responseChallengeCount + advancedChallengeCount, total - localChallengeCount)) {
    return "adaptive";
  }
  if (completed < total) return "local";
  return "select";
}
