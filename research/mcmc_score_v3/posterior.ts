import { responseTransform } from "../../lib/mmm/response";
import type {
  SamplingDecisionDraw,
  SamplingResult,
} from "../../lib/mmm/sampling";
import type { ModelConfig, ModelResult } from "../../lib/mmm/types";
import { evaluatePosteriorDecisionTruth } from "../score_v2/evaluate";
import type { DecisionScenarioId, SyntheticBusiness } from "../score_v2/types";

const CONVERGENCE_GATES = new Set([
  "rhat",
  "ess",
  "divergences",
  "treedepth",
  "bfmi",
  "mcse",
]);

export interface PosteriorDecisionLabel {
  convergencePassed: boolean;
  decisionLoss?: number;
  profitRegret?: number;
  roiError?: number;
  contributionError?: number;
  posteriorRoi: Record<string, number>;
  scenarioProfitRegret?: Record<DecisionScenarioId, number>;
  scenarioDecisionLoss?: Record<DecisionScenarioId, number>;
  scenarioRecommendedSpend?: Record<DecisionScenarioId, number>;
}

function quantile(values: number[], probability: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const position = Math.min(1, Math.max(0, probability)) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

export function samplingConverged(result: SamplingResult): boolean {
  return result.gates
    .filter((gate) => CONVERGENCE_GATES.has(gate.id))
    .every((gate) => gate.passed);
}

function configForDraw(
  base: ModelConfig,
  draw: SamplingDecisionDraw,
): ModelConfig {
  return {
    ...base,
    channelResponses: Object.fromEntries(
      draw.channels.map((channel) => [channel.channel, channel.response]),
    ),
  };
}

function modelForDraw(
  business: SyntheticBusiness,
  base: ModelResult,
  config: ModelConfig,
  draw: SamplingDecisionDraw,
): ModelResult {
  const channels = base.channels.map((estimate) => {
    const sampled = draw.channels.find(
      (channel) => channel.channel.toLowerCase() === estimate.channel.toLowerCase(),
    );
    const truth = business.truth.channels.find(
      (channel) => channel.spendColumn.toLowerCase() === estimate.channel.toLowerCase(),
    );
    if (!sampled || !truth) return estimate;
    const basis = responseTransform(truth.spend, sampled.response).transformed.reduce(
      (total, value) => total + value,
      0,
    );
    const coefficient = sampled.contribution / Math.max(basis, 1e-9);
    return {
      ...estimate,
      coefficient,
      contribution: sampled.contribution,
      roi: sampled.roi,
      roiLow: sampled.roi,
      roiHigh: sampled.roi,
    };
  });
  const totalContribution = channels.reduce(
    (total, channel) => total + Math.max(0, channel.contribution),
    0,
  );
  return {
    ...base,
    channels: channels.map((channel) => ({
      ...channel,
      contributionShare:
        Math.max(0, channel.contribution) / Math.max(totalContribution, 1e-9),
    })),
  };
}

function stratifiedDraws(
  draws: SamplingDecisionDraw[],
  maximum: number,
): SamplingDecisionDraw[] {
  if (draws.length <= maximum) return draws;
  return Array.from({ length: maximum }, (_, index) =>
    draws[Math.floor(((index + 0.5) / maximum) * draws.length)]
  );
}

export function posteriorDecisionLabel(
  business: SyntheticBusiness,
  mapModel: ModelResult,
  mapConfig: ModelConfig,
  result: SamplingResult,
  maximumDecisionDraws = 64,
): PosteriorDecisionLabel {
  const posteriorRoi = Object.fromEntries(
    result.channels.map((channel) => [channel.channel, channel.posteriorMedian]),
  );
  return posteriorDecisionLabelFromDraws(
    business,
    mapModel,
    mapConfig,
    result.decisionDraws ?? [],
    posteriorRoi,
    samplingConverged(result),
    maximumDecisionDraws,
  );
}

/**
 * Shared posterior-action evaluator. The inference engine supplies aligned
 * joint draws and an explicit numerical-quality flag; model validation gates
 * are deliberately not used to decide whether a research label exists.
 */
export function posteriorDecisionLabelFromDraws(
  business: SyntheticBusiness,
  mapModel: ModelResult,
  mapConfig: ModelConfig,
  decisionDraws: SamplingDecisionDraw[],
  posteriorRoi: Record<string, number>,
  inferencePassed: boolean,
  maximumDecisionDraws = 64,
): PosteriorDecisionLabel {
  if (!inferencePassed || !decisionDraws.length) {
    return { convergencePassed: false, posteriorRoi };
  }
  const draws = stratifiedDraws(decisionDraws, maximumDecisionDraws);
  const posterior = draws.map((draw) => {
    const config = configForDraw(mapConfig, draw);
    return {
      config,
      model: modelForDraw(business, mapModel, config, draw),
    };
  });
  const decision = evaluatePosteriorDecisionTruth(business, posterior);
  const totalSpend = business.truth.channels.reduce(
    (total, channel) => total + channel.totalSpend,
    0,
  );
  const roiError = business.truth.channels.reduce((total, truth) => {
    const values = draws.flatMap((draw) => {
      const channel = draw.channels.find(
        (item) => item.channel.toLowerCase() === truth.spendColumn.toLowerCase(),
      );
      return channel ? [Math.max(channel.roi, 1e-6)] : [];
    });
    return total +
      (truth.totalSpend / Math.max(totalSpend, 1)) *
        Math.abs(Math.log(Math.max(quantile(values, 0.5), 1e-6) / truth.targetRoi));
  }, 0);
  const totalTrueContribution = business.truth.channels.reduce(
    (total, channel) => total + channel.totalContribution,
    0,
  );
  const contributionError = business.truth.channels.reduce((total, truth) => {
    const values = draws.flatMap((draw) => {
      const channel = draw.channels.find(
        (item) => item.channel.toLowerCase() === truth.spendColumn.toLowerCase(),
      );
      return channel ? [channel.contribution] : [];
    });
    return total + Math.abs(quantile(values, 0.5) - truth.totalContribution);
  }, 0) / Math.max(totalTrueContribution, 1);
  return {
    convergencePassed: true,
    decisionLoss: decision.decisionLoss,
    profitRegret: decision.profitRegret,
    roiError,
    contributionError,
    posteriorRoi,
    scenarioProfitRegret: decision.scenarioProfitRegret,
    scenarioDecisionLoss: decision.scenarioDecisionLoss,
    scenarioRecommendedSpend: decision.scenarioRecommendedSpend,
  };
}
