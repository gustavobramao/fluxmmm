import {
  responseForChannel,
  responseTransform,
} from "../../lib/mmm/response";
import type { ModelConfig, ModelResult } from "../../lib/mmm/types";
import {
  trueContributionForAllocation,
  trueIncrementalProfitForAllocation,
} from "./simulator";
import type { SyntheticBusiness } from "./types";

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function allocations(
  channels: string[],
  total: number,
  units: number,
): Record<string, number>[] {
  if (total <= 0) return [Object.fromEntries(channels.map((channel) => [channel, 0]))];
  const output: Record<string, number>[] = [];
  const visit = (index: number, remaining: number, values: number[]) => {
    if (index === channels.length - 1) {
      const complete = [...values, remaining];
      output.push(
        Object.fromEntries(
          channels.map((channel, channelIndex) => [
            channel,
            (complete[channelIndex] / units) * total,
          ]),
        ),
      );
      return;
    }
    for (let value = 0; value <= remaining; value += 1) {
      visit(index + 1, remaining - value, [...values, value]);
    }
  };
  visit(0, units, []);
  return output;
}

function modelContributionForAllocation(
  business: SyntheticBusiness,
  model: ModelResult,
  config: ModelConfig,
  additionalBudget: Record<string, number>,
): number {
  return business.truth.channels.reduce((total, truth) => {
    const estimate = model.channels.find(
      (channel) => channel.channel === truth.spendColumn,
    );
    if (!estimate) return total;
    const scale =
      (truth.totalSpend + (additionalBudget[truth.channel] ?? 0)) /
      Math.max(truth.totalSpend, 1);
    const response = responseForChannel(config, truth.spendColumn);
    const baseline = responseTransform(truth.spend, response);
    const counterfactual = responseTransform(
      truth.spend.map((value) => value * scale),
      response,
      baseline.halfSaturation,
    );
    return total + estimate.coefficient * sum(counterfactual.transformed);
  }, 0);
}

export function recommendCandidateAllocation(
  business: SyntheticBusiness,
  model: ModelResult,
  config: ModelConfig,
): { allocation: Record<string, number>; spend: number; predictedProfit: number } {
  const channels = business.truth.channels.map((channel) => channel.channel);
  const currentModelContribution = modelContributionForAllocation(
    business,
    model,
    config,
    Object.fromEntries(channels.map((channel) => [channel, 0])),
  );
  const totalSpend = sum(
    business.truth.channels.map((channel) => channel.totalSpend),
  );
  const margin = business.truth.allocation.effectiveRevenueMargin;
  let best = Object.fromEntries(channels.map((channel) => [channel, 0]));
  let bestSpend = 0;
  let bestValue = 0;
  for (const budgetShare of business.scenario.decisionBudgetShares) {
    const increment = totalSpend * budgetShare;
    for (const allocation of allocations(
      channels,
      increment,
      business.truth.allocation.gridUnits,
    )) {
      const respectsBounds = business.scenario.channels.every(
        (channel) =>
          (allocation[channel.channel] ?? 0) <=
          increment * channel.allocation.maximumShareOfIncrement + 1e-6,
      );
      if (!respectsBounds) continue;
      const modelContribution = modelContributionForAllocation(
        business,
        model,
        config,
        allocation,
      );
      const predictedProfit =
        (modelContribution - currentModelContribution) * margin - increment;
      if (predictedProfit > bestValue) {
        bestValue = predictedProfit;
        best = allocation;
        bestSpend = increment;
      }
    }
  }
  return { allocation: best, spend: bestSpend, predictedProfit: bestValue };
}

export function evaluateCandidateTruth(
  business: SyntheticBusiness,
  model: ModelResult,
  config: ModelConfig,
): {
  weightedLogRoiError: number;
  weightedLogBenchmarkAgreement: number;
  contributionError: number;
  budgetRegret: number;
  profitRegret: number;
  revenueRegret: number;
  recommendedAdditionalBudget: Record<string, number>;
  recommendedSpend: number;
  trueOutcomeUnderRecommendation: number;
  trueProfitUnderRecommendation: number;
} {
  const totalSpend = sum(
    business.truth.channels.map((channel) => channel.totalSpend),
  );
  const weightedLogRoiError = business.truth.channels.reduce(
    (total, truth) => {
      const estimate = model.channels.find(
        (channel) => channel.channel === truth.spendColumn,
      );
      const estimatedRoi = Math.max(estimate?.roi ?? 0, 1e-6);
      return (
        total +
        (truth.totalSpend / Math.max(totalSpend, 1)) *
          Math.abs(Math.log(estimatedRoi / truth.targetRoi))
      );
    },
    0,
  );
  const weightedLogBenchmarkAgreement = business.truth.channels.reduce(
    (total, truth) => {
      const estimate = model.channels.find(
        (channel) => channel.channel === truth.spendColumn,
      );
      const estimatedRoi = Math.max(estimate?.roi ?? 0, 1e-6);
      const benchmark = Math.max(
        business.truth.industryBenchmarks[truth.channel] ?? truth.targetRoi,
        1e-6,
      );
      return (
        total +
        (truth.totalSpend / Math.max(totalSpend, 1)) *
          Math.abs(Math.log(estimatedRoi / benchmark))
      );
    },
    0,
  );
  const totalTrueContribution = sum(
    business.truth.channels.map((channel) => channel.totalContribution),
  );
  const contributionError =
    business.truth.channels.reduce((total, truth) => {
      const estimate = model.channels.find(
        (channel) => channel.channel === truth.spendColumn,
      );
      return total + Math.abs((estimate?.contribution ?? 0) - truth.totalContribution);
    }, 0) / Math.max(totalTrueContribution, 1);
  const recommendation = recommendCandidateAllocation(business, model, config);
  const trueOutcomeUnderRecommendation = trueContributionForAllocation(
    business,
    recommendation.allocation,
  );
  const trueProfitUnderRecommendation = trueIncrementalProfitForAllocation(
    business,
    recommendation.allocation,
  );
  const profitGap = Math.max(
    0,
    business.truth.allocation.optimalIncrementalProfit -
      trueProfitUnderRecommendation,
  );
  const revenueGap = Math.max(
    0,
    business.truth.allocation.optimalContribution -
      trueOutcomeUnderRecommendation,
  );
  // A recommendation can destroy value when the oracle would spend nothing.
  // Normalize by the larger decision magnitude and cap at 100%, so regret is
  // interpretable rather than exploding when the positive opportunity is zero.
  const profitRegret = Math.min(
    1,
    profitGap /
      Math.max(
        business.truth.allocation.optimalIncrementalProfit,
        Math.abs(trueProfitUnderRecommendation),
        1,
      ),
  );
  const revenueRegret = Math.min(
    1,
    revenueGap /
      Math.max(
        business.truth.allocation.optimalIncrementalOutcome,
        Math.abs(
          trueOutcomeUnderRecommendation -
            business.truth.allocation.currentContribution,
        ),
        1,
      ),
  );
  return {
    weightedLogRoiError,
    weightedLogBenchmarkAgreement,
    contributionError,
    budgetRegret: profitRegret,
    profitRegret,
    revenueRegret,
    recommendedAdditionalBudget: recommendation.allocation,
    recommendedSpend: recommendation.spend,
    trueOutcomeUnderRecommendation,
    trueProfitUnderRecommendation,
  };
}
