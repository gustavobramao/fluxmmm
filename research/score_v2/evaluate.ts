import { adstock, weibullAdstock } from "../../lib/mmm/math";
import type { ModelConfig, ModelResult } from "../../lib/mmm/types";
import { fixedHill, trueContributionForAllocation } from "./simulator";
import type { SyntheticBusiness } from "./types";

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function quantile(values: number[], probability: number): number {
  const ordered = values
    .filter((value) => value > 0)
    .sort((left, right) => left - right);
  if (!ordered.length) return 1;
  return ordered[
    Math.min(
      ordered.length - 1,
      Math.max(0, Math.round((ordered.length - 1) * probability)),
    )
  ];
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
    const baselineCarry =
      config.adstockType === "weibull"
        ? weibullAdstock(truth.spend, config.weibullShape, config.weibullScale)
        : adstock(truth.spend, config.adstock);
    const counterfactualCarry =
      config.adstockType === "weibull"
        ? weibullAdstock(
            truth.spend.map((value) => value * scale),
            config.weibullShape,
            config.weibullScale,
          )
        : adstock(
            truth.spend.map((value) => value * scale),
            config.adstock,
          );
    const halfSaturation = quantile(baselineCarry, 0.5);
    const transformed = fixedHill(
      counterfactualCarry,
      config.saturation,
      halfSaturation,
    );
    return total + estimate.coefficient * sum(transformed);
  }, 0);
}

function allocations(
  channels: string[],
  total: number,
  units: number,
): Record<string, number>[] {
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

export function recommendCandidateAllocation(
  business: SyntheticBusiness,
  model: ModelResult,
  config: ModelConfig,
): Record<string, number> {
  let bestValue = Number.NEGATIVE_INFINITY;
  let best: Record<string, number> = {};
  const truth = business.truth.allocation;
  for (const allocation of allocations(
    business.truth.channels.map((channel) => channel.channel),
    truth.extraBudget,
    truth.gridUnits,
  )) {
    const value = modelContributionForAllocation(
      business,
      model,
      config,
      allocation,
    );
    if (value > bestValue) {
      bestValue = value;
      best = allocation;
    }
  }
  return best;
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
  recommendedAdditionalBudget: Record<string, number>;
  trueOutcomeUnderRecommendation: number;
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
  const recommendedAdditionalBudget = recommendCandidateAllocation(
    business,
    model,
    config,
  );
  const trueOutcomeUnderRecommendation = trueContributionForAllocation(
    business,
    recommendedAdditionalBudget,
  );
  const opportunity = Math.max(
    business.truth.allocation.optimalContribution -
      business.truth.allocation.currentContribution,
    1,
  );
  const budgetRegret = Math.max(
    0,
    (business.truth.allocation.optimalContribution -
      trueOutcomeUnderRecommendation) /
      opportunity,
  );
  return {
    weightedLogRoiError,
    weightedLogBenchmarkAgreement,
    contributionError,
    budgetRegret,
    recommendedAdditionalBudget,
    trueOutcomeUnderRecommendation,
  };
}
