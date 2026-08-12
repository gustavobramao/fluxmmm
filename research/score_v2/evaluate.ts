import {
  responseForChannel,
  responseTransform,
} from "../../lib/mmm/response";
import type { ModelConfig, ModelResult } from "../../lib/mmm/types";
import {
  DECISION_ALLOCATION_CANDIDATES,
  DECISION_BUDGET_STEPS,
  DECISION_SCENARIO_CONTRACTS,
  trueContributionForAllocation,
  trueDecisionValueSurface,
  trueIncrementalProfitForAllocation,
} from "./simulator";
import type {
  DecisionScenarioContract,
  DecisionScenarioId,
  SyntheticBusiness,
} from "./types";

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

function normalizeAllocation(
  business: SyntheticBusiness,
  values: Record<string, number>,
  total: number,
): Record<string, number> {
  if (Math.abs(total) <= 1e-9) {
    return Object.fromEntries(
      business.truth.channels.map((channel) => [channel.channel, 0]),
    );
  }
  const direction = Math.sign(total);
  const bounded = business.truth.channels.map((channel) => {
    const config = business.scenario.channels.find(
      (item) => item.channel === channel.channel,
    )!;
    const raw = Math.max(0, direction * (values[channel.channel] ?? 0));
    const cap = direction > 0
      ? Math.abs(total) * config.allocation.maximumShareOfIncrement
      : channel.totalSpend * 0.7;
    return { channel: channel.channel, cap, value: Math.min(raw, cap) };
  });
  let remaining = Math.max(0, Math.abs(total) - sum(bounded.map((item) => item.value)));
  for (let iteration = 0; iteration < 12 && remaining > 1e-6; iteration += 1) {
    const open = bounded.filter((item) => item.value < item.cap - 1e-9);
    if (!open.length) break;
    const addition = remaining / open.length;
    let used = 0;
    open.forEach((item) => {
      const next = Math.min(item.cap, item.value + addition);
      used += next - item.value;
      item.value = next;
    });
    remaining -= used;
  }
  return Object.fromEntries(
    bounded.map((item) => [item.channel, direction * item.value]),
  );
}

function candidateShares(index: number, count: number): number[] {
  const golden = 0.6180339887498949;
  return Array.from({ length: count }, (_, channelIndex) =>
    0.03 + ((index + 1) * (channelIndex + 2) * golden % 1) ** 1.6,
  );
}

function candidateAllocation(
  business: SyntheticBusiness,
  index: number,
  budget: number,
): Record<string, number> {
  const weights = candidateShares(index, business.truth.channels.length);
  const denominator = sum(weights);
  return normalizeAllocation(
    business,
    Object.fromEntries(
      business.truth.channels.map((channel, channelIndex) => [
        channel.channel,
        budget * weights[channelIndex] / Math.max(denominator, 1e-9),
      ]),
    ),
    budget,
  );
}

function reallocationCandidate(
  business: SyntheticBusiness,
  index: number,
): Record<string, number> {
  const channels = business.truth.channels;
  const count = channels.length;
  const donorIndex = index % count;
  const receiverIndex =
    (donorIndex + 1 + Math.floor(index / count) % Math.max(count - 1, 1)) %
    count;
  const step = Math.floor(index / Math.max(count * (count - 1), 1));
  const steps = Math.max(
    Math.ceil(
      DECISION_ALLOCATION_CANDIDATES /
        Math.max(count * (count - 1), 1),
    ),
    1,
  );
  const transfer = channels[donorIndex].totalSpend * 0.7 * (step + 1) / steps;
  return Object.fromEntries(
    channels.map((channel, channelIndex) => [
      channel.channel,
      channelIndex === donorIndex
        ? -transfer
        : channelIndex === receiverIndex
          ? transfer
          : 0,
    ]),
  );
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

function modelDecisionValueSurface(
  business: SyntheticBusiness,
  model: ModelResult,
  config: ModelConfig,
): (allocation: Record<string, number>) => number {
  const points = 401;
  const maximumScale = 3;
  const surfaces = new Map(
    business.truth.channels.map((truth) => {
      const estimate = model.channels.find(
        (channel) => channel.channel === truth.spendColumn,
      );
      const response = responseForChannel(config, truth.spendColumn);
      const baseline = responseTransform(truth.spend, response);
      const values = Array.from({ length: points }, (_, index) => {
        const scale = index / (points - 1) * maximumScale;
        const transformed = responseTransform(
          truth.spend.map((value) => value * scale),
          response,
          baseline.halfSaturation,
        );
        return (estimate?.coefficient ?? 0) * sum(transformed.transformed);
      });
      return [truth.channel, values] as const;
    }),
  );
  const currentContribution = business.truth.channels.reduce((total, channel) => {
    const values = surfaces.get(channel.channel)!;
    const position = (points - 1) / maximumScale;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    const weight = position - lower;
    return total + values[lower] * (1 - weight) + values[upper] * weight;
  }, 0);
  const margin = business.truth.allocation.effectiveRevenueMargin;
  return (allocation) => {
    const contribution = business.truth.channels.reduce((total, channel) => {
      const scale = Math.min(
        maximumScale,
        Math.max(
          0,
          (channel.totalSpend + (allocation[channel.channel] ?? 0)) /
            Math.max(channel.totalSpend, 1),
        ),
      );
      const position = scale / maximumScale * (points - 1);
      const lower = Math.floor(position);
      const upper = Math.min(points - 1, Math.ceil(position));
      const weight = position - lower;
      const values = surfaces.get(channel.channel)!;
      return total + values[lower] * (1 - weight) + values[upper] * weight;
    }, 0);
    return (
      (contribution - currentContribution) * margin -
      sum(Object.values(allocation))
    );
  };
}

function recommendForDecisionScenario(
  business: SyntheticBusiness,
  contract: DecisionScenarioContract,
  predictedValue: (allocation: Record<string, number>) => number,
): { allocation: Record<string, number>; spend: number; predictedProfit: number } {
  const channels = business.truth.channels.map((channel) => channel.channel);
  const totalSpend = sum(business.truth.channels.map((channel) => channel.totalSpend));
  const budgetSteps = contract.fixedBudgetShare === undefined
    ? DECISION_BUDGET_STEPS
    : 1;
  let best = Object.fromEntries(channels.map((channel) => [channel, 0]));
  let bestSpend = 0;
  let bestValue = 0;
  for (let budgetIndex = 0; budgetIndex < budgetSteps; budgetIndex += 1) {
    const fraction = budgetSteps === 1 ? 0 : budgetIndex / (budgetSteps - 1);
    const share = contract.fixedBudgetShare ??
      contract.minimumBudgetShare +
        fraction * (contract.maximumBudgetShare - contract.minimumBudgetShare);
    const increment = totalSpend * share;
    for (
      let index = 0;
      index < DECISION_ALLOCATION_CANDIDATES;
      index += 1
    ) {
      const allocation = contract.id === "fixed-budget-mix"
        ? reallocationCandidate(business, index)
        : candidateAllocation(business, index, increment);
      const predictedProfit = predictedValue(allocation);
      if (predictedProfit > bestValue) {
        bestValue = predictedProfit;
        best = allocation;
        bestSpend = sum(Object.values(allocation));
      }
    }
  }
  return { allocation: best, spend: bestSpend, predictedProfit: bestValue };
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

export function evaluateDecisionScenarioRegret(
  business: SyntheticBusiness,
  model: ModelResult,
  config: ModelConfig,
): {
  regret: Record<DecisionScenarioId, number>;
  recommendedSpend: Record<DecisionScenarioId, number>;
} {
  const trueValue = trueDecisionValueSurface(
    business.scenario,
    business.truth.channels,
    business.truth.allocation.currentContribution,
    business.truth.allocation.effectiveRevenueMargin,
  );
  const predictedValue = modelDecisionValueSurface(business, model, config);
  const regret = {} as Record<DecisionScenarioId, number>;
  const recommendedSpend = {} as Record<DecisionScenarioId, number>;
  DECISION_SCENARIO_CONTRACTS.forEach((contract) => {
    const recommendation = recommendForDecisionScenario(
      business,
      contract,
      predictedValue,
    );
    const truth = business.truth.decisionScenarios.find(
      (decision) => decision.id === contract.id,
    )!;
    const realized = trueValue(recommendation.allocation);
    const gap = Math.max(0, truth.optimalIncrementalProfit - realized);
    regret[contract.id] = Math.min(
      1,
      gap /
        Math.max(
          truth.optimalIncrementalProfit,
          Math.abs(realized),
          1,
        ),
    );
    recommendedSpend[contract.id] = recommendation.spend;
  });
  return { regret, recommendedSpend };
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
  scenarioProfitRegret: Record<DecisionScenarioId, number>;
  scenarioRecommendedSpend: Record<DecisionScenarioId, number>;
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
  const decisionEvaluation = evaluateDecisionScenarioRegret(
    business,
    model,
    config,
  );
  const trueOutcomeUnderRecommendation = trueContributionForAllocation(
    business,
    recommendation.allocation,
  );
  const trueProfitUnderRecommendation = trueIncrementalProfitForAllocation(
    business,
    recommendation.allocation,
  );
  const revenueGap = Math.max(
    0,
    business.truth.allocation.optimalContribution -
      trueOutcomeUnderRecommendation,
  );
  // Average four predeclared decision regrets; individual labels remain
  // inspectable so an offsetting success cannot hide a catastrophic decision.
  const scenarioRegrets = Object.values(decisionEvaluation.regret);
  const profitRegret = scenarioRegrets.reduce(
    (total, value) => total + value,
    0,
  ) / Math.max(scenarioRegrets.length, 1);
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
    scenarioProfitRegret: decisionEvaluation.regret,
    scenarioRecommendedSpend: decisionEvaluation.recommendedSpend,
    recommendedAdditionalBudget: recommendation.allocation,
    recommendedSpend: recommendation.spend,
    trueOutcomeUnderRecommendation,
    trueProfitUnderRecommendation,
  };
}
