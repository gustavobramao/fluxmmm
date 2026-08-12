import { passesAgenticEligibility, type AgenticCandidateRun } from "./agentic";
import { sha256, toNumber } from "./csv";
import { mean } from "./math";
import {
  carryoverForResponse,
  positiveQuantile,
  responseForChannel,
} from "./response";
import type { SamplingResult } from "./sampling";
import type { Dataset, MediaResponseConfig } from "./types";

export const BUDGET_OPTIMIZER_VERSION =
  "flux-budget-frontier-v1.1.0-cadence-aware";

export type BudgetScenarioType = "fixed" | "target" | "economic";
export type BudgetHorizon = "quarter" | "year";
export type BudgetRiskMode = "expected" | "balanced" | "conservative";

export interface BudgetChannelConstraint {
  channel: string;
  lowerMultiplier: number;
  upperMultiplier: number;
  locked: boolean;
  excluded: boolean;
}

export interface BudgetOptimizationContract {
  scenario: BudgetScenarioType;
  horizon: BudgetHorizon;
  budgetChange: number;
  targetOutcome: number;
  targetProbability: number;
  grossMargin: number;
  riskMode: BudgetRiskMode;
  constraints: BudgetChannelConstraint[];
}

export interface BudgetReference {
  periods: number;
  periodUnit: "week" | "month";
  currentBudget: number;
  baselineOutcome: number;
  currentIncrementalOutcome: number;
  currentTotalOutcome: number;
  defaultTargetOutcome: number;
}

export interface BudgetFrontierPoint {
  budget: number;
  incrementalOutcome: number;
  incrementalLow: number;
  incrementalHigh: number;
  totalOutcome: number;
  low: number;
  high: number;
  profit: number;
  targetProbability: number;
  feasible: boolean;
}

export interface BudgetChannelRecommendation {
  channel: string;
  currentBudget: number;
  recommendedBudget: number;
  change: number;
  share: number;
  incrementalOutcome: number;
  low: number;
  high: number;
  averageRoi: number;
  marginalRoi: number;
  saturation: number;
  constraint: "interior" | "lower" | "upper" | "locked" | "excluded";
}

export interface BudgetOptimizationResult {
  kind: "budget";
  fingerprint: string;
  cached: boolean;
  optimizerVersion: string;
  samplingFingerprint: string;
  promotedId: string;
  scenario: BudgetScenarioType;
  contract: BudgetOptimizationContract;
  status: "supported" | "review";
  headline: string;
  reference: BudgetReference;
  recommendedBudget: number;
  budgetChange: number;
  expectedIncrementalOutcome: number;
  incrementalLift: number;
  expectedTotalOutcome: number;
  lowTotalOutcome: number;
  highTotalOutcome: number;
  overallRoi: number;
  marginalRoi: number;
  expectedProfit: number;
  targetProbability: number;
  targetFeasible: boolean;
  frontier: BudgetFrontierPoint[];
  channels: BudgetChannelRecommendation[];
  solver: {
    method: "Multi-start constrained frontier search";
    starts: number;
    stableStarts: number;
    frontierPoints: number;
    converged: boolean;
    posteriorSamples: number;
  };
  warnings: string[];
  runAt: string;
}

export interface BudgetOptimizationProgress {
  stage: "preparing" | "frontier" | "selecting" | "checking";
  completed: number;
  total: number;
  detail: string;
}

export const DEFAULT_BUDGET_CONTRACT: BudgetOptimizationContract = {
  scenario: "fixed",
  horizon: "quarter",
  budgetChange: 0,
  targetOutcome: 0,
  targetProbability: 0.8,
  grossMargin: 0.4,
  riskMode: "balanced",
  constraints: [],
};

export function defaultBudgetConstraints(
  dataset: Dataset,
): BudgetChannelConstraint[] {
  return dataset.mediaColumns.map((channel) => ({
    channel,
    lowerMultiplier: 0.5,
    upperMultiplier: 1.5,
    locked: false,
    excluded: false,
  }));
}

function periodsForHorizon(
  dataset: Dataset,
  horizon: BudgetHorizon,
): number {
  if (dataset.modelCadence === "monthly") {
    return horizon === "year" ? 12 : 3;
  }
  return horizon === "year" ? 52 : 13;
}

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function quantile(values: number[], probability: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const position = clamp(probability, 0, 1) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function fixedHill(value: number, shape: number, half: number): number {
  const powered = Math.max(value, 0) ** shape;
  return powered / Math.max(powered + half ** shape, 1e-12);
}

function deterministicNormal(index: number, seed: number): number {
  const first = Math.max(
    1e-9,
    ((Math.sin((index + 1) * (seed + 11) * 12.9898) * 43758.5453) % 1 + 1) % 1,
  );
  const second =
    ((Math.sin((index + 3) * (seed + 29) * 78.233) * 12345.6789) % 1 + 1) % 1;
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
}

export function createBudgetReference(
  dataset: Dataset,
  run: AgenticCandidateRun,
  contract: Pick<BudgetOptimizationContract, "horizon">,
): BudgetReference {
  const periods = Math.min(
    periodsForHorizon(dataset, contract.horizon),
    dataset.rows.length,
  );
  const start = Math.max(0, dataset.rows.length - periods);
  const model = run.model;
  const currentBudget = dataset.mediaColumns.reduce(
    (total, channel) =>
      total +
      dataset.rows
        .slice(start)
        .reduce((channelTotal, row) => channelTotal + Math.max(toNumber(row[channel]), 0), 0),
    0,
  );
  const baselineOutcome = sum(model?.baseline.slice(start) ?? []);
  const actualOutcome = sum(model?.actual.slice(start) ?? []);
  const currentIncrementalOutcome = Math.max(actualOutcome - baselineOutcome, 0);
  const currentTotalOutcome = Math.max(
    actualOutcome,
    baselineOutcome + currentIncrementalOutcome,
  );
  return {
    periods,
    periodUnit: dataset.periodUnit,
    currentBudget,
    baselineOutcome,
    currentIncrementalOutcome,
    currentTotalOutcome,
    defaultTargetOutcome: currentTotalOutcome * 1.1,
  };
}

interface ChannelSurface {
  channel: string;
  referenceBudget: number;
  share: number;
  budgets: number[];
  multipliers: number[];
  roiSamples: number[];
  riskRoi: number;
}

interface PreparedBudgetModel {
  reference: BudgetReference;
  surfaces: ChannelSurface[];
  sampleCount: number;
  baselineSamples: number[];
}

function futureEffectFactor(
  dataset: Dataset,
  run: AgenticCandidateRun,
  channel: string,
): number {
  const path = run.model?.advanced?.coefficientPaths.find(
    (candidate) => candidate.channel.toLowerCase() === channel.toLowerCase(),
  )?.values;
  if (!path?.length) return 1;
  const overall = mean(path.map((value) => Math.max(value, 0)));
  const recentWindow = dataset.modelCadence === "monthly" ? 3 : 13;
  const recent = mean(path.slice(-Math.min(recentWindow, path.length)).map((value) => Math.max(value, 0)));
  return overall > 1e-9 ? clamp(recent / overall, 0.5, 1.5) : 1;
}

function fallbackRoiSamples(
  median: number,
  low: number,
  high: number,
  channelIndex: number,
  sampleCount = 256,
): number[] {
  const standardDeviation = Math.max((high - low) / 3.92, median * 0.08, 1e-6);
  return Array.from({ length: sampleCount }, (_, index) =>
    Math.max(0, median + standardDeviation * deterministicNormal(index, channelIndex + 17)),
  );
}

function alignedRoiSamples(
  sampling: SamplingResult,
  channelIndex: number,
  sampleCount: number,
): number[] {
  const channel = sampling.channels[channelIndex];
  if (!channel) return Array(sampleCount).fill(0);
  const stored = channel.posteriorSamples;
  if (stored?.length) {
    return Array.from(
      { length: sampleCount },
      (_, index) => stored[index % stored.length],
    );
  }
  return fallbackRoiSamples(
    channel.posteriorMedian,
    channel.posteriorLow,
    channel.posteriorHigh,
    channelIndex,
    sampleCount,
  );
}

function flightingPattern(values: number[], periods: number): number[] {
  const recent = values.slice(-periods);
  const total = sum(recent);
  return total > 0
    ? recent.map((value) => Math.max(value, 0) / total)
    : Array.from({ length: periods }, () => 1 / Math.max(periods, 1));
}

function futureTransformedResponse(
  history: number[],
  future: number[],
  response: MediaResponseConfig,
  half: number,
): number {
  const historyLength = history.length;
  const planned = carryoverForResponse([...history, ...future], response);
  const counterfactual = carryoverForResponse(
    [...history, ...Array(future.length).fill(0)],
    response,
  );
  return planned
    .slice(historyLength)
    .reduce(
      (total, value, index) =>
        total +
        Math.max(
          0,
          fixedHill(value, response.saturation, half) -
            fixedHill(
              counterfactual[historyLength + index],
              response.saturation,
              half,
            ),
        ),
      0,
    );
}

function prepareBudgetModel(
  dataset: Dataset,
  run: AgenticCandidateRun,
  sampling: SamplingResult,
  contract: BudgetOptimizationContract,
): PreparedBudgetModel {
  const reference = createBudgetReference(dataset, run, contract);
  const storedSampleCounts = sampling.channels
    .map((channel) => channel.posteriorSamples?.length ?? 0)
    .filter((length) => length > 0);
  const sampleCount = storedSampleCounts.length
    ? Math.min(500, ...storedSampleCounts)
    : 256;
  const surfaces = dataset.mediaColumns.map((channel, channelIndex) => {
    const values = dataset.rows.map((row) => Math.max(toNumber(row[channel]), 0));
    const historicalSpend = sum(values);
    const referenceBudget = sum(values.slice(-reference.periods));
    const response = responseForChannel(run.spec.config, channel);
    const carried = carryoverForResponse(values, response);
    const half = positiveQuantile(
      carried,
      response.halfSaturationQuantile,
    );
    const trainingResponse = sum(
      carried.map((value) => fixedHill(value, response.saturation, half)),
    );
    const pattern = flightingPattern(values, reference.periods);
    const maximumBudget = Math.max(referenceBudget * 4, reference.currentBudget * 0.05, 1);
    const budgets = Array.from(
      { length: 81 },
      (_, index) => (maximumBudget * index) / 80,
    );
    const recentFactor = futureEffectFactor(dataset, run, channel);
    const multipliers = budgets.map((budget) => {
      const future = pattern.map((share) => share * budget);
      const transformed = futureTransformedResponse(
        values,
        future,
        response,
        half,
      );
      return (
        (historicalSpend * transformed * recentFactor) /
        Math.max(trainingResponse, 1e-9)
      );
    });
    const samplingIndex = sampling.channels.findIndex(
      (candidate) => candidate.channel.toLowerCase() === channel.toLowerCase(),
    );
    const roiSamples = alignedRoiSamples(
      sampling,
      samplingIndex >= 0 ? samplingIndex : channelIndex,
      sampleCount,
    );
    const riskRoi =
      contract.riskMode === "expected"
        ? mean(roiSamples)
        : contract.riskMode === "conservative"
          ? quantile(roiSamples, 0.1)
          : quantile(roiSamples, 0.5);
    return {
      channel,
      referenceBudget,
      share: referenceBudget / Math.max(reference.currentBudget, 1),
      budgets,
      multipliers,
      roiSamples,
      riskRoi,
    };
  });

  const predictiveWidths = sampling.predictive.high.map(
    (value, index) => value - sampling.predictive.low[index],
  );
  const periodNoise = mean(predictiveWidths) / 3.92;
  const horizonNoise = Math.max(
    periodNoise * Math.sqrt(reference.periods),
    reference.baselineOutcome * 0.025,
  );
  const baselineSamples = Array.from({ length: sampleCount }, (_, index) =>
    Math.max(
      0,
      reference.baselineOutcome +
        deterministicNormal(index, 91) * horizonNoise,
    ),
  );
  return { reference, surfaces, sampleCount, baselineSamples };
}

function interpolate(
  budgets: number[],
  values: number[],
  budget: number,
): number {
  if (!budgets.length) return 0;
  if (budget <= budgets[0]) return values[0] ?? 0;
  if (budget >= budgets.at(-1)!) return values.at(-1) ?? 0;
  const step = budgets[1] - budgets[0];
  const lower = clamp(Math.floor(budget / Math.max(step, 1e-9)), 0, budgets.length - 2);
  const fraction =
    (budget - budgets[lower]) /
    Math.max(budgets[lower + 1] - budgets[lower], 1e-9);
  return values[lower] * (1 - fraction) + values[lower + 1] * fraction;
}

function responseMultiplier(surface: ChannelSurface, budget: number): number {
  return interpolate(surface.budgets, surface.multipliers, Math.max(budget, 0));
}

function scalarObjective(
  model: PreparedBudgetModel,
  allocation: number[],
): number {
  return model.surfaces.reduce(
    (total, surface, index) =>
      total + surface.riskRoi * responseMultiplier(surface, allocation[index]),
    0,
  );
}

interface BudgetBounds {
  lower: number[];
  upper: number[];
  labels: BudgetChannelRecommendation["constraint"][];
  feasible: boolean;
}

function constraintFor(
  contract: BudgetOptimizationContract,
  channel: string,
): BudgetChannelConstraint {
  return (
    contract.constraints.find(
      (constraint) => constraint.channel.toLowerCase() === channel.toLowerCase(),
    ) ?? {
      channel,
      lowerMultiplier: 0.5,
      upperMultiplier: 1.5,
      locked: false,
      excluded: false,
    }
  );
}

function boundsForBudget(
  model: PreparedBudgetModel,
  contract: BudgetOptimizationContract,
  budget: number,
): BudgetBounds {
  const lower: number[] = [];
  const upper: number[] = [];
  const labels: BudgetChannelRecommendation["constraint"][] = [];
  model.surfaces.forEach((surface) => {
    const constraint = constraintFor(contract, surface.channel);
    if (constraint.excluded || surface.referenceBudget <= 0) {
      lower.push(0);
      upper.push(0);
      labels.push("excluded");
      return;
    }
    if (constraint.locked) {
      lower.push(surface.referenceBudget);
      upper.push(surface.referenceBudget);
      labels.push("locked");
      return;
    }
    lower.push(
      Math.max(0, constraint.lowerMultiplier * surface.share * budget),
    );
    upper.push(
      Math.min(
        constraint.upperMultiplier * surface.share * budget,
        surface.budgets.at(-1) ?? Number.POSITIVE_INFINITY,
      ),
    );
    labels.push("interior");
  });
  return {
    lower,
    upper,
    labels,
    feasible:
      sum(lower) <= budget + Math.max(1, budget * 1e-7) &&
      sum(upper) >= budget - Math.max(1, budget * 1e-7),
  };
}

function projectAllocation(
  weights: number[],
  budget: number,
  bounds: BudgetBounds,
): number[] {
  const weightTotal = sum(weights.map((value) => Math.max(value, 0))) || 1;
  const allocation = weights.map((weight, index) =>
    clamp((Math.max(weight, 0) / weightTotal) * budget, bounds.lower[index], bounds.upper[index]),
  );
  for (let iteration = 0; iteration < 12; iteration += 1) {
    const difference = budget - sum(allocation);
    if (Math.abs(difference) <= Math.max(0.01, budget * 1e-9)) break;
    const capacities = allocation.map((value, index) =>
      difference > 0
        ? Math.max(bounds.upper[index] - value, 0)
        : Math.max(value - bounds.lower[index], 0),
    );
    const capacityTotal = sum(capacities);
    if (capacityTotal <= 1e-9) break;
    allocation.forEach((value, index) => {
      allocation[index] = clamp(
        value + difference * (capacities[index] / capacityTotal),
        bounds.lower[index],
        bounds.upper[index],
      );
    });
  }
  return allocation;
}

function localImprove(
  model: PreparedBudgetModel,
  initial: number[],
  budget: number,
  bounds: BudgetBounds,
): { allocation: number[]; objective: number } {
  const allocation = [...initial];
  let objective = scalarObjective(model, allocation);
  for (const fraction of [0.12, 0.06, 0.03, 0.015, 0.0075, 0.003]) {
    const baseStep = Math.max(budget * fraction, 0.01);
    for (let iteration = 0; iteration < 14; iteration += 1) {
      let bestObjective = objective;
      let bestAllocation: number[] | undefined;
      for (let from = 0; from < allocation.length; from += 1) {
        for (let to = 0; to < allocation.length; to += 1) {
          if (from === to) continue;
          const transfer = Math.min(
            baseStep,
            allocation[from] - bounds.lower[from],
            bounds.upper[to] - allocation[to],
          );
          if (transfer <= 1e-9) continue;
          const candidate = [...allocation];
          candidate[from] -= transfer;
          candidate[to] += transfer;
          const candidateObjective = scalarObjective(model, candidate);
          if (candidateObjective > bestObjective + Math.max(1e-7, Math.abs(objective) * 1e-9)) {
            bestObjective = candidateObjective;
            bestAllocation = candidate;
          }
        }
      }
      if (!bestAllocation) break;
      allocation.splice(0, allocation.length, ...bestAllocation);
      objective = bestObjective;
    }
  }
  return { allocation, objective };
}

interface SolvedBudget {
  budget: number;
  allocation: number[];
  bounds: BudgetBounds;
  objective: number;
  starts: number;
  stableStarts: number;
  feasible: boolean;
}

function solveFixedBudget(
  model: PreparedBudgetModel,
  contract: BudgetOptimizationContract,
  budget: number,
): SolvedBudget {
  const safeBudget = Math.max(budget, 0);
  const bounds = boundsForBudget(model, contract, safeBudget);
  if (!bounds.feasible) {
    return {
      budget: safeBudget,
      allocation: [...bounds.lower],
      bounds,
      objective: Number.NEGATIVE_INFINITY,
      starts: 0,
      stableStarts: 0,
      feasible: false,
    };
  }
  const active = model.surfaces.map((surface, index) =>
    bounds.upper[index] > bounds.lower[index] ? surface : undefined,
  );
  const starts = [
    model.surfaces.map((surface) => surface.share),
    model.surfaces.map((surface, index) => active[index] ? 1 : 0),
    model.surfaces.map((surface, index) => active[index] ? Math.max(surface.riskRoi, 0.01) : 0),
    ...model.surfaces.slice(0, 5).map((_, favored) =>
      model.surfaces.map((surface, index) =>
        active[index] ? surface.share * (index === favored ? 4 : 0.6) : 0,
      ),
    ),
  ];
  const solutions = starts.map((weights) =>
    localImprove(
      model,
      projectAllocation(weights, safeBudget, bounds),
      safeBudget,
      bounds,
    ),
  );
  solutions.sort((left, right) => right.objective - left.objective);
  const best = solutions[0];
  const tolerance = Math.max(Math.abs(best.objective) * 0.005, 1e-6);
  return {
    budget: safeBudget,
    allocation: best.allocation,
    bounds,
    objective: best.objective,
    starts: solutions.length,
    stableStarts: solutions.filter(
      (solution) => best.objective - solution.objective <= tolerance,
    ).length,
    feasible: true,
  };
}

interface EvaluatedPlan {
  incrementalDraws: number[];
  totalDraws: number[];
  channelDraws: number[][];
  incrementalOutcome: number;
  incrementalLow: number;
  incrementalHigh: number;
  totalOutcome: number;
  low: number;
  high: number;
  targetProbability: number;
  expectedProfit: number;
}

function evaluatePlan(
  model: PreparedBudgetModel,
  allocation: number[],
  target: number,
  grossMargin: number,
): EvaluatedPlan {
  const channelDraws = model.surfaces.map((surface, channelIndex) => {
    const multiplier = responseMultiplier(surface, allocation[channelIndex]);
    return surface.roiSamples.map((roi) => roi * multiplier);
  });
  const incrementalDraws = Array.from({ length: model.sampleCount }, (_, index) =>
    channelDraws.reduce((total, draws) => total + (draws[index] ?? 0), 0),
  );
  const totalDraws = incrementalDraws.map(
    (value, index) => value + model.baselineSamples[index],
  );
  const budget = sum(allocation);
  return {
    incrementalDraws,
    totalDraws,
    channelDraws,
    incrementalOutcome: mean(incrementalDraws),
    incrementalLow: quantile(incrementalDraws, 0.1),
    incrementalHigh: quantile(incrementalDraws, 0.9),
    totalOutcome: mean(totalDraws),
    low: quantile(totalDraws, 0.1),
    high: quantile(totalDraws, 0.9),
    targetProbability:
      target > 0
        ? totalDraws.filter((value) => value >= target).length /
          Math.max(totalDraws.length, 1)
        : 1,
    expectedProfit: mean(incrementalDraws) * grossMargin - budget,
  };
}

function frontierPoint(
  model: PreparedBudgetModel,
  solved: SolvedBudget,
  target: number,
  grossMargin: number,
): BudgetFrontierPoint {
  if (!solved.feasible) {
    return {
      budget: solved.budget,
      incrementalOutcome: 0,
      incrementalLow: 0,
      incrementalHigh: 0,
      totalOutcome: model.reference.baselineOutcome,
      low: model.reference.baselineOutcome,
      high: model.reference.baselineOutcome,
      profit: -solved.budget,
      targetProbability: 0,
      feasible: false,
    };
  }
  const evaluation = evaluatePlan(
    model,
    solved.allocation,
    target,
    grossMargin,
  );
  return {
    budget: solved.budget,
    incrementalOutcome: evaluation.incrementalOutcome,
    incrementalLow: evaluation.incrementalLow,
    incrementalHigh: evaluation.incrementalHigh,
    totalOutcome: evaluation.totalOutcome,
    low: evaluation.low,
    high: evaluation.high,
    profit: evaluation.expectedProfit,
    targetProbability: evaluation.targetProbability,
    feasible: true,
  };
}

function marginalResponse(
  surface: ChannelSurface,
  budget: number,
): number {
  const step = Math.max(surface.referenceBudget * 0.01, budget * 0.005, 1);
  const lower = Math.max(0, budget - step);
  const upper = budget + step;
  return (
    (responseMultiplier(surface, upper) - responseMultiplier(surface, lower)) *
    quantile(surface.roiSamples, 0.5) /
    Math.max(upper - lower, 1e-9)
  );
}

function channelRecommendations(
  model: PreparedBudgetModel,
  solved: SolvedBudget,
  evaluation: EvaluatedPlan,
): BudgetChannelRecommendation[] {
  return model.surfaces.map((surface, index) => {
    const recommendedBudget = solved.allocation[index];
    const draws = evaluation.channelDraws[index];
    const incrementalOutcome = mean(draws);
    const maximumResponse =
      responseMultiplier(surface, surface.budgets.at(-1) ?? recommendedBudget) *
      quantile(surface.roiSamples, 0.5);
    const initialLabel = solved.bounds.labels[index];
    const tolerance = Math.max(1, solved.budget * 1e-5);
    const constraint =
      initialLabel === "locked" || initialLabel === "excluded"
        ? initialLabel
        : Math.abs(recommendedBudget - solved.bounds.lower[index]) <= tolerance
          ? "lower"
          : Math.abs(recommendedBudget - solved.bounds.upper[index]) <= tolerance
            ? "upper"
            : "interior";
    return {
      channel: surface.channel,
      currentBudget: surface.referenceBudget,
      recommendedBudget,
      change:
        surface.referenceBudget > 0
          ? recommendedBudget / surface.referenceBudget - 1
          : 0,
      share: recommendedBudget / Math.max(solved.budget, 1),
      incrementalOutcome,
      low: quantile(draws, 0.1),
      high: quantile(draws, 0.9),
      averageRoi: incrementalOutcome / Math.max(recommendedBudget, 1),
      marginalRoi: marginalResponse(surface, recommendedBudget),
      saturation: clamp(
        incrementalOutcome / Math.max(maximumResponse, 1e-9),
        0,
        1,
      ),
      constraint,
    };
  });
}

function normalisedContract(
  contract: BudgetOptimizationContract,
  dataset: Dataset,
): BudgetOptimizationContract {
  const defaults = defaultBudgetConstraints(dataset);
  return {
    ...contract,
    budgetChange: clamp(contract.budgetChange, -0.5, 0.5),
    targetProbability: clamp(contract.targetProbability, 0.5, 0.99),
    grossMargin: clamp(contract.grossMargin, 0.05, 0.95),
    constraints: defaults.map((fallback) => {
      const selected = constraintFor(contract, fallback.channel);
      return {
        ...selected,
        lowerMultiplier: clamp(selected.lowerMultiplier, 0, 2),
        upperMultiplier: clamp(
          selected.upperMultiplier,
          selected.lowerMultiplier,
          4,
        ),
      };
    }),
  };
}

export async function budgetFingerprint(
  dataset: Dataset,
  sampling: SamplingResult,
  contract: BudgetOptimizationContract,
): Promise<string> {
  const normalized = normalisedContract(contract, dataset);
  return sha256(
    JSON.stringify({
      version: BUDGET_OPTIMIZER_VERSION,
      dataset: dataset.hash,
      sampling: sampling.fingerprint,
      contract: {
        ...normalized,
        constraints: [...normalized.constraints].sort((left, right) =>
          left.channel.localeCompare(right.channel),
        ),
      },
    }),
  );
}

async function yieldToUi(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

export async function runBudgetOptimization(
  dataset: Dataset,
  run: AgenticCandidateRun,
  sampling: SamplingResult,
  contractInput: BudgetOptimizationContract,
  fingerprint: string,
  onProgress?: (progress: BudgetOptimizationProgress) => void,
): Promise<BudgetOptimizationResult> {
  const contract = normalisedContract(contractInput, dataset);
  onProgress?.({
    stage: "preparing",
    completed: 0,
    total: 25,
    detail: "Freezing the promoted posterior and future response assumptions.",
  });
  await yieldToUi();
  const model = prepareBudgetModel(dataset, run, sampling, contract);
  const target =
    contract.targetOutcome > 0
      ? contract.targetOutcome
      : model.reference.defaultTargetOutcome;
  const frontierBudgets = Array.from(
    { length: 25 },
    (_, index) =>
      model.reference.currentBudget * (0.25 + (2.25 * index) / 24),
  );
  const solvedFrontier: SolvedBudget[] = [];
  const frontier: BudgetFrontierPoint[] = [];
  for (let index = 0; index < frontierBudgets.length; index += 1) {
    const solved = solveFixedBudget(model, contract, frontierBudgets[index]);
    solvedFrontier.push(solved);
    frontier.push(frontierPoint(model, solved, target, contract.grossMargin));
    onProgress?.({
      stage: "frontier",
      completed: index + 1,
      total: frontierBudgets.length,
      detail: `Evaluating supported budget ${index + 1} of ${frontierBudgets.length}.`,
    });
    if (index % 4 === 3) await yieldToUi();
  }

  onProgress?.({
    stage: "selecting",
    completed: 24,
    total: 25,
    detail: "Selecting the scenario-specific point on the efficient frontier.",
  });
  let selected: SolvedBudget;
  let targetFeasible = true;
  if (contract.scenario === "fixed") {
    selected = solveFixedBudget(
      model,
      contract,
      model.reference.currentBudget * (1 + contract.budgetChange),
    );
  } else if (contract.scenario === "target") {
    const qualifyingIndex = frontier.findIndex(
      (point) =>
        point.feasible && point.targetProbability >= contract.targetProbability,
    );
    if (qualifyingIndex < 0) {
      targetFeasible = false;
      selected = solvedFrontier.at(-1)!;
    } else if (qualifyingIndex === 0) {
      selected = solvedFrontier[0];
    } else {
      let lowerBudget = frontier[qualifyingIndex - 1].budget;
      let upperBudget = frontier[qualifyingIndex].budget;
      selected = solvedFrontier[qualifyingIndex];
      for (let iteration = 0; iteration < 7; iteration += 1) {
        const candidate = solveFixedBudget(
          model,
          contract,
          (lowerBudget + upperBudget) / 2,
        );
        const candidatePoint = frontierPoint(
          model,
          candidate,
          target,
          contract.grossMargin,
        );
        if (candidatePoint.targetProbability >= contract.targetProbability) {
          selected = candidate;
          upperBudget = candidate.budget;
        } else {
          lowerBudget = candidate.budget;
        }
      }
    }
  } else {
    const feasible = solvedFrontier.filter((solution) => solution.feasible);
    selected = feasible.reduce((best, candidate) => {
      const bestProfit = frontierPoint(
        model,
        best,
        target,
        contract.grossMargin,
      ).profit;
      const candidateProfit = frontierPoint(
        model,
        candidate,
        target,
        contract.grossMargin,
      ).profit;
      return candidateProfit > bestProfit ? candidate : best;
    }, feasible[0]);
  }

  await yieldToUi();
  onProgress?.({
    stage: "checking",
    completed: 25,
    total: 25,
    detail: "Checking posterior risk, active constraints, and solver stability.",
  });
  const evaluation = evaluatePlan(
    model,
    selected.allocation,
    target,
    contract.grossMargin,
  );
  const currentAllocation = model.surfaces.map(
    (surface) => surface.referenceBudget,
  );
  const currentEvaluation = evaluatePlan(
    model,
    currentAllocation,
    target,
    contract.grossMargin,
  );
  const channels = channelRecommendations(model, selected, evaluation);
  const warnings: string[] = [];
  if (sampling.status !== "ready") {
    warnings.push("The Production posterior still needs attention.");
  }
  if (!passesAgenticEligibility(run)) {
    warnings.push("The specification was promoted with unresolved model evidence.");
  }
  if (!targetFeasible) {
    warnings.push("The target is outside the supported posterior frontier.");
  }
  if (selected.stableStarts < 2) {
    warnings.push("Independent solver starts did not converge on a stable allocation.");
  }
  const extrapolated = channels.filter(
    (channel) =>
      channel.currentBudget > 0 && channel.recommendedBudget > channel.currentBudget * 2,
  );
  if (extrapolated.length) {
    warnings.push(
      `${extrapolated.length} channel${extrapolated.length === 1 ? " exceeds" : "s exceed"} twice the comparable historical budget.`,
    );
  }
  const atEconomicLowerBoundary =
    contract.scenario === "economic" &&
    Math.abs(selected.budget - solvedFrontier[0].budget) <=
      model.reference.currentBudget * 0.01;
  const atEconomicUpperBoundary =
    contract.scenario === "economic" &&
    Math.abs(selected.budget - solvedFrontier.at(-1)!.budget) <=
      model.reference.currentBudget * 0.01;
  const atEconomicBoundary =
    atEconomicLowerBoundary || atEconomicUpperBoundary;
  if (atEconomicUpperBoundary) {
    warnings.push("Profit is still rising at the supported budget ceiling; no interior optimum was claimed.");
  } else if (atEconomicLowerBoundary) {
    warnings.push("Profit peaks at the supported budget floor; test a lower-spend operating scenario before claiming an interior optimum.");
  }
  const supported =
    selected.feasible &&
    selected.stableStarts >= 2 &&
    targetFeasible &&
    !atEconomicBoundary &&
    sampling.status === "ready" &&
    passesAgenticEligibility(run);
  const headline =
    contract.scenario === "fixed"
      ? `Reallocate ${Math.abs(selected.budget / Math.max(model.reference.currentBudget, 1) - 1) < 0.005 ? "the current" : "the selected"} budget for the strongest supported response`
      : contract.scenario === "target"
        ? targetFeasible
          ? `Minimum supported budget for a ${Math.round(contract.targetProbability * 100)}% target probability`
          : "Target exceeds the supported planning frontier"
        : atEconomicBoundary
          ? "Economic optimum lies on a supported planning boundary"
          : "Posterior economic ceiling under the selected margin";
  const recommendedBudget = sum(selected.allocation);
  const weightedMarginalRoi = channels.reduce(
    (total, channel) => total + channel.marginalRoi * channel.share,
    0,
  );
  return {
    kind: "budget",
    fingerprint,
    cached: false,
    optimizerVersion: BUDGET_OPTIMIZER_VERSION,
    samplingFingerprint: sampling.fingerprint,
    promotedId: sampling.promotedId,
    scenario: contract.scenario,
    contract,
    status: supported ? "supported" : "review",
    headline,
    reference: model.reference,
    recommendedBudget,
    budgetChange:
      recommendedBudget / Math.max(model.reference.currentBudget, 1) - 1,
    expectedIncrementalOutcome: evaluation.incrementalOutcome,
    incrementalLift:
      evaluation.incrementalOutcome - currentEvaluation.incrementalOutcome,
    expectedTotalOutcome: evaluation.totalOutcome,
    lowTotalOutcome: evaluation.low,
    highTotalOutcome: evaluation.high,
    overallRoi:
      evaluation.incrementalOutcome / Math.max(recommendedBudget, 1),
    marginalRoi: weightedMarginalRoi,
    expectedProfit: evaluation.expectedProfit,
    targetProbability: evaluation.targetProbability,
    targetFeasible,
    frontier,
    channels,
    solver: {
      method: "Multi-start constrained frontier search",
      starts: selected.starts,
      stableStarts: selected.stableStarts,
      frontierPoints: frontier.length,
      converged: selected.feasible && selected.stableStarts >= 2,
      posteriorSamples: model.sampleCount,
    },
    warnings,
    runAt: new Date().toISOString(),
  };
}
