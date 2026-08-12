import type { MediaResponseConfig } from "../../lib/mmm/types";
import {
  responseTransform,
} from "../../lib/mmm/response";
import {
  EVIDENCE_REGISTRY_VERSION,
  sampleHierarchicalRoi,
} from "./evidence";
import { createSeededRandom } from "./random";
import type {
  AllocationTruth,
  ChannelTruth,
  SimulatedExperimentTruth,
  SyntheticBusiness,
  SyntheticChannelConfig,
  SyntheticScenarioConfig,
} from "./types";

export const SIMULATOR_VERSION = "flux-score-v2-dgp-2026.08.2";

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function mean(values: number[]): number {
  return sum(values) / Math.max(values.length, 1);
}

function standardize(values: number[]): number[] {
  const average = mean(values);
  const scale = Math.sqrt(mean(values.map((value) => (value - average) ** 2)));
  return values.map((value) => (value - average) / Math.max(scale, 1e-12));
}

function responseContract(config: SyntheticChannelConfig): MediaResponseConfig {
  return config.response.family === "weibull"
    ? {
        adstockType: "weibull",
        adstock: 0.3,
        weibullShape: config.response.shape,
        weibullScale: config.response.scale,
        saturation: config.response.hillShape,
        halfSaturationQuantile: config.response.halfSaturationQuantile,
        kernelNormalization: config.response.kernelNormalization,
      }
    : {
        adstockType: "geometric",
        adstock: config.response.decay,
        weibullShape: 2,
        weibullScale: 4,
        saturation: config.response.hillShape,
        halfSaturationQuantile: config.response.halfSaturationQuantile,
        kernelNormalization: config.response.kernelNormalization,
      };
}

export function fixedHill(
  values: number[],
  shape: number,
  halfSaturation: number,
): number[] {
  const safeHalf = Math.max(halfSaturation, 1e-12);
  return values.map((value) => {
    const powered = Math.max(value, 0) ** shape;
    return powered / (powered + safeHalf ** shape || 1);
  });
}

function transformDelivery(
  deliveryUnits: number[],
  config: SyntheticChannelConfig,
  fixedHalfSaturation?: number,
): { transformed: number[]; halfSaturation: number } {
  const transformed = responseTransform(
    deliveryUnits,
    responseContract(config),
    fixedHalfSaturation,
  );
  return {
    transformed: transformed.transformed,
    halfSaturation: transformed.halfSaturation,
  };
}

function weeklyDate(index: number): string {
  const date = new Date(Date.UTC(2022, 0, 3 + index * 7));
  return date.toISOString().slice(0, 10);
}

function csvValue(value: string | number): string {
  return typeof value === "number" ? value.toFixed(6) : value;
}

function toCsv(rows: Record<string, string | number>[]): string {
  const columns = Object.keys(rows[0] ?? {});
  return [
    columns.join(","),
    ...rows.map((row) => columns.map((column) => csvValue(row[column])).join(",")),
  ].join("\n");
}

function enumerateAllocations(
  channels: string[],
  total: number,
  units: number,
): Record<string, number>[] {
  if (total <= 0) return [Object.fromEntries(channels.map((channel) => [channel, 0]))];
  const allocations: Record<string, number>[] = [];
  const recurse = (index: number, remainingUnits: number, values: number[]) => {
    if (index === channels.length - 1) {
      const completed = [...values, remainingUnits];
      allocations.push(
        Object.fromEntries(
          channels.map((channel, channelIndex) => [
            channel,
            (completed[channelIndex] / units) * total,
          ]),
        ),
      );
      return;
    }
    for (let value = 0; value <= remainingUnits; value += 1) {
      recurse(index + 1, remainingUnits - value, [...values, value]);
    }
  };
  recurse(0, units, []);
  return allocations;
}

function deliveryScale(
  config: SyntheticChannelConfig,
  budgetScale: number,
): number {
  if (config.delivery.kind === "auction") {
    return Math.max(0, budgetScale ** 0.9);
  }
  if (config.delivery.kind === "reach") {
    const incrementalFrequency =
      config.delivery.frequencyInflation * Math.max(budgetScale - 1, 0);
    return Math.max(0, budgetScale / (1 + incrementalFrequency));
  }
  return Math.max(0, budgetScale);
}

export function trueContributionForAllocation(
  business: Pick<SyntheticBusiness, "scenario" | "truth">,
  additionalBudget: Record<string, number>,
): number {
  return business.truth.channels.reduce((total, channelTruth) => {
    const config = business.scenario.channels.find(
      (item) => item.channel === channelTruth.channel,
    );
    if (!config) throw new Error(`Missing config for ${channelTruth.channel}.`);
    const additional = additionalBudget[channelTruth.channel] ?? 0;
    const budgetScale =
      (channelTruth.totalSpend + additional) /
      Math.max(channelTruth.totalSpend, 1);
    const counterfactualDelivery = channelTruth.deliveryUnits.map(
      (value) => value * deliveryScale(config, budgetScale),
    );
    const transformed = transformDelivery(
      counterfactualDelivery,
      config,
      channelTruth.halfSaturation,
    ).transformed;
    return total + channelTruth.coefficient * sum(transformed);
  }, 0);
}

export function trueIncrementalProfitForAllocation(
  business: Pick<SyntheticBusiness, "scenario" | "truth">,
  additionalBudget: Record<string, number>,
): number {
  const contribution = trueContributionForAllocation(business, additionalBudget);
  const incrementalOutcome = contribution - business.truth.allocation.currentContribution;
  const spend = sum(Object.values(additionalBudget));
  return (
    incrementalOutcome * business.truth.allocation.effectiveRevenueMargin - spend
  );
}

function allocationTruth(
  scenario: SyntheticScenarioConfig,
  channels: ChannelTruth[],
): AllocationTruth {
  const totalSpend = sum(channels.map((channel) => channel.totalSpend));
  const maximumAdditionalBudget = totalSpend * scenario.maximumIncrementShare;
  const currentContribution = sum(channels.map((channel) => channel.totalContribution));
  const effectiveRevenueMargin = Math.min(
    scenario.grossMargin * scenario.ltvRevenueMultiplier,
    1,
  );
  const gridUnits = 20;
  const shell = {
    scenario,
    truth: {
      channels,
      allocation: { currentContribution, effectiveRevenueMargin },
    },
  } as Pick<SyntheticBusiness, "scenario" | "truth">;
  let optimalContribution = currentContribution;
  let optimalIncrementalProfit = 0;
  let optimalAdditionalBudget = Object.fromEntries(
    channels.map((channel) => [channel.channel, 0]),
  );
  let optimalSpend = 0;
  for (const budgetShare of scenario.decisionBudgetShares) {
    const increment = totalSpend * budgetShare;
    for (const allocation of enumerateAllocations(
      channels.map((channel) => channel.channel),
      increment,
      gridUnits,
    )) {
      const respectsBounds = channels.every((channel) =>
        (allocation[channel.channel] ?? 0) <=
        increment *
          (scenario.channels.find((item) => item.channel === channel.channel)
            ?.allocation.maximumShareOfIncrement ?? 1) +
          1e-6,
      );
      if (!respectsBounds) continue;
      const contribution = trueContributionForAllocation(shell, allocation);
      const incrementalProfit =
        (contribution - currentContribution) * effectiveRevenueMargin - increment;
      if (incrementalProfit > optimalIncrementalProfit) {
        optimalIncrementalProfit = incrementalProfit;
        optimalContribution = contribution;
        optimalAdditionalBudget = allocation;
        optimalSpend = increment;
      }
    }
  }
  return {
    maximumAdditionalBudget,
    currentContribution,
    optimalContribution,
    optimalIncrementalOutcome: optimalContribution - currentContribution,
    optimalIncrementalProfit,
    optimalAdditionalBudget,
    optimalSpend,
    effectiveRevenueMargin,
    evaluatedBudgetShares: scenario.decisionBudgetShares,
    gridUnits,
  };
}

function simulatedExperiments(
  scenario: SyntheticScenarioConfig,
  channels: ChannelTruth[],
  random: ReturnType<typeof createSeededRandom>,
): SimulatedExperimentTruth[] {
  return channels.flatMap((channel) => {
    const config = scenario.channels.find((item) => item.channel === channel.channel)!;
    if (config.experiment.design === "none") return [];
    const standardError = Math.max(
      0.12,
      channel.targetRoi * config.experiment.standardErrorShare,
    );
    const observedRoi = Math.max(
      0.05,
      channel.targetRoi * (1 + config.experiment.biasShare) +
        random.normal() * standardError,
    );
    const experiment = {
      channel: channel.spendColumn,
      startDate: weeklyDate(104),
      endDate: weeklyDate(129),
      incrementalOutcome: observedRoi * config.experiment.spend,
      incrementalSpend: config.experiment.spend,
      standardError,
      confidence: 0.9,
      scope: "total" as const,
      source: `Simulated independent ${config.experiment.design} experiment`,
    };
    return [
      {
        channel: channel.channel,
        design: config.experiment.design,
        trueRoi: channel.targetRoi,
        observedRoi,
        standardError,
        biasShare: config.experiment.biasShare,
        experiment,
      },
    ];
  });
}

export function generateSyntheticBusiness(
  scenario: SyntheticScenarioConfig,
): SyntheticBusiness {
  if (scenario.channels.length < 2) {
    throw new Error("A synthetic MMM business needs at least two channels.");
  }
  const random = createSeededRandom(scenario.seed);
  const brandLogEffect = random.normal() * 0.16;
  const demandRaw: number[] = [];
  const commercialRaw: number[] = [];
  const planningRaw: number[] = [];
  for (let index = 0; index < scenario.periods; index += 1) {
    demandRaw.push(
      scenario.latentDemandPersistence * (demandRaw[index - 1] ?? 0) +
        random.normal(),
    );
    commercialRaw.push(
      scenario.commercialPersistence * (commercialRaw[index - 1] ?? 0) +
        random.normal(),
    );
    planningRaw.push(
      scenario.planningPersistence * (planningRaw[index - 1] ?? 0) +
        0.45 * commercialRaw[index] +
        random.normal() * 0.8,
    );
  }
  const latentDemand = standardize(demandRaw);
  const commercialIntensity = standardize(commercialRaw);
  const planningIntensity = standardize(planningRaw);
  const promotion = Array.from({ length: scenario.periods }, (_, index) => {
    const logit = -2 + 0.55 * commercialIntensity[index] + 0.2 * latentDemand[index];
    const probability = 1 / (1 + Math.exp(-logit));
    return random.uniform() < probability ? 1 : 0;
  });
  const eventShock = Array.from({ length: scenario.periods }, () =>
    random.uniform() < scenario.eventShockProbability
      ? Math.max(-1.5, Math.min(2.5, random.normal()))
      : 0,
  );

  const channelTruth: ChannelTruth[] = scenario.channels.map((channel) => {
    const targetRoi =
      channel.targetRoi ??
      sampleHierarchicalRoi(channel.roiEvidenceId, random, brandLogEffect);
    let flightActive = false;
    const spend: number[] = [];
    const deliveryUnits: number[] = [];
    const deliveryEfficiency: number[] = [];
    for (let index = 0; index < scenario.periods; index += 1) {
      const seasonal = 0.1 * Math.sin((2 * Math.PI * index) / 52.18);
      const budgetIntent =
        channel.averageWeeklySpend *
        Math.exp(
          channel.demandCoupling * latentDemand[index] +
            channel.planningCoupling * planningIntensity[index] +
            seasonal +
            channel.spendVolatility * random.normal(),
        );
      const price =
        channel.delivery.averageUnitPrice *
        Math.exp(
          channel.delivery.priceVolatility * random.normal() +
            0.06 * commercialIntensity[index],
        );
      let deliveredSpend = budgetIntent;
      let equivalentUnits = budgetIntent;
      if (channel.delivery.kind === "auction") {
        const averageClicks =
          channel.averageWeeklySpend / channel.delivery.averageUnitPrice;
        const availableClicks =
          averageClicks *
          Math.exp(
            channel.delivery.inventoryDemandCoupling * latentDemand[index] +
              0.1 * random.normal(),
          );
        const clicks = Math.min(budgetIntent / price, availableClicks * 1.35);
        deliveredSpend = clicks * price;
        equivalentUnits = clicks * channel.delivery.averageUnitPrice;
      } else if (channel.delivery.kind === "reach") {
        const impressions = (budgetIntent / price) * 1_000;
        const pressure = budgetIntent / Math.max(channel.averageWeeklySpend, 1);
        const frequency =
          1 + channel.delivery.frequencyInflation * Math.max(pressure, 0) ** 1.35;
        const reach = impressions / frequency;
        equivalentUnits = (reach / 1_000) * channel.delivery.averageUnitPrice;
      } else {
        flightActive = flightActive
          ? random.uniform() < channel.delivery.flightContinuationProbability
          : random.uniform() < channel.delivery.flightStartProbability;
        deliveredSpend = flightActive ? budgetIntent : 0;
        const grps = deliveredSpend / price;
        equivalentUnits = grps * channel.delivery.averageUnitPrice;
      }
      spend.push(Math.max(0, deliveredSpend));
      deliveryUnits.push(Math.max(0, equivalentUnits));
      deliveryEfficiency.push(
        equivalentUnits / Math.max(deliveredSpend, 1),
      );
    }
    const { transformed, halfSaturation } = transformDelivery(
      deliveryUnits,
      channel,
    );
    const totalSpend = sum(spend);
    const coefficient =
      (targetRoi * totalSpend) / Math.max(sum(transformed), 1e-12);
    const contribution = transformed.map((value) => value * coefficient);
    const totalContribution = sum(contribution);
    const onePercentScale = 1.01;
    const marginalTransform = transformDelivery(
      deliveryUnits.map((value) => value * deliveryScale(channel, onePercentScale)),
      channel,
      halfSaturation,
    ).transformed;
    const marginalRoiAtObserved =
      (coefficient * (sum(marginalTransform) - sum(transformed))) /
      Math.max(totalSpend * 0.01, 1);
    return {
      channel: channel.channel,
      spendColumn: channel.spendColumn,
      evidenceId: channel.roiEvidenceId,
      targetRoi,
      realizedRoi: totalContribution / Math.max(totalSpend, 1),
      marginalRoiAtObserved,
      coefficient,
      totalSpend,
      totalContribution,
      halfSaturation,
      spend,
      deliveryUnits,
      deliveryEfficiency,
      transformed,
      contribution,
    };
  });

  const baseline = Array.from({ length: scenario.periods }, (_, index) =>
    Math.max(
      30_000,
      205_000 +
        index * 105 +
        34_000 * latentDemand[index] +
        28_000 * commercialIntensity[index] +
        17_000 * Math.sin((2 * Math.PI * index) / 52.18) +
        promotion[index] * 46_000 +
        eventShock[index] * 24_000,
    ),
  );
  const deterministicOutcome = baseline.map(
    (value, index) =>
      value +
      channelTruth.reduce(
        (total, channel) => total + channel.contribution[index],
        0,
      ),
  );
  const revenue = deterministicOutcome.map((value, index) => {
    const sigma = scenario.noiseShare * (1 + 0.3 * Math.abs(eventShock[index]));
    return Math.max(1, value * Math.exp(sigma * random.normal() - 0.5 * sigma ** 2));
  });
  const observedRows = Array.from({ length: scenario.periods }, (_, index) => ({
    date: weeklyDate(index),
    revenue: revenue[index],
    ...Object.fromEntries(
      channelTruth.map((channel) => [channel.spendColumn, channel.spend[index]]),
    ),
    promotion: promotion[index],
    ...(scenario.observeDemandProxy
      ? { demand_index: latentDemand[index] + random.normal() * 0.12 }
      : {}),
  }));
  const experiments = simulatedExperiments(scenario, channelTruth, random);
  const business = {
    scenario,
    observedCsv: toCsv(observedRows),
    observedRows,
    truth: {
      simulatorVersion: SIMULATOR_VERSION,
      evidenceRegistryVersion: EVIDENCE_REGISTRY_VERSION,
      scenarioId: scenario.id,
      seed: scenario.seed,
      latentDemand,
      planningIntensity,
      commercialIntensity,
      baseline,
      deterministicOutcome,
      channels: channelTruth,
      experiments,
      allocation: {} as AllocationTruth,
      industryBenchmarks: scenario.industryBenchmarks,
    },
  };
  business.truth.allocation = allocationTruth(scenario, channelTruth);
  return business;
}

export function verifyInjectedTruth(business: SyntheticBusiness): void {
  business.truth.channels.forEach((channel) => {
    const difference = Math.abs(channel.realizedRoi - channel.targetRoi);
    if (difference > 1e-9) {
      throw new Error(
        `${channel.channel} realized ROI ${channel.realizedRoi} differs from target ${channel.targetRoi}.`,
      );
    }
    if (
      channel.spend.length !== business.observedRows.length ||
      channel.deliveryUnits.length !== business.observedRows.length ||
      channel.contribution.length !== business.observedRows.length
    ) {
      throw new Error(`${channel.channel} truth vectors are misaligned.`);
    }
  });
  const allocated = sum(Object.values(business.truth.allocation.optimalAdditionalBudget));
  if (Math.abs(allocated - business.truth.allocation.optimalSpend) > 1e-6) {
    throw new Error("The optimal allocation does not conserve selected spend.");
  }
  if (business.truth.allocation.optimalIncrementalProfit < -1e-6) {
    throw new Error("The profit oracle cannot choose a loss-making plan over zero spend.");
  }
  business.truth.experiments.forEach((study) => {
    if (Math.abs(study.observedRoi - study.trueRoi) < 1e-12) {
      throw new Error(`${study.channel} experiment unrealistically reveals exact truth.`);
    }
  });
}
