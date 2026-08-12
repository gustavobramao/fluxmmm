import { adstock, weibullAdstock } from "../../lib/mmm/math";
import { createSeededRandom } from "./random";
import type {
  AllocationTruth,
  ChannelTruth,
  SyntheticBusiness,
  SyntheticChannelConfig,
  SyntheticScenarioConfig,
} from "./types";

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function mean(values: number[]): number {
  return sum(values) / Math.max(values.length, 1);
}

function standardize(values: number[]): number[] {
  const average = mean(values);
  const scale = Math.sqrt(
    mean(values.map((value) => (value - average) ** 2)),
  );
  return values.map((value) => (value - average) / Math.max(scale, 1e-12));
}

function positiveQuantile(values: number[], quantile: number): number {
  const positive = values
    .filter((value) => value > 0)
    .sort((left, right) => left - right);
  if (!positive.length) return 1;
  const index = Math.min(
    positive.length - 1,
    Math.max(0, Math.round((positive.length - 1) * quantile)),
  );
  return positive[index];
}

function carryover(
  spend: number[],
  config: SyntheticChannelConfig,
): number[] {
  return config.response.family === "weibull"
    ? weibullAdstock(
        spend,
        config.response.shape,
        config.response.scale,
      )
    : adstock(spend, config.response.decay);
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

function transformSpend(
  spend: number[],
  config: SyntheticChannelConfig,
  fixedHalfSaturation?: number,
): { transformed: number[]; halfSaturation: number } {
  const carried = carryover(spend, config);
  const halfSaturation =
    fixedHalfSaturation ??
    positiveQuantile(carried, config.response.halfSaturationQuantile);
  return {
    transformed: fixedHill(
      carried,
      config.response.hillShape,
      halfSaturation,
    ),
    halfSaturation,
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
    const scale =
      (channelTruth.totalSpend + additional) /
      Math.max(channelTruth.totalSpend, 1);
    const transformed = transformSpend(
      channelTruth.spend.map((value) => value * scale),
      config,
      channelTruth.halfSaturation,
    ).transformed;
    return total + channelTruth.coefficient * sum(transformed);
  }, 0);
}

function allocationTruth(
  scenario: SyntheticScenarioConfig,
  channels: ChannelTruth[],
): AllocationTruth {
  const minimalBusiness = {
    scenario,
    truth: { channels },
  } as Pick<SyntheticBusiness, "scenario" | "truth">;
  const totalSpend = sum(channels.map((channel) => channel.totalSpend));
  const extraBudget = totalSpend * 0.1;
  const gridUnits = 20;
  const currentContribution = sum(
    channels.map((channel) => channel.totalContribution),
  );
  let optimalContribution = Number.NEGATIVE_INFINITY;
  let optimalAdditionalBudget: Record<string, number> = {};
  for (const allocation of enumerateAllocations(
    channels.map((channel) => channel.channel),
    extraBudget,
    gridUnits,
  )) {
    const contribution = trueContributionForAllocation(
      minimalBusiness,
      allocation,
    );
    if (contribution > optimalContribution) {
      optimalContribution = contribution;
      optimalAdditionalBudget = allocation;
    }
  }
  return {
    extraBudget,
    currentContribution,
    optimalContribution,
    optimalIncrementalOutcome: optimalContribution - currentContribution,
    optimalAdditionalBudget,
    gridUnits,
  };
}

export function generateSyntheticBusiness(
  scenario: SyntheticScenarioConfig,
): SyntheticBusiness {
  if (scenario.channels.length < 2) {
    throw new Error("A synthetic MMM business needs at least two channels.");
  }
  const random = createSeededRandom(scenario.seed);
  const latentDemandRaw: number[] = [];
  const planningRaw: number[] = [];
  for (let index = 0; index < scenario.periods; index += 1) {
    latentDemandRaw.push(
      scenario.latentDemandPersistence * (latentDemandRaw[index - 1] ?? 0) +
        random.normal(),
    );
    planningRaw.push(
      scenario.planningPersistence * (planningRaw[index - 1] ?? 0) +
        random.normal(),
    );
  }
  const latentDemand = standardize(latentDemandRaw);
  const planningIntensity = standardize(planningRaw);
  const promotion = Array.from({ length: scenario.periods }, () =>
    random.uniform() < 0.12 ? 1 : 0,
  );

  const channelTruth: ChannelTruth[] = scenario.channels.map((channel) => {
    const spend = Array.from({ length: scenario.periods }, (_, index) => {
      if (
        channel.flightProbability !== undefined &&
        random.uniform() > channel.flightProbability
      ) {
        return 0;
      }
      const seasonal = 0.12 * Math.sin((2 * Math.PI * index) / 52.18);
      const logMultiplier =
        channel.demandCoupling * latentDemand[index] +
        channel.planningCoupling * planningIntensity[index] +
        seasonal +
        channel.spendVolatility * random.normal();
      return Math.max(0, channel.averageWeeklySpend * Math.exp(logMultiplier));
    });
    const { transformed, halfSaturation } = transformSpend(spend, channel);
    const totalSpend = sum(spend);
    const coefficient =
      (channel.targetRoi * totalSpend) /
      Math.max(sum(transformed), 1e-12);
    const contribution = transformed.map((value) => value * coefficient);
    const totalContribution = sum(contribution);
    return {
      channel: channel.channel,
      spendColumn: channel.spendColumn,
      targetRoi: channel.targetRoi,
      realizedRoi: totalContribution / Math.max(totalSpend, 1),
      coefficient,
      totalSpend,
      totalContribution,
      halfSaturation,
      spend,
      transformed,
      contribution,
    };
  });

  const baseline = Array.from({ length: scenario.periods }, (_, index) =>
    Math.max(
      25_000,
      190_000 +
        index * 115 +
        38_000 * latentDemand[index] +
        16_000 * Math.sin((2 * Math.PI * index) / 52.18) +
        promotion[index] * 42_000,
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
  const noiseScale = mean(deterministicOutcome) * scenario.noiseShare;
  const revenue = deterministicOutcome.map((value) =>
    Math.max(1, value + random.normal() * noiseScale),
  );
  const observedRows = Array.from({ length: scenario.periods }, (_, index) => ({
    date: weeklyDate(index),
    revenue: revenue[index],
    ...Object.fromEntries(
      channelTruth.map((channel) => [channel.spendColumn, channel.spend[index]]),
    ),
    promotion: promotion[index],
    ...(scenario.observeDemandProxy
      ? { demand_index: latentDemand[index] + random.normal() * 0.08 }
      : {}),
  }));
  const truthWithoutAllocation = {
    scenarioId: scenario.id,
    seed: scenario.seed,
    latentDemand,
    planningIntensity,
    baseline,
    deterministicOutcome,
    channels: channelTruth,
    industryBenchmarks: scenario.industryBenchmarks,
  };
  const business = {
    scenario,
    observedCsv: toCsv(observedRows),
    observedRows,
    truth: {
      ...truthWithoutAllocation,
      allocation: {} as AllocationTruth,
    },
  };
  business.truth.allocation = allocationTruth(scenario, channelTruth);
  return business;
}

export function verifyInjectedTruth(business: SyntheticBusiness): void {
  business.truth.channels.forEach((channel) => {
    const difference = Math.abs(channel.realizedRoi - channel.targetRoi);
    if (difference > 1e-10) {
      throw new Error(
        `${channel.channel} realized ROI ${channel.realizedRoi} differs from target ${channel.targetRoi}.`,
      );
    }
    if (
      channel.spend.length !== business.observedRows.length ||
      channel.contribution.length !== business.observedRows.length
    ) {
      throw new Error(`${channel.channel} truth vectors are misaligned.`);
    }
  });
  const allocated = sum(
    Object.values(business.truth.allocation.optimalAdditionalBudget),
  );
  if (Math.abs(allocated - business.truth.allocation.extraBudget) > 1e-6) {
    throw new Error("The optimal allocation does not conserve budget.");
  }
  if (business.truth.allocation.optimalIncrementalOutcome < -1e-6) {
    throw new Error("The optimized allocation cannot reduce true media outcome.");
  }
}
