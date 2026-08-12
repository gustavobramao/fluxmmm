import { EVIDENCE_REGISTRY_VERSION } from "../score_v2/evidence";
import { createSeededRandom } from "../score_v2/random";
import { PILOT_SCENARIOS } from "../score_v2/scenarios";
import {
  DECISION_SCENARIO_CONTRACTS,
  generateSyntheticBusiness,
  SIMULATOR_VERSION,
  verifyInjectedTruth,
} from "../score_v2/simulator";
import type {
  SyntheticChannel,
  SyntheticChannelConfig,
  SyntheticScenarioConfig,
} from "../score_v2/types";
import type {
  AuditBusinessSummary,
  AuditSplit,
  DistributionSummary,
  GeneratorFamilyContract,
  SimulatorAuditV3Artifact,
} from "./types";

export const SIMULATOR_AUDIT_V3_ID = "flux-score-simulator-audit-v3-2026.08.1";

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

export const GENERATOR_FAMILIES: GeneratorFamilyContract[] = [
  {
    id: "balanced-dtc",
    label: "Balanced DTC",
    split: "train",
    businessCount: 100,
    description: "Observable demand, moderate noise, and channel-typical response mechanics.",
  },
  {
    id: "search-demand-harvesting",
    label: "Search demand harvesting",
    split: "train",
    businessCount: 100,
    description: "Search inventory and spend move strongly with hidden underlying demand.",
  },
  {
    id: "social-frequency-pressure",
    label: "Social frequency pressure",
    split: "train",
    businessCount: 100,
    description: "Paid-social reach saturates as frequency rises with spend.",
  },
  {
    id: "delayed-tv",
    label: "Delayed TV",
    split: "validation",
    businessCount: 60,
    description: "Flighted TV has delayed and heterogeneous carryover.",
  },
  {
    id: "correlated-planning",
    label: "Correlated planning",
    split: "validation",
    businessCount: 60,
    description: "A shared commercial plan drives media and baseline simultaneously.",
  },
  {
    id: "wrong-evidence",
    label: "Wrong external evidence",
    split: "audit",
    businessCount: 40,
    description: "Industry benchmarks are intentionally shifted away from causal truth.",
    heldOutReason: "Tests whether a learned score blindly rewards benchmark agreement.",
  },
  {
    id: "swapped-channel-mechanics",
    label: "Swapped mechanics",
    split: "audit",
    businessCount: 40,
    description: "Channel labels no longer imply their usual carryover family.",
    heldOutReason: "Tests whether the score memorizes channel stereotypes instead of diagnostics.",
  },
];

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function between(random: ReturnType<typeof createSeededRandom>, low: number, high: number) {
  return low + random.uniform() * (high - low);
}

function choose<T>(random: ReturnType<typeof createSeededRandom>, values: readonly T[]): T {
  return values[Math.min(values.length - 1, Math.floor(random.uniform() * values.length))];
}

function template(id: string): SyntheticScenarioConfig {
  const map: Record<string, number> = {
    "balanced-dtc": 0,
    "search-demand-harvesting": 1,
    "social-frequency-pressure": 2,
    "delayed-tv": 3,
    "correlated-planning": 4,
    "wrong-evidence": 5,
    "swapped-channel-mechanics": 0,
  };
  return structuredClone(PILOT_SCENARIOS[map[id] ?? 0]);
}

function randomizedResponse(
  channel: SyntheticChannel,
  family: GeneratorFamilyContract,
  random: ReturnType<typeof createSeededRandom>,
): SyntheticChannelConfig["response"] {
  const swapped = family.id === "swapped-channel-mechanics";
  const weibullProbability = swapped
    ? channel === "paid_social" ? 0.85 : channel === "tv" ? 0.15 : 0.45
    : channel === "tv" ? 0.82 : channel === "paid_social" ? 0.18 : 0.08;
  const useWeibull = random.uniform() < weibullProbability;
  const halfSaturationQuantile = between(random, 0.3, 0.82);
  const hillShape = between(
    random,
    channel === "paid_social" ? 0.9 : 0.72,
    channel === "paid_social" ? 3.1 : 2.35,
  );
  if (useWeibull) {
    return {
      family: "weibull",
      shape: between(random, 1.15, 4.8),
      scale: between(
        random,
        channel === "tv" && !swapped ? 3.5 : 1.5,
        channel === "tv" && !swapped ? 13 : 8,
      ),
      hillShape,
      halfSaturationQuantile,
      kernelNormalization: "sum",
    };
  }
  return {
    family: "geometric",
    decay: between(
      random,
      0.01,
      channel === "tv" && !swapped ? 0.78 : 0.55,
    ),
    hillShape,
    halfSaturationQuantile,
    kernelNormalization: "sum",
  };
}

function randomizedExperiment(
  channel: SyntheticChannelConfig,
  split: AuditSplit,
  random: ReturnType<typeof createSeededRandom>,
): SyntheticChannelConfig["experiment"] {
  const availability = split === "audit" ? 0.38 : 0.58;
  if (random.uniform() > availability) return { design: "none" };
  return {
    design: random.uniform() < 0.75 ? "geo" : "platform-holdout",
    spend: channel.averageWeeklySpend * between(random, 3, 10),
    standardErrorShare: between(random, 0.08, 0.38),
    biasShare: between(random, -0.12, 0.12),
  };
}

export function randomizedAuditScenario(
  family: GeneratorFamilyContract,
  familyIndex: number,
): SyntheticScenarioConfig {
  const seed = 31_000_000 + GENERATOR_FAMILIES.indexOf(family) * 100_000 + familyIndex;
  const random = createSeededRandom(seed);
  const base = template(family.id);
  const periods = choose(random, [104, 130, 156, 208, 260] as const);
  const channels = base.channels.map((channel) => {
    const demandBase = family.id === "search-demand-harvesting" && channel.channel === "search"
      ? between(random, 0.75, 1.15)
      : between(random, 0.02, 0.62);
    const planningBase = family.id === "correlated-planning"
      ? between(random, 0.72, 1.08)
      : between(random, 0.04, 0.72);
    const delivery = channel.delivery.kind === "reach"
      ? {
          ...channel.delivery,
          averageUnitPrice: between(random, 8, 28),
          priceVolatility: between(random, 0.08, 0.32),
          frequencyInflation: family.id === "social-frequency-pressure"
            ? between(random, 0.48, 0.95)
            : between(random, 0.16, 0.62),
        }
      : channel.delivery.kind === "auction"
        ? {
            ...channel.delivery,
            averageUnitPrice: between(random, 0.7, 4.8),
            priceVolatility: between(random, 0.06, 0.32),
            inventoryDemandCoupling: family.id === "search-demand-harvesting"
              ? between(random, 0.45, 0.95)
              : between(random, 0.08, 0.55),
          }
        : {
            ...channel.delivery,
            averageUnitPrice: between(random, 350, 1_150),
            priceVolatility: between(random, 0.06, 0.24),
            flightStartProbability: between(random, 0.1, 0.34),
            flightContinuationProbability: between(random, 0.48, 0.9),
          };
    const next: SyntheticChannelConfig = {
      ...channel,
      targetRoi: undefined,
      averageWeeklySpend: channel.averageWeeklySpend * between(random, 0.42, 2.4),
      spendVolatility: between(random, 0.08, 0.65),
      demandCoupling: demandBase,
      planningCoupling: planningBase,
      delivery,
      response: randomizedResponse(channel.channel, family, random),
      allocation: {
        maximumShareOfIncrement: between(random, 0.48, 0.92),
      },
    };
    next.experiment = randomizedExperiment(next, family.split, random);
    return next;
  });
  const industryBenchmarks = { ...base.industryBenchmarks };
  if (family.id === "wrong-evidence") {
    (Object.keys(industryBenchmarks) as SyntheticChannel[]).forEach((channel) => {
      industryBenchmarks[channel] *= random.uniform() < 0.5
        ? between(random, 0.25, 0.55)
        : between(random, 1.9, 3.2);
    });
  } else {
    (Object.keys(industryBenchmarks) as SyntheticChannel[]).forEach((channel) => {
      industryBenchmarks[channel] *= Math.exp(random.normal() * 0.28);
    });
  }
  return {
    ...base,
    id: `${family.id}-${String(familyIndex + 1).padStart(3, "0")}`,
    label: `${family.label} ${familyIndex + 1}`,
    description: family.description,
    seed,
    periods,
    latentDemandPersistence: between(random, 0.35, 0.94),
    planningPersistence: between(random, 0.3, 0.94),
    commercialPersistence: between(random, 0.35, 0.93),
    noiseShare: between(random, 0.018, family.split === "audit" ? 0.14 : 0.1),
    eventShockProbability: between(random, 0.015, 0.2),
    observeDemandProxy: family.id === "balanced-dtc" ? random.uniform() < 0.8 : random.uniform() < 0.25,
    grossMargin: between(random, 0.24, 0.68),
    ltvRevenueMultiplier: between(random, 1, 2.45),
    maximumIncrementShare: between(random, 0.2, 0.75),
    decisionBudgetShares: [0, 0.05, 0.1, 0.2, 0.35],
    channels,
    industryBenchmarks,
  };
}

function quantile(values: number[], probability: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  const position = clamp(probability, 0, 1) * Math.max(sorted.length - 1, 0);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return (sorted[lower] ?? 0) * (1 - weight) + (sorted[upper] ?? 0) * weight;
}

function distribution(values: number[]): DistributionSummary {
  return {
    minimum: Math.min(...values),
    p10: quantile(values, 0.1),
    median: quantile(values, 0.5),
    p90: quantile(values, 0.9),
    maximum: Math.max(...values),
  };
}

function businessSummary(
  family: GeneratorFamilyContract,
  scenario: SyntheticScenarioConfig,
): AuditBusinessSummary {
  const business = generateSyntheticBusiness(scenario);
  verifyInjectedTruth(business);
  const totalSpend = business.truth.channels.reduce(
    (total, channel) => total + channel.totalSpend,
    0,
  );
  const benchmarkLogError = business.truth.channels.reduce((total, channel) =>
    total + channel.totalSpend / Math.max(totalSpend, 1) * Math.abs(
      Math.log(
        Math.max(scenario.industryBenchmarks[channel.channel], 1e-6) /
        Math.max(channel.targetRoi, 1e-6),
      ),
    ), 0);
  return {
    id: scenario.id,
    family: family.id,
    split: family.split,
    seed: scenario.seed,
    periods: scenario.periods,
    noiseShare: scenario.noiseShare,
    effectiveRevenueMargin: business.truth.allocation.effectiveRevenueMargin,
    hiddenDemand: !scenario.observeDemandProxy,
    experimentCount: business.truth.experiments.length,
    benchmarkLogError,
    channels: Object.fromEntries(business.truth.channels.map((channel) => {
      const config = scenario.channels.find((item) => item.channel === channel.channel)!;
      return [channel.channel, {
        roi: channel.targetRoi,
        marginalRoi: channel.marginalRoiAtObserved,
        family: config.response.family,
        memory: config.response.family === "weibull" ? config.response.scale : config.response.decay,
        hillShape: config.response.hillShape,
        halfSaturationQuantile: config.response.halfSaturationQuantile,
        demandCoupling: config.demandCoupling,
        planningCoupling: config.planningCoupling,
        spendShare: channel.totalSpend / Math.max(totalSpend, 1),
      }];
    })) as AuditBusinessSummary["channels"],
    decisions: Object.fromEntries(business.truth.decisionScenarios.map((decision) => [
      decision.id,
      {
        optimalBudgetShare: decision.optimalSpend / Math.max(totalSpend, 1),
        optimalIncrementalProfit: decision.optimalIncrementalProfit,
        mixShiftShare:
          sum(Object.values(decision.optimalAdditionalBudget).map(Math.abs)) /
          (2 * Math.max(totalSpend, 1)),
      },
    ])) as AuditBusinessSummary["decisions"],
  };
}

export function generateAuditPopulation(): SimulatorAuditV3Artifact {
  const businesses = GENERATOR_FAMILIES.flatMap((family) =>
    Array.from({ length: family.businessCount }, (_, index) =>
      businessSummary(family, randomizedAuditScenario(family, index)),
    ),
  );
  const channelCoverage = Object.fromEntries(
    (["paid_social", "search", "tv"] as SyntheticChannel[]).map((channel) => {
      const values = businesses.map((business) => business.channels[channel]);
      return [channel, {
        roi: distribution(values.map((item) => item.roi)),
        marginalRoi: distribution(values.map((item) => item.marginalRoi)),
        memory: distribution(values.map((item) => item.memory)),
        hillShape: distribution(values.map((item) => item.hillShape)),
        halfSaturationQuantile: distribution(values.map((item) => item.halfSaturationQuantile)),
        demandCoupling: distribution(values.map((item) => item.demandCoupling)),
        planningCoupling: distribution(values.map((item) => item.planningCoupling)),
        weibullShare: values.filter((item) => item.family === "weibull").length / values.length,
      }];
    }),
  ) as SimulatorAuditV3Artifact["coverage"]["channel"];
  const decisions = Object.fromEntries(
    DECISION_SCENARIO_CONTRACTS.map((contract) => {
      const values = businesses.map((business) => business.decisions[contract.id]);
      const budgetShares = values.map((item) => item.optimalBudgetShare);
      return [contract.id, {
        label: contract.label,
        optimalBudgetShare: distribution(budgetShares),
        profitOpportunity: distribution(values.map((item) => item.optimalIncrementalProfit)),
        mixShiftShare: distribution(values.map((item) => item.mixShiftShare)),
        decreaseShare: budgetShares.filter((value) => value < -0.005).length / values.length,
        unchangedShare: budgetShares.filter((value) => Math.abs(value) <= 0.005).length / values.length,
        increaseShare: budgetShares.filter((value) => value > 0.005).length / values.length,
      }];
    }),
  ) as SimulatorAuditV3Artifact["decisions"];
  return {
    artifactId: SIMULATOR_AUDIT_V3_ID,
    simulatorVersion: SIMULATOR_VERSION,
    evidenceRegistryVersion: EVIDENCE_REGISTRY_VERSION,
    businessCount: businesses.length,
    splits: {
      train: businesses.filter((business) => business.split === "train").length,
      validation: businesses.filter((business) => business.split === "validation").length,
      audit: businesses.filter((business) => business.split === "audit").length,
    },
    families: GENERATOR_FAMILIES,
    coverage: {
      historyWeeks: distribution(businesses.map((business) => business.periods)),
      noiseShare: distribution(businesses.map((business) => business.noiseShare)),
      effectiveRevenueMargin: distribution(businesses.map((business) => business.effectiveRevenueMargin)),
      benchmarkLogError: distribution(businesses.map((business) => business.benchmarkLogError)),
      experimentCoverage: businesses.filter((business) => business.experimentCount > 0).length / businesses.length,
      channel: channelCoverage,
    },
    decisions,
    businesses,
  };
}
