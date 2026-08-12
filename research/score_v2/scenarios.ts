import type {
  SyntheticChannelConfig,
  SyntheticScenarioConfig,
} from "./types";

function baseChannels(): SyntheticChannelConfig[] {
  return [
    {
      channel: "paid_social",
      spendColumn: "meta_acquisition_spend",
      roiEvidenceId: "dtc-meta-acquisition",
      averageWeeklySpend: 22_000,
      spendVolatility: 0.3,
      demandCoupling: 0.16,
      planningCoupling: 0.42,
      delivery: {
        kind: "reach",
        priceLabel: "CPM",
        averageUnitPrice: 14,
        priceVolatility: 0.2,
        frequencyInflation: 0.32,
      },
      experiment: {
        design: "geo",
        spend: 120_000,
        standardErrorShare: 0.16,
        biasShare: 0,
      },
      allocation: { maximumShareOfIncrement: 0.8 },
      response: {
        family: "geometric",
        decay: 0.22,
        hillShape: 1.5,
        halfSaturationQuantile: 0.55,
        kernelNormalization: "sum",
      },
    },
    {
      channel: "search",
      spendColumn: "google_search_nonbrand_spend",
      roiEvidenceId: "dtc-search-nonbrand",
      averageWeeklySpend: 30_000,
      spendVolatility: 0.2,
      demandCoupling: 0.8,
      planningCoupling: 0.15,
      delivery: {
        kind: "auction",
        priceLabel: "CPC",
        averageUnitPrice: 1.9,
        priceVolatility: 0.16,
        inventoryDemandCoupling: 0.3,
      },
      experiment: { design: "none" },
      allocation: { maximumShareOfIncrement: 0.65 },
      response: {
        family: "geometric",
        decay: 0.08,
        hillShape: 1.15,
        halfSaturationQuantile: 0.68,
        kernelNormalization: "sum",
      },
    },
    {
      channel: "tv",
      spendColumn: "ctv_spend",
      roiEvidenceId: "dtc-ctv",
      averageWeeklySpend: 26_000,
      spendVolatility: 0.34,
      demandCoupling: 0.04,
      planningCoupling: 0.62,
      delivery: {
        kind: "flighted-grp",
        priceLabel: "CPP",
        averageUnitPrice: 650,
        priceVolatility: 0.13,
        flightStartProbability: 0.2,
        flightContinuationProbability: 0.72,
      },
      experiment: {
        design: "geo",
        spend: 180_000,
        standardErrorShare: 0.22,
        biasShare: 0,
      },
      allocation: { maximumShareOfIncrement: 0.75 },
      response: {
        family: "weibull",
        shape: 2,
        scale: 6.5,
        hillShape: 1.05,
        halfSaturationQuantile: 0.52,
        kernelNormalization: "sum",
      },
    },
  ];
}

function scenario(
  overrides: Partial<SyntheticScenarioConfig> &
    Pick<SyntheticScenarioConfig, "id" | "label" | "description">,
): SyntheticScenarioConfig {
  return {
    seed: 20260812,
    periods: 156,
    latentDemandPersistence: 0.76,
    planningPersistence: 0.68,
    commercialPersistence: 0.72,
    noiseShare: 0.045,
    eventShockProbability: 0.09,
    observeDemandProxy: false,
    grossMargin: 0.42,
    ltvRevenueMultiplier: 1.65,
    maximumIncrementShare: 0.35,
    decisionBudgetShares: [0, 0.05, 0.1, 0.2, 0.35],
    channels: baseChannels(),
    industryBenchmarks: {
      paid_social: 2.3,
      search: 1.8,
      tv: 2.5,
    },
    ...overrides,
  };
}

export const PILOT_SCENARIOS: SyntheticScenarioConfig[] = [
  scenario({
    id: "clean-identification",
    label: "Clean identification",
    description: "Low confounding, observable demand, and moderate noise.",
    seed: 20260813,
    observeDemandProxy: true,
    noiseShare: 0.025,
    channels: baseChannels().map((channel) => ({
      ...channel,
      demandCoupling: 0.06,
      planningCoupling: 0.1,
    })),
  }),
  scenario({
    id: "demand-confounded-search",
    label: "Demand-confounded search",
    description:
      "Search auction volume rises with hidden demand, which also raises baseline revenue.",
    seed: 20260814,
  }),
  scenario({
    id: "saturated-paid-social",
    label: "Saturated paid social",
    description:
      "Paid social frequency rises quickly and creates strong diminishing returns.",
    seed: 20260815,
    channels: baseChannels().map((channel) =>
      channel.channel === "paid_social"
        ? {
            ...channel,
            spendVolatility: 0.5,
            delivery: { ...channel.delivery, frequencyInflation: 0.55 },
            response: {
              family: "geometric" as const,
              decay: 0.22,
              hillShape: 2.35,
              halfSaturationQuantile: 0.38,
              kernelNormalization: "sum" as const,
            },
          }
        : channel,
    ),
  }),
  scenario({
    id: "delayed-tv",
    label: "Delayed TV",
    description:
      "Long, flighted TV carryover makes contemporaneous attribution misleading.",
    seed: 20260816,
    channels: baseChannels().map((channel) =>
      channel.channel === "tv"
        ? {
            ...channel,
            response: {
              family: "weibull" as const,
              shape: 2.3,
              scale: 10,
              hillShape: 1.05,
              halfSaturationQuantile: 0.55,
              kernelNormalization: "sum" as const,
            },
          }
        : channel,
    ),
  }),
  scenario({
    id: "correlated-media",
    label: "Correlated media",
    description:
      "Channels share an unobserved commercial planning process and become difficult to separate.",
    seed: 20260817,
    planningPersistence: 0.86,
    commercialPersistence: 0.88,
    channels: baseChannels().map((channel) => ({
      ...channel,
      planningCoupling: 0.92,
      spendVolatility: 0.12,
    })),
  }),
  scenario({
    id: "wrong-industry-benchmark",
    label: "Wrong industry benchmark",
    description:
      "External benchmarks intentionally disagree with randomized truth to test non-circular validation.",
    seed: 20260818,
    channels: baseChannels().map((channel) => ({
      ...channel,
      targetRoi:
        channel.channel === "paid_social"
          ? 1.15
          : channel.channel === "search"
            ? 3.5
            : 1.05,
    })),
  }),
];

export const DEFAULT_PILOT_SCENARIO = PILOT_SCENARIOS.find(
  (item) => item.id === "demand-confounded-search",
)!;

export function scenarioById(id: string): SyntheticScenarioConfig {
  const selected = PILOT_SCENARIOS.find((item) => item.id === id);
  if (!selected) throw new Error(`Unknown score V2 scenario: ${id}.`);
  return selected;
}
