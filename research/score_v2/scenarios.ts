import type {
  SyntheticChannelConfig,
  SyntheticScenarioConfig,
} from "./types";

function baseChannels(): SyntheticChannelConfig[] {
  return [
    {
      channel: "paid_social",
      spendColumn: "paid_social_spend",
      targetRoi: 2.5,
      averageWeeklySpend: 22_000,
      spendVolatility: 0.28,
      demandCoupling: 0.2,
      planningCoupling: 0.35,
      response: {
        family: "geometric",
        decay: 0.25,
        hillShape: 1.4,
        halfSaturationQuantile: 0.6,
      },
    },
    {
      channel: "search",
      spendColumn: "search_spend",
      targetRoi: 1.2,
      averageWeeklySpend: 30_000,
      spendVolatility: 0.2,
      demandCoupling: 0.85,
      planningCoupling: 0.2,
      response: {
        family: "geometric",
        decay: 0.1,
        hillShape: 1.05,
        halfSaturationQuantile: 0.65,
      },
    },
    {
      channel: "tv",
      spendColumn: "tv_spend",
      targetRoi: 1.8,
      averageWeeklySpend: 26_000,
      spendVolatility: 0.35,
      demandCoupling: 0.05,
      planningCoupling: 0.55,
      flightProbability: 0.42,
      response: {
        family: "geometric",
        decay: 0.7,
        hillShape: 1.1,
        halfSaturationQuantile: 0.55,
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
    latentDemandPersistence: 0.75,
    planningPersistence: 0.6,
    noiseShare: 0.035,
    observeDemandProxy: false,
    channels: baseChannels(),
    industryBenchmarks: {
      paid_social: 2.4,
      search: 1.5,
      tv: 1.7,
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
    noiseShare: 0.02,
    channels: baseChannels().map((channel) => ({
      ...channel,
      demandCoupling: 0.08,
      planningCoupling: 0.1,
    })),
  }),
  scenario({
    id: "demand-confounded-search",
    label: "Demand-confounded search",
    description:
      "Search spend rises with hidden consumer demand, which also raises baseline revenue.",
    seed: 20260814,
  }),
  scenario({
    id: "saturated-paid-social",
    label: "Saturated paid social",
    description:
      "Paid social has strong diminishing returns inside the observed spend range.",
    seed: 20260815,
    channels: baseChannels().map((channel) =>
      channel.channel === "paid_social"
        ? {
            ...channel,
            spendVolatility: 0.5,
            response: {
              family: "geometric" as const,
              decay: 0.25,
              hillShape: 2.2,
              halfSaturationQuantile: 0.38,
            },
          }
        : channel,
    ),
  }),
  scenario({
    id: "delayed-tv",
    label: "Delayed TV",
    description: "Long TV carryover makes contemporaneous attribution misleading.",
    seed: 20260816,
    channels: baseChannels().map((channel) =>
      channel.channel === "tv"
        ? {
            ...channel,
            response: {
              family: "weibull" as const,
              shape: 2.2,
              scale: 7,
              hillShape: 1.1,
              halfSaturationQuantile: 0.55,
            },
          }
        : channel,
    ),
  }),
  scenario({
    id: "correlated-media",
    label: "Correlated media",
    description:
      "Channels share a strong latent planning process and become difficult to separate.",
    seed: 20260817,
    planningPersistence: 0.82,
    channels: baseChannels().map((channel) => ({
      ...channel,
      planningCoupling: 0.9,
      spendVolatility: 0.12,
    })),
  }),
  scenario({
    id: "wrong-industry-benchmark",
    label: "Wrong industry benchmark",
    description:
      "External benchmarks intentionally disagree with causal truth to test non-circular validation.",
    seed: 20260818,
    industryBenchmarks: {
      paid_social: 1.1,
      search: 3.8,
      tv: 0.65,
    },
  }),
];

export const DEFAULT_PILOT_SCENARIO = PILOT_SCENARIOS.find(
  (item) => item.id === "demand-confounded-search",
)!;

export function scenarioById(id: string): SyntheticScenarioConfig {
  const selected = PILOT_SCENARIOS.find((item) => item.id === id);
  if (!selected) {
    throw new Error(`Unknown score V2 scenario: ${id}.`);
  }
  return selected;
}
