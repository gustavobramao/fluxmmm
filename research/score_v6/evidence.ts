import type {
  IndustryBenchmark,
  IndustryPriorOverrides,
} from "../../lib/mmm/benchmarks";
import type { SyntheticBusiness } from "../score_v2/types";

function intervalAround(median: number): Pick<IndustryBenchmark, "low" | "high"> {
  return {
    low: Math.max(0.05, median / 2.6),
    high: median * 2.6,
  };
}

/** Model-visible evidence only. Hidden ROI truth never enters this registry. */
export function syntheticIndustryPriorOverrides(
  business: SyntheticBusiness,
): IndustryPriorOverrides {
  return Object.fromEntries(
    business.truth.channels.map((channel) => {
      const median = Math.max(
        0.05,
        business.scenario.industryBenchmarks[channel.channel],
      );
      const interval = intervalAround(median);
      return [
        channel.spendColumn,
        {
          id: `synthetic-${business.scenario.id}-${channel.channel}`,
          label: `Registered synthetic ${channel.channel.replaceAll("_", " ")} benchmark`,
          family: channel.channel,
          median,
          ...interval,
          confidence: "Low",
          evidence: "Scenario-visible external evidence contract",
        } satisfies IndustryBenchmark,
      ];
    }),
  );
}
