import type { Experiment } from "./types";

export type BenchmarkConfidence = "Medium" | "Low" | "Very low";

export interface IndustryBenchmark {
  id: string;
  label: string;
  family: string;
  median: number;
  low: number;
  high: number;
  confidence: BenchmarkConfidence;
  evidence: string;
}

export interface IndustryPriorRecommendation extends IndustryBenchmark {
  channel: string;
  matchQuality: "Exact tactic" | "Channel only" | "Generic fallback";
  standardDeviation: number;
}

export type IndustryPriorSelection = boolean | readonly string[];
export type IndustryPriorOverrides = Readonly<
  Record<string, IndustryBenchmark>
>;

export const INDUSTRY_BENCHMARK_VERSION = "dtc-us-owned-revenue-2026.2";

export const INDUSTRY_BENCHMARKS: IndustryBenchmark[] = [
  {
    id: "google-search-brand",
    label: "Google branded search",
    family: "Paid search",
    median: 1.5,
    low: 0.4,
    high: 5.1,
    confidence: "Low",
    evidence: "DTC geo tests + portfolio triangulation",
  },
  {
    id: "google-search-nonbrand",
    label: "Google non-brand search",
    family: "Paid search",
    median: 1.8,
    low: 0.7,
    high: 4.5,
    confidence: "Low",
    evidence: "DTC geo tests + blended search portfolio",
  },
  {
    id: "google-shopping",
    label: "Google Shopping / PLA",
    family: "Paid search",
    median: 2.1,
    low: 0.8,
    high: 5.4,
    confidence: "Low",
    evidence: "DTC geo tests + blended commerce portfolio",
  },
  {
    id: "google-pmax",
    label: "Google Performance Max",
    family: "Paid search",
    median: 2.6,
    low: 1,
    high: 6.6,
    confidence: "Low",
    evidence: "DTC geo tests + blended commerce portfolio",
  },
  {
    id: "meta-acquisition",
    label: "Meta acquisition / prospecting",
    family: "Paid social",
    median: 2.3,
    low: 1.1,
    high: 4.7,
    confidence: "Medium",
    evidence: "DTC geo tests + acquisition portfolio",
  },
  {
    id: "meta-retargeting",
    label: "Meta retargeting / retention",
    family: "Paid social",
    median: 1.8,
    low: 0.6,
    high: 5.3,
    confidence: "Low",
    evidence: "DTC portfolio + conservative incrementality adjustment",
  },
  {
    id: "meta-awareness",
    label: "Meta awareness / reach",
    family: "Paid social",
    median: 1.6,
    low: 0.5,
    high: 5.2,
    confidence: "Low",
    evidence: "All-Meta geo tests with wider objective uncertainty",
  },
  {
    id: "youtube",
    label: "YouTube / online video",
    family: "Online video",
    median: 2,
    low: 0.8,
    high: 4.8,
    confidence: "Medium",
    evidence: "DTC geo tests + channel portfolio",
  },
  {
    id: "tiktok",
    label: "TikTok prospecting",
    family: "Paid social",
    median: 1.1,
    low: 0.3,
    high: 4,
    confidence: "Low",
    evidence: "DTC benchmarks with high creative variance",
  },
  {
    id: "ctv",
    label: "CTV / streaming TV",
    family: "Television",
    median: 2.5,
    low: 0.8,
    high: 7.5,
    confidence: "Low",
    evidence: "DTC geo tests with high execution variance",
  },
  {
    id: "ooh-dtc-ecommerce",
    label: "OOH / digital OOH (D2C ecommerce)",
    family: "Out-of-home",
    median: 1.5,
    low: 0.5,
    high: 4.5,
    confidence: "Low",
    evidence: "Conservative DTC geo-test base rate + US OOH and retail MMM synthesis",
  },
  {
    id: "pinterest",
    label: "Pinterest",
    family: "Paid social",
    median: 2.1,
    low: 0.7,
    high: 6.5,
    confidence: "Low",
    evidence: "Small geo-test sample + channel portfolio",
  },
  {
    id: "snapchat",
    label: "Snapchat",
    family: "Paid social",
    median: 1.3,
    low: 0.4,
    high: 4.5,
    confidence: "Very low",
    evidence: "Small geo-test sample + channel portfolio",
  },
  {
    id: "applovin",
    label: "AppLovin",
    family: "Paid social",
    median: 1.5,
    low: 0.45,
    high: 5,
    confidence: "Very low",
    evidence: "Single DTC portfolio benchmark",
  },
  {
    id: "generic-fallback",
    label: "Broad paid-media fallback",
    family: "Other paid media",
    median: 1.22,
    low: 0.38,
    high: 3.87,
    confidence: "Very low",
    evidence: "Broad weakly informative ROI prior",
  },
];

const byId = new Map(
  INDUSTRY_BENCHMARKS.map((benchmark) => [benchmark.id, benchmark]),
);

function compactChannel(channel: string): string {
  return channel
    .toLowerCase()
    .replace(/_(s|spend|cost|investment)$/i, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function hasAny(value: string, tokens: string[]): boolean {
  return tokens.some((token) => value.includes(token));
}

function recommendation(
  channel: string,
  id: string,
  matchQuality: IndustryPriorRecommendation["matchQuality"],
): IndustryPriorRecommendation {
  const benchmark =
    byId.get(id) ?? byId.get("generic-fallback") ?? INDUSTRY_BENCHMARKS[0];
  return {
    ...benchmark,
    channel,
    matchQuality,
    standardDeviation:
      (benchmark.high - benchmark.low) / (2 * 1.281551565545),
  };
}

export function inferIndustryPrior(
  channel: string,
  overrides?: IndustryPriorOverrides,
): IndustryPriorRecommendation {
  const override = Object.entries(overrides ?? {}).find(
    ([candidate]) => candidate.toLowerCase() === channel.toLowerCase(),
  )?.[1];
  if (override) {
    return {
      ...override,
      channel,
      matchQuality: "Exact tactic",
      standardDeviation:
        (override.high - override.low) / (2 * 1.281551565545),
    };
  }
  const value = compactChannel(channel);
  const joined = value.replace(/\s+/g, "");
  const isGoogle = hasAny(value, ["google", "adwords"]);
  const isSearch = hasAny(value, ["search", "sem", "paid search"]);
  const isMeta = hasAny(value, ["meta", "facebook", "instagram", "fb "]);
  const isRetargeting = hasAny(value, [
    "retarget",
    "remarket",
    "retention",
    "dpa",
    "catalog",
  ]);
  const isAwareness = hasAny(value, [
    "awareness",
    "reach",
    "video view",
    "brand awareness",
  ]);
  const isNonBrand =
    joined.includes("nonbrand") || hasAny(value, ["generic", "unbranded"]);
  const isBrand =
    !isNonBrand &&
    (hasAny(value, ["branded", "brand search"]) ||
      value.split(" ").includes("brand"));

  if (hasAny(value, ["youtube", "yt video"])) {
    return recommendation(channel, "youtube", "Exact tactic");
  }
  if (hasAny(value, ["performance max", "pmax", "performance_max"])) {
    return recommendation(channel, "google-pmax", "Exact tactic");
  }
  if (hasAny(value, ["shopping", " pla ", "product listing"])) {
    return recommendation(channel, "google-shopping", "Exact tactic");
  }
  if ((isGoogle || isSearch) && isNonBrand) {
    return recommendation(channel, "google-search-nonbrand", "Exact tactic");
  }
  if ((isGoogle || isSearch) && isBrand) {
    return recommendation(channel, "google-search-brand", "Exact tactic");
  }
  if (isSearch) {
    return recommendation(channel, "google-search-nonbrand", "Channel only");
  }
  if (isMeta && isRetargeting) {
    return recommendation(channel, "meta-retargeting", "Exact tactic");
  }
  if (isMeta && isAwareness) {
    return recommendation(channel, "meta-awareness", "Exact tactic");
  }
  if (isMeta) {
    return recommendation(channel, "meta-acquisition", "Channel only");
  }
  if (value.includes("tiktok")) {
    return recommendation(channel, "tiktok", "Channel only");
  }
  if (value.includes("pinterest")) {
    return recommendation(channel, "pinterest", "Channel only");
  }
  if (hasAny(value, ["snapchat", "snap ads"])) {
    return recommendation(channel, "snapchat", "Channel only");
  }
  if (value.includes("applovin")) {
    return recommendation(channel, "applovin", "Channel only");
  }
  if (
    hasAny(value, [
      "ctv",
      "connected tv",
      "streaming tv",
      "tatari",
      "linear tv",
      "television",
    ]) ||
    value === "tv"
  ) {
    return recommendation(channel, "ctv", "Channel only");
  }
  if (
    hasAny(value, [
      "ooh",
      "out of home",
      "outdoor",
      "billboard",
      "dooh",
      "digital out of home",
      "street furniture",
      "transit advertising",
    ])
  ) {
    return recommendation(channel, "ooh-dtc-ecommerce", "Channel only");
  }
  if (
    hasAny(value, [
      "display",
      "programmatic",
      "affiliate",
      "creator",
      "influencer",
      "amazon",
      "retail media",
    ])
  ) {
    return recommendation(channel, "generic-fallback", "Generic fallback");
  }
  return recommendation(channel, "generic-fallback", "Generic fallback");
}

export function channelExperiments(
  experiments: Experiment[],
  channel: string,
): Experiment[] {
  return experiments.filter(
    (experiment) =>
      experiment.channel.toLowerCase() === channel.toLowerCase(),
  );
}

export function activeIndustryPrior(
  channel: string,
  experiments: Experiment[],
  selection: IndustryPriorSelection,
  overrides?: IndustryPriorOverrides,
): IndustryPriorRecommendation | undefined {
  const selected =
    selection === true ||
    (Array.isArray(selection) &&
      selection.some(
        (selectedChannel) =>
          selectedChannel.toLowerCase() === channel.toLowerCase(),
      ));
  if (!selected || channelExperiments(experiments, channel).length) {
    return undefined;
  }
  return inferIndustryPrior(channel, overrides);
}

function normalCdf(value: number): number {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * x);
  const erf =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
      0.284496736) *
      t +
      0.254829592) *
      t) *
      Math.exp(-(x ** 2));
  return 0.5 * (1 + sign * erf);
}

export function industryPriorPercentile(
  roi: number,
  benchmark: IndustryBenchmark,
): number {
  if (!Number.isFinite(roi) || roi <= 0) return 0;
  const z90 = 1.281551565545;
  const logMedian = Math.log(Math.max(benchmark.median, 1e-6));
  const lowerSigma =
    (logMedian - Math.log(Math.max(benchmark.low, 1e-6))) / z90;
  const upperSigma =
    (Math.log(Math.max(benchmark.high, 1e-6)) - logMedian) / z90;
  const logSigma = Math.max((lowerSigma + upperSigma) / 2, 0.1);
  return Math.min(
    1,
    Math.max(0, normalCdf((Math.log(roi) - logMedian) / logSigma)),
  );
}

export function isHighlyImprobableIndustryRoi(
  roi: number,
  benchmark: IndustryBenchmark,
  threshold = 0.95,
): boolean {
  const percentile = industryPriorPercentile(roi, benchmark);
  return percentile <= 1 - threshold || percentile >= threshold;
}

export function industryRoiTail(
  roi: number,
  benchmark: IndustryBenchmark,
  threshold = 0.95,
): "low" | "high" | undefined {
  const percentile = industryPriorPercentile(roi, benchmark);
  if (percentile <= 1 - threshold) return "low";
  if (percentile >= threshold) return "high";
  return undefined;
}
