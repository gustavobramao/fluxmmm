import type {
  AgenticCandidateRun,
  AgenticCandidateSpec,
  AgenticSearchContract,
} from "./agentic";
import { passesAgenticEligibility } from "./agentic";
import type {
  AdvancedModelConfig,
  ModelConfig,
} from "./types";
import type { ValidationModelKind } from "./validation";

export const WEEKLY_CYCLE_PERIODS = [13, 26, 52] as const;
export const MONTHLY_CYCLE_PERIODS = [3, 6, 12] as const;

export const AGENTIC_PARAMETER_BOUNDS = {
  adstock: { min: 0.05, max: 0.9, step: 0.05 },
  weibullShape: { min: 0.6, max: 6, step: 0.1 },
  weibullScale: { min: 1, max: 12, step: 0.5 },
  saturation: { min: 0.5, max: 3, step: 0.1 },
  ridge: { min: 0.01, max: 2, step: 0.01 },
  fourierOrder: [1, 2, 3] as const,
  cyclePeriod: [...MONTHLY_CYCLE_PERIODS, ...WEEKLY_CYCLE_PERIODS] as const,
  kernelKnots: { min: 3, max: 10, step: 1 },
  kernelBandwidth: { min: 0.08, max: 0.35, step: 0.01 },
  studentTDegreesFreedom: { min: 3, max: 30, step: 1 },
  halfSaturationQuantile: { min: 0.2, max: 0.8, step: 0.05 },
} as const;

export interface AgenticSearchCapabilities {
  likelihoodCalibration: boolean;
  mediaColumns?: string[];
}

export interface AgenticChannelResponseCoverage {
  requiredChallenges: number;
  completedChallenges: number;
  complete: boolean;
  channels: {
    channel: string;
    adstockFamilies: string[];
    saturationProfiles: string[];
    halfSaturationProfiles: string[];
    normalizations: string[];
    complete: boolean;
  }[];
}

export interface AgenticStopDecision {
  shouldStop: boolean;
  reason: string;
  evaluationsSinceImprovement: number;
}

export interface AgenticAdvancedCoverageDimension {
  id: string;
  label: string;
  tested: string[];
  required: string[];
  complete: boolean;
}

export interface AgenticAdvancedCoverage {
  requiredChallenges: number;
  completedChallenges: number;
  complete: boolean;
  dimensions: AgenticAdvancedCoverageDimension[];
}

export interface AgenticSearchConfidence {
  level: "high" | "medium" | "low" | "not-established";
  score: number;
  structuralCoverage: number;
  restartAgreement: number;
  restartCount: number;
  localChallengeCount: number;
  localImprovements: number;
  evaluationsSinceImprovement: number;
  boundaryParameters: string[];
  summary: string;
  restartWinners: {
    restart: 1 | 2 | 3;
    candidateId?: string;
    score?: number;
    family?: ValidationModelKind;
  }[];
  frontier: {
    evaluation: number;
    score: number | null;
    phase: AgenticCandidateSpec["searchPhase"];
    restart: 1 | 2 | 3;
  }[];
}

interface EncodedSpecification {
  numeric: Record<string, number>;
  categorical: Record<string, string>;
}

interface ProposalScore {
  spec: AgenticCandidateSpec;
  acquisition: number;
  expectedScore: number;
  eligibilityProbability: number;
  exploration: number;
}

const HALTON_BASES = [
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
  59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
  127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
  191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
  257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
];

function activeFamilies(
  contract: AgenticSearchContract,
): ValidationModelKind[] {
  return (["frequentist", "bayesian", "advanced"] as const).filter(
    (family) => contract.families[family],
  );
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function roundToStep(
  value: number,
  minimum: number,
  maximum: number,
  step: number,
): number {
  const rounded = minimum + Math.round((value - minimum) / step) * step;
  const decimals = step < 0.1 ? 2 : step < 1 ? 1 : 0;
  return Number(clamp(rounded, minimum, maximum).toFixed(decimals));
}

function nearestChoice<T extends number>(
  value: number,
  choices: readonly T[],
): T {
  return choices.reduce((nearest, choice) =>
    Math.abs(choice - value) < Math.abs(nearest - value)
      ? choice
      : nearest,
  );
}

function cyclePeriodChoices(
  config: Pick<ModelConfig, "cyclePeriod">,
): readonly number[] {
  return config.cyclePeriod <= 12
    ? MONTHLY_CYCLE_PERIODS
    : WEEKLY_CYCLE_PERIODS;
}

function halton(index: number, base: number): number {
  let result = 0;
  let fraction = 1;
  let cursor = Math.max(1, index);
  while (cursor > 0) {
    fraction /= base;
    result += fraction * (cursor % base);
    cursor = Math.floor(cursor / base);
  }
  return result;
}

function primeForDimension(dimension: number): number {
  if (dimension < HALTON_BASES.length) return HALTON_BASES[dimension];
  let candidate = HALTON_BASES[HALTON_BASES.length - 1] + 2;
  let found = HALTON_BASES.length;
  while (true) {
    let isPrime = true;
    for (let divisor = 3; divisor * divisor <= candidate; divisor += 2) {
      if (candidate % divisor === 0) {
        isPrime = false;
        break;
      }
    }
    if (isPrime) {
      if (found === dimension) return candidate;
      found += 1;
    }
    candidate += 2;
  }
}

function haltonPoint(index: number, dimensions = HALTON_BASES.length): number[] {
  return Array.from({ length: dimensions }, (_, dimension) =>
    halton(index, primeForDimension(dimension)),
  );
}

function searchDimensions(capabilities: AgenticSearchCapabilities): number {
  return Math.max(16, 16 + (capabilities.mediaColumns?.length ?? 0) * 7);
}

function interpolate(
  unit: number,
  minimum: number,
  maximum: number,
): number {
  return minimum + clamp(unit, 0, 1) * (maximum - minimum);
}

function interpolateLog(
  unit: number,
  minimum: number,
  maximum: number,
): number {
  const lower = Math.log(minimum);
  const upper = Math.log(maximum);
  return Math.exp(lower + clamp(unit, 0, 1) * (upper - lower));
}

function categoricalChoice<T>(
  unit: number,
  values: readonly T[],
): T {
  return values[
    Math.min(values.length - 1, Math.floor(clamp(unit, 0, 0.999999) * values.length))
  ];
}

function boundModelConfig(config: ModelConfig): ModelConfig {
  const cyclePeriods = cyclePeriodChoices(config);
  const channelResponses = config.channelResponses
    ? Object.fromEntries(
        Object.entries(config.channelResponses).map(([channel, response]) => [
          channel,
          {
            ...response,
            adstock: roundToStep(
              response.adstock ?? config.adstock,
              AGENTIC_PARAMETER_BOUNDS.adstock.min,
              AGENTIC_PARAMETER_BOUNDS.adstock.max,
              AGENTIC_PARAMETER_BOUNDS.adstock.step,
            ),
            weibullShape: roundToStep(
              response.weibullShape ?? config.weibullShape,
              AGENTIC_PARAMETER_BOUNDS.weibullShape.min,
              AGENTIC_PARAMETER_BOUNDS.weibullShape.max,
              AGENTIC_PARAMETER_BOUNDS.weibullShape.step,
            ),
            weibullScale: roundToStep(
              response.weibullScale ?? config.weibullScale,
              AGENTIC_PARAMETER_BOUNDS.weibullScale.min,
              AGENTIC_PARAMETER_BOUNDS.weibullScale.max,
              AGENTIC_PARAMETER_BOUNDS.weibullScale.step,
            ),
            saturation: roundToStep(
              response.saturation ?? config.saturation,
              AGENTIC_PARAMETER_BOUNDS.saturation.min,
              AGENTIC_PARAMETER_BOUNDS.saturation.max,
              AGENTIC_PARAMETER_BOUNDS.saturation.step,
            ),
            halfSaturationQuantile: roundToStep(
              response.halfSaturationQuantile ?? 0.5,
              AGENTIC_PARAMETER_BOUNDS.halfSaturationQuantile.min,
              AGENTIC_PARAMETER_BOUNDS.halfSaturationQuantile.max,
              AGENTIC_PARAMETER_BOUNDS.halfSaturationQuantile.step,
            ),
            kernelNormalization: response.kernelNormalization ?? "peak",
          },
        ]),
      )
    : undefined;
  return {
    ...config,
    adstock: roundToStep(
      config.adstock,
      AGENTIC_PARAMETER_BOUNDS.adstock.min,
      AGENTIC_PARAMETER_BOUNDS.adstock.max,
      AGENTIC_PARAMETER_BOUNDS.adstock.step,
    ),
    weibullShape: roundToStep(
      config.weibullShape,
      AGENTIC_PARAMETER_BOUNDS.weibullShape.min,
      AGENTIC_PARAMETER_BOUNDS.weibullShape.max,
      AGENTIC_PARAMETER_BOUNDS.weibullShape.step,
    ),
    weibullScale: roundToStep(
      config.weibullScale,
      AGENTIC_PARAMETER_BOUNDS.weibullScale.min,
      AGENTIC_PARAMETER_BOUNDS.weibullScale.max,
      AGENTIC_PARAMETER_BOUNDS.weibullScale.step,
    ),
    saturation: roundToStep(
      config.saturation,
      AGENTIC_PARAMETER_BOUNDS.saturation.min,
      AGENTIC_PARAMETER_BOUNDS.saturation.max,
      AGENTIC_PARAMETER_BOUNDS.saturation.step,
    ),
    ridge: roundToStep(
      config.ridge,
      AGENTIC_PARAMETER_BOUNDS.ridge.min,
      AGENTIC_PARAMETER_BOUNDS.ridge.max,
      AGENTIC_PARAMETER_BOUNDS.ridge.step,
    ),
    fourierOrder: nearestChoice(
      config.fourierOrder,
      AGENTIC_PARAMETER_BOUNDS.fourierOrder,
    ),
    cyclePeriod: nearestChoice(
      config.cyclePeriod,
      cyclePeriods,
    ),
    channelResponses,
  };
}

function channelResponsesFromPoint(
  baseConfig: ModelConfig,
  point: number[],
  mediaColumns: string[],
  offset = 16,
): ModelConfig["channelResponses"] {
  if (!mediaColumns.length) return baseConfig.channelResponses;
  return Object.fromEntries(
    mediaColumns.map((channel, channelIndex) => {
      const start = offset + channelIndex * 7;
      return [
        channel,
        {
          adstockType: point[start] < 0.5 ? "geometric" : "weibull",
          adstock: interpolate(
            point[start + 1],
            AGENTIC_PARAMETER_BOUNDS.adstock.min,
            AGENTIC_PARAMETER_BOUNDS.adstock.max,
          ),
          weibullShape: interpolate(
            point[start + 2],
            AGENTIC_PARAMETER_BOUNDS.weibullShape.min,
            AGENTIC_PARAMETER_BOUNDS.weibullShape.max,
          ),
          weibullScale: interpolate(
            point[start + 3],
            AGENTIC_PARAMETER_BOUNDS.weibullScale.min,
            AGENTIC_PARAMETER_BOUNDS.weibullScale.max,
          ),
          saturation: interpolate(
            point[start + 4],
            AGENTIC_PARAMETER_BOUNDS.saturation.min,
            AGENTIC_PARAMETER_BOUNDS.saturation.max,
          ),
          halfSaturationQuantile: interpolate(
            point[start + 5],
            AGENTIC_PARAMETER_BOUNDS.halfSaturationQuantile.min,
            AGENTIC_PARAMETER_BOUNDS.halfSaturationQuantile.max,
          ),
          kernelNormalization: point[start + 6] < 0.5 ? "peak" : "sum",
        },
      ];
    }),
  );
}

function localChannelResponses(
  baseConfig: ModelConfig,
  point: number[],
  mediaColumns: string[],
  offset = 16,
): ModelConfig["channelResponses"] {
  if (!mediaColumns.length) return baseConfig.channelResponses;
  return Object.fromEntries(
    mediaColumns.map((channel, channelIndex) => {
      const start = offset + channelIndex * 7;
      const source = baseConfig.channelResponses?.[channel] ?? {};
      const adstockType = source.adstockType ?? baseConfig.adstockType;
      const kernelNormalization = source.kernelNormalization ?? "peak";
      return [
        channel,
        {
          ...source,
          adstockType:
            point[start] > 0.94
              ? adstockType === "geometric"
                ? "weibull"
                : "geometric"
              : adstockType,
          adstock: shifted(
            source.adstock ?? baseConfig.adstock,
            point[start + 1],
            AGENTIC_PARAMETER_BOUNDS.adstock.min,
            AGENTIC_PARAMETER_BOUNDS.adstock.max,
            0.3,
          ),
          weibullShape: shifted(
            source.weibullShape ?? baseConfig.weibullShape,
            point[start + 2],
            AGENTIC_PARAMETER_BOUNDS.weibullShape.min,
            AGENTIC_PARAMETER_BOUNDS.weibullShape.max,
            0.3,
          ),
          weibullScale: shifted(
            source.weibullScale ?? baseConfig.weibullScale,
            point[start + 3],
            AGENTIC_PARAMETER_BOUNDS.weibullScale.min,
            AGENTIC_PARAMETER_BOUNDS.weibullScale.max,
            0.3,
          ),
          saturation: shifted(
            source.saturation ?? baseConfig.saturation,
            point[start + 4],
            AGENTIC_PARAMETER_BOUNDS.saturation.min,
            AGENTIC_PARAMETER_BOUNDS.saturation.max,
            0.3,
          ),
          halfSaturationQuantile: shifted(
            source.halfSaturationQuantile ?? 0.5,
            point[start + 5],
            AGENTIC_PARAMETER_BOUNDS.halfSaturationQuantile.min,
            AGENTIC_PARAMETER_BOUNDS.halfSaturationQuantile.max,
            0.3,
          ),
          kernelNormalization:
            point[start + 6] > 0.94
              ? kernelNormalization === "peak"
                ? "sum"
                : "peak"
              : kernelNormalization,
        },
      ];
    }),
  );
}

function boundAdvancedConfig(
  config: AdvancedModelConfig,
  capabilities: AgenticSearchCapabilities,
): AdvancedModelConfig {
  return {
    ...config,
    kernelKnots: roundToStep(
      config.kernelKnots,
      AGENTIC_PARAMETER_BOUNDS.kernelKnots.min,
      AGENTIC_PARAMETER_BOUNDS.kernelKnots.max,
      AGENTIC_PARAMETER_BOUNDS.kernelKnots.step,
    ),
    kernelBandwidth: roundToStep(
      config.kernelBandwidth,
      AGENTIC_PARAMETER_BOUNDS.kernelBandwidth.min,
      AGENTIC_PARAMETER_BOUNDS.kernelBandwidth.max,
      AGENTIC_PARAMETER_BOUNDS.kernelBandwidth.step,
    ),
    calibrationMode: capabilities.likelihoodCalibration
      ? config.calibrationMode
      : "prior",
    studentTDegreesFreedom: roundToStep(
      config.studentTDegreesFreedom,
      AGENTIC_PARAMETER_BOUNDS.studentTDegreesFreedom.min,
      AGENTIC_PARAMETER_BOUNDS.studentTDegreesFreedom.max,
      AGENTIC_PARAMETER_BOUNDS.studentTDegreesFreedom.step,
    ),
  };
}

function configurationFromPoint(
  family: ValidationModelKind,
  point: number[],
  baseConfig: ModelConfig,
  baseAdvancedConfig: AdvancedModelConfig,
  capabilities: AgenticSearchCapabilities,
): {
  config: ModelConfig;
  advancedConfig: AdvancedModelConfig;
} {
  const adstockType = point[0] < 0.5 ? "geometric" : "weibull";
  const cyclePeriods = cyclePeriodChoices(baseConfig);
  const config = boundModelConfig({
    ...baseConfig,
    adstockType,
    adstock: interpolate(
      point[1],
      AGENTIC_PARAMETER_BOUNDS.adstock.min,
      AGENTIC_PARAMETER_BOUNDS.adstock.max,
    ),
    weibullShape: interpolate(
      point[2],
      AGENTIC_PARAMETER_BOUNDS.weibullShape.min,
      AGENTIC_PARAMETER_BOUNDS.weibullShape.max,
    ),
    weibullScale: interpolate(
      point[3],
      AGENTIC_PARAMETER_BOUNDS.weibullScale.min,
      AGENTIC_PARAMETER_BOUNDS.weibullScale.max,
    ),
    saturation: interpolate(
      point[4],
      AGENTIC_PARAMETER_BOUNDS.saturation.min,
      AGENTIC_PARAMETER_BOUNDS.saturation.max,
    ),
    ridge: interpolateLog(
      point[5],
      AGENTIC_PARAMETER_BOUNDS.ridge.min,
      AGENTIC_PARAMETER_BOUNDS.ridge.max,
    ),
    fourierOrder: categoricalChoice(
      point[6],
      AGENTIC_PARAMETER_BOUNDS.fourierOrder,
    ),
    cyclePeriod: categoricalChoice(
      point[7],
      cyclePeriods,
    ),
    channelResponses: channelResponsesFromPoint(
      baseConfig,
      point,
      capabilities.mediaColumns ?? [],
    ),
  });
  const advancedConfig = boundAdvancedConfig(
    family === "advanced"
      ? {
          ...baseAdvancedConfig,
          timeVarying: point[8] >= 0.45,
          kernelKnots: interpolate(
            point[9],
            AGENTIC_PARAMETER_BOUNDS.kernelKnots.min,
            AGENTIC_PARAMETER_BOUNDS.kernelKnots.max,
          ),
          kernelBandwidth: interpolate(
            point[10],
            AGENTIC_PARAMETER_BOUNDS.kernelBandwidth.min,
            AGENTIC_PARAMETER_BOUNDS.kernelBandwidth.max,
          ),
          planningIntensity: point[11] >= 0.7,
          calibrationMode:
            capabilities.likelihoodCalibration && point[12] >= 0.6
              ? "likelihood"
              : "prior",
          priorDistribution:
            point[13] < 0.5 ? "half-normal" : "log-normal",
          likelihoodDistribution: categoricalChoice(
            point[14],
            ["gaussian", "student-t", "log-normal"] as const,
          ),
          studentTDegreesFreedom: interpolate(
            point[15],
            AGENTIC_PARAMETER_BOUNDS.studentTDegreesFreedom.min,
            AGENTIC_PARAMETER_BOUNDS.studentTDegreesFreedom.max,
          ),
        }
      : baseAdvancedConfig,
    capabilities,
  );
  return { config, advancedConfig };
}

function shifted(
  current: number,
  unit: number,
  minimum: number,
  maximum: number,
  localShare: number,
): number {
  return current + (unit - 0.5) * (maximum - minimum) * localShare;
}

function localConfiguration(
  anchor: AgenticCandidateSpec,
  point: number[],
  capabilities: AgenticSearchCapabilities,
): {
  config: ModelConfig;
  advancedConfig: AdvancedModelConfig;
} {
  const source = anchor.config;
  const sourceAdvanced = anchor.advancedConfig;
  const cyclePeriods = cyclePeriodChoices(source);
  const adstockType =
    point[0] > 0.9
      ? source.adstockType === "geometric"
        ? "weibull"
        : "geometric"
      : source.adstockType;
  const cycleIndex = cyclePeriods.indexOf(
    nearestChoice(
      source.cyclePeriod,
      cyclePeriods,
    ),
  );
  const fourierDelta = point[6] < 0.18 ? -1 : point[6] > 0.82 ? 1 : 0;
  const cycleDelta = point[7] < 0.14 ? -1 : point[7] > 0.86 ? 1 : 0;
  const logRidge = Math.log(source.ridge);
  const ridgeRange =
    Math.log(AGENTIC_PARAMETER_BOUNDS.ridge.max) -
    Math.log(AGENTIC_PARAMETER_BOUNDS.ridge.min);
  const config = boundModelConfig({
    ...source,
    adstockType,
    adstock: shifted(
      source.adstock,
      point[1],
      AGENTIC_PARAMETER_BOUNDS.adstock.min,
      AGENTIC_PARAMETER_BOUNDS.adstock.max,
      0.35,
    ),
    weibullShape: shifted(
      source.weibullShape,
      point[2],
      AGENTIC_PARAMETER_BOUNDS.weibullShape.min,
      AGENTIC_PARAMETER_BOUNDS.weibullShape.max,
      0.35,
    ),
    weibullScale: shifted(
      source.weibullScale,
      point[3],
      AGENTIC_PARAMETER_BOUNDS.weibullScale.min,
      AGENTIC_PARAMETER_BOUNDS.weibullScale.max,
      0.35,
    ),
    saturation: shifted(
      source.saturation,
      point[4],
      AGENTIC_PARAMETER_BOUNDS.saturation.min,
      AGENTIC_PARAMETER_BOUNDS.saturation.max,
      0.35,
    ),
    ridge: Math.exp(logRidge + (point[5] - 0.5) * ridgeRange * 0.45),
    fourierOrder: source.fourierOrder + fourierDelta,
    cyclePeriod:
      cyclePeriods[
        clamp(
          cycleIndex + cycleDelta,
          0,
          cyclePeriods.length - 1,
        )
      ],
    channelResponses: localChannelResponses(
      source,
      point,
      capabilities.mediaColumns ?? Object.keys(source.channelResponses ?? {}),
    ),
  });
  const advancedConfig = boundAdvancedConfig(
    anchor.family === "advanced"
      ? {
          ...sourceAdvanced,
          timeVarying:
            point[8] > 0.92
              ? !sourceAdvanced.timeVarying
              : sourceAdvanced.timeVarying,
          kernelKnots: shifted(
            sourceAdvanced.kernelKnots,
            point[9],
            AGENTIC_PARAMETER_BOUNDS.kernelKnots.min,
            AGENTIC_PARAMETER_BOUNDS.kernelKnots.max,
            0.45,
          ),
          kernelBandwidth: shifted(
            sourceAdvanced.kernelBandwidth,
            point[10],
            AGENTIC_PARAMETER_BOUNDS.kernelBandwidth.min,
            AGENTIC_PARAMETER_BOUNDS.kernelBandwidth.max,
            0.4,
          ),
          planningIntensity:
            point[11] > 0.92
              ? !sourceAdvanced.planningIntensity
              : sourceAdvanced.planningIntensity,
          calibrationMode:
            capabilities.likelihoodCalibration && point[12] > 0.9
              ? sourceAdvanced.calibrationMode === "prior"
                ? "likelihood"
                : "prior"
              : sourceAdvanced.calibrationMode,
          priorDistribution:
            point[13] > 0.92
              ? sourceAdvanced.priorDistribution === "half-normal"
                ? "log-normal"
                : "half-normal"
              : sourceAdvanced.priorDistribution,
          likelihoodDistribution:
            point[14] > 0.88
              ? categoricalChoice(
                  point[1],
                  ["gaussian", "student-t", "log-normal"] as const,
                )
              : sourceAdvanced.likelihoodDistribution,
          studentTDegreesFreedom: shifted(
            sourceAdvanced.studentTDegreesFreedom,
            point[15],
            AGENTIC_PARAMETER_BOUNDS.studentTDegreesFreedom.min,
            AGENTIC_PARAMETER_BOUNDS.studentTDegreesFreedom.max,
            0.35,
          ),
        }
      : sourceAdvanced,
    capabilities,
  );
  return { config, advancedConfig };
}

function familyLabel(family: ValidationModelKind): string {
  return family === "frequentist"
    ? "Frequentist"
    : family === "bayesian"
      ? "Bayesian"
      : "Advanced";
}

function summaryFor(
  family: ValidationModelKind,
  config: ModelConfig,
  advancedConfig: AdvancedModelConfig,
): string {
  const carryover =
    config.adstockType === "geometric"
      ? `Geometric θ ${config.adstock.toFixed(2)}`
      : `Weibull ${config.weibullShape.toFixed(1)}/${config.weibullScale.toFixed(1)}`;
  const cycleUnit = config.cyclePeriod <= 12 ? "m" : "w";
  const base = `${carryover} · Hill ${config.saturation.toFixed(1)} · ridge ${config.ridge.toFixed(2)} · F${config.fourierOrder}/${config.cyclePeriod}${cycleUnit}`;
  const responseProfiles = Object.keys(config.channelResponses ?? {}).length;
  const responseLabel = responseProfiles
    ? ` · ${responseProfiles} channel-specific responses`
    : " · global media response";
  if (family !== "advanced") return `${base}${responseLabel}`;
  return `${base}${responseLabel} · ${advancedConfig.timeVarying ? "dynamic" : "static"} · ${advancedConfig.likelihoodDistribution}`;
}

function makeSpec(
  id: string,
  family: ValidationModelKind,
  config: ModelConfig,
  advancedConfig: AdvancedModelConfig,
  searchPhase: AgenticCandidateSpec["searchPhase"],
  proposal: AgenticCandidateSpec["proposal"],
  restart: 1 | 2 | 3 = 1,
): AgenticCandidateSpec {
  const methodLabel =
    searchPhase === "seed"
      ? "Seed"
      : searchPhase === "advanced-challenge"
        ? "Paired challenge"
        : searchPhase === "response-coverage"
          ? "Response grid"
        : searchPhase === "local-challenge"
          ? "Local challenge"
          : "Adaptive";
  const carryover =
    config.adstockType === "geometric" ? "geometric" : "Weibull";
  return {
    id,
    family,
    label: `${familyLabel(family)} · ${methodLabel} ${carryover}`,
    summary: summaryFor(family, config, advancedConfig),
    hypothesis: proposal.reason,
    config,
    advancedConfig,
    searchPhase,
    proposal,
    restart,
  };
}

export function agenticCandidateSignature(
  candidate: Pick<
    AgenticCandidateSpec,
    "family" | "config" | "advancedConfig"
  >,
): string {
  return JSON.stringify({
    family: candidate.family,
    config: candidate.config,
    advanced:
      candidate.family === "advanced"
        ? candidate.advancedConfig
        : undefined,
  });
}

export function agenticSeedBudget(
  contract: AgenticSearchContract,
): number {
  return Math.min(24, contract.candidateBudget);
}

export function agenticChannelResponseBudget(
  contract: AgenticSearchContract,
  capabilities: AgenticSearchCapabilities,
): number {
  if (!(capabilities.mediaColumns?.length ?? 0)) return 0;
  return contract.candidateBudget >= 384
    ? 72
    : contract.candidateBudget >= 192
      ? 48
      : 24;
}

export function agenticAdvancedChallengeBudget(
  contract: AgenticSearchContract,
): number {
  if (!contract.families.advanced) return 0;
  return Math.min(
    12,
    Math.max(0, contract.candidateBudget - agenticSeedBudget(contract)),
  );
}

export function agenticLocalChallengeBudget(
  contract: AgenticSearchContract,
): number {
  return contract.candidateBudget >= 384
    ? 36
    : contract.candidateBudget >= 192
      ? 24
      : 12;
}

export function agenticAdaptiveMaximumBudget(
  contract: AgenticSearchContract,
  capabilities: AgenticSearchCapabilities = { likelihoodCalibration: true },
): number {
  return Math.max(
    0,
    contract.candidateBudget -
      agenticSeedBudget(contract) -
      agenticChannelResponseBudget(contract, capabilities) -
      agenticAdvancedChallengeBudget(contract) -
      agenticLocalChallengeBudget(contract),
  );
}

export function agenticAdaptiveMinimumBudget(
  contract: AgenticSearchContract,
  capabilities: AgenticSearchCapabilities = { likelihoodCalibration: true },
): number {
  const families = activeFamilies(contract);
  if (!families.length) return 0;
  const available = Math.max(
    0,
    agenticAdaptiveMaximumBudget(contract, capabilities),
  );
  const perFamilyRestart = contract.candidateBudget >= 384 ? 4 : contract.candidateBudget >= 192 ? 3 : 2;
  return Math.min(available, families.length * 3 * perFamilyRestart);
}

export function generateAgenticSeeds(
  baseConfig: ModelConfig,
  baseAdvancedConfig: AdvancedModelConfig,
  contract: AgenticSearchContract,
  capabilities: AgenticSearchCapabilities,
): AgenticCandidateSpec[] {
  const families = activeFamilies(contract);
  if (!families.length) return [];
  const target = agenticSeedBudget(contract);
  const drafts: AgenticCandidateSpec[] = [];
  const seen = new Set<string>();

  families.forEach((family) => {
    if (drafts.length >= target) return;
    const config = boundModelConfig(baseConfig);
    const advancedConfig = boundAdvancedConfig(
      baseAdvancedConfig,
      capabilities,
    );
    const spec = makeSpec(
      "",
      family,
      config,
      advancedConfig,
      "seed",
      {
        method: "space-filling",
        reason:
          "Reference seed anchors the adaptive search to the current working specification.",
      },
      1,
    );
    seen.add(agenticCandidateSignature(spec));
    drafts.push(spec);
  });

  let attempt = 1;
  while (drafts.length < target && attempt < 1000) {
    const family = families[(attempt - 1) % families.length];
    const restart = (Math.floor((attempt - 1) / families.length) % 3 +
      1) as 1 | 2 | 3;
    const { config, advancedConfig } = configurationFromPoint(
      family,
      haltonPoint(attempt + 7, searchDimensions(capabilities)),
      baseConfig,
      baseAdvancedConfig,
      capabilities,
    );
    const spec = makeSpec(
      "",
      family,
      config,
      advancedConfig,
      "seed",
      {
        method: "space-filling",
        reason:
          "Deterministic space-filling seed broadens coverage before the optimizer learns where to refine.",
      },
      restart,
    );
    const signature = agenticCandidateSignature(spec);
    if (!seen.has(signature)) {
      seen.add(signature);
      drafts.push(spec);
    }
    attempt += 1;
  }

  return drafts.map((spec, index) => ({
    ...spec,
    id: `C${String(index + 1).padStart(2, "0")}`,
  }));
}

const RESPONSE_DECAYS = [0.1, 0.3, 0.55, 0.8] as const;
const RESPONSE_WEIBULL_SHAPES = [0.8, 1.5, 3, 5] as const;
const RESPONSE_WEIBULL_SCALES = [1.5, 3.5, 7, 11] as const;
const RESPONSE_SATURATIONS = [0.6, 1.3, 2.2, 3] as const;
const RESPONSE_HALF_SATURATION = [0.2, 0.5, 0.8] as const;

function coveringChannelResponses(
  channels: string[],
  challengeIndex: number,
): NonNullable<ModelConfig["channelResponses"]> {
  return Object.fromEntries(
    channels.map((channel, channelIndex) => {
      const offset = challengeIndex + channelIndex * 5;
      return [
        channel,
        {
          adstockType: offset % 2 === 0 ? "geometric" : "weibull",
          adstock: RESPONSE_DECAYS[offset % RESPONSE_DECAYS.length],
          weibullShape:
            RESPONSE_WEIBULL_SHAPES[(challengeIndex * 3 + channelIndex) % RESPONSE_WEIBULL_SHAPES.length],
          weibullScale:
            RESPONSE_WEIBULL_SCALES[(challengeIndex + channelIndex * 3) % RESPONSE_WEIBULL_SCALES.length],
          saturation:
            RESPONSE_SATURATIONS[(challengeIndex * 3 + channelIndex * 2) % RESPONSE_SATURATIONS.length],
          halfSaturationQuantile:
            RESPONSE_HALF_SATURATION[(challengeIndex + channelIndex) % RESPONSE_HALF_SATURATION.length],
          kernelNormalization: offset % 4 < 2 ? "peak" : "sum",
        },
      ];
    }),
  );
}

export function generateAgenticChannelResponseChallenge(
  baseConfig: ModelConfig,
  baseAdvancedConfig: AdvancedModelConfig,
  contract: AgenticSearchContract,
  candidateNumber: number,
  capabilities: AgenticSearchCapabilities,
): AgenticCandidateSpec {
  const responseBudget = agenticChannelResponseBudget(contract, capabilities);
  const challengeIndex = candidateNumber - agenticSeedBudget(contract) - 1;
  if (
    challengeIndex < 0 ||
    challengeIndex >= responseBudget ||
    !(capabilities.mediaColumns?.length ?? 0)
  ) {
    throw new Error("Candidate is outside the channel-response coverage phase.");
  }
  const families = activeFamilies(contract);
  const family = families[challengeIndex % families.length];
  const point = haltonPoint(
    candidateNumber * 149 + 31,
    searchDimensions(capabilities),
  );
  const generated = configurationFromPoint(
    family,
    point,
    baseConfig,
    baseAdvancedConfig,
    capabilities,
  );
  const config = boundModelConfig({
    ...generated.config,
    channelResponses: coveringChannelResponses(
      capabilities.mediaColumns ?? [],
      challengeIndex,
    ),
  });
  return makeSpec(
    `C${String(candidateNumber).padStart(3, "0")}`,
    family,
    config,
    generated.advancedConfig,
    "response-coverage",
    {
      method: "response-covering-array",
      reason:
        "Predeclared channel-response covering array tests distinct carryover, saturation, half-saturation, and normalization contracts before adaptive refinement.",
    },
    ((Math.floor(challengeIndex / Math.max(families.length, 1)) % 3) + 1) as 1 | 2 | 3,
  );
}

function responseProfile(value: number): string {
  return value <= 0.8 ? "Low" : value >= 2.2 ? "High" : "Medium";
}

function halfSaturationProfile(value: number): string {
  return value <= 0.3 ? "Early" : value >= 0.7 ? "Late" : "Middle";
}

export function agenticChannelResponseCoverage(
  runs: AgenticCandidateRun[],
  contract: AgenticSearchContract,
  capabilities: AgenticSearchCapabilities,
): AgenticChannelResponseCoverage {
  const channels = capabilities.mediaColumns ?? [];
  const requiredChallenges = agenticChannelResponseBudget(contract, capabilities);
  const challenges = completedWithScores(runs).filter(
    (run) => run.spec.searchPhase === "response-coverage",
  );
  const coverage = channels.map((channel) => {
    const responses = challenges.flatMap((run) => {
      const match = Object.entries(run.spec.config.channelResponses ?? {}).find(
        ([key]) => key.toLowerCase() === channel.toLowerCase(),
      );
      return match?.[1] ? [match[1]] : [];
    });
    const adstockFamilies = uniqueValues(
      responses.map((response) => response.adstockType ?? "geometric"),
    );
    const saturationProfiles = uniqueValues(
      responses.map((response) => responseProfile(response.saturation ?? 1)),
    );
    const halfSaturationProfiles = uniqueValues(
      responses.map((response) =>
        halfSaturationProfile(response.halfSaturationQuantile ?? 0.5),
      ),
    );
    const normalizations = uniqueValues(
      responses.map((response) => response.kernelNormalization ?? "peak"),
    );
    const complete =
      ["geometric", "weibull"].every((value) => adstockFamilies.includes(value)) &&
      ["Low", "Medium", "High"].every((value) => saturationProfiles.includes(value)) &&
      ["Early", "Middle", "Late"].every((value) => halfSaturationProfiles.includes(value)) &&
      ["peak", "sum"].every((value) => normalizations.includes(value));
    return {
      channel,
      adstockFamilies,
      saturationProfiles,
      halfSaturationProfiles,
      normalizations,
      complete,
    };
  });
  return {
    requiredChallenges,
    completedChallenges: challenges.length,
    complete:
      challenges.length >= requiredChallenges &&
      coverage.every((channel) => channel.complete),
    channels: coverage,
  };
}

const ADVANCED_CHALLENGE_DESIGNS: AdvancedModelConfig[] = [
  {
    timeVarying: false,
    kernelKnots: 3,
    kernelBandwidth: 0.08,
    planningIntensity: false,
    calibrationMode: "prior",
    priorDistribution: "log-normal",
    likelihoodDistribution: "gaussian",
    studentTDegreesFreedom: 4,
  },
  {
    timeVarying: false,
    kernelKnots: 3,
    kernelBandwidth: 0.08,
    planningIntensity: true,
    calibrationMode: "likelihood",
    priorDistribution: "half-normal",
    likelihoodDistribution: "student-t",
    studentTDegreesFreedom: 4,
  },
  {
    timeVarying: true,
    kernelKnots: 3,
    kernelBandwidth: 0.08,
    planningIntensity: false,
    calibrationMode: "likelihood",
    priorDistribution: "half-normal",
    likelihoodDistribution: "log-normal",
    studentTDegreesFreedom: 8,
  },
  {
    timeVarying: true,
    kernelKnots: 4,
    kernelBandwidth: 0.12,
    planningIntensity: true,
    calibrationMode: "prior",
    priorDistribution: "log-normal",
    likelihoodDistribution: "student-t",
    studentTDegreesFreedom: 8,
  },
  {
    timeVarying: true,
    kernelKnots: 6,
    kernelBandwidth: 0.18,
    planningIntensity: false,
    calibrationMode: "prior",
    priorDistribution: "half-normal",
    likelihoodDistribution: "gaussian",
    studentTDegreesFreedom: 12,
  },
  {
    timeVarying: true,
    kernelKnots: 6,
    kernelBandwidth: 0.22,
    planningIntensity: true,
    calibrationMode: "likelihood",
    priorDistribution: "log-normal",
    likelihoodDistribution: "log-normal",
    studentTDegreesFreedom: 12,
  },
  {
    timeVarying: true,
    kernelKnots: 8,
    kernelBandwidth: 0.28,
    planningIntensity: false,
    calibrationMode: "likelihood",
    priorDistribution: "log-normal",
    likelihoodDistribution: "student-t",
    studentTDegreesFreedom: 15,
  },
  {
    timeVarying: true,
    kernelKnots: 10,
    kernelBandwidth: 0.35,
    planningIntensity: true,
    calibrationMode: "prior",
    priorDistribution: "half-normal",
    likelihoodDistribution: "gaussian",
    studentTDegreesFreedom: 20,
  },
  {
    timeVarying: false,
    kernelKnots: 10,
    kernelBandwidth: 0.35,
    planningIntensity: true,
    calibrationMode: "likelihood",
    priorDistribution: "log-normal",
    likelihoodDistribution: "gaussian",
    studentTDegreesFreedom: 20,
  },
  {
    timeVarying: true,
    kernelKnots: 5,
    kernelBandwidth: 0.14,
    planningIntensity: false,
    calibrationMode: "prior",
    priorDistribution: "log-normal",
    likelihoodDistribution: "log-normal",
    studentTDegreesFreedom: 24,
  },
  {
    timeVarying: true,
    kernelKnots: 7,
    kernelBandwidth: 0.24,
    planningIntensity: true,
    calibrationMode: "likelihood",
    priorDistribution: "half-normal",
    likelihoodDistribution: "student-t",
    studentTDegreesFreedom: 30,
  },
  {
    timeVarying: true,
    kernelKnots: 9,
    kernelBandwidth: 0.31,
    planningIntensity: false,
    calibrationMode: "prior",
    priorDistribution: "half-normal",
    likelihoodDistribution: "gaussian",
    studentTDegreesFreedom: 30,
  },
];

function strongestBaseSpecification(
  runs: AgenticCandidateRun[],
  fallbackConfig: ModelConfig,
): { id: string; family: ValidationModelKind; config: ModelConfig } {
  const baseRuns = completedWithScores(runs)
    .filter((run) => run.spec.family !== "advanced")
    .sort((left, right) => {
      const leftEligible = passesAgenticEligibility(left);
      const rightEligible = passesAgenticEligibility(right);
      if (leftEligible !== rightEligible) return leftEligible ? -1 : 1;
      return (
        (right.validation?.finalScore ?? -1) -
        (left.validation?.finalScore ?? -1)
      );
    });
  const anchor = baseRuns[0];
  return anchor
    ? {
        id: anchor.spec.id,
        family: anchor.spec.family,
        config: anchor.spec.config,
      }
    : { id: "working specification", family: "bayesian", config: fallbackConfig };
}

export function generateAgenticAdvancedChallenge(
  baseConfig: ModelConfig,
  contract: AgenticSearchContract,
  evaluatedRuns: AgenticCandidateRun[],
  candidateNumber: number,
  capabilities: AgenticSearchCapabilities,
): AgenticCandidateSpec {
  const challengeBudget = agenticAdvancedChallengeBudget(contract);
  const challengeIndex =
    candidateNumber -
    agenticSeedBudget(contract) -
    agenticChannelResponseBudget(contract, capabilities) -
    1;
  if (
    !contract.families.advanced ||
    challengeIndex < 0 ||
    challengeIndex >= challengeBudget
  ) {
    throw new Error("Candidate is outside the paired Advanced challenge phase.");
  }
  const anchor = strongestBaseSpecification(evaluatedRuns, baseConfig);
  const advancedConfig = boundAdvancedConfig(
    ADVANCED_CHALLENGE_DESIGNS[
      challengeIndex % ADVANCED_CHALLENGE_DESIGNS.length
    ],
    capabilities,
  );
  return makeSpec(
    `C${String(candidateNumber).padStart(2, "0")}`,
    "advanced",
    boundModelConfig(anchor.config),
    advancedConfig,
    "advanced-challenge",
    {
      method: "covering-array",
      reason: `Paired Advanced challenge holds ${anchor.id}'s ${familyLabel(anchor.family)} media response constant while testing a predeclared combination of dynamic effects, planning, calibration, and distributions.`,
    },
    ((challengeIndex % 3) + 1) as 1 | 2 | 3,
  );
}

function uniqueValues(values: string[]): string[] {
  return Array.from(new Set(values));
}

export function agenticAdvancedCoverage(
  runs: AgenticCandidateRun[],
  contract: AgenticSearchContract,
  capabilities: AgenticSearchCapabilities,
): AgenticAdvancedCoverage {
  const requiredChallenges = agenticAdvancedChallengeBudget(contract);
  if (!requiredChallenges) {
    return {
      requiredChallenges: 0,
      completedChallenges: 0,
      complete: true,
      dimensions: [],
    };
  }
  const challenges = completedWithScores(runs).filter(
    (run) =>
      run.spec.family === "advanced" &&
      run.spec.searchPhase === "advanced-challenge",
  );
  const configs = challenges.map((run) => run.spec.advancedConfig);
  const kernelProfiles = uniqueValues(
    configs
      .filter((config) => config.timeVarying)
      .map((config) =>
        config.kernelKnots <= 4
          ? "Low"
          : config.kernelKnots >= 8
            ? "High"
            : "Medium",
      ),
  );
  const studentTProfiles = uniqueValues(
    configs
      .filter((config) => config.likelihoodDistribution === "student-t")
      .map((config) =>
        config.studentTDegreesFreedom <= 6
          ? "Low"
          : config.studentTDegreesFreedom > 15
            ? "High"
            : "Medium",
      ),
  );
  const dimensions: AgenticAdvancedCoverageDimension[] = [
    {
      id: "time-varying",
      label: "Coefficient structure",
      tested: uniqueValues(
        configs.map((config) =>
          config.timeVarying ? "Dynamic" : "Static",
        ),
      ),
      required: ["Static", "Dynamic"],
      complete: false,
    },
    {
      id: "kernel",
      label: "Kernel flexibility",
      tested: kernelProfiles,
      required: ["Low", "Medium", "High"],
      complete: false,
    },
    {
      id: "planning",
      label: "Planning factor",
      tested: uniqueValues(
        configs.map((config) =>
          config.planningIntensity ? "On" : "Off",
        ),
      ),
      required: ["Off", "On"],
      complete: false,
    },
    {
      id: "calibration",
      label: "Calibration route",
      tested: uniqueValues(
        configs.map((config) =>
          config.calibrationMode === "likelihood" ? "Likelihood" : "Prior",
        ),
      ),
      required: capabilities.likelihoodCalibration
        ? ["Prior", "Likelihood"]
        : ["Prior"],
      complete: false,
    },
    {
      id: "prior",
      label: "Prior family",
      tested: uniqueValues(
        configs.map((config) =>
          config.priorDistribution === "log-normal"
            ? "Log-normal"
            : "Half-normal",
        ),
      ),
      required: ["Half-normal", "Log-normal"],
      complete: false,
    },
    {
      id: "likelihood",
      label: "Outcome likelihood",
      tested: uniqueValues(
        configs.map((config) =>
          config.likelihoodDistribution === "student-t"
            ? "Student-t"
            : config.likelihoodDistribution === "log-normal"
              ? "Log-normal"
              : "Gaussian",
        ),
      ),
      required: ["Gaussian", "Student-t", "Log-normal"],
      complete: false,
    },
    {
      id: "student-t",
      label: "Student-t robustness",
      tested: studentTProfiles,
      required: ["Low", "Medium", "High"],
      complete: false,
    },
  ].map((dimension) => ({
    ...dimension,
    complete: dimension.required.every((value) =>
      dimension.tested.includes(value),
    ),
  }));
  return {
    requiredChallenges,
    completedChallenges: challenges.length,
    complete:
      challenges.length >= requiredChallenges &&
      dimensions.every((dimension) => dimension.complete),
    dimensions,
  };
}

function normalize(
  value: number,
  minimum: number,
  maximum: number,
): number {
  return clamp((value - minimum) / Math.max(maximum - minimum, 1e-9), 0, 1);
}

function normalizeLog(
  value: number,
  minimum: number,
  maximum: number,
): number {
  return normalize(Math.log(value), Math.log(minimum), Math.log(maximum));
}

function encodeSpecification(
  spec: Pick<
    AgenticCandidateSpec,
    "family" | "config" | "advancedConfig"
  >,
): EncodedSpecification {
  const numeric: Record<string, number> = {
    saturation: normalize(
      spec.config.saturation,
      AGENTIC_PARAMETER_BOUNDS.saturation.min,
      AGENTIC_PARAMETER_BOUNDS.saturation.max,
    ),
    ridge: normalizeLog(
      spec.config.ridge,
      AGENTIC_PARAMETER_BOUNDS.ridge.min,
      AGENTIC_PARAMETER_BOUNDS.ridge.max,
    ),
  };
  const categorical: Record<string, string> = {
    family: spec.family,
    adstockType: spec.config.adstockType,
    fourierOrder: String(spec.config.fourierOrder),
    cyclePeriod: String(spec.config.cyclePeriod),
  };
  if (spec.config.adstockType === "geometric") {
    numeric.adstock = normalize(
      spec.config.adstock,
      AGENTIC_PARAMETER_BOUNDS.adstock.min,
      AGENTIC_PARAMETER_BOUNDS.adstock.max,
    );
  } else {
    numeric.weibullShape = normalize(
      spec.config.weibullShape,
      AGENTIC_PARAMETER_BOUNDS.weibullShape.min,
      AGENTIC_PARAMETER_BOUNDS.weibullShape.max,
    );
    numeric.weibullScale = normalize(
      spec.config.weibullScale,
      AGENTIC_PARAMETER_BOUNDS.weibullScale.min,
      AGENTIC_PARAMETER_BOUNDS.weibullScale.max,
    );
  }
  Object.entries(spec.config.channelResponses ?? {})
    .sort(([left], [right]) => left.localeCompare(right))
    .forEach(([channel, response]) => {
      const prefix = `channel:${channel.toLowerCase()}`;
      const adstockType = response.adstockType ?? spec.config.adstockType;
      categorical[`${prefix}:adstockType`] = adstockType;
      categorical[`${prefix}:normalization`] =
        response.kernelNormalization ?? "peak";
      numeric[`${prefix}:saturation`] = normalize(
        response.saturation ?? spec.config.saturation,
        AGENTIC_PARAMETER_BOUNDS.saturation.min,
        AGENTIC_PARAMETER_BOUNDS.saturation.max,
      );
      numeric[`${prefix}:halfSaturation`] = normalize(
        response.halfSaturationQuantile ?? 0.5,
        AGENTIC_PARAMETER_BOUNDS.halfSaturationQuantile.min,
        AGENTIC_PARAMETER_BOUNDS.halfSaturationQuantile.max,
      );
      if (adstockType === "geometric") {
        numeric[`${prefix}:decay`] = normalize(
          response.adstock ?? spec.config.adstock,
          AGENTIC_PARAMETER_BOUNDS.adstock.min,
          AGENTIC_PARAMETER_BOUNDS.adstock.max,
        );
      } else {
        numeric[`${prefix}:weibullShape`] = normalize(
          response.weibullShape ?? spec.config.weibullShape,
          AGENTIC_PARAMETER_BOUNDS.weibullShape.min,
          AGENTIC_PARAMETER_BOUNDS.weibullShape.max,
        );
        numeric[`${prefix}:weibullScale`] = normalize(
          response.weibullScale ?? spec.config.weibullScale,
          AGENTIC_PARAMETER_BOUNDS.weibullScale.min,
          AGENTIC_PARAMETER_BOUNDS.weibullScale.max,
        );
      }
    });
  if (spec.family === "advanced") {
    categorical.timeVarying = String(spec.advancedConfig.timeVarying);
    categorical.planningIntensity = String(
      spec.advancedConfig.planningIntensity,
    );
    categorical.calibrationMode = spec.advancedConfig.calibrationMode;
    categorical.priorDistribution = spec.advancedConfig.priorDistribution;
    categorical.likelihoodDistribution =
      spec.advancedConfig.likelihoodDistribution;
    if (spec.advancedConfig.timeVarying) {
      numeric.kernelKnots = normalize(
        spec.advancedConfig.kernelKnots,
        AGENTIC_PARAMETER_BOUNDS.kernelKnots.min,
        AGENTIC_PARAMETER_BOUNDS.kernelKnots.max,
      );
      numeric.kernelBandwidth = normalize(
        spec.advancedConfig.kernelBandwidth,
        AGENTIC_PARAMETER_BOUNDS.kernelBandwidth.min,
        AGENTIC_PARAMETER_BOUNDS.kernelBandwidth.max,
      );
    }
    if (spec.advancedConfig.likelihoodDistribution === "student-t") {
      numeric.studentTDegreesFreedom = normalize(
        spec.advancedConfig.studentTDegreesFreedom,
        AGENTIC_PARAMETER_BOUNDS.studentTDegreesFreedom.min,
        AGENTIC_PARAMETER_BOUNDS.studentTDegreesFreedom.max,
      );
    }
  }
  return { numeric, categorical };
}

function mixedDistance(
  left: EncodedSpecification,
  right: EncodedSpecification,
): number {
  let distance = 0;
  let terms = 0;
  const numericKeys = new Set([
    ...Object.keys(left.numeric),
    ...Object.keys(right.numeric),
  ]);
  numericKeys.forEach((key) => {
    const leftValue = left.numeric[key];
    const rightValue = right.numeric[key];
    distance +=
      leftValue === undefined || rightValue === undefined
        ? 0.65
        : Math.abs(leftValue - rightValue);
    terms += 1;
  });
  const categoricalKeys = new Set([
    ...Object.keys(left.categorical),
    ...Object.keys(right.categorical),
  ]);
  categoricalKeys.forEach((key) => {
    distance +=
      left.categorical[key] === right.categorical[key] ? 0 : 1;
    terms += 1;
  });
  return distance / Math.max(terms, 1);
}

function kernel(distance: number, bandwidth: number): number {
  return Math.exp(-(distance ** 2) / (2 * bandwidth ** 2));
}

function density(
  candidate: EncodedSpecification,
  samples: EncodedSpecification[],
  bandwidth = 0.2,
): number {
  if (!samples.length) return 1e-6;
  return (
    samples.reduce(
      (total, sample) =>
        total + kernel(mixedDistance(candidate, sample), bandwidth),
      0,
    ) /
      samples.length +
    1e-6
  );
}

function completedWithScores(
  runs: AgenticCandidateRun[],
): AgenticCandidateRun[] {
  return runs.filter(
    (run) =>
      run.state === "complete" &&
      run.validation?.finalScore !== null &&
      run.validation?.finalScore !== undefined,
  );
}

function diverseEliteRuns(
  runs: AgenticCandidateRun[],
  count: number,
  minimumDistance = 0.12,
): AgenticCandidateRun[] {
  const ranked = [...runs].sort(
    (left, right) =>
      (right.validation?.finalScore ?? -1) -
      (left.validation?.finalScore ?? -1),
  );
  const selected: AgenticCandidateRun[] = [];
  ranked.forEach((run) => {
    if (selected.length >= count) return;
    const encoded = encodeSpecification(run.spec);
    const sufficientlyDifferent = selected.every(
      (existing) =>
        mixedDistance(encoded, encodeSpecification(existing.spec)) >=
        minimumDistance,
    );
    if (sufficientlyDifferent) selected.push(run);
  });
  ranked.forEach((run) => {
    if (selected.length >= count || selected.includes(run)) return;
    selected.push(run);
  });
  return selected;
}

function scoreProposal(
  spec: AgenticCandidateSpec,
  observed: AgenticCandidateRun[],
  good: AgenticCandidateRun[],
  bad: AgenticCandidateRun[],
): ProposalScore {
  const encoded = encodeSpecification(spec);
  const observedEncoded = observed.map((run) =>
    encodeSpecification(run.spec),
  );
  const distances = observedEncoded.map((item) =>
    mixedDistance(encoded, item),
  );
  const weights = distances.map((distance) => kernel(distance, 0.24));
  const totalWeight = weights.reduce((total, value) => total + value, 0);
  const scores = observed.map(
    (run) => run.validation?.finalScore ?? 0,
  );
  const feasibleIndexes = observed
    .map((run, index) => ({ run, index }))
    .filter(({ run }) => passesAgenticEligibility(run))
    .map(({ index }) => index);
  const scoreIndexes = feasibleIndexes.length
    ? feasibleIndexes
    : observed.map((_, index) => index);
  const scoreWeight = scoreIndexes.reduce(
    (total, index) => total + weights[index],
    0,
  );
  const expectedScore = scoreWeight > 1e-8
    ? scoreIndexes.reduce(
        (total, index) => total + scores[index] * weights[index],
        0,
      ) / scoreWeight
    : scoreIndexes.reduce((total, index) => total + scores[index], 0) /
      Math.max(scoreIndexes.length, 1);
  const eligibilityProbability =
    (1 +
      observed.reduce(
        (total, run, index) =>
          total +
          (passesAgenticEligibility(run) ? weights[index] : 0),
        0,
      )) /
    (2 + totalWeight);
  const goodDensity = density(
    encoded,
    good.map((run) => encodeSpecification(run.spec)),
  );
  const badDensity = density(
    encoded,
    bad.map((run) => encodeSpecification(run.spec)),
  );
  const densityRatio = clamp(
    Math.log(goodDensity / Math.max(badDensity, 1e-9)),
    -4,
    4,
  );
  const exploration = distances.length ? Math.min(...distances) : 1;
  const scoreVariance =
    scoreWeight > 1e-8
      ? scoreIndexes.reduce(
          (total, index) =>
            total + weights[index] * (scores[index] - expectedScore) ** 2,
          0,
        ) / scoreWeight
      : 0;
  const uncertainty =
    Math.sqrt(Math.max(scoreVariance, 0)) + exploration * 12;
  const bestEligibleScore = Math.max(
    ...observed
      .filter(passesAgenticEligibility)
      .map((run) => run.validation?.finalScore ?? 0),
    0,
  );
  const expectedImprovement = Math.max(
    0,
    expectedScore + uncertainty * 0.6 - bestEligibleScore,
  ) / 100;
  const acquisition =
    eligibilityProbability *
      (Math.max(densityRatio, -1) * 0.8 +
        expectedImprovement * 2.2 +
        expectedScore / 100) +
    exploration * 0.45;
  return {
    spec,
    acquisition,
    expectedScore,
    eligibilityProbability,
    exploration,
  };
}

export function proposeAgenticCandidate(
  baseConfig: ModelConfig,
  baseAdvancedConfig: AdvancedModelConfig,
  contract: AgenticSearchContract,
  evaluatedRuns: AgenticCandidateRun[],
  candidateNumber: number,
  capabilities: AgenticSearchCapabilities,
): AgenticCandidateSpec {
  const families = activeFamilies(contract);
  if (!families.length) {
    throw new Error("At least one model family must remain enabled.");
  }
  const adaptiveIndex = Math.max(
    0,
    candidateNumber -
      agenticSeedBudget(contract) -
      agenticChannelResponseBudget(contract, capabilities) -
      agenticAdvancedChallengeBudget(contract) -
      1,
  );
  const targetFamily = families[adaptiveIndex % families.length];
  const targetRestart = (Math.floor(adaptiveIndex / families.length) % 3 +
    1) as 1 | 2 | 3;
  const familyObserved = completedWithScores(evaluatedRuns).filter(
    (run) => run.spec.family === targetFamily,
  );
  const restartObserved = familyObserved.filter(
    (run) => run.spec.restart === targetRestart,
  );
  // Keep each adaptive path statistically separate. The shared seed screen
  // establishes broad coverage, while a restart's surrogate only learns from
  // its own completed path once that path has an observation.
  const observed = restartObserved.length
    ? restartObserved
    : familyObserved;
  const ranked = [...observed].sort(
    (left, right) =>
      (right.validation?.finalScore ?? -1) -
      (left.validation?.finalScore ?? -1),
  );
  const eligible = ranked.filter(passesAgenticEligibility);
  const sourceForGood = eligible.length >= 3 ? eligible : ranked;
  const goodCount = Math.max(
    2,
    Math.min(
      sourceForGood.length,
      Math.ceil(Math.max(observed.length, 1) * 0.25),
    ),
  );
  const good = diverseEliteRuns(sourceForGood, goodCount);
  const goodIds = new Set(good.map((run) => run.spec.id));
  const bad = observed.filter((run) => !goodIds.has(run.spec.id));
  const existing = new Set(
    evaluatedRuns.map((run) => agenticCandidateSignature(run.spec)),
  );
  const proposals: AgenticCandidateSpec[] = [];
  const proposalSignatures = new Set<string>();
  const poolSize = 160;

  for (let attempt = 0; attempt < poolSize; attempt += 1) {
    const point = haltonPoint(
      candidateNumber * 389 + attempt + 29,
      searchDimensions(capabilities),
    );
    const generated = configurationFromPoint(
      targetFamily,
      point,
      baseConfig,
      baseAdvancedConfig,
      capabilities,
    );
    const spec = makeSpec(
      `C${String(candidateNumber).padStart(2, "0")}`,
      targetFamily,
      generated.config,
      generated.advancedConfig,
      "adaptive",
      {
        method: "tpe",
        reason: "Adaptive proposal awaiting surrogate diagnostics.",
      },
      targetRestart,
    );
    const signature = agenticCandidateSignature(spec);
    if (!existing.has(signature) && !proposalSignatures.has(signature)) {
      proposalSignatures.add(signature);
      proposals.push(spec);
    }
  }

  if (good.length) {
    for (let attempt = 0; attempt < poolSize; attempt += 1) {
      const anchor = good[attempt % good.length].spec;
      const point = haltonPoint(
        candidateNumber * 521 + attempt + 71,
        searchDimensions(capabilities),
      );
      const generated = localConfiguration(
        anchor,
        point,
        capabilities,
      );
      const spec = makeSpec(
        `C${String(candidateNumber).padStart(2, "0")}`,
        anchor.family,
        generated.config,
        generated.advancedConfig,
        "adaptive",
        {
          method: "tpe",
          reason: "Adaptive proposal awaiting surrogate diagnostics.",
        },
        targetRestart,
      );
      const signature = agenticCandidateSignature(spec);
      if (!existing.has(signature) && !proposalSignatures.has(signature)) {
        proposalSignatures.add(signature);
        proposals.push(spec);
      }
    }
  }

  const scored = proposals
    .map((spec) => scoreProposal(spec, observed, good, bad))
    .sort((left, right) => right.acquisition - left.acquisition);
  const selected = scored[0];
  if (!selected) {
    throw new Error("The bounded Agentic search space was exhausted.");
  }
  const reason =
    selected.exploration >= 0.18
      ? `Restart ${targetRestart} proposes joint ${familyLabel(selected.spec.family)} settings with an expected eligible score near ${selected.expectedScore.toFixed(1)} and ${Math.round(selected.eligibilityProbability * 100)}% estimated feasibility, while covering a less-tested basin.`
      : `Restart ${targetRestart} refines a distinct promising ${familyLabel(selected.spec.family)} basin with an expected eligible score near ${selected.expectedScore.toFixed(1)} and ${Math.round(selected.eligibilityProbability * 100)}% estimated feasibility.`;
  return {
    ...selected.spec,
    proposal: {
      method: "tpe",
      expectedScore: selected.expectedScore,
      eligibilityProbability: selected.eligibilityProbability,
      acquisition: selected.acquisition,
      reason,
    },
    hypothesis: reason,
  };
}

function localChallengeConfiguration(
  champion: AgenticCandidateSpec,
  challengeIndex: number,
  capabilities: AgenticSearchCapabilities,
): { config: ModelConfig; advancedConfig: AdvancedModelConfig } {
  const config = { ...champion.config };
  config.channelResponses = champion.config.channelResponses
    ? Object.fromEntries(
        Object.entries(champion.config.channelResponses).map(([channel, response]) => [
          channel,
          { ...response },
        ]),
      )
    : undefined;
  const advancedConfig = { ...champion.advancedConfig };
  const cyclePeriods = cyclePeriodChoices(config);
  const cycleIndex = cyclePeriods.indexOf(
    nearestChoice(config.cyclePeriod, cyclePeriods),
  );
  const channels = Object.keys(config.channelResponses ?? {});
  if (channels.length && challengeIndex >= 12) {
    const channel = channels[(challengeIndex - 12) % channels.length];
    const response = { ...(config.channelResponses?.[channel] ?? {}) };
    const dimension = Math.floor((challengeIndex - 12) / channels.length) % 7;
    if (dimension === 0) {
      response.adstockType = response.adstockType === "weibull" ? "geometric" : "weibull";
    } else if (dimension === 1) {
      response.adstock = (response.adstock ?? config.adstock) * 0.8;
    } else if (dimension === 2) {
      response.weibullShape = (response.weibullShape ?? config.weibullShape) * 1.2;
    } else if (dimension === 3) {
      response.weibullScale = (response.weibullScale ?? config.weibullScale) * 1.2;
    } else if (dimension === 4) {
      response.saturation = (response.saturation ?? config.saturation) * 1.2;
    } else if (dimension === 5) {
      response.halfSaturationQuantile =
        (response.halfSaturationQuantile ?? 0.5) >= 0.5 ? 0.35 : 0.65;
    } else {
      response.kernelNormalization =
        response.kernelNormalization === "sum" ? "peak" : "sum";
    }
    config.channelResponses = {
      ...config.channelResponses,
      [channel]: response,
    };
  } else switch (challengeIndex % 12) {
    case 0:
      config.adstock *= 0.8;
      config.weibullScale *= 0.8;
      break;
    case 1:
      config.adstock *= 1.2;
      config.weibullScale *= 1.2;
      break;
    case 2:
      config.saturation *= 0.8;
      break;
    case 3:
      config.saturation *= 1.2;
      break;
    case 4:
      config.ridge *= 0.65;
      break;
    case 5:
      config.ridge *= 1.35;
      break;
    case 6:
      config.fourierOrder = config.fourierOrder === 3
        ? 2
        : config.fourierOrder + 1;
      break;
    case 7:
      config.cyclePeriod =
        cyclePeriods[(cycleIndex + 1) % cyclePeriods.length];
      break;
    case 8:
      config.adstockType =
        config.adstockType === "geometric" ? "weibull" : "geometric";
      break;
    case 9:
      if (champion.family === "advanced") {
        advancedConfig.timeVarying = !advancedConfig.timeVarying;
      } else {
        config.saturation *= 0.9;
        config.ridge *= 1.1;
      }
      break;
    case 10:
      if (champion.family === "advanced") {
        advancedConfig.planningIntensity =
          !advancedConfig.planningIntensity;
      } else {
        config.adstock *= 0.9;
        config.saturation *= 1.1;
      }
      break;
    default:
      if (champion.family === "advanced") {
        advancedConfig.priorDistribution =
          advancedConfig.priorDistribution === "log-normal"
            ? "half-normal"
            : "log-normal";
        advancedConfig.likelihoodDistribution =
          advancedConfig.likelihoodDistribution === "gaussian"
            ? "student-t"
            : "gaussian";
      } else {
        config.weibullShape *= 1.15;
        config.ridge *= 0.85;
      }
  }
  return {
    config: boundModelConfig(config),
    advancedConfig: boundAdvancedConfig(advancedConfig, capabilities),
  };
}

export function generateAgenticLocalChallenge(
  contract: AgenticSearchContract,
  evaluatedRuns: AgenticCandidateRun[],
  candidateNumber: number,
  capabilities: AgenticSearchCapabilities,
): AgenticCandidateSpec {
  const localBudget = agenticLocalChallengeBudget(contract);
  const localStart =
    agenticSeedBudget(contract) +
    agenticChannelResponseBudget(contract, capabilities) +
    agenticAdvancedChallengeBudget(contract) +
    agenticAdaptiveMaximumBudget(contract, capabilities);
  const challengeIndex = candidateNumber - localStart - 1;
  if (challengeIndex < 0 || challengeIndex >= localBudget) {
    throw new Error("Candidate is outside the local champion challenge phase.");
  }
  const completed = completedWithScores(evaluatedRuns).filter(
    (run) => run.spec.searchPhase !== "rescue",
  );
  const eligible = completed.filter(passesAgenticEligibility);
  const source = (eligible.length ? eligible : completed).sort(
    (left, right) =>
      (right.validation?.finalScore ?? -1) -
      (left.validation?.finalScore ?? -1),
  )[0];
  if (!source) {
    throw new Error("A completed candidate is required for local challenge.");
  }
  const existing = new Set(
    evaluatedRuns.map((run) => agenticCandidateSignature(run.spec)),
  );
  let configuration = localChallengeConfiguration(
    source.spec,
    challengeIndex,
    capabilities,
  );
  let attempt = 0;
  while (
    existing.has(
      agenticCandidateSignature({
        ...source.spec,
        config: configuration.config,
        advancedConfig: configuration.advancedConfig,
      }),
    ) &&
    attempt < 64
  ) {
    configuration = localConfiguration(
      source.spec,
      haltonPoint(
        candidateNumber * 733 + attempt + 17,
        searchDimensions(capabilities),
      ),
      capabilities,
    );
    attempt += 1;
  }
  const spec = makeSpec(
    `C${String(candidateNumber).padStart(2, "0")}`,
    source.spec.family,
    configuration.config,
    configuration.advancedConfig,
    "local-challenge",
    {
      method: "local-challenge",
      reason: `Challenges ${source.spec.id} with a predeclared neighboring perturbation; it must improve the eligible score rather than merely remain close.`,
    },
    source.spec.restart,
  );
  return {
    ...spec,
    challengeOf: source.spec.id,
  };
}

export function agenticStoppingDecision(
  runs: AgenticCandidateRun[],
  contract: AgenticSearchContract,
  capabilities: AgenticSearchCapabilities = {
    likelihoodCalibration: true,
  },
): AgenticStopDecision {
  const completed = completedWithScores(runs);
  if (completed.length >= contract.candidateBudget) {
    return {
      shouldStop: true,
      reason: `Candidate budget reached after ${completed.length} evaluated specifications.`,
      evaluationsSinceImprovement: 0,
    };
  }
  const responseCoverage = agenticChannelResponseCoverage(
    runs,
    contract,
    capabilities,
  );
  if (!responseCoverage.complete) {
    return {
      shouldStop: false,
      reason: `Channel-response coverage is ${responseCoverage.completedChallenges}/${responseCoverage.requiredChallenges}; convergence is locked until every media channel spans carryover, saturation, half-saturation, and normalization profiles.`,
      evaluationsSinceImprovement: 0,
    };
  }
  const advancedCoverage = agenticAdvancedCoverage(
    runs,
    contract,
    capabilities,
  );
  if (!advancedCoverage.complete) {
    return {
      shouldStop: false,
      reason: `Advanced challenge coverage is ${advancedCoverage.completedChallenges}/${advancedCoverage.requiredChallenges}; convergence is locked until the predeclared assumptions have been tested.`,
      evaluationsSinceImprovement: 0,
    };
  }
  const families = activeFamilies(contract);
  const adaptiveMinimum = agenticAdaptiveMinimumBudget(contract, capabilities);
  const requiredPerFamilyRestart = contract.candidateBudget >= 384 ? 4 : contract.candidateBudget >= 192 ? 3 : 2;
  const missingAdaptiveCells = families.flatMap((family) =>
    ([1, 2, 3] as const).flatMap((restart) =>
      completed.filter(
        (run) =>
          run.spec.searchPhase === "adaptive" &&
          run.spec.family === family &&
          run.spec.restart === restart,
      ).length < requiredPerFamilyRestart
        ? [`${family}-${restart}`]
        : [],
    ),
  );
  if (missingAdaptiveCells.length) {
    const completedAdaptive = completed.filter(
      (run) => run.spec.searchPhase === "adaptive",
    ).length;
    return {
      shouldStop: false,
      reason: `Restart-aware refinement is ${completedAdaptive}/${adaptiveMinimum}; convergence remains locked until every estimator family is challenged within each independent start.`,
      evaluationsSinceImprovement: 0,
    };
  }
  const localChallengeBudget = agenticLocalChallengeBudget(contract);
  const completedLocalChallenges = completed.filter(
    (run) => run.spec.searchPhase === "local-challenge",
  ).length;
  if (completedLocalChallenges < localChallengeBudget) {
    return {
      shouldStop: false,
      reason: `Champion-neighborhood testing is ${completedLocalChallenges}/${localChallengeBudget}; the best-known model is not declared until every reserved local challenge is complete.`,
      evaluationsSinceImprovement: 0,
    };
  }
  const minimumEvaluations = Math.min(
    contract.candidateBudget,
    Math.max(
      24,
      agenticSeedBudget(contract) +
        agenticChannelResponseBudget(contract, capabilities) +
        agenticAdvancedChallengeBudget(contract) +
        adaptiveMinimum,
    ),
  );
  const patience = contract.candidateBudget >= 384 ? 30 : contract.candidateBudget >= 192 ? 22 : 16;
  if (completed.length < minimumEvaluations) {
    return {
      shouldStop: false,
      reason: "Building enough evidence for an adaptive stopping decision.",
      evaluationsSinceImprovement: 0,
    };
  }
  const split = completed.length - patience;
  if (split <= 0) {
    return {
      shouldStop: false,
      reason: "Adaptive refinement is still inside the minimum evidence window.",
      evaluationsSinceImprovement: 0,
    };
  }
  const eligibleBefore = completed
    .slice(0, split)
    .filter(passesAgenticEligibility);
  if (!eligibleBefore.length) {
    return {
      shouldStop: false,
      reason: "No earlier eligible reference exists yet.",
      evaluationsSinceImprovement: 0,
    };
  }
  const previousBest = Math.max(
    ...eligibleBefore.map((run) => run.validation?.finalScore ?? 0),
  );
  const recentBest = Math.max(
    ...completed
      .slice(split)
      .filter(passesAgenticEligibility)
      .map((run) => run.validation?.finalScore ?? -Infinity),
  );
  const minimumImprovement = 0.25;
  if (recentBest < previousBest + minimumImprovement) {
    return {
      shouldStop: true,
      reason: `Converged after ${patience} trials without at least ${minimumImprovement.toFixed(2)} points of eligible-score improvement.`,
      evaluationsSinceImprovement: patience,
    };
  }
  return {
    shouldStop: false,
    reason: "The eligible frontier is still improving.",
    evaluationsSinceImprovement: 0,
  };
}

export function isAgenticSpecWithinBounds(
  spec: Pick<AgenticCandidateSpec, "config" | "advancedConfig">,
): boolean {
  const { config, advancedConfig } = spec;
  return (
    config.adstock >= AGENTIC_PARAMETER_BOUNDS.adstock.min &&
    config.adstock <= AGENTIC_PARAMETER_BOUNDS.adstock.max &&
    config.weibullShape >= AGENTIC_PARAMETER_BOUNDS.weibullShape.min &&
    config.weibullShape <= AGENTIC_PARAMETER_BOUNDS.weibullShape.max &&
    config.weibullScale >= AGENTIC_PARAMETER_BOUNDS.weibullScale.min &&
    config.weibullScale <= AGENTIC_PARAMETER_BOUNDS.weibullScale.max &&
    config.saturation >= AGENTIC_PARAMETER_BOUNDS.saturation.min &&
    config.saturation <= AGENTIC_PARAMETER_BOUNDS.saturation.max &&
    config.ridge >= AGENTIC_PARAMETER_BOUNDS.ridge.min &&
    config.ridge <= AGENTIC_PARAMETER_BOUNDS.ridge.max &&
    AGENTIC_PARAMETER_BOUNDS.fourierOrder.includes(
      config.fourierOrder as 1 | 2 | 3,
    ) &&
    cyclePeriodChoices(config).some(
      (period) => period === config.cyclePeriod,
    ) &&
    advancedConfig.kernelKnots >=
      AGENTIC_PARAMETER_BOUNDS.kernelKnots.min &&
    advancedConfig.kernelKnots <=
      AGENTIC_PARAMETER_BOUNDS.kernelKnots.max &&
    advancedConfig.kernelBandwidth >=
      AGENTIC_PARAMETER_BOUNDS.kernelBandwidth.min &&
    advancedConfig.kernelBandwidth <=
      AGENTIC_PARAMETER_BOUNDS.kernelBandwidth.max &&
    advancedConfig.studentTDegreesFreedom >=
      AGENTIC_PARAMETER_BOUNDS.studentTDegreesFreedom.min &&
    advancedConfig.studentTDegreesFreedom <=
      AGENTIC_PARAMETER_BOUNDS.studentTDegreesFreedom.max &&
    Object.values(config.channelResponses ?? {}).every((response) =>
      (response.adstock ?? config.adstock) >= AGENTIC_PARAMETER_BOUNDS.adstock.min &&
      (response.adstock ?? config.adstock) <= AGENTIC_PARAMETER_BOUNDS.adstock.max &&
      (response.weibullShape ?? config.weibullShape) >= AGENTIC_PARAMETER_BOUNDS.weibullShape.min &&
      (response.weibullShape ?? config.weibullShape) <= AGENTIC_PARAMETER_BOUNDS.weibullShape.max &&
      (response.weibullScale ?? config.weibullScale) >= AGENTIC_PARAMETER_BOUNDS.weibullScale.min &&
      (response.weibullScale ?? config.weibullScale) <= AGENTIC_PARAMETER_BOUNDS.weibullScale.max &&
      (response.saturation ?? config.saturation) >= AGENTIC_PARAMETER_BOUNDS.saturation.min &&
      (response.saturation ?? config.saturation) <= AGENTIC_PARAMETER_BOUNDS.saturation.max &&
      (response.halfSaturationQuantile ?? 0.5) >= AGENTIC_PARAMETER_BOUNDS.halfSaturationQuantile.min &&
      (response.halfSaturationQuantile ?? 0.5) <= AGENTIC_PARAMETER_BOUNDS.halfSaturationQuantile.max,
    )
  );
}

export function agenticBoundaryParameters(
  spec: Pick<
    AgenticCandidateSpec,
    "family" | "config" | "advancedConfig"
  >,
): string[] {
  const hits: string[] = [];
  const atBoundary = (
    value: number,
    bounds: { min: number; max: number; step: number },
  ) =>
    value <= bounds.min + bounds.step / 2 ||
    value >= bounds.max - bounds.step / 2;
  if (
    spec.config.adstockType === "geometric" &&
    atBoundary(spec.config.adstock, AGENTIC_PARAMETER_BOUNDS.adstock)
  ) {
    hits.push("Geometric decay");
  }
  if (spec.config.adstockType === "weibull") {
    if (
      atBoundary(
        spec.config.weibullShape,
        AGENTIC_PARAMETER_BOUNDS.weibullShape,
      )
    ) {
      hits.push("Weibull shape");
    }
    if (
      atBoundary(
        spec.config.weibullScale,
        AGENTIC_PARAMETER_BOUNDS.weibullScale,
      )
    ) {
      hits.push("Weibull scale");
    }
  }
  if (
    atBoundary(
      spec.config.saturation,
      AGENTIC_PARAMETER_BOUNDS.saturation,
    )
  ) {
    hits.push("Hill shape");
  }
  if (atBoundary(spec.config.ridge, AGENTIC_PARAMETER_BOUNDS.ridge)) {
    hits.push("Ridge penalty");
  }
  if (spec.family === "advanced" && spec.advancedConfig.timeVarying) {
    if (
      atBoundary(
        spec.advancedConfig.kernelKnots,
        AGENTIC_PARAMETER_BOUNDS.kernelKnots,
      )
    ) {
      hits.push("Kernel knots");
    }
    if (
      atBoundary(
        spec.advancedConfig.kernelBandwidth,
        AGENTIC_PARAMETER_BOUNDS.kernelBandwidth,
      )
    ) {
      hits.push("Kernel bandwidth");
    }
  }
  if (
    spec.family === "advanced" &&
    spec.advancedConfig.likelihoodDistribution === "student-t" &&
    atBoundary(
      spec.advancedConfig.studentTDegreesFreedom,
      AGENTIC_PARAMETER_BOUNDS.studentTDegreesFreedom,
    )
  ) {
    hits.push("Student-t degrees of freedom");
  }
  Object.entries(spec.config.channelResponses ?? {}).forEach(
    ([channel, response]) => {
      const responseAdstock = response.adstockType ?? spec.config.adstockType;
      if (
        responseAdstock === "geometric" &&
        atBoundary(
          response.adstock ?? spec.config.adstock,
          AGENTIC_PARAMETER_BOUNDS.adstock,
        )
      ) {
        hits.push(`${channel} decay`);
      }
      if (responseAdstock === "weibull") {
        if (
          atBoundary(
            response.weibullShape ?? spec.config.weibullShape,
            AGENTIC_PARAMETER_BOUNDS.weibullShape,
          )
        ) {
          hits.push(`${channel} Weibull shape`);
        }
        if (
          atBoundary(
            response.weibullScale ?? spec.config.weibullScale,
            AGENTIC_PARAMETER_BOUNDS.weibullScale,
          )
        ) {
          hits.push(`${channel} Weibull scale`);
        }
      }
      if (
        atBoundary(
          response.saturation ?? spec.config.saturation,
          AGENTIC_PARAMETER_BOUNDS.saturation,
        )
      ) {
        hits.push(`${channel} Hill shape`);
      }
      if (
        atBoundary(
          response.halfSaturationQuantile ?? 0.5,
          AGENTIC_PARAMETER_BOUNDS.halfSaturationQuantile,
        )
      ) {
        hits.push(`${channel} half-saturation`);
      }
    },
  );
  return hits;
}

export function agenticSearchConfidence(
  runs: AgenticCandidateRun[],
  contract: AgenticSearchContract,
  capabilities: AgenticSearchCapabilities = {
    likelihoodCalibration: true,
  },
): AgenticSearchConfidence {
  const evaluated = completedWithScores(runs);
  const completed = evaluated.filter(
    (run) => run.spec.searchPhase !== "rescue",
  );
  let bestEligible = -Infinity;
  let evaluationsSinceImprovement = 0;
  const frontier = evaluated.map((run, index) => {
    if (
      passesAgenticEligibility(run) &&
      (run.validation?.finalScore ?? -Infinity) > bestEligible + 0.25
    ) {
      bestEligible = run.validation?.finalScore ?? bestEligible;
      evaluationsSinceImprovement = 0;
    } else {
      evaluationsSinceImprovement += 1;
    }
    return {
      evaluation: index + 1,
      score: Number.isFinite(bestEligible) ? bestEligible : null,
      phase: run.spec.searchPhase,
      restart: run.spec.restart,
    };
  });
  const eligible = evaluated.filter(passesAgenticEligibility);
  const champion = [...eligible].sort(
    (left, right) =>
      (right.validation?.finalScore ?? -1) -
      (left.validation?.finalScore ?? -1),
  )[0];
  const restartWinners = ([1, 2, 3] as const).map((restart) => {
    const winner = evaluated
      .filter(
        (run) => run.spec.restart === restart && passesAgenticEligibility(run),
      )
      .sort(
        (left, right) =>
          (right.validation?.finalScore ?? -1) -
          (left.validation?.finalScore ?? -1),
      )[0];
    return {
      restart,
      candidateId: winner?.spec.id,
      score: winner?.validation?.finalScore ?? undefined,
      family: winner?.spec.family,
    };
  });
  const restartRuns = restartWinners.flatMap((winner) => {
    const run = evaluated.find((candidate) => candidate.spec.id === winner.candidateId);
    return run ? [run] : [];
  });
  const pairAgreement: number[] = [];
  restartRuns.forEach((left, leftIndex) => {
    restartRuns.slice(leftIndex + 1).forEach((right) => {
      const distance = mixedDistance(
        encodeSpecification(left.spec),
        encodeSpecification(right.spec),
      );
      const scoreGap = Math.abs(
        (left.validation?.finalScore ?? 0) -
          (right.validation?.finalScore ?? 0),
      );
      pairAgreement.push(
        0.7 * (1 - clamp(distance / 0.35, 0, 1)) +
          0.3 * (1 - clamp(scoreGap / 3, 0, 1)),
      );
    });
  });
  const restartAgreement = pairAgreement.length
    ? pairAgreement.reduce((total, value) => total + value, 0) /
      pairAgreement.length
    : 0;
  const families = activeFamilies(contract);
  const structureCells = new Set(
    completed.map(
      (run) => `${run.spec.family}-${run.spec.config.adstockType}`,
    ),
  );
  const baseCoverage =
    structureCells.size / Math.max(families.length * 2, 1);
  const advancedCoverage = agenticAdvancedCoverage(
    runs,
    contract,
    capabilities,
  );
  const advancedShare = advancedCoverage.requiredChallenges
    ? advancedCoverage.completedChallenges /
      Math.max(advancedCoverage.requiredChallenges, 1)
    : 1;
  const responseCoverage = agenticChannelResponseCoverage(
    runs,
    contract,
    capabilities,
  );
  const responseDimensionShare = responseCoverage.channels.length
    ? responseCoverage.channels.reduce((total, channel) => {
        const dimensions = [
          channel.adstockFamilies.includes("geometric") &&
            channel.adstockFamilies.includes("weibull"),
          ["Low", "Medium", "High"].every((profile) =>
            channel.saturationProfiles.includes(profile),
          ),
          ["Early", "Middle", "Late"].every((profile) =>
            channel.halfSaturationProfiles.includes(profile),
          ),
          channel.normalizations.includes("peak") &&
            channel.normalizations.includes("sum"),
        ];
        return (
          total +
          dimensions.filter(Boolean).length / Math.max(dimensions.length, 1)
        );
      }, 0) / responseCoverage.channels.length
    : 1;
  const responseShare = responseCoverage.requiredChallenges
    ? 0.5 *
        clamp(
          responseCoverage.completedChallenges /
            Math.max(responseCoverage.requiredChallenges, 1),
          0,
          1,
        ) +
      0.5 * responseDimensionShare
    : 1;
  const structuralCoverage = clamp(
    0.35 * baseCoverage + 0.25 * advancedShare + 0.4 * responseShare,
    0,
    1,
  );
  const localChallenges = completed.filter(
    (run) => run.spec.searchPhase === "local-challenge",
  );
  const preLocalBest = Math.max(
    ...completed
      .filter(
        (run) =>
          run.spec.searchPhase !== "local-challenge" &&
          passesAgenticEligibility(run),
      )
      .map((run) => run.validation?.finalScore ?? -Infinity),
    -Infinity,
  );
  const localImprovements = localChallenges.filter(
    (run) =>
      passesAgenticEligibility(run) &&
      (run.validation?.finalScore ?? -Infinity) > preLocalBest + 0.25,
  ).length;
  const localBudget = agenticLocalChallengeBudget(contract);
  const localConfidence = localBudget
    ? clamp(localChallenges.length / localBudget, 0, 1) *
      (localImprovements ? 0.75 : 1)
    : 0.35;
  const plateauTarget =
    contract.candidateBudget >= 384
      ? 24
      : contract.candidateBudget >= 192
        ? 16
        : 12;
  const plateauConfidence = clamp(
    evaluationsSinceImprovement / plateauTarget,
    0,
    1,
  );
  const boundaryParameters = champion
    ? agenticBoundaryParameters(champion.spec)
    : [];
  const boundaryConfidence = boundaryParameters.length ? 0.35 : 1;
  const restartCount = restartRuns.length;
  const score = champion
    ? Math.round(
        100 *
          (0.3 * structuralCoverage +
            0.3 * restartAgreement +
            0.2 * localConfidence +
            0.1 * plateauConfidence +
            0.1 * boundaryConfidence),
      )
    : 0;
  const level = !champion
    ? "not-established"
    : score >= 80
      ? "high"
      : score >= 60
        ? "medium"
        : "low";
  return {
    level,
    score,
    structuralCoverage,
    restartAgreement,
    restartCount,
    localChallengeCount: localChallenges.length,
    localImprovements,
    evaluationsSinceImprovement,
    boundaryParameters,
    summary: !champion
      ? "No eligible model exists, so search confidence cannot be established."
      : level === "high"
        ? `${champion.spec.searchPhase === "rescue" ? "Evidence rescue, independent starts" : "Independent starts"}, structural coverage, and local challenges support a stable best-known specification.`
        : level === "medium"
          ? "The best-known model is credible, but at least one restart, coverage, plateau, or boundary diagnostic remains incomplete."
          : "The current leader is search-budget dependent; expand the search or resolve disagreement before treating it as stable.",
    restartWinners,
    frontier,
  };
}
