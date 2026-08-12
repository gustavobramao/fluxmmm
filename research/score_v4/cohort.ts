import { DEFAULT_ADVANCED_CONFIG } from "../../lib/mmm/advanced";
import { parseCsv } from "../../lib/mmm/csv";
import { DEFAULT_CONFIG, modelFingerprint, runModel } from "../../lib/mmm/models";
import { createDataset, validateDataset } from "../../lib/mmm/schema";
import type { ModelConfig } from "../../lib/mmm/types";
import {
  runModelValidation,
  validationFingerprint,
  type ValidationLayerId,
} from "../../lib/mmm/validation";
import { evaluateCandidateTruth } from "../score_v2/evaluate";
import { generateSyntheticBusiness } from "../score_v2/simulator";
import {
  GENERATOR_FAMILIES,
  randomizedAuditScenario,
} from "../score_v3/population";
import type { AuditSplit } from "../score_v3/types";
import type { LearnedScoreCandidateRow } from "./types";

export const SCORE_V4_CANDIDATES: { id: string; config: ModelConfig }[] = [
  {
    id: "geometric-short",
    config: { ...DEFAULT_CONFIG, adstock: 0.12, saturation: 0.9, ridge: 0.08 },
  },
  {
    id: "geometric-balanced",
    config: { ...DEFAULT_CONFIG, adstock: 0.35, saturation: 1.2, ridge: 0.1 },
  },
  {
    id: "geometric-long",
    config: { ...DEFAULT_CONFIG, adstock: 0.72, saturation: 1.2, ridge: 0.14 },
  },
  {
    id: "strong-saturation",
    config: { ...DEFAULT_CONFIG, adstock: 0.3, saturation: 2.2, ridge: 0.12 },
  },
  {
    id: "weibull-fast",
    config: {
      ...DEFAULT_CONFIG,
      adstockType: "weibull",
      weibullShape: 1.4,
      weibullScale: 3,
      saturation: 1.1,
    },
  },
  {
    id: "weibull-delayed",
    config: {
      ...DEFAULT_CONFIG,
      adstockType: "weibull",
      weibullShape: 2.4,
      weibullScale: 8,
      saturation: 1.3,
    },
  },
  {
    id: "channel-typical",
    config: {
      ...DEFAULT_CONFIG,
      channelResponses: {
        meta_acquisition_spend: {
          adstockType: "geometric",
          adstock: 0.22,
          saturation: 1.55,
          halfSaturationQuantile: 0.5,
          kernelNormalization: "sum",
        },
        google_search_nonbrand_spend: {
          adstockType: "geometric",
          adstock: 0.06,
          saturation: 1.05,
          halfSaturationQuantile: 0.65,
          kernelNormalization: "sum",
        },
        ctv_spend: {
          adstockType: "weibull",
          weibullShape: 2.2,
          weibullScale: 7,
          saturation: 1.1,
          halfSaturationQuantile: 0.55,
          kernelNormalization: "sum",
        },
      },
    },
  },
  {
    id: "channel-swapped",
    config: {
      ...DEFAULT_CONFIG,
      channelResponses: {
        meta_acquisition_spend: {
          adstockType: "weibull",
          weibullShape: 2.8,
          weibullScale: 7,
          saturation: 1.8,
          halfSaturationQuantile: 0.45,
          kernelNormalization: "sum",
        },
        google_search_nonbrand_spend: {
          adstockType: "geometric",
          adstock: 0.22,
          saturation: 1.25,
          halfSaturationQuantile: 0.55,
          kernelNormalization: "sum",
        },
        ctv_spend: {
          adstockType: "geometric",
          adstock: 0.18,
          saturation: 1.35,
          halfSaturationQuantile: 0.6,
          kernelNormalization: "sum",
        },
      },
    },
  },
];

export const SCORE_V4_BUSINESSES_PER_FAMILY: Record<AuditSplit, number> = {
  train: 40,
  validation: 20,
  audit: 20,
};

function selectedFamilyScenarios() {
  return GENERATOR_FAMILIES.flatMap((family) =>
    Array.from(
      { length: SCORE_V4_BUSINESSES_PER_FAMILY[family.split] },
      (_, index) => ({ family, scenario: randomizedAuditScenario(family, index) }),
    ),
  );
}

export async function generateLearnedScoreCohort(
  onProgress?: (detail: string) => void,
): Promise<LearnedScoreCandidateRow[]> {
  const rows: LearnedScoreCandidateRow[] = [];
  const scenarios = selectedFamilyScenarios();
  for (let businessIndex = 0; businessIndex < scenarios.length; businessIndex += 1) {
    const { family, scenario } = scenarios[businessIndex];
    onProgress?.(
      `${businessIndex + 1}/${scenarios.length} · ${family.label} · ${scenario.id}`,
    );
    const business = generateSyntheticBusiness(scenario);
    const parsed = parseCsv(business.observedCsv);
    const dataset = await createDataset(
      `${scenario.id}.csv`,
      business.observedCsv,
      parsed.columns,
      parsed.rows,
    );
    const contract = validateDataset(dataset);
    if (contract.status === "blocked") {
      throw new Error(`${scenario.id} failed its generated data contract.`);
    }
    const experiments = business.truth.experiments.map((study) => study.experiment);
    const candidateRows = await Promise.all(
      SCORE_V4_CANDIDATES.flatMap((candidate) =>
        ([
        "experiments-only",
        "benchmark-gap-fill",
        ] as const).map(async (evidenceArm) => {
        const industryPriorChannels =
          evidenceArm === "benchmark-gap-fill"
            ? ["google_search_nonbrand_spend"]
            : [];
        const fingerprint = await modelFingerprint(
          dataset,
          candidate.config,
          experiments,
          "bayesian",
          industryPriorChannels,
        );
        const model = await runModel(
          dataset,
          candidate.config,
          experiments,
          "bayesian",
          fingerprint,
          industryPriorChannels,
        );
        const options = {
          anchorIndependenceConfirmed: true,
          industryPriorChannels,
          industryBenchmarkScreeningEnabled:
            evidenceArm === "benchmark-gap-fill",
          materialSpendShareThreshold: 0.02,
        };
        const validationId = await validationFingerprint(
          dataset,
          model,
          candidate.config,
          DEFAULT_ADVANCED_CONFIG,
          experiments,
          options,
        );
        const validation = await runModelValidation(
          dataset,
          model,
          candidate.config,
          DEFAULT_ADVANCED_CONFIG,
          experiments,
          validationId,
          undefined,
          options,
        );
        const truth = evaluateCandidateTruth(business, model, candidate.config);
        return {
          businessId: scenario.id,
          family: family.id,
          split: family.split,
          candidateId: `${candidate.id} · ${evidenceArm}`,
          evidenceArm,
          eligible: validation.eligible,
          failedGateCount: validation.gates.filter(
            (gate) => gate.applicable && !gate.passed,
          ).length,
          layers: Object.fromEntries(
            Object.entries(validation.layers).map(([id, layer]) => [
              id,
              layer.score,
            ]),
          ) as Record<ValidationLayerId, number>,
          heuristicScore: validation.heuristicScore ?? 0,
          profitRegret: truth.profitRegret,
          roiError: truth.weightedLogRoiError,
          contributionError: truth.contributionError,
        } satisfies LearnedScoreCandidateRow;
      })),
    );
    rows.push(...candidateRows);
  }
  return rows;
}
