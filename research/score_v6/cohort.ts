import {
  advancedModelFingerprint,
  DEFAULT_ADVANCED_CONFIG,
  runAdvancedModel,
} from "../../lib/mmm/advanced";
import type { AgenticCandidateSpec } from "../../lib/mmm/agentic";
import {
  generateAgenticChannelResponseChallenge,
  generateAgenticSeeds,
} from "../../lib/mmm/agentic-search";
import { parseCsv } from "../../lib/mmm/csv";
import { DEFAULT_CONFIG, modelFingerprint, runModel } from "../../lib/mmm/models";
import { validationDiagnosticValues } from "../../lib/mmm/score-diagnostics";
import { createDataset, validateDataset } from "../../lib/mmm/schema";
import {
  runModelValidation,
  validationFingerprint,
  type ValidationOptions,
} from "../../lib/mmm/validation";
import { evaluateCandidateTruth } from "../score_v2/evaluate";
import { generateSyntheticBusiness } from "../score_v2/simulator";
import {
  GENERATOR_FAMILIES,
  randomizedAuditScenario,
} from "../score_v3/population";
import type { AuditSplit } from "../score_v3/types";
import { syntheticIndustryPriorOverrides } from "./evidence";
import {
  type ScoreV6CandidateRow,
  type ScoreV6EvidenceArm,
} from "./types";

export const SCORE_V6_BUSINESSES_PER_FAMILY: Record<AuditSplit, number> = {
  train: 60,
  validation: 30,
  audit: 30,
};

const PILOT_SEARCH_CONTRACT = {
  families: { frequentist: false, bayesian: true, advanced: true },
  candidateBudget: 96 as const,
};

function candidateSpecifications(): AgenticCandidateSpec[] {
  const capabilities = {
    likelihoodCalibration: true,
    mediaColumns: [
      "meta_acquisition_spend",
      "google_search_nonbrand_spend",
      "ctv_spend",
    ],
  };
  const seeds = generateAgenticSeeds(
    DEFAULT_CONFIG,
    DEFAULT_ADVANCED_CONFIG,
    PILOT_SEARCH_CONTRACT,
    capabilities,
  );
  const response = Array.from({ length: 12 }, (_, index) =>
    generateAgenticChannelResponseChallenge(
      DEFAULT_CONFIG,
      DEFAULT_ADVANCED_CONFIG,
      PILOT_SEARCH_CONTRACT,
      25 + index,
      capabilities,
    ),
  );
  return [...seeds.slice(0, 12), ...response].map((spec, index) => ({
    ...spec,
    id: `V6-C${String(index + 1).padStart(2, "0")}`,
  }));
}

export const SCORE_V6_CANDIDATES = candidateSpecifications();

function selectedFamilyScenarios() {
  return GENERATOR_FAMILIES.flatMap((family) =>
    Array.from(
      { length: SCORE_V6_BUSINESSES_PER_FAMILY[family.split] },
      (_, index) => ({ family, scenario: randomizedAuditScenario(family, 500 + index) }),
    ),
  );
}

function industryChannels(
  business: ReturnType<typeof generateSyntheticBusiness>,
  arm: ScoreV6EvidenceArm,
): string[] {
  if (arm === "experiments-only") return [];
  const experiments = new Set(
    business.truth.experiments.map((study) => study.experiment.channel.toLowerCase()),
  );
  return business.truth.channels
    .map((channel) => channel.spendColumn)
    .filter((channel) => !experiments.has(channel.toLowerCase()));
}

async function fitCandidate(
  dataset: Awaited<ReturnType<typeof createDataset>>,
  spec: AgenticCandidateSpec,
  experiments: ReturnType<typeof generateSyntheticBusiness>["truth"]["experiments"][number]["experiment"][],
  options: ValidationOptions,
) {
  if (spec.family === "advanced") {
    const fingerprint = await advancedModelFingerprint(
      dataset,
      spec.config,
      spec.advancedConfig,
      experiments,
      options.industryPriorChannels,
      options.industryPriorOverrides,
    );
    return runAdvancedModel(
      dataset,
      spec.config,
      spec.advancedConfig,
      experiments,
      fingerprint,
      options.industryPriorChannels,
      options.industryPriorOverrides,
    );
  }
  const fingerprint = await modelFingerprint(
    dataset,
    spec.config,
    experiments,
    spec.family,
    options.industryPriorChannels,
    options.industryPriorOverrides,
  );
  return runModel(
    dataset,
    spec.config,
    experiments,
    spec.family,
    fingerprint,
    options.industryPriorChannels,
    options.industryPriorOverrides,
  );
}

export async function generateScoreV6Cohort(
  onProgress?: (detail: string) => void,
): Promise<ScoreV6CandidateRow[]> {
  const rows: ScoreV6CandidateRow[] = [];
  const scenarios = selectedFamilyScenarios();
  for (let businessIndex = 0; businessIndex < scenarios.length; businessIndex += 1) {
    const { family, scenario } = scenarios[businessIndex];
    onProgress?.(`${businessIndex + 1}/${scenarios.length} · ${family.label} · ${scenario.id}`);
    const business = generateSyntheticBusiness(scenario);
    const parsed = parseCsv(business.observedCsv);
    const dataset = await createDataset(
      `${scenario.id}.csv`,
      business.observedCsv,
      parsed.columns,
      parsed.rows,
    );
    if (validateDataset(dataset).status === "blocked") {
      throw new Error(`${scenario.id} failed its generated data contract.`);
    }
    const experiments = business.truth.experiments.map((study) => study.experiment);
    const overrides = syntheticIndustryPriorOverrides(business);
    for (const evidenceArm of ["experiments-only", "benchmark-gap-fill"] as const) {
      const industryPriorChannels = industryChannels(business, evidenceArm);
      const options: ValidationOptions = {
        anchorIndependenceConfirmed: true,
        industryPriorChannels,
        industryPriorOverrides: overrides,
        industryBenchmarkScreeningEnabled: evidenceArm === "benchmark-gap-fill",
        materialSpendShareThreshold: 0.02,
      };
      const armRows = await Promise.all(SCORE_V6_CANDIDATES.map(async (spec) => {
        const model = await fitCandidate(dataset, spec, experiments, options);
        const validationId = await validationFingerprint(
          dataset,
          model,
          spec.config,
          spec.advancedConfig,
          experiments,
          options,
        );
        const validation = await runModelValidation(
          dataset,
          model,
          spec.config,
          spec.advancedConfig,
          experiments,
          validationId,
          undefined,
          options,
        );
        const truth = evaluateCandidateTruth(business, model, spec.config);
        const failedGateCount = validation.gates.filter(
          (gate) => gate.applicable && !gate.passed,
        ).length;
        const reviewEligible = failedGateCount === 0;
        return {
          businessId: scenario.id,
          family: family.id,
          split: family.split,
          candidateId: `${spec.id} · ${evidenceArm}`,
          evidenceArm,
          modelFamily: spec.family,
          eligible: validation.eligible,
          reviewEligible,
          eligibilityTier: validation.eligible ? "decision-grade" : "review",
          failedGateCount,
          layerScores: Object.fromEntries(
            Object.entries(validation.layers).map(([id, layer]) => [id, layer.score]),
          ) as ScoreV6CandidateRow["layerScores"],
          diagnostics: validationDiagnosticValues(validation.layers),
          heuristicScore: validation.heuristicScore ?? 0,
          decisionLoss: truth.decisionLoss,
          cappedRegret: truth.profitRegret,
          roiError: truth.weightedLogRoiError,
          contributionError: truth.contributionError,
        } satisfies ScoreV6CandidateRow;
      }));
      rows.push(...armRows);
    }
  }
  return rows;
}
