import {
  advancedModelFingerprint,
  runAdvancedModel,
} from "../../lib/mmm/advanced";
import type {
  AgenticCandidateRun,
  AgenticCandidateSpec,
} from "../../lib/mmm/agentic";
import { parseCsv } from "../../lib/mmm/csv";
import { modelFingerprint, runModel } from "../../lib/mmm/models";
import { validationDiagnosticValues } from "../../lib/mmm/score-diagnostics";
import { createDataset, validateDataset } from "../../lib/mmm/schema";
import {
  runModelValidation,
  validationFingerprint,
  type ValidationOptions,
} from "../../lib/mmm/validation";
import type { Dataset, Experiment } from "../../lib/mmm/types";
import { evaluateCandidateTruth } from "../score_v2/evaluate";
import { generateSyntheticBusiness } from "../score_v2/simulator";
import type {
  SyntheticBusiness,
  SyntheticScenarioConfig,
} from "../score_v2/types";
import { syntheticIndustryPriorOverrides } from "../score_v6/evidence";
import { SCORE_V6_CANDIDATES } from "../score_v6/cohort";
import type {
  ScoreV6CandidateRow,
  ScoreV6EvidenceArm,
} from "../score_v6/types";

export interface PreparedResearchCandidate {
  row: ScoreV6CandidateRow;
  run: AgenticCandidateRun;
  industryPriorChannels: string[];
}

export interface PreparedResearchBusiness {
  business: SyntheticBusiness;
  dataset: Dataset;
  experiments: Experiment[];
  candidates: PreparedResearchCandidate[];
}

function industryChannels(
  business: SyntheticBusiness,
  arm: ScoreV6EvidenceArm,
): string[] {
  if (arm === "experiments-only") return [];
  const experiments = new Set(
    business.truth.experiments.map((study) =>
      study.experiment.channel.toLowerCase()
    ),
  );
  return business.truth.channels
    .map((channel) => channel.spendColumn)
    .filter((channel) => !experiments.has(channel.toLowerCase()));
}

async function fitCandidate(
  dataset: Dataset,
  spec: AgenticCandidateSpec,
  experiments: Experiment[],
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

export async function prepareMcmcResearchBusiness(
  scenario: SyntheticScenarioConfig,
  family: string,
  split: ScoreV6CandidateRow["split"],
  onProgress?: (detail: string) => void,
  preparationOptions: { includeHiddenTruthLabels?: boolean } = {},
): Promise<PreparedResearchBusiness> {
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
  const candidates: PreparedResearchCandidate[] = [];
  for (const evidenceArm of ["experiments-only", "benchmark-gap-fill"] as const) {
    const activeChannels = industryChannels(business, evidenceArm);
    const validationOptions: ValidationOptions = {
      anchorIndependenceConfirmed: true,
      industryPriorChannels: activeChannels,
      industryPriorOverrides: overrides,
      industryBenchmarkScreeningEnabled: evidenceArm === "benchmark-gap-fill",
      materialSpendShareThreshold: 0.02,
    };
    for (let index = 0; index < SCORE_V6_CANDIDATES.length; index += 1) {
      const baseSpec = SCORE_V6_CANDIDATES[index];
      const candidateId = `${baseSpec.id} · ${evidenceArm}`;
      onProgress?.(`${scenario.id} · ${evidenceArm} · ${index + 1}/${SCORE_V6_CANDIDATES.length}`);
      const spec: AgenticCandidateSpec = {
        ...baseSpec,
        id: candidateId,
        evidencePriorChannels: activeChannels,
      };
      const model = await fitCandidate(
        dataset,
        spec,
        experiments,
        validationOptions,
      );
      const validationId = await validationFingerprint(
        dataset,
        model,
        spec.config,
        spec.advancedConfig,
        experiments,
        validationOptions,
      );
      const validation = await runModelValidation(
        dataset,
        model,
        spec.config,
        spec.advancedConfig,
        experiments,
        validationId,
        undefined,
        validationOptions,
      );
      // Feature refreshes must not read hidden economic truth. Posterior-label
      // jobs retain the historical default and evaluate truth only after fit.
      const truth = preparationOptions.includeHiddenTruthLabels === false
        ? undefined
        : evaluateCandidateTruth(business, model, spec.config);
      const failedGateCount = validation.gates.filter(
        (gate) => gate.applicable && !gate.passed,
      ).length;
      const reviewEligible = failedGateCount === 0;
      const row: ScoreV6CandidateRow = {
        businessId: scenario.id,
        family,
        split,
        candidateId,
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
        decisionLoss: truth?.decisionLoss ?? Number.NaN,
        cappedRegret: truth?.profitRegret ?? Number.NaN,
        roiError: truth?.weightedLogRoiError ?? Number.NaN,
        contributionError: truth?.contributionError ?? Number.NaN,
      };
      candidates.push({
        row,
        run: { spec, state: "complete", model, validation },
        industryPriorChannels: activeChannels,
      });
    }
  }
  return { business, dataset, experiments, candidates };
}
