import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  advancedModelFingerprint,
  DEFAULT_ADVANCED_CONFIG,
  logNormalLaplaceFromMedian,
  runAdvancedModel,
} from "../lib/mmm/advanced";
import {
  createAgenticForcePromotionAudit,
  DEFAULT_AGENTIC_SEARCH_CONTRACT,
  findAgenticBenchmarkRescueRecommendations,
  findAgenticRoiGuardrailViolations,
  passesAgenticEligibility,
  passesApplicableGates,
  rankAgenticCandidates,
  selectAgenticWinner,
} from "../lib/mmm/agentic";
import {
  agenticAdaptiveMinimumBudget,
  agenticAdaptiveMaximumBudget,
  agenticAdvancedChallengeBudget,
  agenticAdvancedCoverage,
  agenticBoundaryParameters,
  agenticCandidateSignature,
  agenticChannelResponseBudget,
  agenticChannelResponseCoverage,
  agenticSeedBudget,
  agenticLocalChallengeBudget,
  agenticSearchConfidence,
  agenticStoppingDecision,
  generateAgenticAdvancedChallenge,
  generateAgenticChannelResponseChallenge,
  generateAgenticLocalChallenge,
  generateAgenticSeeds,
  isAgenticSpecWithinBounds,
  proposeAgenticCandidate,
} from "../lib/mmm/agentic-search";
import {
  inferIndustryPrior,
  industryPriorPercentile,
  isHighlyImprobableIndustryRoi,
} from "../lib/mmm/benchmarks";
import {
  budgetFingerprint,
  DEFAULT_BUDGET_CONTRACT,
  defaultBudgetConstraints,
  runBudgetOptimization,
} from "../lib/mmm/budget";
import { parseCsv } from "../lib/mmm/csv";
import {
  defaultExperimentsForDataset,
  DEFAULT_CONFIG,
  modelFingerprint,
  SAMPLE_EXPERIMENTS,
  runModel,
} from "../lib/mmm/models";
import { modelExperimentWindowRoi } from "../lib/mmm/experiment-window";
import {
  diagonalPenalty,
  matrixVector,
  solveLeastSquares,
} from "../lib/mmm/math";
import {
  ACTIVE_SCORE_CONTRACT,
  HEURISTIC_SCORE_WEIGHTS,
  scoreValidationLayers,
  scoreValidationResult,
} from "../lib/mmm/score-contract";
import { createDataset } from "../lib/mmm/schema";
import {
  compileSamplingModel,
  DEFAULT_SAMPLING_CONTRACT,
  SAMPLING_PRESETS,
  samplingFingerprint,
} from "../lib/mmm/sampling";
import type { SamplingResult } from "../lib/mmm/sampling";
import type { Experiment, ModelResult } from "../lib/mmm/types";
import { refitValidationModel } from "../lib/mmm/validation/causal";
import { fitValidationFold } from "../lib/mmm/validation/adapter";
import {
  assessEvidenceCoherence,
  assessExternalAnchorEligibility,
  rescueIndustryPriorChannels,
  runModelValidation,
  validationFingerprint,
} from "../lib/mmm/validation";
import type { ValidationResult as ModelValidationResult } from "../lib/mmm/validation";
import {
  createWorkspaceCheckpoint,
  isCompatibleWorkspaceCheckpoint,
  WORKSPACE_CHECKPOINT_VERSION,
} from "../lib/mmm/workspace-checkpoint";

function monthlyFixtureCsv(count = 48): string {
  const rows = Array.from({ length: count }, (_, index) => {
    const date = new Date(Date.UTC(2020, index, 1)).toISOString().slice(0, 10);
    const search = 80 + (index % 8) * 17;
    const social = 55 + ((index * 3) % 10) * 11;
    const revenue =
      9_000 + index * 37 + search * 5 + social * 3 + (index % 12) * 21;
    return [date, revenue, search, social, 40 + (index % 6)].join(",");
  });
  return [
    "date,revenue,search_spend,social_spend,control",
    ...rows,
  ].join("\n");
}

test("workspace checkpoints are versioned and reject stale contracts", () => {
  const checkpoint = createWorkspaceCheckpoint(
    "dataset-hash",
    { view: "agentic", promotedId: "C384" },
    "2026-08-15T10:00:00.000Z",
  );
  assert.equal(checkpoint.version, WORKSPACE_CHECKPOINT_VERSION);
  assert.equal(checkpoint.datasetHash, "dataset-hash");
  assert.equal(isCompatibleWorkspaceCheckpoint(checkpoint), true);
  assert.equal(
    isCompatibleWorkspaceCheckpoint({
      ...checkpoint,
      version: "obsolete-workspace-contract",
    }),
    false,
  );
});

test("stable least-squares solver uses QR for regular designs", () => {
  const matrix = [
    [1, 0],
    [1, 1],
    [1, 2],
  ];
  const outcome = [1, 3, 5];
  const solution = solveLeastSquares(matrix, outcome);

  assert.equal(solution.diagnostics.method, "pivoted-qr");
  assert.equal(solution.diagnostics.status, "stable");
  assert.equal(solution.diagnostics.rank, 2);
  assert.ok(Math.abs(solution.coefficients[0] - 1) < 1e-10);
  assert.ok(Math.abs(solution.coefficients[1] - 2) < 1e-10);
  assert.ok(
    matrixVector(matrix, solution.coefficients).every(
      (value, index) => Math.abs(value - outcome[index]) < 1e-10,
    ),
  );
});

test("stable least-squares solver falls back to SVD for duplicate and extreme-scale columns", () => {
  const duplicate = [
    [1, 1, 1],
    [1, 2, 2],
    [1, 3, 3],
    [1, 4, 4],
  ];
  const outcome = [3, 5, 7, 9];
  const rankDeficient = solveLeastSquares(duplicate, outcome);
  const extremeScale = solveLeastSquares(
    duplicate.map(([intercept, value]) => [
      intercept,
      value * 1e9,
      value * 1e-9,
    ]),
    outcome,
  );

  for (const solution of [rankDeficient, extremeScale]) {
    assert.equal(solution.diagnostics.method, "svd");
    assert.equal(solution.diagnostics.status, "rank-deficient");
    assert.ok(solution.coefficients.every(Number.isFinite));
  }
  assert.ok(
    matrixVector(duplicate, rankDeficient.coefficients).every(
      (value, index) => Math.abs(value - outcome[index]) < 1e-8,
    ),
  );
});

test("regularization stabilizes zero-variance and sparse designs", () => {
  const matrix = [
    [1, 0, 0, 0],
    [1, 0, 0, 4],
    [1, 0, 0, 0],
    [1, 0, 0, 8],
  ];
  const solution = solveLeastSquares(matrix, [2, 6, 2, 10], {
    penalties: [
      diagonalPenalty(4, 1, 0.1),
      diagonalPenalty(4, 2, 0.1),
      diagonalPenalty(4, 3, 0.1),
    ],
  });

  assert.equal(solution.diagnostics.rank, 4);
  assert.ok(solution.coefficients.every(Number.isFinite));
  assert.ok(solution.precisionInverse.flat().every(Number.isFinite));
});

test("monthly datasets run natively and keep agentic cycle choices monthly", async () => {
  const rawCsv = monthlyFixtureCsv();
  const parsed = parseCsv(rawCsv);
  const dataset = await createDataset(
    "monthly-advertiser.csv",
    rawCsv,
    parsed.columns,
    parsed.rows,
  );
  const config = { ...DEFAULT_CONFIG, cyclePeriod: 6 };
  const result = await runModel(
    dataset,
    config,
    [],
    "frequentist",
    "monthly-model-test",
  );
  const validation = await runModelValidation(
    dataset,
    result,
    config,
    DEFAULT_ADVANCED_CONFIG,
    [],
    "monthly-validation-test",
  );
  const seeds = generateAgenticSeeds(
    config,
    DEFAULT_ADVANCED_CONFIG,
    DEFAULT_AGENTIC_SEARCH_CONTRACT,
    { likelihoodCalibration: false },
  );

  assert.equal(dataset.modelCadence, "monthly");
  assert.equal(result.predicted.length, dataset.rows.length);
  assert.ok(result.predicted.every(Number.isFinite));
  assert.ok(validation.layers.generalization.tests.length > 0);
  assert.ok(validation.layers.structure.tests.length > 0);
  assert.ok(
    seeds.every((seed) => [3, 6, 12].includes(seed.config.cyclePeriod)),
    "monthly search must never propose weekly cycle periods",
  );
});

test("bundled Robyn experiments never leak into advertiser uploads", () => {
  const robynExperiments = defaultExperimentsForDataset("robyn-demo");
  const advertiserExperiments = defaultExperimentsForDataset("advertiser-upload");

  assert.equal(robynExperiments.length, SAMPLE_EXPERIMENTS.length);
  assert.deepEqual(advertiserExperiments, []);
  assert.notEqual(robynExperiments, SAMPLE_EXPERIMENTS);
  assert.notEqual(robynExperiments[0], SAMPLE_EXPERIMENTS[0]);
});

test("experiment priors materially distinguish the Bayesian model", async () => {
  const rawCsv = await readFile(
    new URL("../public/data/robyn_weekly.csv", import.meta.url),
    "utf8",
  );
  const parsed = parseCsv(rawCsv);
  const dataset = await createDataset(
    "Robyn weekly demo.csv",
    rawCsv,
    parsed.columns,
    parsed.rows,
  );

  const frequentist = await runModel(
    dataset,
    DEFAULT_CONFIG,
    SAMPLE_EXPERIMENTS,
    "frequentist",
    "frequentist-test",
  );
  const bayesian = await runModel(
    dataset,
    DEFAULT_CONFIG,
    SAMPLE_EXPERIMENTS,
    "bayesian",
    "bayesian-test",
  );
  const weibull = await runModel(
    dataset,
    { ...DEFAULT_CONFIG, adstockType: "weibull" },
    SAMPLE_EXPERIMENTS,
    "frequentist",
    "weibull-test",
  );

  assert.notEqual(
    frequentist.r2.toFixed(6),
    bayesian.r2.toFixed(6),
    "Bayesian calibration must change the fitted model",
  );
  assert.equal(
    bayesian.diagnostics.find(
      (diagnostic) => diagnostic.label === "Experiment priors",
    )?.value,
    "2 calibrated",
  );
  assert.notEqual(
    frequentist.r2.toFixed(6),
    weibull.r2.toFixed(6),
    "Weibull and geometric kernels must produce distinct fits",
  );
  assert.equal(frequentist.numerical?.solver, "pivoted-qr");
  assert.equal(bayesian.numerical?.solver, "pivoted-qr");
  assert.equal(
    bayesian.numerical?.rank,
    bayesian.numerical?.parameterCount,
  );
  assert.ok(
    bayesian.numerical?.priorInfluence.every((channel) =>
      Number.isFinite(channel.precisionShare),
    ),
  );
  assert.ok(
    Number.isFinite(
      bayesian.numerical?.clipping.predictionShiftShare ?? Number.NaN,
    ),
  );

  for (const experiment of SAMPLE_EXPERIMENTS) {
    const priorRoi =
      experiment.incrementalOutcome / experiment.incrementalSpend;
    const frequentistRoi = modelExperimentWindowRoi(
      dataset,
      frequentist,
      DEFAULT_CONFIG,
      experiment,
    )?.roi;
    const bayesianRoi = modelExperimentWindowRoi(
      dataset,
      bayesian,
      DEFAULT_CONFIG,
      experiment,
    )?.roi;

    assert.ok(frequentistRoi !== undefined && bayesianRoi !== undefined);
    assert.ok(
      Math.abs(bayesianRoi - priorRoi) <
        Math.abs(frequentistRoi - priorRoi),
      `${experiment.channel} same-window posterior should move toward its ROI prior`,
    );
  }
});

test("temporal validation folds cannot use experiments whose outcome window is still in the future", async () => {
  const rawCsv = await readFile(
    new URL("../public/data/robyn_weekly.csv", import.meta.url),
    "utf8",
  );
  const parsed = parseCsv(rawCsv);
  const dataset = await createDataset(
    "Robyn weekly demo.csv",
    rawCsv,
    parsed.columns,
    parsed.rows,
  );
  const firstExperiment = SAMPLE_EXPERIMENTS[0];
  const trainIndexes = dataset.rows
    .map((row, index) => ({
      index,
      date: Date.parse(String(row[dataset.dateColumn])),
    }))
    .filter(({ date }) => date < Date.parse(firstExperiment.startDate))
    .map(({ index }) => index);
  const testIndexes = Array.from(
    { length: 4 },
    (_, offset) => trainIndexes.at(-1)! + offset + 1,
  );
  const commonSpec = {
    kind: "bayesian" as const,
    config: DEFAULT_CONFIG,
    advancedConfig: DEFAULT_ADVANCED_CONFIG,
    validationOptions: { anchorIndependenceConfirmed: false },
  };
  const withFutureEvidence = fitValidationFold(
    dataset,
    { ...commonSpec, experiments: [firstExperiment] },
    trainIndexes,
    testIndexes,
  );
  const withoutEvidence = fitValidationFold(
    dataset,
    { ...commonSpec, experiments: [] },
    trainIndexes,
    testIndexes,
  );
  assert.deepEqual(withFutureEvidence.predicted, withoutEvidence.predicted);
});

test("production sampling compiles the promoted specification and fingerprints every contract", async () => {
  const rawCsv = await readFile(
    new URL("../public/data/robyn_weekly.csv", import.meta.url),
    "utf8",
  );
  const parsed = parseCsv(rawCsv);
  const dataset = await createDataset(
    "Robyn weekly demo.csv",
    rawCsv,
    parsed.columns,
    parsed.rows,
  );
  const spec = generateAgenticSeeds(
    DEFAULT_CONFIG,
    DEFAULT_ADVANCED_CONFIG,
    DEFAULT_AGENTIC_SEARCH_CONTRACT,
    { likelihoodCalibration: true },
  ).find((candidate) => candidate.family === "bayesian")!;
  const model = await runModel(
    dataset,
    spec.config,
    SAMPLE_EXPERIMENTS,
    "bayesian",
    "sampling-compile-model",
    ["print_S"],
  );
  const validation = {
    finalScore: 83.4,
    gates: [
      {
        id: "sampling-fixture",
        label: "Sampling fixture",
        applicable: true,
        passed: true,
        detail: "",
      },
    ],
  } as ModelValidationResult;
  const compiled = compileSamplingModel(
    dataset,
    { spec, model, validation, state: "complete" },
    SAMPLE_EXPERIMENTS,
    ["print_S"],
  );

  assert.equal(compiled.promotedFamily, "bayesian");
  assert.equal(compiled.matrix.length, dataset.rows.length);
  assert.equal(compiled.media.length, dataset.mediaColumns.length);
  assert.equal(compiled.parameterNames.length, compiled.matrix[0].length);
  assert.equal(compiled.validationEligible, true);
  assert.ok(compiled.media.some((channel) => channel.priorEvidence?.source === "experiment"));
  assert.ok(compiled.media.some((channel) => channel.priorEvidence?.source === "industry"));
  assert.ok(
    compiled.media
      .filter((channel) => channel.priorEvidence)
      .every((channel) => (channel.priorEvidence?.rows.length ?? 0) > 0),
  );
  assert.equal(compiled.calibrations.length, SAMPLE_EXPERIMENTS.length);
  assert.ok(
    compiled.calibrations.every(
      (calibration) =>
        calibration.route === "prior" &&
        calibration.basis === "experiment-window" &&
        calibration.spend > 0 &&
        calibration.spendRows.length > 0 &&
        calibration.outcomeRows.length >= calibration.spendRows.length,
    ),
    "Sampling must calibrate experiment ROI on the declared spend and outcome windows.",
  );
  assert.match(compiled.version, /full-response/);
  assert.equal(compiled.response.adstockType, spec.config.adstockType);
  assert.equal(compiled.response.hillShape, spec.config.saturation);
  assert.ok(compiled.media.every((channel) => channel.rawSpend.length === dataset.rows.length));
  assert.equal("mapPredicted" in compiled, false);
  assert.ok(
    compiled.media.every(
      (channel) =>
        !("mapRoi" in channel) &&
        !("mapLow" in channel) &&
        !("mapHigh" in channel),
    ),
    "Production payloads must label analytic values as screening estimates rather than imply a shared PyMC posterior.",
  );

  const forcedCompiled = compileSamplingModel(
    dataset,
    {
      spec,
      model,
      validation,
      state: "complete",
      roiGuardrailViolations: [
        {
          channel: "search_S",
          roi: 24.41,
          benchmarkLabel: "Google non-brand search",
          percentile: 0.999,
        },
      ],
    },
    SAMPLE_EXPERIMENTS,
    ["print_S"],
  );
  assert.equal(
    forcedCompiled.validationEligible,
    false,
    "A forced ROI override must remain review-only in Production",
  );

  const productionFingerprint = await samplingFingerprint(
    compiled,
    DEFAULT_SAMPLING_CONTRACT,
  );
  assert.equal(
    productionFingerprint,
    await samplingFingerprint(compiled, DEFAULT_SAMPLING_CONTRACT),
  );
  assert.notEqual(
    productionFingerprint,
    await samplingFingerprint(compiled, SAMPLING_PRESETS.robust),
  );
});

test("budget optimizer conserves spend and answers all three planning scenarios", async () => {
  const rawCsv = await readFile(
    new URL("../public/data/robyn_weekly.csv", import.meta.url),
    "utf8",
  );
  const parsed = parseCsv(rawCsv);
  const dataset = await createDataset(
    "Robyn weekly demo.csv",
    rawCsv,
    parsed.columns,
    parsed.rows,
  );
  const spec = generateAgenticSeeds(
    DEFAULT_CONFIG,
    DEFAULT_ADVANCED_CONFIG,
    DEFAULT_AGENTIC_SEARCH_CONTRACT,
    { likelihoodCalibration: true },
  ).find((candidate) => candidate.family === "bayesian")!;
  const model = await runModel(
    dataset,
    spec.config,
    SAMPLE_EXPERIMENTS,
    "bayesian",
    "budget-model",
    ["print_S"],
  );
  const validation = {
    finalScore: 84,
    gates: [
      {
        id: "budget-fixture",
        label: "Budget fixture",
        applicable: true,
        passed: true,
        detail: "",
      },
    ],
  } as ModelValidationResult;
  const run = { spec, model, validation, state: "complete" as const };
  const posteriorSamples = (roi: number, channelIndex: number) =>
    Array.from(
      { length: 128 },
      (_, index) =>
        Math.max(
          0,
          roi *
            (0.82 +
              0.36 * (((index * (channelIndex * 2 + 3)) % 127) / 126)),
        ),
    );
  const sampling = {
    kind: "mcmc",
    fingerprint: "budget-posterior",
    promotedId: spec.id,
    status: "ready",
    channels: model.channels.map((channel, channelIndex) => ({
      channel: channel.channel,
      mapRoi: channel.roi,
      mapLow: channel.roiLow,
      mapHigh: channel.roiHigh,
      posteriorMean: channel.roi,
      posteriorMedian: channel.roi,
      posteriorLow: Math.max(0, channel.roi * 0.65),
      posteriorHigh: channel.roi * 1.35,
      posteriorSamples: posteriorSamples(channel.roi, channelIndex),
      difference: 0,
      uncertaintyRatio: 1,
      materialShift: false,
    })),
    predictive: {
      dates: dataset.rows.map((row) => String(row[dataset.dateColumn])),
      actual: model.actual,
      map: model.predicted,
      median: model.predicted,
      low: model.predicted.map((value) => value * 0.9),
      high: model.predicted.map((value) => value * 1.1),
    },
  } as SamplingResult;
  const fixedContract = {
    ...DEFAULT_BUDGET_CONTRACT,
    constraints: defaultBudgetConstraints(dataset),
  };
  const fixedFingerprint = await budgetFingerprint(
    dataset,
    sampling,
    fixedContract,
  );
  const fixed = await runBudgetOptimization(
    dataset,
    run,
    sampling,
    fixedContract,
    fixedFingerprint,
  );

  assert.equal(fixed.kind, "budget");
  assert.equal(fixed.status, "supported");
  assert.equal(fixed.solver.converged, true);
  assert.equal(fixed.frontier.length, 25);
  assert.equal(fixed.solver.posteriorSamples, 128);
  assert.ok(
    Math.abs(fixed.recommendedBudget - fixed.reference.currentBudget) < 1,
  );
  assert.ok(
    Math.abs(
      fixed.channels.reduce(
        (total, channel) => total + channel.recommendedBudget,
        0,
      ) - fixed.recommendedBudget,
    ) < 1,
  );
  assert.ok(
    fixed.channels.every(
      (channel) =>
        channel.recommendedBudget >= channel.currentBudget * 0.49 &&
        channel.recommendedBudget <= channel.currentBudget * 1.51,
    ),
  );

  const targetContract = {
    ...fixedContract,
    scenario: "target" as const,
    targetOutcome: fixed.expectedTotalOutcome * 0.98,
    targetProbability: 0.5,
  };
  const target = await runBudgetOptimization(
    dataset,
    run,
    sampling,
    targetContract,
    await budgetFingerprint(dataset, sampling, targetContract),
  );
  assert.equal(target.targetFeasible, true);
  assert.ok(target.targetProbability >= 0.5);

  const economicContract = {
    ...fixedContract,
    scenario: "economic" as const,
    grossMargin: 0.4,
  };
  const economic = await runBudgetOptimization(
    dataset,
    run,
    sampling,
    economicContract,
    await budgetFingerprint(dataset, sampling, economicContract),
  );
  assert.ok(economic.recommendedBudget > 0);
  assert.ok(
    economic.recommendedBudget <= economic.reference.currentBudget * 2.51,
  );
  assert.ok(
    economic.frontier.every(
      (point) =>
        point.incrementalLow <= point.incrementalOutcome &&
        point.incrementalOutcome <= point.incrementalHigh,
    ),
  );
  assert.ok(
    economic.frontier.every((point) => {
      const lowProfit =
        point.incrementalLow * economic.contract.grossMargin - point.budget;
      const highProfit =
        point.incrementalHigh * economic.contract.grossMargin - point.budget;
      return lowProfit <= point.profit && point.profit <= highProfit;
    }),
  );
  assert.notEqual(fixed.fingerprint, economic.fingerprint);
});

test("agentic search is diverse, bounded, and ranks gate-passing models first", () => {
  const capabilities = {
    likelihoodCalibration: true,
    mediaColumns: ["tv_S", "search_S", "facebook_S"],
  };
  const candidates = generateAgenticSeeds(
    DEFAULT_CONFIG,
    DEFAULT_ADVANCED_CONFIG,
    DEFAULT_AGENTIC_SEARCH_CONTRACT,
    capabilities,
  );
  assert.equal(
    candidates.length,
    agenticSeedBudget(DEFAULT_AGENTIC_SEARCH_CONTRACT),
  );
  assert.deepEqual(
    new Set(candidates.map((candidate) => candidate.family)),
    new Set(["frequentist", "bayesian", "advanced"]),
  );
  assert.equal(
    new Set(
      candidates.map((candidate) =>
        JSON.stringify({
          family: candidate.family,
          config: candidate.config,
          advanced: candidate.advancedConfig,
        }),
      ),
    ).size,
    candidates.length,
  );
  assert.ok(candidates.every(isAgenticSpecWithinBounds));

  const validation = (
    finalScore: number,
    gatePassed: boolean,
  ): ModelValidationResult =>
    ({
      finalScore,
      gates: [
        {
          id: "test-gate",
          label: "Test gate",
          applicable: true,
          passed: gatePassed,
          detail: "",
        },
      ],
    }) as ModelValidationResult;
  const runs = [
    {
      spec: candidates[0],
      state: "complete" as const,
      validation: validation(94, true),
      roiGuardrailViolations: [
        {
          channel: "search_S",
          roi: 24.41,
          benchmarkLabel: "Google non-brand search",
          percentile: 0.999,
        },
      ],
    },
    {
      spec: candidates[1],
      state: "complete" as const,
      validation: validation(90, false),
    },
    {
      spec: candidates[2],
      state: "complete" as const,
      validation: validation(82, true),
    },
  ];

  assert.equal(passesApplicableGates(runs[0].validation), true);
  assert.equal(passesAgenticEligibility(runs[0]), false);
  assert.equal(passesApplicableGates(runs[1].validation), false);
  assert.equal(rankAgenticCandidates(runs)[0].spec.id, candidates[2].id);
  assert.equal(selectAgenticWinner(runs)?.spec.id, candidates[2].id);

  const forcedRun = {
    ...runs[1],
    roiGuardrailViolations: runs[0].roiGuardrailViolations,
  };
  const forceAudit = createAgenticForcePromotionAudit(
    forcedRun,
    "2026-08-04T12:00:00.000Z",
  );
  assert.equal(passesAgenticEligibility(forcedRun), false);
  assert.equal(forceAudit.type, "forced");
  assert.equal(forceAudit.forcedAt, "2026-08-04T12:00:00.000Z");
  assert.deepEqual(forceAudit.failedGates, [
    { id: "test-gate", label: "Test gate" },
  ]);
  assert.deepEqual(
    forceAudit.roiGuardrailViolations,
    runs[0].roiGuardrailViolations,
  );

  const evaluatedSeeds = candidates.map((spec, index) => ({
    spec,
    state: "complete" as const,
    validation: validation(65 + index, true),
  }));
  const responseChallenges = Array.from(
    {
      length: agenticChannelResponseBudget(
        DEFAULT_AGENTIC_SEARCH_CONTRACT,
        capabilities,
      ),
    },
    (_, index) => {
      const spec = generateAgenticChannelResponseChallenge(
        DEFAULT_CONFIG,
        DEFAULT_ADVANCED_CONFIG,
        DEFAULT_AGENTIC_SEARCH_CONTRACT,
        candidates.length + index + 1,
        capabilities,
      );
      return {
        spec,
        state: "complete" as const,
        validation: validation(72 + (index % 5), true),
      };
    },
  );
  const responseCoverage = agenticChannelResponseCoverage(
    responseChallenges,
    DEFAULT_AGENTIC_SEARCH_CONTRACT,
    capabilities,
  );
  assert.equal(responseCoverage.complete, true);
  assert.ok(responseCoverage.channels.every((channel) => channel.complete));
  assert.ok(
    responseChallenges.every(
      (run) =>
        Object.keys(run.spec.config.channelResponses ?? {}).length ===
        capabilities.mediaColumns.length,
    ),
  );
  const advancedChallenges = Array.from(
    {
      length: agenticAdvancedChallengeBudget(
        DEFAULT_AGENTIC_SEARCH_CONTRACT,
      ),
    },
    (_, index) => {
      const spec = generateAgenticAdvancedChallenge(
        DEFAULT_CONFIG,
        DEFAULT_AGENTIC_SEARCH_CONTRACT,
        [...evaluatedSeeds, ...responseChallenges],
        candidates.length + responseChallenges.length + index + 1,
        capabilities,
      );
      return {
        spec,
        state: "complete" as const,
        validation: validation(80.1, true),
      };
    },
  );
  const coverage = agenticAdvancedCoverage(
    advancedChallenges,
    DEFAULT_AGENTIC_SEARCH_CONTRACT,
    capabilities,
  );
  assert.equal(coverage.complete, true);
  assert.ok(coverage.dimensions.every((dimension) => dimension.complete));
  assert.ok(
    advancedChallenges.every(
      (run) =>
        JSON.stringify(run.spec.config) ===
        JSON.stringify(advancedChallenges[0].spec.config),
    ),
    "paired Advanced challenges must hold the base specification constant",
  );

  const firstAdaptiveNumber =
    candidates.length + responseChallenges.length + advancedChallenges.length + 1;
  const adaptive = proposeAgenticCandidate(
    DEFAULT_CONFIG,
    DEFAULT_ADVANCED_CONFIG,
    DEFAULT_AGENTIC_SEARCH_CONTRACT,
    [...evaluatedSeeds, ...responseChallenges, ...advancedChallenges],
    firstAdaptiveNumber,
    capabilities,
  );
  const repeated = proposeAgenticCandidate(
    DEFAULT_CONFIG,
    DEFAULT_ADVANCED_CONFIG,
    DEFAULT_AGENTIC_SEARCH_CONTRACT,
    [...evaluatedSeeds, ...responseChallenges, ...advancedChallenges],
    firstAdaptiveNumber,
    capabilities,
  );
  assert.equal(adaptive.searchPhase, "adaptive");
  assert.equal(adaptive.proposal.method, "tpe");
  assert.ok(isAgenticSpecWithinBounds(adaptive));
  assert.deepEqual(
    agenticBoundaryParameters({
      family: "bayesian",
      config: {
        ...DEFAULT_CONFIG,
        adstockType: "weibull",
        weibullScale: 12,
        saturation: 0.5,
      },
      advancedConfig: DEFAULT_ADVANCED_CONFIG,
    }),
    ["Weibull scale", "Hill shape"],
  );
  assert.deepEqual(adaptive, repeated, "adaptive proposals must be deterministic");
  assert.ok(
    !new Set(
      [
        ...candidates,
        ...responseChallenges.map((run) => run.spec),
        ...advancedChallenges.map((run) => run.spec),
      ].map(
        agenticCandidateSignature,
      ),
    ).has(
      agenticCandidateSignature(adaptive),
    ),
  );

  const adaptiveFamilies = [0, 1, 2].map(
    (offset) =>
      proposeAgenticCandidate(
        DEFAULT_CONFIG,
        DEFAULT_ADVANCED_CONFIG,
        DEFAULT_AGENTIC_SEARCH_CONTRACT,
        [...evaluatedSeeds, ...responseChallenges, ...advancedChallenges],
        firstAdaptiveNumber + offset,
        capabilities,
      ).family,
  );
  assert.deepEqual(adaptiveFamilies, ["frequentist", "bayesian", "advanced"]);

  const convergedRuns = [
    ...evaluatedSeeds.map((run) => ({
      ...run,
      validation: validation(80, true),
    })),
    ...responseChallenges,
    ...advancedChallenges,
  ];
  const stopping = agenticStoppingDecision(
    convergedRuns,
    DEFAULT_AGENTIC_SEARCH_CONTRACT,
    capabilities,
  );
  assert.equal(
    agenticAdaptiveMinimumBudget(DEFAULT_AGENTIC_SEARCH_CONTRACT, capabilities),
    27,
  );
  assert.equal(
    agenticAdaptiveMaximumBudget(DEFAULT_AGENTIC_SEARCH_CONTRACT, capabilities),
    84,
  );
  assert.equal(agenticLocalChallengeBudget(DEFAULT_AGENTIC_SEARCH_CONTRACT), 24);
  assert.equal(stopping.shouldStop, false);
  assert.match(stopping.reason, /Restart-aware refinement/);

  const adaptivelyRefinedRuns = [...convergedRuns];
  for (
    let index = 0;
    index < agenticAdaptiveMinimumBudget(DEFAULT_AGENTIC_SEARCH_CONTRACT, capabilities);
    index += 1
  ) {
    const spec = proposeAgenticCandidate(
      DEFAULT_CONFIG,
      DEFAULT_ADVANCED_CONFIG,
      DEFAULT_AGENTIC_SEARCH_CONTRACT,
      adaptivelyRefinedRuns,
      firstAdaptiveNumber + index,
      capabilities,
    );
    adaptivelyRefinedRuns.push({
      spec,
      state: "complete" as const,
      validation: validation(80.1, true),
    });
  }
  const refinedStopping = agenticStoppingDecision(
    adaptivelyRefinedRuns,
    DEFAULT_AGENTIC_SEARCH_CONTRACT,
    capabilities,
  );
  assert.equal(refinedStopping.shouldStop, false);
  assert.match(refinedStopping.reason, /Champion-neighborhood testing/);
  assert.equal(
    new Set(
      adaptivelyRefinedRuns
        .filter((run) => run.spec.searchPhase === "adaptive")
        .map((run) => `${run.spec.family}-${run.spec.restart}`),
    ).size,
    9,
    "every family must receive an adaptive trial within each restart",
  );

  for (
    let index = agenticAdaptiveMinimumBudget(DEFAULT_AGENTIC_SEARCH_CONTRACT, capabilities);
    index < agenticAdaptiveMaximumBudget(DEFAULT_AGENTIC_SEARCH_CONTRACT, capabilities);
    index += 1
  ) {
    const spec = proposeAgenticCandidate(
      DEFAULT_CONFIG,
      DEFAULT_ADVANCED_CONFIG,
      DEFAULT_AGENTIC_SEARCH_CONTRACT,
      adaptivelyRefinedRuns,
      firstAdaptiveNumber + index,
      capabilities,
    );
    adaptivelyRefinedRuns.push({
      spec,
      state: "complete" as const,
      validation: validation(80.1, true),
    });
  }
  const firstLocalNumber =
    agenticSeedBudget(DEFAULT_AGENTIC_SEARCH_CONTRACT) +
    agenticChannelResponseBudget(DEFAULT_AGENTIC_SEARCH_CONTRACT, capabilities) +
    agenticAdvancedChallengeBudget(DEFAULT_AGENTIC_SEARCH_CONTRACT) +
    agenticAdaptiveMaximumBudget(DEFAULT_AGENTIC_SEARCH_CONTRACT, capabilities) +
    1;
  for (
    let index = 0;
    index < agenticLocalChallengeBudget(DEFAULT_AGENTIC_SEARCH_CONTRACT);
    index += 1
  ) {
    const spec = generateAgenticLocalChallenge(
      DEFAULT_AGENTIC_SEARCH_CONTRACT,
      adaptivelyRefinedRuns,
      firstLocalNumber + index,
      capabilities,
    );
    assert.equal(spec.searchPhase, "local-challenge");
    assert.ok(spec.challengeOf);
    adaptivelyRefinedRuns.push({
      spec,
      state: "complete" as const,
      validation: validation(80.1, true),
    });
  }
  const finalStopping = agenticStoppingDecision(
    adaptivelyRefinedRuns,
    DEFAULT_AGENTIC_SEARCH_CONTRACT,
    capabilities,
  );
  assert.equal(finalStopping.shouldStop, true);
  assert.match(finalStopping.reason, /Candidate budget reached/);
  const confidence = agenticSearchConfidence(
    adaptivelyRefinedRuns,
    DEFAULT_AGENTIC_SEARCH_CONTRACT,
    capabilities,
  );
  assert.equal(confidence.restartCount, 3);
  assert.equal(confidence.localChallengeCount, 24);
  assert.ok(confidence.structuralCoverage > 0.8);
  assert.equal(confidence.frontier.length, 192);

  const rescuedConfidence = agenticSearchConfidence(
    [
      {
        spec: candidates[0],
        state: "complete" as const,
        validation: validation(74, true),
      },
      {
        spec: {
          ...candidates[0],
          id: `${candidates[0].id}R`,
          searchPhase: "rescue" as const,
          rescueOf: candidates[0].id,
        },
        state: "complete" as const,
        validation: validation(80, true),
      },
    ],
    DEFAULT_AGENTIC_SEARCH_CONTRACT,
    capabilities,
  );
  assert.notEqual(
    rescuedConfidence.level,
    "not-established",
    "an eligible evidence-rescue must not be hidden from the confidence receipt",
  );
  assert.equal(rescuedConfidence.restartCount, 1);

  const incompleteCoverage = agenticStoppingDecision(
    Array.from({ length: 24 }, (_, index) => ({
      spec: {
        ...candidates[index % candidates.length],
        id: `U${index + 1}`,
      },
      state: "complete" as const,
      validation: validation(80, true),
    })),
    DEFAULT_AGENTIC_SEARCH_CONTRACT,
    capabilities,
  );
  assert.equal(incompleteCoverage.shouldStop, false);
  assert.match(incompleteCoverage.reason, /Channel-response coverage/);
});

test("agentic ROI review catches unanchored extremes and preserves experiment anchors", () => {
  const model = {
    channels: [
      { channel: "search_S", roi: 24.411989507726833 },
      { channel: "facebook_S", roi: 25 },
    ],
  } as ModelResult;
  const violations = findAgenticRoiGuardrailViolations(
    model,
    SAMPLE_EXPERIMENTS,
  );

  assert.deepEqual(
    violations.map((violation) => violation.channel),
    ["search_S"],
    "Search should require review while the experiment-backed Facebook channel is exempt",
  );
  assert.ok(violations[0].percentile >= 0.95);
  assert.equal(
    findAgenticRoiGuardrailViolations(model, SAMPLE_EXPERIMENTS, false)
      .length,
    0,
  );

  const lowTail = findAgenticRoiGuardrailViolations(
    {
      channels: [{ channel: "search_S", roi: 0 }],
    } as ModelResult,
    SAMPLE_EXPERIMENTS,
  );
  assert.equal(lowTail.length, 1);
  assert.equal(lowTail[0].tail, "low");
  assert.ok(lowTail[0].percentile <= 0.05);
});

test("agentic evidence rescue pairs channel benchmarks and weak fallbacks without overriding experiments", async () => {
  const rawCsv = await readFile(
    new URL("../public/data/robyn_weekly.csv", import.meta.url),
    "utf8",
  );
  const parsed = parseCsv(rawCsv);
  const dataset = await createDataset(
    "Robyn weekly demo.csv",
    rawCsv,
    parsed.columns,
    parsed.rows,
  );
  const recommendations = findAgenticBenchmarkRescueRecommendations(
    {
      channels: [
        { channel: "search_S", roi: 0.03 },
        { channel: "print_S", roi: 0.11 },
        { channel: "facebook_S", roi: 25 },
      ],
    } as ModelResult,
    SAMPLE_EXPERIMENTS,
    true,
    dataset,
  );

  assert.deepEqual(
    recommendations.map((item) => item.channel),
    ["search_S", "print_S"],
    "material implausible channels should receive paired refits while the experiment-backed channel stays fixed",
  );
  assert.equal(
    recommendations.find((item) => item.channel === "search_S")?.role,
    "channel-benchmark",
  );
  assert.equal(
    recommendations.find((item) => item.channel === "print_S")?.role,
    "weak-fallback",
  );
  assert.deepEqual(
    findAgenticBenchmarkRescueRecommendations(
      { channels: [{ channel: "search_S", roi: 0.03 }] } as ModelResult,
      SAMPLE_EXPERIMENTS,
      true,
      dataset,
      ["search_S"],
    ),
    [],
    "a benchmark already in the candidate contract must not trigger another rescue",
  );
});

test("material zero ROI is capped and routed to a non-circular rescue refit", async () => {
  const rawCsv = await readFile(
    new URL("../public/data/robyn_weekly.csv", import.meta.url),
    "utf8",
  );
  const parsed = parseCsv(rawCsv);
  const dataset = await createDataset(
    "Robyn weekly demo.csv",
    rawCsv,
    parsed.columns,
    parsed.rows,
  );
  const fitted = await runModel(
    dataset,
    DEFAULT_CONFIG,
    SAMPLE_EXPERIMENTS,
    "bayesian",
    "evidence-collapse-fixture",
  );
  const search = fitted.channels.find(
    (channel) => channel.channel.toLowerCase() === "search_s",
  );
  assert.ok(search);
  const collapsed: ModelResult = {
    ...fitted,
    channels: fitted.channels.map((channel) =>
      channel.channel.toLowerCase() === "search_s"
        ? {
            ...channel,
            roi: 0,
            roiLow: 0,
            roiHigh: 43.6,
            contribution: 0,
            contributionShare: 0,
            coefficient: 0,
            priorRoi: undefined,
            priorSource: undefined,
            priorLabel: undefined,
          }
        : channel,
    ),
    numerical: fitted.numerical
      ? {
          ...fitted.numerical,
          clipping: {
            ...fitted.numerical.clipping,
            applied: true,
            material: true,
            severity: "warning",
            channels: [
              ...fitted.numerical.clipping.channels.filter(
                (channel) => channel.channel.toLowerCase() !== "search_s",
              ),
              {
                channel: search.channel,
                unconstrainedCoefficient: -0.1,
                constrainedCoefficient: 0,
                contributionChange: 0,
                contributionChangeShare: 0,
                unconstrainedRoi: -1,
                constrainedRoi: 0,
                roiIntervalMaterial: true,
              },
            ],
          },
        }
      : undefined,
  };
  const assessment = assessEvidenceCoherence(
    dataset,
    collapsed,
    SAMPLE_EXPERIMENTS,
    {
      anchorIndependenceConfirmed: false,
      industryPriorChannels: [],
      industryBenchmarkScreeningEnabled: true,
    },
  );
  const searchAssessment = assessment.channels.find(
    (channel) => channel.channel.toLowerCase() === "search_s",
  );

  assert.equal(searchAssessment?.material, true);
  assert.equal(searchAssessment?.status, "boundary-collapse");
  assert.equal(searchAssessment?.blocking, true);
  assert.ok(
    assessment.causalScoreCap <= 59,
    "a boundary collapse may coexist with a stricter same-window experiment conflict",
  );
  assert.ok(assessment.blockingChannels.includes(search.channel));
  assert.deepEqual(
    rescueIndustryPriorChannels(assessment),
    [search.channel],
  );
  assert.deepEqual(
    rescueIndustryPriorChannels(assessment, [search.channel]),
    [],
    "a benchmark already used for calibration must not trigger a second rescue",
  );
  const validation = await runModelValidation(
    dataset,
    collapsed,
    DEFAULT_CONFIG,
    DEFAULT_ADVANCED_CONFIG,
    SAMPLE_EXPERIMENTS,
    "evidence-collapse-validation",
    undefined,
    {
      anchorIndependenceConfirmed: false,
      industryPriorChannels: [],
      industryBenchmarkScreeningEnabled: true,
    },
  );
  assert.ok(validation.layers.decision.score < 55);
  assert.equal(validation.layers.decision.status, "fail");
  assert.equal(
    validation.gates.find((gate) => gate.id === "evidence-coherence")
      ?.passed,
    false,
  );
});

test("material OOH below its D2C benchmark is routed into evidence rescue", async () => {
  const rawCsv = await readFile(
    new URL("../public/data/robyn_weekly.csv", import.meta.url),
    "utf8",
  );
  const parsed = parseCsv(rawCsv);
  const dataset = await createDataset(
    "Robyn weekly demo.csv",
    rawCsv,
    parsed.columns,
    parsed.rows,
  );
  const fitted = await runModel(
    dataset,
    DEFAULT_CONFIG,
    SAMPLE_EXPERIMENTS,
    "bayesian",
    "ooh-evidence-rescue-fixture",
  );
  const stressed: ModelResult = {
    ...fitted,
    channels: fitted.channels.map((channel) =>
      channel.channel.toLowerCase() === "ooh_s"
        ? {
            ...channel,
            roi: 0.03,
            roiLow: 0,
            roiHigh: 0.13,
            priorRoi: undefined,
            priorSource: undefined,
            priorLabel: undefined,
          }
        : channel,
    ),
  };
  const assessment = assessEvidenceCoherence(
    dataset,
    stressed,
    SAMPLE_EXPERIMENTS,
    {
      anchorIndependenceConfirmed: false,
      industryPriorChannels: [],
      industryBenchmarkScreeningEnabled: true,
    },
  );
  const ooh = assessment.channels.find(
    (channel) => channel.channel.toLowerCase() === "ooh_s",
  );

  assert.equal(ooh?.evidenceLabel, "OOH / digital OOH (D2C ecommerce)");
  assert.equal(ooh?.status, "benchmark-tension");
  assert.equal(ooh?.blocking, true);
  assert.ok(assessment.blockingChannels.includes("ooh_S"));
  assert.ok(rescueIndustryPriorChannels(assessment).includes("ooh_S"));
});

test("industry priors fill only channels without experiment evidence", async () => {
  const rawCsv = await readFile(
    new URL("../public/data/robyn_weekly.csv", import.meta.url),
    "utf8",
  );
  const parsed = parseCsv(rawCsv);
  const dataset = await createDataset(
    "Robyn weekly demo.csv",
    rawCsv,
    parsed.columns,
    parsed.rows,
  );
  const disabledFingerprint = await modelFingerprint(
    dataset,
    DEFAULT_CONFIG,
    SAMPLE_EXPERIMENTS,
    "bayesian",
    false,
  );
  const enabledFingerprint = await modelFingerprint(
    dataset,
    DEFAULT_CONFIG,
    SAMPLE_EXPERIMENTS,
    "bayesian",
    true,
  );
  assert.notEqual(disabledFingerprint, enabledFingerprint);

  const fallbackModel = await runModel(
    dataset,
    DEFAULT_CONFIG,
    SAMPLE_EXPERIMENTS,
    "bayesian",
    enabledFingerprint,
    true,
  );
  assert.equal(
    fallbackModel.channels.find((channel) => channel.channel === "facebook_S")
      ?.priorSource,
    "experiment",
  );
  assert.equal(
    fallbackModel.channels.find((channel) => channel.channel === "tv_S")
      ?.priorSource,
    "experiment",
  );
  const searchFallback = fallbackModel.channels.find(
    (channel) => channel.channel === "search_S",
  );
  assert.equal(searchFallback?.priorSource, "industry");
  assert.equal(searchFallback?.priorRoi, 1.8);
  const searchIdentification = fallbackModel.numerical?.priorInfluence.find(
    (channel) => channel.channel === "search_S",
  );
  assert.equal(searchIdentification?.source, "industry");
  assert.ok(searchIdentification?.sourceLabel);
  assert.ok(
    ["data-led", "data-and-prior", "prior-led"].includes(
      searchIdentification?.classification ?? "",
    ),
  );

  const searchExperiment: Experiment = {
    ...SAMPLE_EXPERIMENTS[0],
    channel: "search_S",
    incrementalOutcome: 70000,
    incrementalSpend: 10000,
    source: "Search geo holdout",
  };
  const experimentModel = await runModel(
    dataset,
    DEFAULT_CONFIG,
    [...SAMPLE_EXPERIMENTS, searchExperiment],
    "bayesian",
    "experiment-precedence",
    true,
  );
  const searchAnchored = experimentModel.channels.find(
    (channel) => channel.channel === "search_S",
  );
  assert.equal(searchAnchored?.priorSource, "experiment");
  assert.ok(Math.abs((searchAnchored?.priorRoi ?? 0) - 7) < 1e-9);
  assert.equal(searchAnchored?.priorLabel, "Search geo holdout");
  assert.equal(
    experimentModel.numerical?.priorInfluence.find(
      (channel) => channel.channel === "search_S",
    )?.source,
    "experiment",
  );
});

test("channel-level guardrails affect only selected improbable media", async () => {
  const rawCsv = await readFile(
    new URL("../public/data/robyn_weekly.csv", import.meta.url),
    "utf8",
  );
  const parsed = parseCsv(rawCsv);
  const dataset = await createDataset(
    "Robyn weekly demo.csv",
    rawCsv,
    parsed.columns,
    parsed.rows,
  );
  const searchBenchmark = inferIndustryPrior("search_S");
  const oohBenchmark = inferIndustryPrior("ooh_S");
  assert.ok(isHighlyImprobableIndustryRoi(14, searchBenchmark));
  assert.ok(industryPriorPercentile(14, searchBenchmark) > 0.95);
  assert.ok(!isHighlyImprobableIndustryRoi(2, searchBenchmark));
  assert.equal(oohBenchmark.id, "ooh-dtc-ecommerce");
  assert.equal(oohBenchmark.matchQuality, "Channel only");
  assert.equal(oohBenchmark.confidence, "Low");
  assert.equal(oohBenchmark.median, 1.5);
  assert.deepEqual([oohBenchmark.low, oohBenchmark.high], [0.5, 4.5]);
  assert.ok(isHighlyImprobableIndustryRoi(0.03, oohBenchmark));

  const fingerprint = await modelFingerprint(
    dataset,
    DEFAULT_CONFIG,
    SAMPLE_EXPERIMENTS,
    "bayesian",
    ["search_S"],
  );
  const selected = await runModel(
    dataset,
    DEFAULT_CONFIG,
    SAMPLE_EXPERIMENTS,
    "bayesian",
    fingerprint,
    ["search_S"],
  );
  assert.equal(
    selected.channels.find((channel) => channel.channel === "search_S")
      ?.priorSource,
    "industry",
  );
  assert.equal(
    selected.channels.find((channel) => channel.channel === "ooh_S")
      ?.priorSource,
    undefined,
  );
  assert.equal(
    selected.channels.find((channel) => channel.channel === "print_S")
      ?.priorSource,
    undefined,
  );
  assert.equal(
    selected.channels.find((channel) => channel.channel === "facebook_S")
      ?.priorSource,
    "experiment",
  );

  const ordered = await modelFingerprint(
    dataset,
    DEFAULT_CONFIG,
    SAMPLE_EXPERIMENTS,
    "bayesian",
    ["search_S", "ooh_S"],
  );
  const reversed = await modelFingerprint(
    dataset,
    DEFAULT_CONFIG,
    SAMPLE_EXPERIMENTS,
    "bayesian",
    ["ooh_S", "search_S"],
  );
  assert.equal(ordered, reversed);
});

test("advanced assumptions produce dynamic, separately calibrated artifacts", async () => {
  const rawCsv = await readFile(
    new URL("../public/data/robyn_weekly.csv", import.meta.url),
    "utf8",
  );
  const parsed = parseCsv(rawCsv);
  const dataset = await createDataset(
    "Robyn weekly demo.csv",
    rawCsv,
    parsed.columns,
    parsed.rows,
  );
  const priorConfig = {
    ...DEFAULT_ADVANCED_CONFIG,
    planningIntensity: true,
    calibrationMode: "prior" as const,
  };
  const likelihoodConfig = {
    ...priorConfig,
    calibrationMode: "likelihood" as const,
  };
  const priorFingerprint = await advancedModelFingerprint(
    dataset,
    DEFAULT_CONFIG,
    priorConfig,
    SAMPLE_EXPERIMENTS,
    ["search_S"],
  );
  const likelihoodFingerprint = await advancedModelFingerprint(
    dataset,
    DEFAULT_CONFIG,
    likelihoodConfig,
    SAMPLE_EXPERIMENTS,
    ["search_S"],
  );
  const prior = await runAdvancedModel(
    dataset,
    DEFAULT_CONFIG,
    priorConfig,
    SAMPLE_EXPERIMENTS,
    priorFingerprint,
    ["search_S"],
  );
  const likelihood = await runAdvancedModel(
    dataset,
    DEFAULT_CONFIG,
    likelihoodConfig,
    SAMPLE_EXPERIMENTS,
    likelihoodFingerprint,
    ["search_S"],
  );

  assert.equal(prior.kind, "advanced");
  assert.notEqual(priorFingerprint, likelihoodFingerprint);
  assert.equal(prior.advanced?.planningIntensity.length, dataset.rows.length);
  assert.equal(
    prior.advanced?.coefficientPaths.length,
    dataset.mediaColumns.length,
  );
  assert.equal(prior.advanced?.calibratedExperiments, SAMPLE_EXPERIMENTS.length);
  assert.equal(
    likelihood.advanced?.calibratedExperiments,
    SAMPLE_EXPERIMENTS.length,
  );
  assert.equal(prior.advanced?.calibratedIndustryPriors, 1);
  assert.equal(prior.advanced?.calibrationReceipts.length, 3);
  assert.equal(
    prior.channels.find((channel) => channel.channel === "search_S")
      ?.priorSource,
    "industry",
  );
  assert.equal(
    prior.channels.find((channel) => channel.channel === "tv_S")?.priorSource,
    "experiment",
  );

  const changingPaths =
    prior.advanced?.coefficientPaths.filter((path) => {
      const rounded = new Set(path.values.map((value) => value.toFixed(6)));
      return rounded.size > 1;
    }) ?? [];
  assert.ok(
    changingPaths.length > 0,
    "time-varying coefficients must produce at least one non-flat path",
  );
  assert.notDeepEqual(
    prior.channels.map((channel) => channel.roi.toFixed(6)),
    likelihood.channels.map((channel) => channel.roi.toFixed(6)),
    "prior and likelihood calibration must not collapse to the same ROI estimates",
  );

  const likelihoodReceipts = likelihood.advanced?.calibrationReceipts ?? [];
  const tvReceipt = likelihoodReceipts.find(
    (receipt) => receipt.channel === "tv_S",
  );
  const benchmarkReceipt = likelihoodReceipts.find(
    (receipt) => receipt.channel === "search_S",
  );
  assert.equal(tvReceipt?.comparisonBasis, "experiment-window");
  assert.ok(
    Math.abs((tvReceipt?.modelRoi ?? 0) - (tvReceipt?.targetRoi ?? 0)) < 0.5,
    `experiment-window ROI must remain comparable to its likelihood anchor: ${JSON.stringify(tvReceipt)}`,
  );
  assert.equal(benchmarkReceipt?.source, "industry");
  assert.equal(benchmarkReceipt?.comparisonBasis, "full-history");
  assert.ok(
    (benchmarkReceipt?.modelRoi ?? 0) >= (benchmarkReceipt?.targetLow ?? Infinity) &&
      (benchmarkReceipt?.modelRoi ?? Infinity) <=
        (benchmarkReceipt?.targetHigh ?? -Infinity),
    `the selected industry prior must replace the weak fallback prior instead of being multiplied by it: ${JSON.stringify(benchmarkReceipt)}`,
  );
  const comparableEvidence = assessEvidenceCoherence(
    dataset,
    likelihood,
    SAMPLE_EXPERIMENTS,
    {
      anchorIndependenceConfirmed: false,
      industryPriorChannels: ["search_S"],
      industryBenchmarkScreeningEnabled: true,
    },
  );
  const tvEvidence = comparableEvidence.channels.find(
    (channel) => channel.channel === "tv_S",
  );
  assert.equal(tvEvidence?.comparisonBasis, "experiment-window");
  assert.notEqual(tvEvidence?.status, "experiment-conflict");

  const searchBenchmark = inferIndustryPrior("search_S");
  const approximation = logNormalLaplaceFromMedian(
    searchBenchmark.median,
    searchBenchmark.low,
    searchBenchmark.high,
  );
  assert.equal(approximation.median, searchBenchmark.median);
  assert.ok(approximation.mode < approximation.median);
  assert.ok(
    Math.abs(
      approximation.mode -
        searchBenchmark.median * Math.exp(-(approximation.logSigma ** 2)),
    ) < 1e-12,
    "the Laplace mode must be derived from the declared log-normal median",
  );
});

test("advanced distribution families change the fit and remain numerically valid", async () => {
  const rawCsv = await readFile(
    new URL("../public/data/robyn_weekly.csv", import.meta.url),
    "utf8",
  );
  const parsed = parseCsv(rawCsv);
  const dataset = await createDataset(
    "Robyn weekly demo.csv",
    rawCsv,
    parsed.columns,
    parsed.rows,
  );
  const configurations = [
    DEFAULT_ADVANCED_CONFIG,
    {
      ...DEFAULT_ADVANCED_CONFIG,
      likelihoodDistribution: "student-t" as const,
      studentTDegreesFreedom: 4,
    },
    {
      ...DEFAULT_ADVANCED_CONFIG,
      priorDistribution: "log-normal" as const,
      likelihoodDistribution: "log-normal" as const,
    },
  ];
  const results = await Promise.all(
    configurations.map(async (advancedConfig) => {
      const fingerprint = await advancedModelFingerprint(
        dataset,
        DEFAULT_CONFIG,
        advancedConfig,
        SAMPLE_EXPERIMENTS,
      );
      return runAdvancedModel(
        dataset,
        DEFAULT_CONFIG,
        advancedConfig,
        SAMPLE_EXPERIMENTS,
        fingerprint,
      );
    }),
  );

  assert.equal(new Set(results.map((result) => result.fingerprint)).size, 3);
  assert.equal(
    new Set(results.map((result) => result.r2.toFixed(6))).size,
    3,
    "distribution families must materially change the fitted model",
  );
  results.forEach((result) => {
    assert.ok(Number.isFinite(result.r2));
    assert.ok(Number.isFinite(result.mape));
    assert.ok(result.predicted.every((value) => Number.isFinite(value)));
    assert.ok(result.predicted.every((value) => value > 0));
    assert.ok(
      result.channels.every(
        (channel) =>
          Number.isFinite(channel.roi) &&
          Number.isFinite(channel.roiLow) &&
          Number.isFinite(channel.roiHigh),
      ),
    );
  });
});

test("validation runs all four layers with deterministic scoring", async () => {
  const rawCsv = await readFile(
    new URL("../public/data/robyn_weekly.csv", import.meta.url),
    "utf8",
  );
  const parsed = parseCsv(rawCsv);
  const dataset = await createDataset(
    "Robyn weekly demo.csv",
    rawCsv,
    parsed.columns,
    parsed.rows,
  );
  const model = await runModel(
    dataset,
    DEFAULT_CONFIG,
    SAMPLE_EXPERIMENTS,
    "bayesian",
    "validation-base-model",
  );
  const fingerprint = await validationFingerprint(
    dataset,
    model,
    DEFAULT_CONFIG,
    DEFAULT_ADVANCED_CONFIG,
    SAMPLE_EXPERIMENTS,
  );
  const validation = await runModelValidation(
    dataset,
    model,
    DEFAULT_CONFIG,
    DEFAULT_ADVANCED_CONFIG,
    SAMPLE_EXPERIMENTS,
    fingerprint,
  );

  assert.equal(validation.kind, "validation");
  assert.equal(validation.modelKind, "bayesian");
  assert.deepEqual(Object.keys(validation.layers).sort(), [
    "causal",
    "decision",
    "generalization",
    "structure",
  ]);
  Object.values(validation.layers).forEach((layer) => {
    assert.ok(Number.isFinite(layer.score));
    assert.ok(layer.score >= 0 && layer.score <= 100);
    assert.ok(layer.tests.length >= 4);
    assert.equal(layer.evidence.kind, layer.id);
  });
  assert.equal(validation.layers.generalization.evidence.kind, "generalization");
  assert.ok(validation.layers.generalization.evidence.folds.length === 3);
  assert.equal(validation.layers.structure.evidence.kind, "structure");
  assert.equal(
    validation.layers.structure.evidence.residuals.length,
    dataset.rows.length,
  );
  assert.equal(validation.layers.causal.evidence.kind, "causal");
  assert.ok(validation.layers.causal.evidence.stability.length > 0);
  assert.equal(
    validation.layers.causal.evidence.anchorAssessment.status,
    "not-testable",
  );
  const anchorGate = validation.gates.find(
    (gate) => gate.id === "external-anchor",
  );
  assert.equal(anchorGate?.applicable, false);
  assert.equal(
    validation.layers.causal.tests.find(
      (item) => item.id === "anchor-recovery",
    )?.status,
    "incomplete",
  );
  assert.equal(validation.layers.decision.evidence.kind, "decision");
  assert.ok(validation.layers.decision.evidence.channels.length > 0);
  assert.ok(validation.layers.decision.tests.length >= 5);
  assert.equal(validation.gates.length, 6);
  assert.ok(
    validation.gates.some((gate) => gate.id === "evidence-coherence"),
  );
  assert.ok(validation.evidenceCoherence.channels.length > 0);
  assert.ok(
    validation.finalScore === null ||
      (Number.isFinite(validation.finalScore) &&
        validation.finalScore >= 0 &&
        validation.finalScore <= 100),
  );
  if (validation.finalScore !== null) {
    const layerScores = Object.fromEntries(
      Object.entries(validation.layers).map(([id, layer]) => [id, layer.score]),
    ) as Record<"generalization" | "structure" | "causal" | "decision", number>;
    const expected = scoreValidationResult(validation.layers);
    assert.ok(Math.abs(validation.finalScore - expected) < 1e-9);
    assert.equal(validation.scoreContract.version, ACTIVE_SCORE_CONTRACT.version);
    assert.equal(validation.scoreContract.kind, "learned");
    assert.ok(validation.heuristicScore !== null);
    assert.ok(
      Math.abs(
        (validation.heuristicScore ?? 0) -
          scoreValidationLayers(layerScores, HEURISTIC_SCORE_WEIGHTS),
      ) < 1e-9,
    );
  }
});

test("causal refits preserve selected channel-level industry guardrails", async () => {
  const rawCsv = await readFile(
    new URL("../public/data/robyn_weekly.csv", import.meta.url),
    "utf8",
  );
  const parsed = parseCsv(rawCsv);
  const dataset = await createDataset(
    "Robyn weekly demo.csv",
    rawCsv,
    parsed.columns,
    parsed.rows,
  );
  const validationOptions = {
    anchorIndependenceConfirmed: false,
    industryPriorChannels: ["print_S"],
  };
  const [bayesianRefit, advancedRefit] = await Promise.all([
    refitValidationModel(
      dataset,
      {
        kind: "bayesian",
        config: DEFAULT_CONFIG,
        advancedConfig: DEFAULT_ADVANCED_CONFIG,
        experiments: SAMPLE_EXPERIMENTS,
        validationOptions,
      },
      SAMPLE_EXPERIMENTS,
      "guarded-bayesian",
    ),
    refitValidationModel(
      dataset,
      {
        kind: "advanced",
        config: DEFAULT_CONFIG,
        advancedConfig: DEFAULT_ADVANCED_CONFIG,
        experiments: SAMPLE_EXPERIMENTS,
        validationOptions,
      },
      SAMPLE_EXPERIMENTS,
      "guarded-advanced",
    ),
  ]);

  [bayesianRefit, advancedRefit].forEach((result) => {
    assert.equal(
      result.channels.find((channel) => channel.channel === "print_S")
        ?.priorSource,
      "industry",
    );
    assert.equal(
      result.channels.find((channel) => channel.channel === "ooh_S")
        ?.priorSource,
      undefined,
    );
    assert.equal(
      result.channels.find((channel) => channel.channel === "tv_S")
        ?.priorSource,
      "experiment",
    );
  });
});

test("external anchor prediction requires comparable independent evidence", () => {
  const singletonAssessment = assessExternalAnchorEligibility(
    SAMPLE_EXPERIMENTS,
  );
  assert.equal(singletonAssessment.status, "not-testable");
  assert.equal(singletonAssessment.qualifiedCount, 0);
  assert.equal(singletonAssessment.scoreWeight, 0);

  const repeatedSameChannel = [
    ...SAMPLE_EXPERIMENTS,
    {
      ...SAMPLE_EXPERIMENTS[0],
      startDate: "2019-01-01",
      endDate: "2019-02-15",
      incrementalOutcome: 46000,
      incrementalSpend: 13000,
      source: "Independent geo holdout",
    },
  ];
  const qualifiedAssessment = assessExternalAnchorEligibility(
    repeatedSameChannel,
    true,
  );
  assert.equal(qualifiedAssessment.status, "qualified");
  assert.equal(qualifiedAssessment.qualifiedCount, 2);
  assert.equal(qualifiedAssessment.scoreWeight, 40);
  const unconfirmedAssessment = assessExternalAnchorEligibility(
    repeatedSameChannel,
  );
  assert.equal(unconfirmedAssessment.status, "needs-confirmation");
  assert.equal(unconfirmedAssessment.scoreWeight, 0);
});
