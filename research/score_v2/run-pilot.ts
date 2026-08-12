import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseCsv } from "../../lib/mmm/csv";
import {
  DEFAULT_CONFIG,
  modelFingerprint,
  runModel,
} from "../../lib/mmm/models";
import { createDataset, validateDataset } from "../../lib/mmm/schema";
import { DEFAULT_ADVANCED_CONFIG } from "../../lib/mmm/advanced";
import {
  runModelValidation,
  validationFingerprint,
} from "../../lib/mmm/validation";
import type { ModelConfig } from "../../lib/mmm/types";
import { evaluateCandidateTruth } from "./evaluate";
import {
  DEFAULT_PILOT_SCENARIO,
  PILOT_SCENARIOS,
  scenarioById,
} from "./scenarios";
import {
  generateSyntheticBusiness,
  verifyInjectedTruth,
} from "./simulator";
import type {
  CandidateEvaluation,
  SyntheticBusiness,
  SyntheticScenarioConfig,
} from "./types";

const CHANNEL_RESPONSE_ARCHETYPE: NonNullable<
  ModelConfig["channelResponses"]
> = {
  meta_acquisition_spend: {
    adstockType: "geometric",
    adstock: 0.24,
    saturation: 1.45,
    halfSaturationQuantile: 0.55,
    kernelNormalization: "sum",
  },
  google_search_nonbrand_spend: {
    adstockType: "geometric",
    adstock: 0.08,
    saturation: 1.1,
    halfSaturationQuantile: 0.67,
    kernelNormalization: "sum",
  },
  ctv_spend: {
    adstockType: "weibull",
    weibullShape: 2.1,
    weibullScale: 6.5,
    saturation: 1.05,
    halfSaturationQuantile: 0.53,
    kernelNormalization: "sum",
  },
};

const CANDIDATES: { id: string; config: ModelConfig }[] = [
  {
    id: "short-memory",
    config: { ...DEFAULT_CONFIG, adstock: 0.12, saturation: 0.9 },
  },
  {
    id: "balanced",
    config: { ...DEFAULT_CONFIG, adstock: 0.35, saturation: 1.2 },
  },
  {
    id: "long-memory",
    config: { ...DEFAULT_CONFIG, adstock: 0.72, saturation: 1.2 },
  },
  {
    id: "strong-saturation",
    config: { ...DEFAULT_CONFIG, adstock: 0.3, saturation: 2.2 },
  },
  {
    id: "weibull",
    config: {
      ...DEFAULT_CONFIG,
      adstockType: "weibull",
      weibullShape: 2.2,
      weibullScale: 6,
      saturation: 1.1,
    },
  },
  {
    id: "channel-specific",
    config: {
      ...DEFAULT_CONFIG,
      channelResponses: CHANNEL_RESPONSE_ARCHETYPE,
    },
  },
  {
    id: "channel-specific-long",
    config: {
      ...DEFAULT_CONFIG,
      channelResponses: {
        ...CHANNEL_RESPONSE_ARCHETYPE,
        meta_acquisition_spend: {
          ...CHANNEL_RESPONSE_ARCHETYPE.meta_acquisition_spend,
          adstock: 0.42,
          saturation: 1.85,
        },
        ctv_spend: {
          ...CHANNEL_RESPONSE_ARCHETYPE.ctv_spend,
          weibullScale: 10,
        },
      },
    },
  },
];

async function evaluateScenario(
  scenario: SyntheticScenarioConfig,
): Promise<{ business: SyntheticBusiness; candidates: CandidateEvaluation[] }> {
  const business = generateSyntheticBusiness(scenario);
  verifyInjectedTruth(business);
  const parsed = parseCsv(business.observedCsv);
  const dataset = await createDataset(
    `${scenario.id}.csv`,
    business.observedCsv,
    parsed.columns,
    parsed.rows,
  );
  const contract = validateDataset(dataset);
  if (contract.status === "blocked") {
    throw new Error(
      `${scenario.id} failed the data contract: ${contract.issues
        .map((issue) => issue.title)
        .join(", ")}`,
    );
  }
  const experiments = business.truth.experiments.map((study) => study.experiment);
  const candidates: CandidateEvaluation[] = [];
  for (const evidenceArm of [
    "experiments-only",
    "benchmark-gap-fill",
  ] as const) {
    const industryPriorChannels =
      evidenceArm === "benchmark-gap-fill"
        ? ["google_search_nonbrand_spend"]
        : [];
    for (const candidate of CANDIDATES) {
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
      const validationOptions = {
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
        validationOptions,
      );
      const validation = await runModelValidation(
        dataset,
        model,
        candidate.config,
        DEFAULT_ADVANCED_CONFIG,
        experiments,
        validationId,
        undefined,
        validationOptions,
      );
      candidates.push({
        id: `${candidate.id} · ${evidenceArm}`,
        evidenceArm,
        config: candidate.config,
        model,
        validation,
        ...evaluateCandidateTruth(business, model, candidate.config),
      });
    }
  }
  return { business, candidates };
}

function compactReport(
  scenario: SyntheticScenarioConfig,
  business: SyntheticBusiness,
  candidates: CandidateEvaluation[],
) {
  const totalSpend = business.truth.channels.reduce(
    (total, channel) => total + channel.totalSpend,
    0,
  );
  const benchmarkTruthError = business.truth.channels.reduce(
    (total, channel) =>
      total +
      (channel.totalSpend / Math.max(totalSpend, 1)) *
        Math.abs(
          Math.log(
            Math.max(
              business.truth.industryBenchmarks[channel.channel] ??
                channel.targetRoi,
              1e-6,
            ) / channel.targetRoi,
          ),
        ),
    0,
  );
  return {
    artifactId: `score-v2-${scenario.id}-${scenario.seed}`,
    scenario: {
      id: scenario.id,
      label: scenario.label,
      seed: scenario.seed,
      description: scenario.description,
    },
    truth: {
      channels: business.truth.channels.map((channel) => ({
        channel: channel.channel,
        targetRoi: channel.targetRoi,
        realizedRoi: channel.realizedRoi,
        totalSpend: channel.totalSpend,
        totalContribution: channel.totalContribution,
        marginalRoiAtObserved: channel.marginalRoiAtObserved,
        evidenceId: channel.evidenceId,
        response: scenario.channels.find(
          (candidate) => candidate.channel === channel.channel,
        )?.response,
        delivery: scenario.channels.find(
          (candidate) => candidate.channel === channel.channel,
        )?.delivery,
      })),
      experiments: business.truth.experiments.map((study) => ({
        channel: study.channel,
        design: study.design,
        trueRoi: study.trueRoi,
        observedRoi: study.observedRoi,
        standardError: study.standardError,
      })),
      industryBenchmarks: business.truth.industryBenchmarks,
      weightedLogBenchmarkTruthError: benchmarkTruthError,
      allocation: business.truth.allocation,
    },
    candidates: candidates.map((candidate) => ({
      id: candidate.id,
      evidenceArm: candidate.evidenceArm,
      config: candidate.config,
      fluxScore: candidate.validation.finalScore,
      layers: Object.fromEntries(
        Object.entries(candidate.validation.layers).map(([id, layer]) => [
          id,
          layer.score,
        ]),
      ),
      roi: Object.fromEntries(
        business.truth.channels.map((truth) => [
          truth.channel,
          candidate.model.channels.find(
            (channel) => channel.channel === truth.spendColumn,
          )?.roi ?? null,
        ]),
      ),
      weightedLogRoiError: candidate.weightedLogRoiError,
      weightedLogBenchmarkAgreement:
        candidate.weightedLogBenchmarkAgreement,
      contributionError: candidate.contributionError,
      budgetRegret: candidate.budgetRegret,
      profitRegret: candidate.profitRegret,
      revenueRegret: candidate.revenueRegret,
      recommendedSpend: candidate.recommendedSpend,
      recommendedAdditionalBudget: candidate.recommendedAdditionalBudget,
    })),
  };
}

function markdownReport(report: ReturnType<typeof compactReport>): string {
  const truthRows = report.truth.channels
    .map(
      (channel) =>
        `| ${channel.channel} | ${channel.targetRoi.toFixed(2)}x | ${channel.realizedRoi.toFixed(6)}x |`,
    )
    .join("\n");
  const candidateRows = [...report.candidates]
    .sort((left, right) => (right.fluxScore ?? -1) - (left.fluxScore ?? -1))
    .map(
      (candidate) =>
        `| ${candidate.id} | ${candidate.fluxScore?.toFixed(1) ?? "—"} | ${(candidate.weightedLogRoiError * 100).toFixed(1)}% | ${(candidate.weightedLogBenchmarkAgreement * 100).toFixed(1)}% | ${(candidate.contributionError * 100).toFixed(1)}% | ${(candidate.profitRegret * 100).toFixed(1)}% |`,
    )
    .join("\n");
  return `# Score V2 pilot: ${report.scenario.label}\n\n${report.scenario.description}\n\nSeed: \`${report.scenario.seed}\`. Truth was held out of model fitting. The declared benchmark has ${(report.truth.weightedLogBenchmarkTruthError * 100).toFixed(1)}% spend-weighted log error versus causal truth and is recorded as fallible evidence, not an answer key.\n\n## Injected truth\n\n| Channel | Target ROI | Realized ROI |\n|---|---:|---:|\n${truthRows}\n\n## Candidate results\n\n| Candidate | Current Flux score | ROI truth error | Benchmark agreement error | Contribution error | Profit regret |\n|---|---:|---:|---:|---:|---:|\n${candidateRows}\n\nProfit regret is the share of the simulator-known incremental profit opportunity lost by following the candidate instead of the hidden profit-optimal spend and mix. Benchmark agreement is reported separately from causal truth and is never treated as the answer key. This single scenario is diagnostic evidence, not a learned score or a production claim.\n`;
}

function average(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) /
    Math.max(values.length, 1);
}

function pearson(left: number[], right: number[]): number | null {
  if (left.length !== right.length || left.length < 2) return null;
  const leftMean = average(left);
  const rightMean = average(right);
  const numerator = left.reduce(
    (total, value, index) =>
      total + (value - leftMean) * (right[index] - rightMean),
    0,
  );
  const denominator = Math.sqrt(
    left.reduce((total, value) => total + (value - leftMean) ** 2, 0) *
      right.reduce((total, value) => total + (value - rightMean) ** 2, 0),
  );
  return denominator > 0 ? numerator / denominator : null;
}

function matrixSummary(reports: ReturnType<typeof compactReport>[]) {
  const completeCandidates = reports.flatMap((report) =>
    report.candidates
      .filter((candidate) => candidate.fluxScore !== null)
      .map((candidate) => ({
        scenario: report.scenario.id,
        ...candidate,
      })),
  );
  const scenarioSelections = reports.map((report) => {
    const rankedByFlux = [...report.candidates]
      .filter((candidate) => candidate.fluxScore !== null)
      .sort((left, right) => (right.fluxScore ?? -1) - (left.fluxScore ?? -1));
    const rankedByRegret = [...report.candidates].sort(
      (left, right) => left.budgetRegret - right.budgetRegret,
    );
    const fluxWinner = rankedByFlux[0];
    const regretWinner = rankedByRegret[0];
    return {
      scenario: report.scenario.id,
      fluxWinner: fluxWinner?.id ?? null,
      fluxWinnerScore: fluxWinner?.fluxScore ?? null,
      fluxWinnerRegret: fluxWinner?.budgetRegret ?? null,
      lowestRegretCandidate: regretWinner?.id ?? null,
      lowestAvailableRegret: regretWinner?.budgetRegret ?? null,
      selectedLowestRegret:
        fluxWinner !== undefined &&
        regretWinner !== undefined &&
        Math.abs(fluxWinner.budgetRegret - regretWinner.budgetRegret) < 1e-10,
    };
  });
  const currentWinnerRegrets = scenarioSelections
    .map((item) => item.fluxWinnerRegret)
    .filter((value): value is number => value !== null);
  const bestAvailableRegrets = scenarioSelections
    .map((item) => item.lowestAvailableRegret)
    .filter((value): value is number => value !== null);
  return {
    artifactId: "score-v2-pilot-matrix-v1",
    scenarioCount: reports.length,
    candidateCount: completeCandidates.length,
    correlation: {
      fluxScoreWithNegativeBudgetRegret: pearson(
        completeCandidates.map((candidate) => candidate.fluxScore!),
        completeCandidates.map((candidate) => -candidate.budgetRegret),
      ),
      fluxScoreWithNegativeRoiError: pearson(
        completeCandidates.map((candidate) => candidate.fluxScore!),
        completeCandidates.map((candidate) => -candidate.weightedLogRoiError),
      ),
    },
    selection: {
      meanCurrentWinnerRegret: average(currentWinnerRegrets),
      meanLowestAvailableRegret: average(bestAvailableRegrets),
      lowestRegretSelectionRate: average(
        scenarioSelections.map((item) => (item.selectedLowestRegret ? 1 : 0)),
      ),
    },
    scenarios: scenarioSelections,
  };
}

function matrixMarkdown(summary: ReturnType<typeof matrixSummary>): string {
  const rows = summary.scenarios
    .map(
      (scenario) =>
        `| ${scenario.scenario} | ${scenario.fluxWinner ?? "—"} | ${scenario.fluxWinnerRegret === null ? "—" : `${(scenario.fluxWinnerRegret * 100).toFixed(1)}%`} | ${scenario.lowestRegretCandidate ?? "—"} | ${scenario.lowestAvailableRegret === null ? "—" : `${(scenario.lowestAvailableRegret * 100).toFixed(1)}%`} |`,
    )
    .join("\n");
  const scoreRegret = summary.correlation.fluxScoreWithNegativeBudgetRegret;
  const scoreRoi = summary.correlation.fluxScoreWithNegativeRoiError;
  return `# Score V2 pilot matrix\n\nThis report contains ${summary.candidateCount} candidate evaluations across ${summary.scenarioCount} deterministic synthetic businesses.\n\n- Correlation of current Flux score with lower profit regret: **${scoreRegret === null ? "not estimable" : scoreRegret.toFixed(3)}**\n- Correlation of current Flux score with lower ROI error: **${scoreRoi === null ? "not estimable" : scoreRoi.toFixed(3)}**\n- Mean profit regret of the current Flux winner: **${(summary.selection.meanCurrentWinnerRegret * 100).toFixed(1)}%**\n- Mean profit regret of the best evaluated candidate: **${(summary.selection.meanLowestAvailableRegret * 100).toFixed(1)}%**\n- Current score selected a lowest-profit-regret candidate in **${(summary.selection.lowestRegretSelectionRate * 100).toFixed(1)}%** of scenarios.\n\n| Scenario | Current Flux winner | Winner profit regret | Lowest-regret candidate | Lowest profit regret |\n|---|---|---:|---|---:|\n${rows}\n\nThis is a feasibility diagnostic over a deliberately small candidate set. It is not sufficient to estimate new production weights.\n`;
}

async function writeArtifacts(
  scenario: SyntheticScenarioConfig,
  business: SyntheticBusiness,
  candidates: CandidateEvaluation[],
) {
  const directory = resolve("research/score_v2/artifacts", scenario.id);
  await mkdir(directory, { recursive: true });
  const report = compactReport(scenario, business, candidates);
  await Promise.all([
    writeFile(resolve(directory, "observed.csv"), business.observedCsv),
    writeFile(
      resolve(directory, "truth.json"),
      JSON.stringify(business.truth, null, 2),
    ),
    writeFile(resolve(directory, "report.json"), JSON.stringify(report, null, 2)),
    writeFile(resolve(directory, "report.md"), markdownReport(report)),
  ]);
  return report;
}

const args = process.argv.slice(2);
const scenarios = args.includes("--all")
  ? PILOT_SCENARIOS
  : [
      args.find((argument) => argument.startsWith("--scenario="))
        ? scenarioById(
            args
              .find((argument) => argument.startsWith("--scenario="))!
              .slice("--scenario=".length),
          )
        : DEFAULT_PILOT_SCENARIO,
    ];

const completedReports: ReturnType<typeof compactReport>[] = [];
for (const scenario of scenarios) {
  const { business, candidates } = await evaluateScenario(scenario);
  const report = await writeArtifacts(scenario, business, candidates);
  completedReports.push(report);
  process.stdout.write(`${markdownReport(report)}\n`);
}

if (args.includes("--all")) {
  const summary = matrixSummary(completedReports);
  const directory = resolve("research/score_v2/artifacts");
  await Promise.all([
    writeFile(
      resolve(directory, "matrix-summary.json"),
      JSON.stringify(summary, null, 2),
    ),
    writeFile(resolve(directory, "matrix-summary.md"), matrixMarkdown(summary)),
  ]);
  process.stdout.write(`${matrixMarkdown(summary)}\n`);
}
