import { runAdvancedModel } from "../advanced";
import { toNumber } from "../csv";
import { modelExperimentWindowRoi } from "../experiment-window";
import { mean, normalise } from "../math";
import { runModel } from "../models";
import type {
  ColumnSpec,
  Dataset,
  Experiment,
  ModelResult,
} from "../types";
import type { ValidationModelSpec } from "./adapter";
import { clampScore, correlation, weightedScore } from "./math";
import type {
  CausalEvidence,
  ValidationLayerResult,
  ValidationStatus,
  ValidationTest,
} from "./types";

function statusForScore(score: number): ValidationStatus {
  return score >= 75 ? "pass" : score >= 55 ? "review" : "fail";
}

function test(
  id: string,
  name: string,
  score: number,
  metric: string,
  detail: string,
  importance: ValidationTest["importance"],
  status?: ValidationStatus,
): ValidationTest {
  return {
    id,
    name,
    score: clampScore(score),
    status: status ?? statusForScore(score),
    metric,
    detail,
    importance,
  };
}

function nonOverlappingWindows(a: Experiment, b: Experiment): boolean {
  const aStart = Date.parse(a.startDate);
  const aEnd = Date.parse(a.outcomeEndDate ?? a.endDate);
  const bStart = Date.parse(b.startDate);
  const bEnd = Date.parse(b.outcomeEndDate ?? b.endDate);
  return (
    [aStart, aEnd, bStart, bEnd].every(Number.isFinite) &&
    (aEnd < bStart || bEnd < aStart)
  );
}

export function assessExternalAnchorEligibility(
  experiments: Experiment[],
  independenceConfirmed = false,
): CausalEvidence["anchorAssessment"] & { qualifiedIndexes: number[] } {
  const comparablePair = (requireScope: boolean) =>
    experiments.some((experiment, index) =>
      experiments.some(
        (candidate, candidateIndex) =>
          candidateIndex !== index &&
          candidate.channel.toLowerCase() ===
            experiment.channel.toLowerCase() &&
          nonOverlappingWindows(experiment, candidate) &&
          (!requireScope || candidate.scope === experiment.scope),
      ),
    );
  const repeatedSameChannel = experiments.some((experiment, index) =>
    experiments.some(
      (candidate, candidateIndex) =>
        candidateIndex !== index &&
        candidate.channel.toLowerCase() ===
          experiment.channel.toLowerCase(),
    ),
  );
  const nonOverlapping = comparablePair(false);
  const matchingScope = comparablePair(true);
  const qualifiedIndexes = experiments
    .map((experiment, index) => ({ experiment, index }))
    .filter(({ experiment, index }) =>
      experiments.some(
        (candidate, candidateIndex) =>
          candidateIndex !== index &&
          candidate.channel.toLowerCase() ===
            experiment.channel.toLowerCase() &&
          candidate.scope === experiment.scope &&
          nonOverlappingWindows(experiment, candidate),
      ),
    )
    .map(({ index }) => index);
  const status = !qualifiedIndexes.length
    ? "not-testable"
    : independenceConfirmed
      ? "qualified"
      : "needs-confirmation";
  const summary =
    status === "qualified"
      ? `${qualifiedIndexes.length} anchor${qualifiedIndexes.length === 1 ? "" : "s"} can be withheld while comparable same-channel evidence remains in calibration.`
      : status === "needs-confirmation"
        ? `${qualifiedIndexes.length} structurally eligible anchor${qualifiedIndexes.length === 1 ? "" : "s"} found. Confirm that the holdout never influenced prior tuning, model selection, or assumption choices.`
    : experiments.length === 0
      ? "No experiments are available for external anchor prediction."
      : repeatedSameChannel
        ? "Repeated anchors exist, but none have non-overlapping windows with the same ROI scope."
        : "Each calibrated channel has only one anchor; removing it would also remove the prior used to identify that channel.";
  return {
    status,
    qualifiedCount: qualifiedIndexes.length,
    totalCount: experiments.length,
    summary,
    scoreWeight: status === "qualified" ? 40 : 0,
    qualifiedIndexes,
    criteria: [
      {
        label: "Repeated same-channel evidence",
        state: repeatedSameChannel ? "pass" : "fail",
        detail: repeatedSameChannel
          ? "At least one channel has multiple experimental estimates."
          : "A channel needs at least two experiments so one can remain in calibration.",
      },
      {
        label: "Non-overlapping windows",
        state: nonOverlapping ? "pass" : "fail",
        detail: nonOverlapping
          ? "A candidate holdout does not overlap its calibration counterpart."
          : "Overlapping experiments are not treated as independent validation evidence.",
      },
      {
        label: "Comparable ROI scope",
        state: matchingScope ? "pass" : "fail",
        detail: matchingScope
          ? "At least one independent pair measures the same immediate or total effect."
          : "Immediate and total ROI estimates are not compared as though they were the same estimand.",
      },
      {
        label: "Pipeline independence",
        state: independenceConfirmed ? "pass" : "confirm",
        detail: independenceConfirmed
          ? "The modeler confirmed that the holdout did not influence prior tuning, model selection, or assumption choices."
          : "Confirm that the holdout did not influence prior tuning, model selection, or assumption choices.",
      },
    ],
  };
}

export async function refitValidationModel(
  dataset: Dataset,
  spec: ValidationModelSpec,
  experiments: Experiment[],
  suffix: string,
): Promise<ModelResult> {
  const fingerprint = `validation-${spec.kind}-${suffix}-${dataset.hash}`;
  const industryPriorChannels =
    spec.validationOptions.industryPriorChannels ?? [];
  if (spec.kind === "advanced") {
    return runAdvancedModel(
      dataset,
      spec.config,
      spec.advancedConfig,
      experiments,
      fingerprint,
      industryPriorChannels,
      spec.validationOptions.industryPriorOverrides,
    );
  }
  return runModel(
    dataset,
    spec.config,
    experiments,
    spec.kind,
    fingerprint,
    industryPriorChannels,
    spec.validationOptions.industryPriorOverrides,
  );
}

function truncatedDataset(dataset: Dataset, length: number): Dataset {
  const rows = dataset.rows.slice(0, length);
  return {
    ...dataset,
    name: `${dataset.name} through ${String(rows.at(-1)?.[dataset.dateColumn] ?? length)}`,
    rows,
    hash: `${dataset.hash}-through-${length}`,
  };
}

function roiStabilityScore(
  reference: ModelResult,
  comparisons: ModelResult[],
  dataset: Dataset,
): {
  score: number;
  medianDrift: number;
  weightedDrift: number;
  tailDrift: number;
} {
  const totalSpendByChannel = new Map(
    dataset.mediaColumns.map((channel) => [
      channel.toLowerCase(),
      dataset.rows.reduce(
        (total, row) => total + Math.max(0, toNumber(row[channel])),
        0,
      ),
    ]),
  );
  const totalSpend = [...totalSpendByChannel.values()].reduce(
    (total, value) => total + value,
    0,
  );
  const channelDrifts = reference.channels.map((channel) => {
    const drifts = comparisons.flatMap((comparison) => {
      const other = comparison.channels.find(
        (candidate) => candidate.channel === channel.channel,
      );
      if (!other) return [];
      return [
        Math.abs(other.roi - channel.roi) /
          Math.max(Math.abs(channel.roi), 0.1),
      ];
    });
    return {
      channel: channel.channel,
      drift: mean(drifts.length ? drifts : [1]),
      weight:
        (totalSpendByChannel.get(channel.channel.toLowerCase()) ?? 0) /
        Math.max(totalSpend, 1),
    };
  });
  const sorted = channelDrifts.map((item) => item.drift).sort((a, b) => a - b);
  const medianDrift = sorted[Math.floor(sorted.length / 2)] ?? 1;
  const tailDrift =
    sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.8))] ?? 1;
  const weightTotal = channelDrifts.reduce(
    (total, channel) => total + channel.weight,
    0,
  );
  const weightedDrift =
    channelDrifts.reduce(
      (total, channel) => total + channel.drift * channel.weight,
      0,
    ) / Math.max(weightTotal, 1e-9);
  const decisionDrift = 0.5 * weightedDrift + 0.5 * tailDrift;
  return {
    score: clampScore(100 - decisionDrift * 170),
    medianDrift,
    weightedDrift,
    tailDrift,
  };
}

function confounderDataset(
  dataset: Dataset,
  result: ModelResult,
  strength: number,
): Dataset {
  const name = "__latent_confounder__";
  const totalSpend = dataset.rows.map((row) =>
    dataset.mediaColumns.reduce(
      (total, column) => total + Math.max(0, toNumber(row[column])),
      0,
    ),
  );
  const spendSignal = normalise(totalSpend);
  const outcomeSignal = normalise(result.residuals);
  const remaining = Math.sqrt(Math.max(1 - 2 * strength ** 2, 0));
  const values = dataset.rows.map(
    (_, index) =>
      strength * spendSignal[index] +
      strength * outcomeSignal[index] +
      remaining * Math.sin((2 * Math.PI * index) / 17),
  );
  const specs: ColumnSpec[] = [
    {
      name,
      role: "control",
      numeric: true,
    },
    ...dataset.specs.filter((column) => column.name !== name),
  ];
  return {
    ...dataset,
    name: `${dataset.name} confounder stress ${strength}`,
    rows: dataset.rows.map((row, index) => ({
      ...row,
      [name]: values[index],
    })),
    columns: [name, ...dataset.columns.filter((column) => column !== name)],
    specs,
    controlColumns: [
      name,
      ...dataset.controlColumns.filter((column) => column !== name).slice(0, 3),
    ],
    hash: `${dataset.hash}-confounder-${strength}`,
  };
}

export async function runCausalValidation(
  dataset: Dataset,
  spec: ValidationModelSpec,
  result: ModelResult,
): Promise<ValidationLayerResult> {
  const anchorAssessment = assessExternalAnchorEligibility(
    spec.experiments,
    spec.validationOptions.anchorIndependenceConfirmed,
  );
  const qualifiedIndexSet = new Set(
    anchorAssessment.status === "qualified"
      ? anchorAssessment.qualifiedIndexes
      : [],
  );
  const anchorComparisons = await Promise.all(
    spec.experiments.map(async (experiment, index) => {
      if (!qualifiedIndexSet.has(index)) return null;
      const heldOut = spec.experiments.filter(
        (_, experimentIndex) => experimentIndex !== index,
      );
      const recovered = await refitValidationModel(
        dataset,
        spec,
        heldOut,
        `anchor-${index}`,
      );
      const channel = recovered.channels.find(
        (candidate) =>
          candidate.channel.toLowerCase() ===
          experiment.channel.toLowerCase(),
      );
      if (!channel) return null;
      const observed =
        experiment.incrementalOutcome /
        Math.max(experiment.incrementalSpend, 1);
      const comparable = modelExperimentWindowRoi(
        dataset,
        recovered,
        spec.config,
        experiment,
      );
      if (!comparable) return null;
      const modelRoi = comparable.roi;
      const modelLow = comparable.low;
      const modelHigh = comparable.high;
      const modelSe =
        Math.max(modelHigh - modelLow, 0) / (2 * 1.96);
      const combinedSe = Math.sqrt(
        modelSe ** 2 + experiment.standardError ** 2,
      );
      const standardizedError =
        Math.abs(modelRoi - observed) / Math.max(combinedSe, 1e-6);
      const relativeError =
        Math.abs(modelRoi - observed) / Math.max(Math.abs(observed), 0.1);
      return {
        channel: experiment.channel,
        experimentRoi: observed,
        experimentLow: observed - 1.96 * experiment.standardError,
        experimentHigh: observed + 1.96 * experiment.standardError,
        modelRoi,
        modelLow,
        modelHigh,
        standardizedError,
        relativeError,
        covered:
          observed >= modelLow && observed <= modelHigh,
        comparisonBasis: comparable.basis,
        comparisonLabel: comparable.label,
      };
    }),
  );
  const usableAnchors = anchorComparisons.filter(
    (value): value is NonNullable<typeof value> => Boolean(value),
  );
  const anchorScore = usableAnchors.length
    ? clampScore(
        100 -
          mean(
            usableAnchors.map(
              (anchor) =>
                anchor.standardizedError * 18 +
                anchor.relativeError * 28,
            ),
          ),
      )
    : 0;
  const anchorCoverage = usableAnchors.length
    ? usableAnchors.filter((anchor) => anchor.covered).length /
      usableAnchors.length
    : 0;

  const lengths = [0.7, 0.85].map((fraction) =>
    Math.max(
      dataset.modelCadence === "monthly" ? 24 : 52,
      Math.floor(dataset.rows.length * fraction),
    ),
  );
  const snapshots = await Promise.all(
    lengths.map((length, index) => {
      const snapshot = truncatedDataset(dataset, length);
      const endDate = Date.parse(
        String(snapshot.rows.at(-1)?.[snapshot.dateColumn] ?? ""),
      );
      const evidence = spec.experiments.filter(
        (experiment) =>
          !Number.isFinite(endDate) ||
          Date.parse(experiment.outcomeEndDate ?? experiment.endDate) <= endDate,
      );
      return refitValidationModel(
        snapshot,
        spec,
        evidence,
        `snapshot-${index}`,
      );
    }),
  );
  const stability = roiStabilityScore(result, snapshots, dataset);

  const confounderStrengths = [0.2, 0.4, 0.6];
  const confounderFits = await Promise.all(
    confounderStrengths.map((strength) =>
      refitValidationModel(
        confounderDataset(dataset, result, strength),
        spec,
        spec.experiments,
        `confounder-${strength}`,
      ),
    ),
  );
  const confounderStability = confounderFits.map((fit) =>
    roiStabilityScore(result, [fit], dataset),
  );
  const firstFragileIndex = confounderStability.findIndex(
    (item) => item.tailDrift >= 0.25,
  );
  const fragileStrength =
    firstFragileIndex >= 0
      ? confounderStrengths[firstFragileIndex]
      : 0.6;
  const confounderScore = clampScore(
    mean(
      confounderStability.map(
        (item, index) =>
          100 -
          (0.5 * item.weightedDrift + 0.5 * item.tailDrift) *
            (index === 0 ? 260 : index === 1 ? 160 : 100),
      ),
    ),
  );

  const totalSpend = dataset.rows.map((row) =>
    dataset.mediaColumns.reduce(
      (total, column) => total + Math.max(0, toNumber(row[column])),
      0,
    ),
  );
  const placebo = [1, 4, 8]
    .filter((lead) => lead < result.residuals.length / 3)
    .map((lead) => ({
      lead,
      correlation: correlation(
        result.residuals.slice(0, -lead),
        totalSpend.slice(lead),
      ),
    }));
  const maximumPlaceboCorrelation = Math.max(
    ...placebo.map((point) => Math.abs(point.correlation)),
    0,
  );
  const placeboScore = clampScore(
    100 - maximumPlaceboCorrelation * 220,
  );

  const tests: ValidationTest[] = [
    usableAnchors.length
      ? test(
          "anchor-recovery",
          "External anchor prediction",
          anchorScore,
          `${Math.round(anchorCoverage * 100)}% interval coverage`,
          `${usableAnchors.length} qualified anchor${usableAnchors.length === 1 ? " was" : "s were"} withheld while non-overlapping, same-channel, same-scope evidence remained in calibration.`,
          "critical",
        )
      : test(
          "anchor-recovery",
          "External anchor prediction",
          0,
          `${anchorAssessment.qualifiedCount}/${anchorAssessment.totalCount} qualify`,
          `${anchorAssessment.summary} This test is excluded from the numerical causal score.`,
          "supporting",
          "incomplete",
        ),
    test(
      "new-data-stability",
      "Sensitivity to new data",
      stability.score,
      `${Math.round(stability.weightedDrift * 100)}% spend-weighted · ${Math.round(stability.tailDrift * 100)}% P80 drift`,
      "Compares channel ROI from 70%, 85%, and the full history. Spend-weighted and tail movement prevent one collapsed material channel from hiding behind a good median.",
      "critical",
    ),
    test(
      "confounder-sensitivity",
      "Unobserved-confounder stress",
      confounderScore,
      `25% tail drift at ${firstFragileIndex >= 0 ? fragileStrength.toFixed(1) : ">0.6"} strength`,
      "Adds a latent control correlated with both media pressure and unexplained outcome, then combines spend-weighted and vulnerable-tail ROI movement.",
      "critical",
    ),
    test(
      "future-media-placebo",
      "Future-media placebo",
      placeboScore,
      `max |correlation| ${maximumPlaceboCorrelation.toFixed(2)}`,
      "Future spend should not explain current residuals. A strong relationship suggests unresolved demand anticipation or timing bias.",
      "high",
    ),
  ];
  const score = weightedScore([
    ...(usableAnchors.length
      ? [{ score: anchorScore, weight: 40 }]
      : []),
    { score: stability.score, weight: 20 },
    { score: confounderScore, weight: 25 },
    { score: placeboScore, weight: 15 },
  ]);
  const status = statusForScore(score);
  const channelSpend = new Map(
    dataset.mediaColumns.map((channel) => [
      channel.toLowerCase(),
      dataset.rows.reduce(
        (total, row) => total + Math.max(0, toNumber(row[channel])),
        0,
      ),
    ]),
  );
  const totalChannelSpend = [...channelSpend.values()].reduce(
    (total, value) => total + value,
    0,
  );
  const materialEvidenceChannels = [...result.channels]
    .filter(
      (channel) =>
        (channelSpend.get(channel.channel.toLowerCase()) ?? 0) /
          Math.max(totalChannelSpend, 1) >=
        0.02,
    )
    .sort(
      (left, right) =>
        (channelSpend.get(right.channel.toLowerCase()) ?? 0) -
        (channelSpend.get(left.channel.toLowerCase()) ?? 0),
    );
  return {
    id: "causal",
    score,
    status,
    summary:
      !usableAnchors.length
        ? `External anchor prediction is not testable for this evidence set. The ${Math.round(score)}/100 causal score uses stability, confounder stress, and placebo diagnostics only.`
        : status === "pass"
        ? "External anchors and sensitivity tests support the model’s causal use."
        : status === "review"
          ? "Causal conclusions are usable only with the reported sensitivity warnings."
          : "Causal recovery or robustness is too weak for decision-grade ROI.",
    tests,
    evidence: {
      kind: "causal",
      anchorAssessment: {
        status: anchorAssessment.status,
        qualifiedCount: anchorAssessment.qualifiedCount,
        totalCount: anchorAssessment.totalCount,
        summary: anchorAssessment.summary,
        scoreWeight: anchorAssessment.scoreWeight,
        criteria: anchorAssessment.criteria,
      },
      anchors: usableAnchors.map((anchor) => ({
        channel: anchor.channel,
        experimentRoi: anchor.experimentRoi,
        experimentLow: anchor.experimentLow,
        experimentHigh: anchor.experimentHigh,
        modelRoi: anchor.modelRoi,
        modelLow: anchor.modelLow,
        modelHigh: anchor.modelHigh,
        covered: anchor.covered,
        comparisonBasis: anchor.comparisonBasis,
        comparisonLabel: anchor.comparisonLabel,
      })),
      stability: materialEvidenceChannels
        .map((channel) => ({
          channel: channel.channel,
          points: [
            ...snapshots.map((snapshot, index) => ({
              historyShare: lengths[index] / dataset.rows.length,
              roi:
                snapshot.channels.find(
                  (candidate) =>
                    candidate.channel === channel.channel,
                )?.roi ?? channel.roi,
            })),
            { historyShare: 1, roi: channel.roi },
          ],
        })),
      confounders: materialEvidenceChannels
        .map((channel) => ({
          channel: channel.channel,
          points: [
            { strength: 0, normalizedRoi: 1 },
            ...confounderFits.map((fit, index) => ({
              strength: confounderStrengths[index],
              normalizedRoi:
                (fit.channels.find(
                  (candidate) =>
                    candidate.channel === channel.channel,
                )?.roi ?? channel.roi) /
                Math.max(Math.abs(channel.roi), 0.1),
            })),
          ],
        })),
      placebo,
    },
  };
}
