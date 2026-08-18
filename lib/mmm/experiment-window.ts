import { toNumber } from "./csv";
import { responseForChannel, responseTransform } from "./response";
import type {
  Dataset,
  Experiment,
  ModelConfig,
  ModelResult,
} from "./types";

export interface ExperimentWindowIndexes {
  spendRows: number[];
  outcomeRows: number[];
}

export interface ExperimentWindowRoiEstimate {
  roi: number;
  low: number;
  high: number;
  basis: "experiment-window";
  label: string;
  spendRows: number;
  outcomeRows: number;
}

function indexesBetween(
  dataset: Dataset,
  startDate: string,
  endDate: string,
): number[] {
  const start = Date.parse(startDate);
  const end = Date.parse(endDate);
  return dataset.rows
    .map((row, index) => ({
      index,
      date: Date.parse(String(row[dataset.dateColumn])),
    }))
    .filter(
      ({ date }) =>
        Number.isFinite(date) &&
        (!Number.isFinite(start) || date >= start) &&
        (!Number.isFinite(end) || date <= end),
    )
    .map(({ index }) => index);
}

export function experimentWindowIndexes(
  dataset: Dataset,
  experiment: Experiment,
): ExperimentWindowIndexes | undefined {
  const availableDates = dataset.rows
    .map((row) => Date.parse(String(row[dataset.dateColumn])))
    .filter(Number.isFinite);
  const availableStart = Math.min(...availableDates);
  const availableEnd = Math.max(...availableDates);
  const campaignStart = Date.parse(experiment.startDate);
  const campaignEnd = Date.parse(experiment.endDate);
  const outcomeEnd = Date.parse(
    experiment.outcomeEndDate ?? experiment.endDate,
  );
  if (
    !availableDates.length ||
    ![campaignStart, campaignEnd, outcomeEnd].every(Number.isFinite) ||
    campaignStart < availableStart ||
    campaignEnd < campaignStart ||
    campaignEnd > availableEnd ||
    outcomeEnd < campaignEnd ||
    outcomeEnd > availableEnd
  ) {
    return undefined;
  }
  const spendRows = indexesBetween(
    dataset,
    experiment.startDate,
    experiment.endDate,
  );
  if (!spendRows.length) return undefined;
  const outcomeRows = indexesBetween(
    dataset,
    experiment.startDate,
    experiment.outcomeEndDate ?? experiment.endDate,
  );
  if (!outcomeRows.length) return undefined;
  return { spendRows, outcomeRows };
}

function sumAt(values: number[], indexes: number[]): number {
  return indexes.reduce(
    (total, index) => total + Math.max(0, values[index] ?? 0),
    0,
  );
}

/**
 * Returns the transformed-media contrast attributable to spend inside the
 * declared experiment window. The counterfactual removes only that window's
 * spend, preserves the fitted response contract, and measures any remaining
 * carryover through outcomeEndDate.
 */
export function experimentResponseContrast(
  dataset: Dataset,
  config: ModelConfig,
  channel: string,
  experiment: Experiment,
): {
  spend: number;
  deltaTransformed: number[];
  spendRows: number[];
  outcomeRows: number[];
} | undefined {
  const mediaColumn = dataset.mediaColumns.find(
    (candidate) => candidate.toLowerCase() === channel.toLowerCase(),
  );
  if (!mediaColumn) return undefined;
  const windows = experimentWindowIndexes(dataset, experiment);
  if (!windows) return undefined;
  const spend = dataset.rows.map((row) =>
    Math.max(0, toNumber(row[mediaColumn])),
  );
  const testedSpend = sumAt(spend, windows.spendRows);
  if (testedSpend <= 0) return undefined;
  const response = responseForChannel(config, mediaColumn);
  const actual = responseTransform(spend, response);
  const spendRowSet = new Set(windows.spendRows);
  const counterfactualSpend = spend.map((value, index) =>
    spendRowSet.has(index) ? 0 : value,
  );
  const counterfactual = responseTransform(
    counterfactualSpend,
    response,
    actual.halfSaturation,
  );
  return {
    spend: testedSpend,
    deltaTransformed: actual.transformed.map(
      (value, index) => value - counterfactual.transformed[index],
    ),
    spendRows: windows.spendRows,
    outcomeRows: windows.outcomeRows,
  };
}

function scaledInterval(
  roi: number,
  fullRoi: number,
  fullLow: number,
  fullHigh: number,
  responseScale: number,
  fullResponseScale: number,
): { low: number; high: number } {
  const fullSe = Math.max(fullHigh - fullLow, 0) / (2 * 1.96);
  const ratio = Math.abs(fullRoi) > 1e-9
    ? Math.abs(roi / fullRoi)
    : Math.abs(responseScale / Math.max(fullResponseScale, 1e-9));
  const se = fullSe * Math.max(ratio, 0.05);
  return {
    low: Math.max(0, roi - 1.96 * se),
    high: Math.max(0, roi + 1.96 * se),
  };
}

export function modelExperimentWindowRoi(
  dataset: Dataset,
  model: ModelResult,
  config: ModelConfig,
  experiment: Experiment,
): ExperimentWindowRoiEstimate | undefined {
  const channel = model.channels.find(
    (candidate) =>
      candidate.channel.toLowerCase() === experiment.channel.toLowerCase(),
  );
  if (!channel) return undefined;
  const contrast = experimentResponseContrast(
    dataset,
    config,
    channel.channel,
    experiment,
  );
  if (!contrast) return undefined;
  const deltaScale = contrast.outcomeRows.reduce(
    (total, rowIndex) =>
      total + (contrast.deltaTransformed[rowIndex] ?? 0),
    0,
  ) / contrast.spend;
  let contribution = 0;
  const path = model.advanced?.coefficientPaths.find(
    (candidate) =>
      candidate.channel.toLowerCase() === channel.channel.toLowerCase(),
  )?.values;
  if (path?.length) {
    if (model.advanced?.likelihoodDistribution === "log-normal") {
      contribution = contrast.outcomeRows.reduce((total, rowIndex) => {
        const deltaEffect =
          (contrast.deltaTransformed[rowIndex] ?? 0) *
          (path[rowIndex] ?? channel.coefficient);
        const actual = Math.max(0, model.predicted[rowIndex] ?? 0);
        return total + actual * (1 - Math.exp(-deltaEffect));
      }, 0);
    } else {
      contribution = contrast.outcomeRows.reduce(
        (total, rowIndex) =>
          total +
          (contrast.deltaTransformed[rowIndex] ?? 0) *
            (path[rowIndex] ?? channel.coefficient),
        0,
      );
    }
  } else {
    contribution = contrast.outcomeRows.reduce(
      (total, rowIndex) =>
        total +
        (contrast.deltaTransformed[rowIndex] ?? 0) * channel.coefficient,
      0,
    );
  }
  const roi = contribution / Math.max(contrast.spend, 1);
  const fullSpend = dataset.rows.reduce(
    (total, row) => total + Math.max(0, toNumber(row[channel.channel])),
    0,
  );
  const fullResponseScale = channel.contribution /
    Math.max(channel.coefficient * fullSpend, 1e-9);
  const interval = scaledInterval(
    roi,
    channel.roi,
    channel.roiLow,
    channel.roiHigh,
    deltaScale,
    fullResponseScale,
  );
  return {
    roi,
    ...interval,
    basis: "experiment-window",
    label: experiment.outcomeEndDate
      ? `${experiment.startDate}–${experiment.endDate} spend · outcomes through ${experiment.outcomeEndDate}`
      : `${experiment.startDate}–${experiment.endDate}`,
    spendRows: contrast.spendRows.length,
    outcomeRows: contrast.outcomeRows.length,
  };
}
