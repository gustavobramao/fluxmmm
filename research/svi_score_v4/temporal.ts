import {
  carryoverForResponse,
  responseForChannel,
} from "../../lib/mmm/response";
import type {
  Dataset,
  MediaResponseConfig,
  ModelConfig,
  ModelResult,
} from "../../lib/mmm/types";
import { toNumber } from "../../lib/mmm/csv";
import { fitValidationFold } from "../../lib/mmm/validation/adapter";
import type { ValidationModelSpec } from "../../lib/mmm/validation/adapter";
import type { SviScoreV4TemporalReceipt } from "./types";

const IMPULSE_HORIZON = 52;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function average(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) /
    Math.max(values.length, 1);
}

function quantile(values: number[], probability: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const position = clamp(probability, 0, 1) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function medianAbsoluteDeviation(values: number[]): number {
  const median = quantile(values, 0.5);
  return Math.max(
    quantile(values.map((value) => Math.abs(value - median)), 0.5) * 1.4826,
    1e-9,
  );
}

function correlation(left: number[], right: number[]): number {
  const length = Math.min(left.length, right.length);
  if (length < 3) return 0;
  const leftMean = average(left.slice(0, length));
  const rightMean = average(right.slice(0, length));
  let numerator = 0;
  let leftScale = 0;
  let rightScale = 0;
  for (let index = 0; index < length; index += 1) {
    const x = left[index] - leftMean;
    const y = right[index] - rightMean;
    numerator += x * y;
    leftScale += x * x;
    rightScale += y * y;
  }
  const denominator = Math.sqrt(leftScale * rightScale);
  return denominator <= 1e-12 ? 0 : clamp(numerator / denominator, -1, 1);
}

function numericColumn(dataset: Dataset, column: string): number[] {
  return dataset.rows.map((row) => {
    const value = Number(row[column]);
    return Number.isFinite(value) ? Math.max(0, value) : 0;
  });
}

function activity(values: number[]): boolean[] {
  const positive = values.filter((value) => value > 0);
  const threshold = Math.max(
    quantile(positive, 0.5) * 0.03,
    Math.max(0, ...values) * 0.001,
    1e-12,
  );
  return values.map((value) => value > threshold);
}

interface FlightPattern {
  activeShare: number;
  flightCount: number;
  ends: number[];
  gaps: number[];
  runs: number[];
}

interface FlightRun {
  start: number;
  end: number;
}

function flightRuns(values: number[]): FlightRun[] {
  const active = activity(values);
  const runs: FlightRun[] = [];
  let start: number | undefined;
  active.forEach((isActive, index) => {
    if (isActive && start === undefined) start = index;
    if (!isActive && start !== undefined) {
      runs.push({ start, end: index - 1 });
      start = undefined;
    }
  });
  if (start !== undefined) runs.push({ start, end: active.length - 1 });
  return runs;
}

function flightPattern(values: number[]): FlightPattern {
  const active = activity(values);
  const ends: number[] = [];
  const gaps: number[] = [];
  const runs: number[] = [];
  let activeRun = 0;
  let inactiveRun = 0;
  active.forEach((isActive, index) => {
    if (isActive) {
      if (inactiveRun > 0) gaps.push(inactiveRun);
      inactiveRun = 0;
      activeRun += 1;
    } else {
      if (activeRun > 0) {
        runs.push(activeRun);
        ends.push(index - 1);
      }
      activeRun = 0;
      inactiveRun += 1;
    }
  });
  if (activeRun > 0) runs.push(activeRun);
  if (inactiveRun > 0) gaps.push(inactiveRun);
  return {
    activeShare: average(active.map((value) => value ? 1 : 0)),
    flightCount: runs.length,
    ends,
    gaps,
    runs,
  };
}

interface KernelReceipt {
  effectiveLag: number;
  lag95: number;
  peakLag: number;
}

export function kernelReceipt(response: MediaResponseConfig): KernelReceipt {
  const impulse = [1, ...Array.from({ length: IMPULSE_HORIZON }, () => 0)];
  const carried = carryoverForResponse(impulse, response).map((value) =>
    Math.max(0, value)
  );
  const total = carried.reduce((sum, value) => sum + value, 0);
  if (total <= 1e-12) return { effectiveLag: 0, lag95: 0, peakLag: 0 };
  const weights = carried.map((value) => value / total);
  const effectiveLag = weights.reduce(
    (sum, weight, lag) => sum + weight * lag,
    0,
  );
  let cumulative = 0;
  let lag95 = 0;
  weights.some((weight, lag) => {
    cumulative += weight;
    lag95 = lag;
    return cumulative >= 0.95;
  });
  const peakLag = weights.reduce(
    (best, value, lag) => value > weights[best] ? lag : best,
    0,
  );
  return { effectiveLag, lag95, peakLag };
}

function observedTemporalContext(dataset: Dataset): number {
  const preferred = dataset.mediaColumns.filter((column) =>
    /(?:^|_)(?:tv|ctv|ooh|video)(?:_|$)/i.test(column)
  );
  const columns = preferred.length ? preferred : dataset.mediaColumns;
  const contexts = columns.map((column) => {
    const values = numericColumn(dataset, column);
    const pattern = flightPattern(values);
    const lagOne = Math.abs(correlation(values.slice(1), values.slice(0, -1)));
    const gap = quantile(pattern.gaps, 0.75);
    const run = quantile(pattern.runs, 0.75);
    return clamp(
      0.45 * (1 - pattern.activeShare) +
        0.25 * clamp(gap / 12, 0, 1) +
        0.15 * clamp(run / 8, 0, 1) +
        0.15 * lagOne,
      0,
      1,
    );
  });
  return clamp(Math.max(0, ...contexts), 0, 1);
}

function postFlightResidualScore(
  values: number[],
  residuals: number[],
  horizon: number,
): { score: number; flights: number } {
  const pattern = flightPattern(values);
  const active = activity(values);
  const indexes = new Set<number>();
  pattern.ends.forEach((end) => {
    for (let offset = 1; offset <= horizon; offset += 1) {
      const index = end + offset;
      if (index >= residuals.length || active[index]) break;
      indexes.add(index);
    }
  });
  if (pattern.ends.length < 2 || indexes.size < Math.max(4, horizon)) {
    return { score: 50, flights: pattern.ends.length };
  }
  const postFlight = [...indexes].map((index) => residuals[index]);
  const scale = medianAbsoluteDeviation(residuals);
  const locationShift = Math.abs(average(postFlight) - quantile(residuals, 0.5));
  const positiveTail = average(
    postFlight.map((value) => Math.max(0, value - quantile(residuals, 0.5))),
  );
  const standardized = (0.7 * locationShift + 0.3 * positiveTail) / scale;
  return {
    score: 100 * Math.exp(-0.8 * standardized),
    flights: pattern.ends.length,
  };
}

/**
 * Truth-blind temporal evidence. This reads only the advertiser-visible
 * dataset, the fitted candidate residuals, and the candidate's declared
 * response contract. No synthetic ROI, injected kernel, or decision label is
 * consulted.
 */
export function temporalEvidenceForCandidate(
  dataset: Dataset,
  model: ModelResult,
  config: ModelConfig,
): SviScoreV4TemporalReceipt {
  const totalSpend = dataset.mediaColumns.reduce(
    (total, column) =>
      total + numericColumn(dataset, column).reduce((sum, value) => sum + value, 0),
    0,
  );
  const material = dataset.mediaColumns.filter((column) => {
    const spend = numericColumn(dataset, column).reduce((sum, value) => sum + value, 0);
    return spend / Math.max(totalSpend, 1) >= 0.02;
  });
  const columns = material.length ? material : dataset.mediaColumns;
  const horizon = dataset.periodUnit === "month" ? 3 : 12;
  const channelReceipts = columns.map((column) => {
    const values = numericColumn(dataset, column);
    const response = responseForChannel(config, column);
    const kernel = kernelReceipt(response);
    const pattern = flightPattern(values);
    const historySupport = clamp(
      dataset.rows.length / Math.max(104, 8 * Math.max(kernel.lag95, 1)),
      0,
      1,
    );
    const eligibleGaps = pattern.gaps.filter((gap) => gap >= Math.max(1, kernel.lag95));
    const blackoutSupport = kernel.lag95 <= 1
      ? 1
      : eligibleGaps.length / Math.max(pattern.gaps.length, 1);
    const carryoverSupport = 100 * (
      0.45 * historySupport + 0.55 * blackoutSupport
    );
    const carried = carryoverForResponse(values, response);
    const rawCorrelation = Math.abs(correlation(values, carried));
    const kernelDistinguishability = kernel.lag95 <= 1
      ? 100
      : 100 * clamp((1 - rawCorrelation) / 0.3, 0, 1);
    const postFlight = postFlightResidualScore(values, model.residuals, horizon);
    return {
      column,
      spend: values.reduce((sum, value) => sum + value, 0),
      kernel,
      carryoverSupport,
      kernelDistinguishability,
      postFlightResidualStability: postFlight.score,
      flightCount: postFlight.flights,
    };
  });
  const channelWeight = (spend: number) => spend / Math.max(
    channelReceipts.reduce((sum, receipt) => sum + receipt.spend, 0),
    1,
  );
  const weighted = (key: "carryoverSupport" | "kernelDistinguishability") =>
    channelReceipts.reduce(
      (sum, receipt) => sum + channelWeight(receipt.spend) * receipt[key],
      0,
    );
  // Post-flight misspecification in one material channel is not allowed to be
  // hidden by good residual behavior in the others.
  const postFlightResidualStability = quantile(
    channelReceipts.map((receipt) => receipt.postFlightResidualStability),
    0.25,
  );
  const carryoverSupport = weighted("carryoverSupport");
  const kernelDistinguishability = weighted("kernelDistinguishability");
  const temporalIdentification = 100 * Math.pow(
    clamp(carryoverSupport / 100, 0.01, 1) *
      clamp(postFlightResidualStability / 100, 0.01, 1) *
      clamp(kernelDistinguishability / 100, 0.01, 1),
    1 / 3,
  );
  const maximumEffectiveLag = Math.max(
    0,
    ...channelReceipts.map((receipt) => receipt.kernel.effectiveLag),
  );
  const maximumLag95 = Math.max(
    0,
    ...channelReceipts.map((receipt) => receipt.kernel.lag95),
  );
  const materialFlightCount = channelReceipts.reduce(
    (sum, receipt) => sum + receipt.flightCount,
    0,
  );
  return {
    context: observedTemporalContext(dataset),
    wholeFlightGeneralization: 50,
    carryoverSupport,
    postFlightResidualStability,
    kernelDistinguishability,
    temporalIdentification,
    maximumEffectiveLag,
    maximumLag95,
    materialFlightCount,
    detail:
      `${columns.length} material channel${columns.length === 1 ? "" : "s"}; ` +
      `max L95 ${maximumLag95.toFixed(0)} ${dataset.periodUnit}s; ` +
      `${materialFlightCount} observed flights.`,
  };
}

function wape(actual: number[], predicted: number[]): number {
  return actual.reduce(
    (sum, value, index) => sum + Math.abs(value - predicted[index]),
    0,
  ) / Math.max(
    actual.reduce((sum, value) => sum + Math.abs(value), 0),
    1,
  );
}

/**
 * Fits only on observations before an entire flight, using a fixed
 * candidate-independent purge, then evaluates the full flight and its fixed
 * post-flight tail. Injected response truth is never read.
 */
export function wholeFlightGeneralization(
  dataset: Dataset,
  spec: ValidationModelSpec,
): { score: number; folds: number; meanWape: number; meanSkill: number } {
  const horizon = dataset.modelCadence === "monthly" ? 3 : 12;
  const purge = dataset.modelCadence === "monthly" ? 3 : 12;
  const minimumTraining = dataset.modelCadence === "monthly" ? 18 : 52;
  const preferred = dataset.mediaColumns.filter((column) =>
    /(?:^|_)(?:tv|ctv|ooh|video)(?:_|$)/i.test(column)
  );
  const columns = preferred.length ? preferred : dataset.mediaColumns;
  const candidateFlights = columns.flatMap((column) =>
    flightRuns(numericColumn(dataset, column))
      .filter((flight) => flight.start - purge >= minimumTraining)
      .map((flight) => ({ ...flight, column }))
  ).sort(
    (left, right) =>
      right.start - left.start ||
      (right.end - right.start) - (left.end - left.start) ||
      left.column.localeCompare(right.column),
  );
  const selectedFlights: typeof candidateFlights = [];
  candidateFlights.forEach((flight) => {
    if (selectedFlights.length >= 3) return;
    const testEnd = Math.min(dataset.rows.length - 1, flight.end + horizon);
    const overlaps = selectedFlights.some((selected) => {
      const selectedEnd = Math.min(
        dataset.rows.length - 1,
        selected.end + horizon,
      );
      return flight.start <= selectedEnd && selected.start <= testEnd;
    });
    if (!overlaps) selectedFlights.push(flight);
  });
  const outcome = dataset.rows.map((row) => toNumber(row[dataset.outcomeColumn]));
  const receipts = selectedFlights.map((flight) => {
    const trainingEnd = flight.start - purge;
    const trainIndexes = Array.from({ length: trainingEnd }, (_, index) => index);
    const testEnd = Math.min(dataset.rows.length - 1, flight.end + horizon);
    const testIndexes = Array.from(
      { length: testEnd - flight.start + 1 },
      (_, index) => flight.start + index,
    );
    const prediction = fitValidationFold(dataset, spec, trainIndexes, testIndexes);
    const trainMean = average(trainIndexes.map((index) => outcome[index]));
    const seasonalLag = Math.max(1, Math.round(dataset.periodsPerYear));
    const trainSet = new Set(trainIndexes);
    const baseline = testIndexes.map((index) =>
      trainSet.has(index - seasonalLag) ? outcome[index - seasonalLag] : trainMean
    );
    const modelWape = wape(prediction.actual, prediction.predicted);
    const baselineWape = wape(prediction.actual, baseline);
    const skill = 1 - modelWape / Math.max(baselineWape, 1e-6);
    const score = clamp(
      55 + 80 * skill - Math.max(0, modelWape - 0.2) * 100,
      0,
      100,
    );
    return { modelWape, skill, score };
  });
  return {
    score: receipts.length ? average(receipts.map((receipt) => receipt.score)) : 50,
    folds: receipts.length,
    meanWape: receipts.length
      ? average(receipts.map((receipt) => receipt.modelWape))
      : Number.NaN,
    meanSkill: receipts.length
      ? average(receipts.map((receipt) => receipt.skill))
      : Number.NaN,
  };
}
