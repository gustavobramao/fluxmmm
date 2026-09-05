import type { McmcScoreV3SampleRecord } from "./types";

function average(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) /
    Math.max(values.length, 1);
}

function quantile(values: number[], probability: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const position = Math.min(1, Math.max(0, probability)) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function seededUnit(seed: number): number {
  let value = seed | 0;
  value ^= value << 13;
  value ^= value >>> 17;
  value ^= value << 5;
  return (value >>> 0) / 4_294_967_296;
}

function clusteredShiftInterval(
  businessShifts: number[],
  replicates = 2_000,
): [number, number] {
  if (!businessShifts.length) return [0, 0];
  const estimates = Array.from({ length: replicates }, (_, replicate) => {
    const sample = Array.from({ length: businessShifts.length }, (_, index) => {
      const draw = Math.floor(
        seededUnit(73_819 + replicate * 10_007 + index * 97) * businessShifts.length,
      );
      return businessShifts[Math.min(businessShifts.length - 1, draw)];
    });
    return average(sample);
  });
  return [quantile(estimates, 0.025), quantile(estimates, 0.975)];
}

export function summarizePairedEvidence(records: McmcScoreV3SampleRecord[]) {
  const labelled = records.filter(
    (record) =>
      record.status === "labelled" &&
      Number.isFinite(record.mapDecisionLoss) &&
      Number.isFinite(record.mcmcDecisionLoss),
  );
  if (!labelled.length) return undefined;

  const grouped = new Map<string, McmcScoreV3SampleRecord[]>();
  labelled.forEach((record) => {
    grouped.set(record.businessId, [...(grouped.get(record.businessId) ?? []), record]);
  });
  const businessShifts = [...grouped.values()].map((business) =>
    average(
      business.map(
        (record) => record.mcmcDecisionLoss! - record.mapDecisionLoss,
      ),
    )
  );
  const [meanLossShiftLow, meanLossShiftHigh] =
    clusteredShiftInterval(businessShifts);
  const mapLosses = labelled.map((record) => record.mapDecisionLoss);
  const mcmcLosses = labelled.map((record) => record.mcmcDecisionLoss!);
  const meanLossShift = average(businessShifts);
  const interpretation = meanLossShiftHigh < 0
    ? "mcmc-lower" as const
    : meanLossShiftLow > 0
      ? "map-lower" as const
      : "inconclusive" as const;

  return {
    pairs: labelled.length,
    businesses: grouped.size,
    mapMeanLoss: average(mapLosses),
    mcmcMeanLoss: average(mcmcLosses),
    meanLossShift,
    meanLossShiftLow,
    meanLossShiftHigh,
    mcmcWinRate: average(
      labelled.map((record) =>
        record.mcmcDecisionLoss! < record.mapDecisionLoss ? 1 : 0
      ),
    ),
    mapMedianLoss: quantile(mapLosses, 0.5),
    mcmcMedianLoss: quantile(mcmcLosses, 0.5),
    mapP90Loss: quantile(mapLosses, 0.9),
    mcmcP90Loss: quantile(mcmcLosses, 0.9),
    interpretation,
  };
}

export function summarizeSamplerEvidence(records: McmcScoreV3SampleRecord[]) {
  const attempted = records.filter((record) => record.status !== "prepared");
  if (!attempted.length) return undefined;
  const runtimes = attempted
    .map((record) => record.runtimeSeconds)
    .filter((value): value is number => Number.isFinite(value));
  return {
    attemptedFits: attempted.length,
    convergenceRate: average(
      attempted.map((record) => record.convergencePassed ? 1 : 0),
    ),
    retryRate: average(attempted.map((record) => (record.attempts ?? 0) > 1 ? 1 : 0)),
    medianRuntimeSeconds: quantile(runtimes, 0.5),
    p90RuntimeSeconds: quantile(runtimes, 0.9),
  };
}
