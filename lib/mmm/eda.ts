import { correlation } from "./math";
import { toNumber } from "./csv";
import type { Dataset, EdaResult } from "./types";

export function runEda(dataset: Dataset): EdaResult {
  const outcome = dataset.rows.map((row) =>
    toNumber(row[dataset.outcomeColumn]),
  );
  const spendVectors = dataset.mediaColumns.map((column) =>
    dataset.rows.map((row) => toNumber(row[column])),
  );
  const totalSpend = spendVectors
    .flat()
    .reduce((total, value) => total + value, 0);
  const totalOutcome = outcome.reduce((total, value) => total + value, 0);

  const channels = dataset.mediaColumns
    .map((channel, index) => {
      const values = spendVectors[index];
      const spend = values.reduce((total, value) => total + value, 0);
      const activePeriods = values.filter((value) => value > 0).length;
      return {
        channel,
        spend,
        share: spend / Math.max(totalSpend, 1),
        activePeriods,
        zeroShare: 1 - activePeriods / Math.max(values.length, 1),
        correlation: correlation(values, outcome),
      };
    })
    .sort((a, b) => b.spend - a.spend);

  const correlations = spendVectors.map((left) =>
    spendVectors.map((right) => correlation(left, right)),
  );

  const readiness: string[] = [];
  const correlatedPairs: string[] = [];
  correlations.forEach((row, left) =>
    row.forEach((value, right) => {
      if (right > left && Math.abs(value) > 0.75) {
        correlatedPairs.push(
          `${dataset.mediaColumns[left]} and ${dataset.mediaColumns[right]}`,
        );
      }
    }),
  );
  if (correlatedPairs.length) {
    readiness.push(
      `${correlatedPairs[0]} move together; interpret their separate ROIs carefully.`,
    );
  }
  const sparse = channels.find((channel) => channel.zeroShare > 0.7);
  if (sparse) {
    readiness.push(
      `${sparse.channel} is inactive in ${Math.round(sparse.zeroShare * 100)}% of periods.`,
    );
  }
  if (dataset.rows.length >= dataset.periodsPerYear * 3) {
    readiness.push(
      `${dataset.rows.length} ${dataset.modelCadence} observations provide three or more annual cycles.`,
    );
  }
  if (!readiness.length) {
    readiness.push("Media variation and history look suitable for a baseline MMM.");
  }

  return {
    totalOutcome,
    totalSpend,
    outcomeTrend: outcome,
    spendTrend: dataset.rows.map((_, rowIndex) =>
      spendVectors.reduce(
        (total, vector) => total + (vector[rowIndex] ?? 0),
        0,
      ),
    ),
    channels,
    correlations,
    readiness,
  };
}
