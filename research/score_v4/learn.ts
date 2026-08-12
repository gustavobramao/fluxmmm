import type { ValidationLayerId } from "../../lib/mmm/validation";
import type {
  LearnedScoreCandidateRow,
  LearnedScoreSplitMetrics,
  LearnedScoreWeights,
} from "./types";

export const HEURISTIC_WEIGHTS: LearnedScoreWeights = {
  generalization: 0.2,
  structure: 0.15,
  causal: 0.25,
  decision: 0.4,
};

const LAYERS: ValidationLayerId[] = [
  "generalization",
  "structure",
  "causal",
  "decision",
];

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function average(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) /
    Math.max(values.length, 1);
}

function quantile(values: number[], probability: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  const position = clamp(probability, 0, 1) * Math.max(sorted.length - 1, 0);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return (sorted[lower] ?? 0) * (1 - weight) +
    (sorted[upper] ?? 0) * weight;
}

export function scoreFromWeights(
  layers: Record<ValidationLayerId, number>,
  weights: LearnedScoreWeights,
): number {
  const logScore = LAYERS.reduce(
    (total, layer) =>
      total + weights[layer] * Math.log(clamp(layers[layer] / 100, 1e-6, 1)),
    0,
  );
  return 100 * Math.exp(logScore);
}

function groups(rows: LearnedScoreCandidateRow[]): LearnedScoreCandidateRow[][] {
  const grouped = new Map<string, LearnedScoreCandidateRow[]>();
  rows.forEach((row) => {
    grouped.set(row.businessId, [...(grouped.get(row.businessId) ?? []), row]);
  });
  return Array.from(grouped.values());
}

function selectionPool(
  candidates: LearnedScoreCandidateRow[],
): LearnedScoreCandidateRow[] {
  const passing = candidates.filter(
    (candidate) => candidate.failedGateCount === 0,
  );
  return passing.length ? passing : candidates;
}

function selectCandidate(
  candidates: LearnedScoreCandidateRow[],
  weights: LearnedScoreWeights,
): LearnedScoreCandidateRow {
  return [...selectionPool(candidates)].sort((left, right) => {
    const scoreGap =
      scoreFromWeights(right.layers, weights) -
      scoreFromWeights(left.layers, weights);
    if (Math.abs(scoreGap) > 1e-10) return scoreGap;
    return left.candidateId.localeCompare(right.candidateId);
  })[0];
}

function familyBalancedRisk(
  rows: LearnedScoreCandidateRow[],
  weights: LearnedScoreWeights,
): number {
  const byFamily = new Map<string, number[]>();
  groups(rows).forEach((candidates) => {
    const selected = selectCandidate(candidates, weights);
    byFamily.set(selected.family, [
      ...(byFamily.get(selected.family) ?? []),
      selected.profitRegret,
    ]);
  });
  const familyRisk = Array.from(byFamily.values()).map(
    (regrets) => average(regrets) + 0.2 * quantile(regrets, 0.9),
  );
  return average(familyRisk);
}

function distanceFromHeuristic(weights: LearnedScoreWeights): number {
  return Math.sqrt(
    LAYERS.reduce(
      (total, layer) =>
        total + (weights[layer] - HEURISTIC_WEIGHTS[layer]) ** 2,
      0,
    ),
  );
}

export function enumerateWeightGrid(
  step = 0.025,
  minimumWeight = 0.05,
): LearnedScoreWeights[] {
  const units = Math.round(1 / step);
  const minimumUnits = Math.round(minimumWeight / step);
  const output: LearnedScoreWeights[] = [];
  for (let generalization = minimumUnits; generalization <= units; generalization += 1) {
    for (let structure = minimumUnits; structure <= units; structure += 1) {
      for (let causal = minimumUnits; causal <= units; causal += 1) {
        const decision = units - generalization - structure - causal;
        if (decision < minimumUnits) continue;
        output.push({
          generalization: generalization / units,
          structure: structure / units,
          causal: causal / units,
          decision: decision / units,
        });
      }
    }
  }
  return output;
}

export function learnWeights(
  trainRows: LearnedScoreCandidateRow[],
  validationRows: LearnedScoreCandidateRow[],
): {
  weights: LearnedScoreWeights;
  gridSize: number;
} {
  const grid = enumerateWeightGrid();
  const trainRanked = grid
    .map((weights) => ({
      weights,
      risk:
        familyBalancedRisk(trainRows, weights) +
        0.015 * distanceFromHeuristic(weights),
    }))
    .sort(
      (left, right) =>
        left.risk - right.risk ||
        distanceFromHeuristic(left.weights) -
          distanceFromHeuristic(right.weights),
    )
    .filter(
      (candidate) =>
        candidate.risk <=
        familyBalancedRisk(trainRows, HEURISTIC_WEIGHTS) * 1.02,
    );
  const heuristicValidation = evaluateWeights(
    validationRows,
    HEURISTIC_WEIGHTS,
  );
  const selected = trainRanked
    .map((candidate) => ({
      ...candidate,
      validation: evaluateWeights(validationRows, candidate.weights),
    }))
    .filter(
      (candidate) =>
        candidate.validation.p90Regret <=
          heuristicValidation.p90Regret * 1.000001 &&
        Object.entries(heuristicValidation.familyMeanRegret).every(
          ([family, regret]) =>
            (candidate.validation.familyMeanRegret[family] ?? Infinity) <=
            regret * 1.01,
        ),
    )
    .sort(
      (left, right) =>
        left.validation.meanRegret - right.validation.meanRegret ||
        left.risk - right.risk,
    )[0];
  return {
    weights: selected?.weights ?? HEURISTIC_WEIGHTS,
    gridSize: grid.length,
  };
}

export function evaluateWeights(
  rows: LearnedScoreCandidateRow[],
  weights: LearnedScoreWeights,
): LearnedScoreSplitMetrics {
  return evaluateSelection(rows, (candidate) =>
    scoreFromWeights(candidate.layers, weights),
  );
}

function evaluateSelection(
  rows: LearnedScoreCandidateRow[],
  score: (candidate: LearnedScoreCandidateRow) => number,
): LearnedScoreSplitMetrics {
  const businessGroups = groups(rows);
  const selections = businessGroups.map((candidates) => {
    const pool = selectionPool(candidates);
    const selected = [...pool].sort((left, right) =>
      score(right) - score(left) || left.candidateId.localeCompare(right.candidateId),
    )[0];
    const best = [...pool].sort(
      (left, right) => left.profitRegret - right.profitRegret,
    )[0];
    return {
      family: selected.family,
      eligible: candidates.some((candidate) => candidate.failedGateCount === 0),
      regret: selected.profitRegret,
      excess: Math.max(0, selected.profitRegret - best.profitRegret),
      lowest: Math.abs(selected.profitRegret - best.profitRegret) < 1e-10,
    };
  });
  const familyMeanRegret = Object.fromEntries(
    Array.from(new Set(selections.map((selection) => selection.family))).map(
      (family) => [
        family,
        average(
          selections
            .filter((selection) => selection.family === family)
            .map((selection) => selection.regret),
        ),
      ],
    ),
  );
  const regrets = selections.map((selection) => selection.regret);
  return {
    businesses: businessGroups.length,
    eligibleBusinessShare: average(
      selections.map((selection) => (selection.eligible ? 1 : 0)),
    ),
    meanRegret: average(regrets),
    medianRegret: quantile(regrets, 0.5),
    p90Regret: quantile(regrets, 0.9),
    meanExcessRegret: average(selections.map((selection) => selection.excess)),
    lowestRegretSelectionRate: average(
      selections.map((selection) => (selection.lowest ? 1 : 0)),
    ),
    familyMeanRegret,
  };
}
