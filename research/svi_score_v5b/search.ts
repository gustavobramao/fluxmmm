import { createHash } from "node:crypto";
import {
  SVI_SCORE_V5B_CONTRACT,
  type V5BSearchAlgorithm,
} from "./contract";
import { v5bSquaredDistance } from "./embedding";
import type {
  V5BObservedCandidate,
  V5BSearchCandidate,
  V5BSearchTrace,
} from "./types";

type RevealCandidate = (candidate: V5BSearchCandidate) => V5BObservedCandidate;

function centerDistance(point: readonly number[]): number {
  return point.reduce((sum, value) => sum + (value - 0.5) ** 2, 0) /
    Math.max(point.length, 1);
}

export function v5bMaximinOrder(
  candidates: readonly V5BSearchCandidate[],
): V5BSearchCandidate[] {
  if (!candidates.length) return [];
  const remaining = new Map(candidates.map((candidate) => [candidate.candidateId, candidate]));
  const first = [...remaining.values()].sort(
    (left, right) =>
      centerDistance(left.point) - centerDistance(right.point) ||
      left.candidateId.localeCompare(right.candidateId),
  )[0];
  const selected = [first];
  remaining.delete(first.candidateId);
  while (remaining.size) {
    const next = [...remaining.values()].sort((left, right) => {
      const leftDistance = Math.min(
        ...selected.map((item) => v5bSquaredDistance(left.point, item.point)),
      );
      const rightDistance = Math.min(
        ...selected.map((item) => v5bSquaredDistance(right.point, item.point)),
      );
      return rightDistance - leftDistance ||
        left.candidateId.localeCompare(right.candidateId);
    })[0];
    selected.push(next);
    remaining.delete(next.candidateId);
  }
  return selected;
}

function pseudoRandom(seed: string): () => number {
  const digest = createHash("sha256").update(seed).digest();
  let state = digest.readUInt32LE(0) || 0x9e3779b9;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled(
  candidates: readonly V5BSearchCandidate[],
  seed: string,
): V5BSearchCandidate[] {
  const output = [...candidates];
  const random = pseudoRandom(seed);
  for (let index = output.length - 1; index > 0; index -= 1) {
    const target = Math.floor(random() * (index + 1));
    [output[index], output[target]] = [output[target], output[index]];
  }
  return output;
}

function cholesky(matrix: number[][]): number[][] {
  const size = matrix.length;
  const lower = Array.from({ length: size }, () => Array(size).fill(0));
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column <= row; column += 1) {
      let value = matrix[row][column];
      for (let index = 0; index < column; index += 1) {
        value -= lower[row][index] * lower[column][index];
      }
      if (row === column) {
        lower[row][column] = Math.sqrt(Math.max(value, 1e-10));
      } else {
        lower[row][column] = value / Math.max(lower[column][column], 1e-10);
      }
    }
  }
  return lower;
}

function solveCholesky(lower: number[][], target: readonly number[]): number[] {
  const size = lower.length;
  const forward = Array(size).fill(0);
  for (let row = 0; row < size; row += 1) {
    let value = target[row];
    for (let column = 0; column < row; column += 1) {
      value -= lower[row][column] * forward[column];
    }
    forward[row] = value / Math.max(lower[row][row], 1e-10);
  }
  const output = Array(size).fill(0);
  for (let row = size - 1; row >= 0; row -= 1) {
    let value = forward[row];
    for (let column = row + 1; column < size; column += 1) {
      value -= lower[column][row] * output[column];
    }
    output[row] = value / Math.max(lower[row][row], 1e-10);
  }
  return output;
}

function kernel(left: readonly number[], right: readonly number[]): number {
  const lengthScale = SVI_SCORE_V5B_CONTRACT.adaptivePolicy.gpLengthScale;
  return Math.exp(-v5bSquaredDistance(left, right) / (2 * lengthScale * lengthScale));
}

function observedUtility(candidate: V5BObservedCandidate): number {
  return candidate.selectorScore -
    (candidate.safetyAccepted
      ? 0
      : SVI_SCORE_V5B_CONTRACT.adaptivePolicy.unsafeUtilityPenalty);
}

function gpAcquisitions(
  candidates: readonly V5BSearchCandidate[],
  observed: readonly V5BObservedCandidate[],
): Map<string, number> {
  const rawTargets = observed.map(observedUtility);
  const mean = rawTargets.reduce((sum, value) => sum + value, 0) /
    Math.max(rawTargets.length, 1);
  const variance = rawTargets.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    Math.max(rawTargets.length - 1, 1);
  const scale = Math.max(Math.sqrt(variance), 0.05);
  const targets = rawTargets.map((value) => (value - mean) / scale);
  const covariance = observed.map((left, row) =>
    observed.map((right, column) =>
      kernel(left.point, right.point) +
      (row === column ? SVI_SCORE_V5B_CONTRACT.adaptivePolicy.gpNoise : 0)
    )
  );
  const lower = cholesky(covariance);
  const alpha = solveCholesky(lower, targets);
  return new Map(candidates.map((candidate) => {
    const cross = observed.map((item) => kernel(candidate.point, item.point));
    const prediction = cross.reduce(
      (sum, value, index) => sum + value * alpha[index],
      0,
    );
    const projected = solveCholesky(lower, cross);
    const posteriorVariance = Math.max(
      1e-8,
      1 - cross.reduce((sum, value, index) => sum + value * projected[index], 0),
    );
    return [
      candidate.candidateId,
      prediction + SVI_SCORE_V5B_CONTRACT.adaptivePolicy.gpExploration *
        Math.sqrt(posteriorVariance),
    ];
  }));
}

function bestObserved(observed: readonly V5BObservedCandidate[]): V5BObservedCandidate {
  const safe = observed.filter((candidate) => candidate.safetyAccepted);
  return [...(safe.length ? safe : observed)].sort(
    (left, right) =>
      right.selectorScore - left.selectorScore ||
      left.candidateId.localeCompare(right.candidateId),
  )[0];
}

function adaptiveOrder(
  candidates: readonly V5BSearchCandidate[],
  reveal: RevealCandidate,
  hybrid: boolean,
): V5BSearchCandidate[] {
  const initialCount = hybrid
    ? SVI_SCORE_V5B_CONTRACT.adaptivePolicy.hybridInitialDesign
    : SVI_SCORE_V5B_CONTRACT.adaptivePolicy.gpInitialDesign;
  const initial = v5bMaximinOrder(candidates).slice(0, initialCount);
  const remaining = new Map(candidates.map((candidate) => [candidate.candidateId, candidate]));
  const ordered: V5BSearchCandidate[] = [];
  const observed: V5BObservedCandidate[] = [];
  const evaluate = (candidate: V5BSearchCandidate) => {
    ordered.push(candidate);
    observed.push(reveal(candidate));
    remaining.delete(candidate.candidateId);
  };
  initial.forEach(evaluate);
  let adaptiveStep = 0;
  while (remaining.size) {
    adaptiveStep += 1;
    let next: V5BSearchCandidate;
    if (
      hybrid &&
      adaptiveStep % SVI_SCORE_V5B_CONTRACT.adaptivePolicy.hybridLocalFrequency === 0
    ) {
      const champion = bestObserved(observed);
      next = [...remaining.values()].sort(
        (left, right) =>
          v5bSquaredDistance(left.point, champion.point) -
            v5bSquaredDistance(right.point, champion.point) ||
          left.candidateId.localeCompare(right.candidateId),
      )[0];
    } else {
      const acquisitions = gpAcquisitions([...remaining.values()], observed);
      next = [...remaining.values()].sort(
        (left, right) =>
          (acquisitions.get(right.candidateId) ?? -Infinity) -
            (acquisitions.get(left.candidateId) ?? -Infinity) ||
          left.candidateId.localeCompare(right.candidateId),
      )[0];
    }
    evaluate(next);
  }
  return ordered;
}

export function runV5BSearch(
  businessId: string,
  algorithm: V5BSearchAlgorithm,
  candidates: readonly V5BSearchCandidate[],
  reveal: RevealCandidate,
): V5BSearchTrace {
  if (!candidates.length) throw new Error("V5B search requires a candidate universe.");
  let order: V5BSearchCandidate[];
  if (algorithm === "fixed-coverage") {
    order = [...candidates];
    order.forEach(reveal);
  } else if (algorithm === "uniform-random") {
    order = shuffled(candidates, `${businessId}\u0000${algorithm}`);
    order.forEach(reveal);
  } else if (algorithm === "maximin-space-filling") {
    order = v5bMaximinOrder(candidates);
    order.forEach(reveal);
  } else {
    order = adaptiveOrder(candidates, reveal, algorithm === "hybrid-global-local");
  }
  const orderedCandidateIds = order.map((candidate) => candidate.candidateId);
  if (
    orderedCandidateIds.length !== candidates.length ||
    new Set(orderedCandidateIds).size !== orderedCandidateIds.length
  ) {
    throw new Error(`${algorithm} violated the V5B equal-budget search contract.`);
  }
  return { businessId, algorithm, orderedCandidateIds };
}
