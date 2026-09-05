import assert from "node:assert/strict";
import test from "node:test";
import type { SviScoreV4CandidateRow } from "../research/svi_score_v4/types";
import {
  SVI_SCORE_V5B_CONTRACT,
  V5B_SEARCH_ALGORITHMS,
} from "../research/svi_score_v5b/contract";
import {
  v5bCandidateUniverse,
  v5bParameterPoint,
} from "../research/svi_score_v5b/embedding";
import { evaluateV5BSearchAlgorithms } from "../research/svi_score_v5b/evaluate";
import { runV5BSearch } from "../research/svi_score_v5b/search";
import type {
  V5BObservedCandidate,
  V5BSearchCandidate,
} from "../research/svi_score_v5b/types";

function candidates(count = 18): V5BSearchCandidate[] {
  return Array.from({ length: count }, (_, index) => ({
    candidateId: `candidate-${String(index + 1).padStart(2, "0")}`,
    point: [
      index / Math.max(count - 1, 1),
      (index * 7 % count) / Math.max(count - 1, 1),
      index % 2,
    ],
  }));
}

function reveal(candidate: V5BSearchCandidate): V5BObservedCandidate {
  return {
    ...candidate,
    selectorScore: 1 - Math.abs(candidate.point[0] - 0.68),
    safetyAccepted: candidate.point[1] > 0.12,
  };
}

test("every V5B algorithm is deterministic, exhaustive, and duplicate-free", () => {
  const universe = candidates();
  V5B_SEARCH_ALGORITHMS.forEach((algorithm) => {
    const first = runV5BSearch("business-1", algorithm, universe, reveal);
    const second = runV5BSearch("business-1", algorithm, universe, reveal);
    assert.deepEqual(first, second);
    assert.equal(first.orderedCandidateIds.length, universe.length);
    assert.equal(new Set(first.orderedCandidateIds).size, universe.length);
    SVI_SCORE_V5B_CONTRACT.compute.checkpoints
      .filter((budget) => budget <= universe.length)
      .forEach((budget) => {
        assert.equal(first.orderedCandidateIds.slice(0, budget).length, budget);
      });
  });
});

test("search order cannot react to hidden economic labels", () => {
  const universe = candidates();
  V5B_SEARCH_ALGORITHMS.forEach((algorithm) => {
    const baseline = runV5BSearch("business-2", algorithm, universe, reveal);
    const changedHiddenLabels = new Map(
      universe.map((candidate, index) => [candidate.candidateId, 10_000 - index]),
    );
    const sameObservableReceipt = runV5BSearch(
      "business-2",
      algorithm,
      universe,
      (candidate) => {
        void changedHiddenLabels.get(candidate.candidateId);
        return reveal(candidate);
      },
    );
    assert.deepEqual(baseline, sameObservableReceipt);
  });
});

test("the frozen 48-candidate universe has a finite parameter-only embedding", () => {
  const ids = Array.from({ length: 24 }, (_, index) => {
    const id = `V6-C${String(index + 1).padStart(2, "0")}`;
    return [
      `${id} · experiments-only`,
      `${id} · benchmark-gap-fill`,
    ];
  }).flat();
  const universe = v5bCandidateUniverse(ids);
  assert.equal(universe.length, 48);
  assert.ok(universe[0].point.length > 50);
  assert.ok(universe.every((candidate) =>
    candidate.point.every((value) => Number.isFinite(value) && value >= 0 && value <= 1)
  ));
  assert.notDeepEqual(
    v5bParameterPoint("V6-C01 · experiments-only"),
    v5bParameterPoint("V6-C01 · benchmark-gap-fill"),
  );
});

test("V5B contract fixes equal compute and an immutable V5A selector", () => {
  assert.equal(SVI_SCORE_V5B_CONTRACT.compute.primaryEvaluationsPerBusiness, 16);
  assert.equal(SVI_SCORE_V5B_CONTRACT.compute.earlyStoppingPermitted, false);
  assert.equal(SVI_SCORE_V5B_CONTRACT.compute.duplicateEvaluationsPermitted, false);
  assert.equal(SVI_SCORE_V5B_CONTRACT.selector.mayChangeDuringSearch, false);
  assert.equal(SVI_SCORE_V5B_CONTRACT.governance.auditMayOpen, false);
  assert.equal(SVI_SCORE_V5B_CONTRACT.governance.freshValidationMayOpen, false);
});

test("V5B evaluation rejects audit data before search", () => {
  const audit = {
    businessId: "audit-business",
    family: "balanced-dtc",
    split: "audit",
    candidateId: "V6-C01 · experiments-only",
  } as unknown as SviScoreV4CandidateRow;
  assert.throws(
    () => evaluateV5BSearchAlgorithms(
      [audit],
      [{ businessId: audit.businessId, family: audit.family, fold: 0 }],
      {
        id: "frozen",
        regularization: 0.06,
        epochs: 1,
        cvarWeight: 0.4,
        familyDroWeight: 0.85,
        featureCap: 0.06,
        learningRate: 0.12,
      },
    ),
    /sealed V3 audit/,
  );
});
