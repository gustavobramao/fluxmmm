import assert from "node:assert/strict";
import test from "node:test";
import type { SviScoreV4CandidateRow } from "../research/svi_score_v4/types";
import {
  fitV5CSelectorEnsemble,
  scoreV5CDistribution,
  v5cCandidateRegions,
} from "../research/svi_score_v5c/crossfit";
import { confidencePromoted, evaluateV5C } from "../research/svi_score_v5c/evaluate";
import { v5cFeatureNames, v5cFeatureVector } from "../research/svi_score_v5c/features";
import { v5cUncappedGap } from "../research/svi_score_v5c/ranker";
import type { V5CScoreDistribution } from "../research/svi_score_v5c/types";

function row(
  businessId: string,
  family: string,
  candidateId: string,
  loss: number,
  diagnostic = 70,
): SviScoreV4CandidateRow {
  return {
    businessId,
    family,
    split: "train",
    candidateId,
    evidenceArm: candidateId.includes("benchmark-gap-fill")
      ? "benchmark-gap-fill"
      : "experiments-only",
    modelFamily: Number(candidateId.match(/C(\d+)/)?.[1] ?? 1) % 2
      ? "bayesian"
      : "advanced",
    eligible: true,
    reviewEligible: true,
    eligibilityTier: "decision-grade",
    failedGateCount: 0,
    layerScores: {
      generalization: diagnostic,
      structure: diagnostic,
      causal: diagnostic,
      decision: diagnostic,
    },
    diagnostics: {
      "rolling-oos": diagnostic,
      "roi-posterior-plausibility": diagnostic,
      "roi-decision-stability": diagnostic,
    },
    heuristicScore: diagnostic,
    economicScale: 1,
    decisionLoss: loss,
    cappedRegret: Math.min(loss, 1),
    roiError: loss,
    contributionError: loss,
    temporal: {
      context: family === "delayed-tv" ? 0.9 : 0.1,
      wholeFlightGeneralization: diagnostic,
      carryoverSupport: diagnostic,
      postFlightResidualStability: diagnostic,
      kernelDistinguishability: diagnostic,
      temporalIdentification: diagnostic,
      maximumEffectiveLag: 4,
      maximumLag95: 10,
      materialFlightCount: 5,
      detail: "test",
    },
  };
}

function candidateIds(count = 8): string[] {
  return Array.from({ length: count }, (_, index) => {
    const id = `V6-C${String(index + 1).padStart(2, "0")}`;
    return [`${id} · experiments-only`, `${id} · benchmark-gap-fill`];
  }).flat();
}

test("V5C removes proposal metadata and cannot read hidden labels", () => {
  const first = row("b1", "balanced-dtc", "V6-C01 · experiments-only", 0.1);
  const changed = {
    ...first,
    businessId: "different",
    family: "delayed-tv",
    split: "validation" as const,
    decisionLoss: 999,
    roiError: 888,
    contributionError: 777,
  };
  const names = v5cFeatureNames(first);
  assert.ok(names.every((name) => !name.includes("search-phase")));
  assert.deepEqual(names, v5cFeatureNames(changed));
  assert.deepEqual(v5cFeatureVector(first), v5cFeatureVector(changed));
});

test("V5C economic target remains uncapped", () => {
  assert.equal(v5cUncappedGap(0.5, 0.2, 1), 0.3);
  assert.equal(v5cUncappedGap(12, 2, 1), 10);
  assert.ok(v5cUncappedGap(12, 2, 1) > 1);
});

test("candidate regions are balanced and keep evidence arms together", () => {
  const ids = candidateIds(24);
  const regions = v5cCandidateRegions(ids);
  const counts = Object.values(regions).reduce<Record<number, number>>((output, region) => {
    output[region] = (output[region] ?? 0) + 1;
    return output;
  }, {});
  assert.deepEqual(Object.values(counts).sort((a, b) => a - b), [12, 12, 12, 12]);
  for (let index = 1; index <= 24; index += 1) {
    const id = `V6-C${String(index).padStart(2, "0")}`;
    assert.equal(
      regions[`${id} · experiments-only`],
      regions[`${id} · benchmark-gap-fill`],
    );
  }
});

test("region-excluded ensemble produces finite uncertainty", () => {
  const ids = candidateIds();
  const rows = Array.from({ length: 9 }, (_, businessIndex) =>
    ids.map((candidateId, candidateIndex) =>
      row(
        `b${businessIndex + 1}`,
        businessIndex % 2 ? "delayed-tv" : "balanced-dtc",
        candidateId,
        0.2 + candidateIndex / 20 + businessIndex / 100,
        55 + candidateIndex,
      )
    )
  ).flat();
  const regions = v5cCandidateRegions(ids);
  const ensemble = fitV5CSelectorEnsemble(rows, regions);
  assert.equal(ensemble.regions.length, 4);
  assert.ok(ensemble.regions.every((region) => region.models.length === 3));
  const distribution = scoreV5CDistribution(rows[0], ensemble);
  assert.ok(Number.isFinite(distribution.mean));
  assert.ok(Number.isFinite(distribution.standardDeviation));
  assert.equal(distribution.members.length, 3);
});

test("uncertain challengers cannot displace the incumbent without margin", () => {
  const incumbent = row("b1", "balanced-dtc", "V6-C01 · experiments-only", 0.2);
  const challenger = row("b1", "balanced-dtc", "V6-C02 · experiments-only", 0.1);
  const distributions = new Map<string, V5CScoreDistribution>([
    [incumbent.candidateId, {
      mean: 0.7,
      standardDeviation: 0.02,
      robust: 0.68,
      members: [0.68, 0.7, 0.72],
    }],
    [challenger.candidateId, {
      mean: 0.75,
      standardDeviation: 0.2,
      robust: 0.55,
      members: [0.55, 0.75, 0.95],
    }],
  ]);
  assert.equal(
    confidencePromoted([incumbent, challenger], distributions).row.candidateId,
    incumbent.candidateId,
  );
  distributions.set(challenger.candidateId, {
    mean: 0.8,
    standardDeviation: 0.01,
    robust: 0.79,
    members: [0.79, 0.8, 0.81],
  });
  assert.equal(
    confidencePromoted([incumbent, challenger], distributions).row.candidateId,
    challenger.candidateId,
  );
});

test("V5C rejects sealed audit rows before fitting", () => {
  const audit = { ...row("audit", "balanced-dtc", "V6-C01 · experiments-only", 1), split: "audit" as const };
  assert.throws(
    () => evaluateV5C(
      [audit],
      [{ businessId: audit.businessId, family: audit.family, fold: 0 }],
    ),
    /sealed V3 audit/,
  );
});
