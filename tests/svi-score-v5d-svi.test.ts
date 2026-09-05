import assert from "node:assert/strict";
import test from "node:test";
import type { SviScoreV4CandidateRow } from "../research/svi_score_v4/types";
import { SVI_SCORE_V5D_SVI_CONTRACT } from "../research/svi_score_v5d_svi/contract";
import {
  type V5DSviPosteriorRecord,
  v5dSviFeatureProvider,
  v5dSviKey,
  v5dSviSelectionPolicy,
} from "../research/svi_score_v5d_svi/posterior";

function candidate(loss = 1): SviScoreV4CandidateRow {
  return {
    businessId: "business",
    family: "balanced-dtc",
    split: "train",
    candidateId: "V6-C01 · experiments-only",
    evidenceArm: "experiments-only",
    modelFamily: "bayesian",
    eligible: true,
    reviewEligible: true,
    eligibilityTier: "decision-grade",
    failedGateCount: 0,
    layerScores: {
      generalization: 75,
      structure: 75,
      causal: 75,
      decision: 75,
    },
    diagnostics: {
      "rolling-oos": 75,
      "roi-posterior-plausibility": 75,
      "roi-decision-stability": 75,
    },
    heuristicScore: 75,
    economicScale: 1,
    decisionLoss: loss,
    cappedRegret: Math.min(loss, 1),
    roiError: loss,
    contributionError: loss,
    temporal: {
      context: 0.2,
      wholeFlightGeneralization: 75,
      carryoverSupport: 75,
      postFlightResidualStability: 75,
      kernelDistinguishability: 75,
      temporalIdentification: 75,
      maximumEffectiveLag: 4,
      maximumLag95: 8,
      materialFlightCount: 4,
      detail: "test",
    },
  };
}

function posterior(
  overrides: Partial<V5DSviPosteriorRecord["diagnostics"]> = {},
): V5DSviPosteriorRecord {
  return {
    status: "labelled",
    posteriorRoi: {
      meta_acquisition_spend: 2,
      google_search_nonbrand_spend: 1.2,
      ctv_spend: 1.6,
    },
    diagnostics: {
      finite: true,
      elboStable: true,
      maximumElboDrift: 0.01,
      seedAgreement: true,
      maximumSeedLogRoiDifference: 0.03,
      adjudicationUsed: false,
      predictiveCoverage: 0.9,
      maximumImplausibleProbability: 0.1,
      maximumRelativeRoiWidth: 2,
      ...overrides,
    },
    seedDiagnostics: [
      {
        seed: 1,
        iterations: 5_000,
        finalLoss: 100,
        elboDrift: 0.01,
        finite: true,
        runtimeSeconds: 1,
      },
    ],
  };
}

test("V5D-SVI reuses posterior fits and forbids posterior truth labels", () => {
  assert.equal(SVI_SCORE_V5D_SVI_CONTRACT.posterior.refitRequired, false);
  assert.equal(SVI_SCORE_V5D_SVI_CONTRACT.posterior.fitsAlreadyAvailable, 24_000);
  assert.ok(SVI_SCORE_V5D_SVI_CONTRACT.forbiddenFeatures.includes("svi-decision-loss"));
  assert.equal(SVI_SCORE_V5D_SVI_CONTRACT.governance.auditAccessed, false);
  assert.equal(SVI_SCORE_V5D_SVI_CONTRACT.governance.publicResearchReleasePermitted, true);
  assert.equal(SVI_SCORE_V5D_SVI_CONTRACT.governance.freshGeneralizationEstablished, false);
  assert.equal(SVI_SCORE_V5D_SVI_CONTRACT.governance.productionActivationPermitted, false);
});

test("V5D-SVI features use posterior observables but not hidden economic loss", () => {
  const first = candidate(0.2);
  const records = new Map([[v5dSviKey(first), posterior()]]);
  const provider = v5dSviFeatureProvider(records);
  const changed = { ...first, decisionLoss: 999, roiError: 888, contributionError: 777 };
  assert.deepEqual(provider.names(first), provider.names(changed));
  assert.deepEqual(provider.vector(first), provider.vector(changed));
  assert.ok(provider.names(first).includes("svi:predictive:coverage"));
  assert.ok(provider.names(first).includes("svi:roi:log-center:ctv"));
});

test("V5D-SVI posterior safety uses the declared decision gates", () => {
  const row = candidate();
  const records = new Map([[v5dSviKey(row), posterior()]]);
  assert.equal(v5dSviSelectionPolicy(records).passes(row), true);

  records.set(v5dSviKey(row), posterior({ maximumRelativeRoiWidth: 3.01 }));
  assert.equal(v5dSviSelectionPolicy(records).passes(row), false);

  records.set(v5dSviKey(row), posterior({ maximumImplausibleProbability: 0.21 }));
  assert.equal(v5dSviSelectionPolicy(records).passes(row), false);

  records.set(v5dSviKey(row), posterior({ predictiveCoverage: 0.79 }));
  assert.equal(v5dSviSelectionPolicy(records).passes(row), false);
});
