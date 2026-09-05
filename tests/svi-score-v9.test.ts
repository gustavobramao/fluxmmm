import assert from "node:assert/strict";
import test from "node:test";
import {
  SVI_SCORE_V9_CONTRACT,
  V9_POSTERIOR_CHANNEL_METRICS,
  V9_POSTERIOR_TOKEN_NAMES,
} from "../research/svi_score_v9/contract";
import { SVI_SCORE_V9_AUDIT_CONTRACT } from "../research/svi_score_v9/audit-contract";
import { v9SealedAuditBusinesses } from "../research/svi_score_v9/audit-design";
import {
  v9PosteriorDecisionTokens,
  type V9SviResult,
} from "../research/svi_score_v9/audit-tokens";
import { sviDesignBusinesses } from "../research/svi_score_v3/design";
import { integerArgument } from "../research/svi_score_v9/arguments";

test("V9 is an isolated SVI-only set-wise research contract", () => {
  assert.equal(SVI_SCORE_V9_CONTRACT.inference.contract.method, "fullrank-advi");
  assert.equal(SVI_SCORE_V9_CONTRACT.inference.nutsPermittedInResearch, false);
  assert.equal(SVI_SCORE_V9_CONTRACT.cohort.v8SealedAuditPermitted, false);
  assert.equal(SVI_SCORE_V9_CONTRACT.cohort.newSealedAuditRequiredAfterDevelopment, true);
  assert.equal(SVI_SCORE_V9_CONTRACT.evaluation.leaveOneGeneratorFamilyOut, true);
  assert.equal(SVI_SCORE_V9_CONTRACT.model.family.includes("DeepSets"), true);
  assert.equal(SVI_SCORE_V9_CONTRACT.tokens.candidateOrderInvariant, true);
  assert.equal(SVI_SCORE_V9_CONTRACT.tokens.channelOrderInvariant, true);
});

test("V9 posterior-token registry is complete and unique", () => {
  assert.equal(V9_POSTERIOR_CHANNEL_METRICS.length, 23);
  assert.equal(V9_POSTERIOR_TOKEN_NAMES.length, 96);
  assert.equal(new Set(V9_POSTERIOR_TOKEN_NAMES).size, 96);
  assert.equal(SVI_SCORE_V9_CONTRACT.tokens.count, 232);
  assert.equal(SVI_SCORE_V9_CONTRACT.targets.heads.length, 10);
});

test("V9 sealed audit is new, balanced, SVI-only, and predeclared", () => {
  const audit = v9SealedAuditBusinesses();
  const developmentSeeds = new Set(sviDesignBusinesses().map((item) => item.scenario.seed));
  assert.equal(audit.length, 140);
  assert.equal(new Set(audit.map((item) => item.scenario.id)).size, 140);
  assert.equal(new Set(audit.map((item) => item.scenario.seed)).size, 140);
  assert.equal(audit.some((item) => developmentSeeds.has(item.scenario.seed)), false);
  assert.deepEqual(
    Object.fromEntries(SVI_SCORE_V9_AUDIT_CONTRACT.cohort.families.map((family) => [
      family,
      audit.filter((item) => item.family.id === family).length,
    ])),
    Object.fromEntries(SVI_SCORE_V9_AUDIT_CONTRACT.cohort.families.map((family) => [family, 20])),
  );
  assert.equal(SVI_SCORE_V9_AUDIT_CONTRACT.inference.nutsPermitted, false);
  assert.equal(SVI_SCORE_V9_AUDIT_CONTRACT.governance.tuningAfterOpenPermitted, false);
  assert.equal(SVI_SCORE_V9_AUDIT_CONTRACT.endpoints.primaryInference.resamples, 10_000);
  assert.equal(SVI_SCORE_V9_AUDIT_CONTRACT.auditId, "003");
  assert.equal(SVI_SCORE_V9_AUDIT_CONTRACT.cohort.seedIndexStart, 90_000);
  assert.equal(SVI_SCORE_V9_AUDIT_CONTRACT.orchestration.tasks, 6_720);
  assert.equal(SVI_SCORE_V9_AUDIT_CONTRACT.orchestration.parallelism, 180);
});

test("V9 audit posterior tokens are invariant to channel ordering", () => {
  const channel = (name: string, scale: number) => ({
    channel: name,
    posteriorMean: scale * 1.1,
    posteriorMedian: scale,
    posteriorLow: scale * 0.6,
    posteriorHigh: scale * 1.7,
    posteriorSamples: [scale * 0.7, scale, scale * 1.5],
    implausibleProbability: 0.1,
    evidenceSource: name === "a" ? "experiment" : "industry benchmark",
    explanation: {
      locationShiftSd: 0.2,
      intervalOverlap: 0.7,
      posteriorSkewness: 0.1,
      nearZeroProbability: 0.02,
    },
  });
  const drawChannel = (name: string, scale: number) => ({
    channel: name,
    roi: scale,
    contribution: scale * 100,
    response: {
      adstockType: name === "c" ? "weibull" as const : "geometric" as const,
      adstock: 0.3 * scale,
      weibullShape: 1.2,
      weibullScale: 3,
      saturation: 1.1,
      halfSaturationQuantile: 0.5,
      kernelNormalization: "sum" as const,
    },
  });
  const base = {
    fingerprint: "fixture",
    contract: {},
    status: "labelled" as const,
    diagnostics: {
      finite: true,
      elboStable: true,
      seedAgreement: true,
      adjudicationUsed: false,
      maximumElboDrift: 0.01,
      maximumSeedLogRoiDifference: 0.02,
      predictiveCoverage: 0.9,
      maximumImplausibleProbability: 0.1,
      maximumRelativeRoiWidth: 1,
    },
    seeds: [],
    channels: [channel("a", 1), channel("b", 2), channel("c", 3)],
    decisionDraws: [
      {
        kernelBandwidth: 0.2,
        channels: [drawChannel("a", 1), drawChannel("b", 2), drawChannel("c", 3)],
      },
      {
        kernelBandwidth: 0.2,
        channels: [drawChannel("a", 1.1), drawChannel("b", 1.9), drawChannel("c", 3.2)],
      },
    ],
  } satisfies V9SviResult;
  const reversed = {
    ...base,
    channels: [...base.channels].reverse(),
    decisionDraws: base.decisionDraws.map((draw) => ({
      kernelBandwidth: draw.kernelBandwidth,
      channels: [...draw.channels].reverse(),
    })),
  } satisfies V9SviResult;
  const originalTokens = v9PosteriorDecisionTokens(base);
  const reversedTokens = v9PosteriorDecisionTokens(reversed);
  assert.equal(originalTokens.length, 96);
  originalTokens.forEach((value, index) =>
    assert.ok(Math.abs(value - reversedTokens[index]) < 1e-12)
  );
});

test("V9 audit preparation preserves zero-based shard zero", () => {
  assert.equal(
    integerArgument(["node", "script", "--shard=0"], "shard", 9, 0),
    0,
  );
  assert.equal(
    integerArgument(["node", "script", "--workers=8"], "workers", 1),
    8,
  );
});
