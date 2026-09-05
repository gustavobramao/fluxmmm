import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_CONFIG } from "../lib/mmm/models";
import type { SamplingDecisionDraw, SamplingResult } from "../lib/mmm/sampling";
import type { ModelResult } from "../lib/mmm/types";
import {
  evaluateAlignedPosteriorDecisionTruth,
} from "../research/score_v2/evaluate";
import { PILOT_SCENARIOS } from "../research/score_v2/scenarios";
import { generateSyntheticBusiness } from "../research/score_v2/simulator";
import {
  posteriorDecisionLabel,
} from "../research/mcmc_score_v3/posterior";
import {
  SVI_SCORE_V8_CONTRACT,
  V8_DECISION_SCENARIOS,
  V8_RISK_PREFERENCE,
} from "../research/svi_score_v8/contract";

test("V8 freezes an SVI-only development contract and leaves audit closed", () => {
  assert.equal(SVI_SCORE_V8_CONTRACT.inference.contract.method, "fullrank-advi");
  assert.equal(SVI_SCORE_V8_CONTRACT.inference.contract.iterations, 5_000);
  assert.equal(SVI_SCORE_V8_CONTRACT.inference.contract.draws, 256);
  assert.equal(SVI_SCORE_V8_CONTRACT.inference.nutsPermittedInResearch, false);
  assert.equal(SVI_SCORE_V8_CONTRACT.tokens.count, 136);
  assert.equal(SVI_SCORE_V8_CONTRACT.cohort.developmentBusinesses, 420);
  assert.equal(SVI_SCORE_V8_CONTRACT.cohort.sealedAuditBusinesses, 80);
  assert.equal(SVI_SCORE_V8_CONTRACT.cohort.auditMayOpenDuringDevelopment, false);
  assert.equal(SVI_SCORE_V8_CONTRACT.objective.pairSamplingPermitted, false);
  assert.deepEqual(V8_RISK_PREFERENCE, { mean: 0.65, p90: 0.35 });
});

test("V8 reconstructs the same aggregate SVI label and preserves all scenarios", () => {
  const business = generateSyntheticBusiness(PILOT_SCENARIOS[0]);
  const channelResponses = Object.fromEntries(
    business.truth.channels.map((channel) => [
      channel.spendColumn,
      business.scenario.channels.find(
        (configured) => configured.channel === channel.channel,
      )!.response,
    ]),
  );
  const config = { ...DEFAULT_CONFIG, channelResponses };
  const channels = business.truth.channels.map((channel) => ({
    channel: channel.spendColumn,
    contribution: channel.totalContribution,
    contributionShare: 1 / business.truth.channels.length,
    roi: channel.targetRoi,
    roiLow: channel.targetRoi * 0.8,
    roiHigh: channel.targetRoi * 1.2,
    coefficient: 1,
  }));
  const model = { channels } as unknown as ModelResult;
  const decisionDraws = Array.from({ length: 8 }, () => ({
    channels: business.truth.channels.map((channel) => ({
      channel: channel.spendColumn,
      roi: channel.targetRoi,
      contribution: channel.totalContribution,
      response: business.scenario.channels.find(
        (configured) => configured.channel === channel.channel,
      )!.response,
    })),
    kernelBandwidth: 0.18,
  })) as unknown as SamplingDecisionDraw[];
  const result = {
    gates: ["rhat", "ess", "divergences", "treedepth", "bfmi", "mcse"].map(
      (id) => ({ id, passed: true }),
    ),
    channels: channels.map((channel) => ({
      channel: channel.channel,
      posteriorMedian: channel.roi,
    })),
    decisionDraws,
  } as unknown as SamplingResult;
  const historical = posteriorDecisionLabel(business, model, config, result, 8);
  const reconstructed = evaluateAlignedPosteriorDecisionTruth(
    business,
    decisionDraws,
    8,
  );
  assert.equal(reconstructed.decisionLoss, historical.decisionLoss);
  assert.equal(reconstructed.profitRegret, historical.profitRegret);
  assert.deepEqual(
    reconstructed.scenarioDecisionLoss,
    historical.scenarioDecisionLoss,
  );
  assert.deepEqual(
    Object.keys(reconstructed.scenarioDecisionLoss).sort(),
    [...V8_DECISION_SCENARIOS].sort(),
  );
});

test("V8 reconstruction preserves historical response-bound semantics", () => {
  const business = generateSyntheticBusiness(PILOT_SCENARIOS[0]);
  const channelResponses = Object.fromEntries(
    business.truth.channels.map((channel) => [
      channel.spendColumn,
      business.scenario.channels.find(
        (configured) => configured.channel === channel.channel,
      )!.response,
    ]),
  );
  const config = { ...DEFAULT_CONFIG, channelResponses };
  const channels = business.truth.channels.map((channel) => ({
    channel: channel.spendColumn,
    contribution: channel.totalContribution,
    contributionShare: 1 / business.truth.channels.length,
    roi: channel.targetRoi,
    roiLow: channel.targetRoi * 0.8,
    roiHigh: channel.targetRoi * 1.2,
    coefficient: 1,
  }));
  const model = { channels } as unknown as ModelResult;
  const decisionDraws = Array.from({ length: 8 }, (_, drawIndex) => ({
    channels: business.truth.channels.map((channel, channelIndex) => ({
      channel: channel.spendColumn,
      roi: channel.targetRoi,
      contribution: channel.totalContribution,
      response: {
        ...business.scenario.channels.find(
          (configured) => configured.channel === channel.channel,
        )!.response,
        adstock: channelIndex === 0 && drawIndex % 2 === 0 ? 1.08 : 0.005,
        saturation: channelIndex === 1 ? 6.2 : 0.18,
      },
    })),
    kernelBandwidth: 0.18,
  })) as unknown as SamplingDecisionDraw[];
  const result = {
    gates: ["rhat", "ess", "divergences", "treedepth", "bfmi", "mcse"].map(
      (id) => ({ id, passed: true }),
    ),
    channels: channels.map((channel) => ({
      channel: channel.channel,
      posteriorMedian: channel.roi,
    })),
    decisionDraws,
  } as unknown as SamplingResult;
  const historical = posteriorDecisionLabel(business, model, config, result, 8);
  const reconstructed = evaluateAlignedPosteriorDecisionTruth(
    business,
    decisionDraws,
    8,
  );
  assert.equal(reconstructed.decisionLoss, historical.decisionLoss);
  assert.deepEqual(
    reconstructed.scenarioDecisionLoss,
    historical.scenarioDecisionLoss,
  );
});
