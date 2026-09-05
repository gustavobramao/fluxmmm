import assert from "node:assert/strict";
import test from "node:test";
import learnedScoreV6Artifact from "../research/score_v6/artifacts/learned-score-v6-pilot.json";
import { DEFAULT_CONFIG } from "../lib/mmm/models";
import type { SamplingResult } from "../lib/mmm/sampling";
import type { ModelResult } from "../lib/mmm/types";
import { generateSyntheticBusiness } from "../research/score_v2/simulator";
import { PILOT_SCENARIOS } from "../research/score_v2/scenarios";
import type { ScoreV6CandidateRow } from "../research/score_v6/types";
import {
  posteriorDecisionLabel,
  samplingConverged,
} from "../research/mcmc_score_v3/posterior";
import { selectMcmcCandidates } from "../research/mcmc_score_v3/selection";
import { summarizePairedEvidence } from "../research/mcmc_score_v3/summary";
import type { McmcScoreV3SampleRecord } from "../research/mcmc_score_v3/types";

const diagnosticKeys = Object.keys(learnedScoreV6Artifact.model.weights);

function candidate(index: number, decisionLoss: number): { row: ScoreV6CandidateRow } {
  return {
    row: {
      businessId: "synthetic-business",
      family: "test",
      split: "train",
      candidateId: `candidate-${index}`,
      evidenceArm: index % 2 ? "experiments-only" : "benchmark-gap-fill",
      modelFamily: index % 3 ? "advanced" : "bayesian",
      eligible: true,
      reviewEligible: true,
      eligibilityTier: "decision-grade",
      failedGateCount: 0,
      layerScores: {
        generalization: 70,
        structure: 70,
        causal: 70,
        decision: 70,
      },
      diagnostics: Object.fromEntries(
        diagnosticKeys.map((key, featureIndex) => [
          key,
          35 + ((index * 17 + featureIndex * 11) % 65),
        ]),
      ) as ScoreV6CandidateRow["diagnostics"],
      heuristicScore: 70,
      decisionLoss,
      cappedRegret: 0.2,
      roiError: 0.3,
      contributionError: 0.25,
    },
  };
}

test("MCMC shortlist cannot use hidden MAP decision loss", () => {
  const original = Array.from({ length: 18 }, (_, index) => candidate(index, index));
  const inverted = original.map((item, index) => ({
    row: { ...item.row, decisionLoss: 10_000 - index * 101 },
  }));
  const first = selectMcmcCandidates(original, 10).map(
    (item) => `${item.candidate.row.candidateId}:${item.reason}`,
  );
  const second = selectMcmcCandidates(inverted, 10).map(
    (item) => `${item.candidate.row.candidateId}:${item.reason}`,
  );
  assert.deepEqual(first, second);
  assert.equal(new Set(first).size, 10);
});

test("posterior decision label requires converged aligned draws", () => {
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
  const convergenceIds = ["rhat", "ess", "divergences", "treedepth", "bfmi", "mcse"];
  const result = {
    gates: convergenceIds.map((id) => ({ id, passed: true })),
    channels: channels.map((channel) => ({
      channel: channel.channel,
      posteriorMedian: channel.roi,
    })),
    decisionDraws: Array.from({ length: 8 }, () => ({
      channels: business.truth.channels.map((channel) => ({
        channel: channel.spendColumn,
        roi: channel.targetRoi,
        contribution: channel.totalContribution,
        response: business.scenario.channels.find(
          (configured) => configured.channel === channel.channel,
        )!.response,
      })),
      kernelBandwidth: 0.18,
    })),
  } as unknown as SamplingResult;
  assert.equal(samplingConverged(result), true);
  const label = posteriorDecisionLabel(business, model, config, result, 8);
  assert.equal(label.convergencePassed, true);
  assert.ok(Number.isFinite(label.decisionLoss));

  result.gates[0].passed = false;
  const rejected = posteriorDecisionLabel(business, model, config, result, 8);
  assert.equal(rejected.convergencePassed, false);
  assert.equal(rejected.decisionLoss, undefined);
});

test("paired evidence clusters the uncertainty interval by business", () => {
  const record = (
    businessId: string,
    candidateId: string,
    mapDecisionLoss: number,
    mcmcDecisionLoss: number,
  ) => ({
    businessId,
    candidateId,
    mapDecisionLoss,
    mcmcDecisionLoss,
    status: "labelled",
    convergencePassed: true,
  }) as McmcScoreV3SampleRecord;
  const summary = summarizePairedEvidence([
    record("business-a", "a-1", 2, 1),
    record("business-a", "a-2", 4, 3),
    record("business-b", "b-1", 3, 2.5),
    record("business-b", "b-2", 5, 4.5),
  ]);
  assert.equal(summary?.pairs, 4);
  assert.equal(summary?.businesses, 2);
  assert.equal(summary?.mapMeanLoss, 3.5);
  assert.equal(summary?.mcmcMeanLoss, 2.75);
  assert.equal(summary?.meanLossShift, -0.75);
  assert.equal(summary?.mcmcWinRate, 1);
  assert.equal(summary?.interpretation, "mcmc-lower");
});
