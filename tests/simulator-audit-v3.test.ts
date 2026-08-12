import assert from "node:assert/strict";
import test from "node:test";
import { roiEvidence } from "../research/score_v2/evidence";
import auditArtifact from "../research/score_v3/artifacts/simulator-audit-v3-summary.json";
import {
  GENERATOR_FAMILIES,
  randomizedAuditScenario,
} from "../research/score_v3/population";
import type { SimulatorAuditV3PublicArtifact } from "../research/score_v3/types";

const population = auditArtifact as SimulatorAuditV3PublicArtifact;

test("V3 predeclares a 500-business family-level split", () => {
  assert.equal(population.businessCount, 500);
  assert.deepEqual(population.splits, {
    train: 300,
    validation: 120,
    audit: 80,
  });
  assert.equal(
    population.families
      .filter((family) => family.split === "audit")
      .every((family) => Boolean(family.heldOutReason)),
    true,
  );
});

test("V3 generation is deterministic and keeps evidence draws in range", () => {
  const family = GENERATOR_FAMILIES[1];
  const first = randomizedAuditScenario(family, 7);
  const second = randomizedAuditScenario(family, 7);
  assert.deepEqual(first, second);
  first.channels.forEach((channel) => {
    assert.equal(channel.targetRoi, undefined);
    const evidence = roiEvidence(channel.roiEvidenceId);
    assert.ok(evidence.low > 0);
    assert.ok(evidence.high > evidence.low);
  });
});

test("V3 spans response families and economically distinct decisions", () => {
  assert.ok(population.coverage.channel.paid_social.weibullShare > 0.1);
  assert.ok(population.coverage.channel.paid_social.weibullShare < 0.5);
  assert.ok(population.coverage.channel.tv.weibullShare > 0.6);
  assert.ok(population.coverage.channel.search.demandCoupling.p90 > 0.8);
  assert.ok(population.decisions["budget-reduction"].decreaseShare > 0.2);
  assert.ok(population.decisions["budget-growth"].increaseShare > 0.2);
  assert.ok(population.decisions["economic-ceiling"].decreaseShare > 0.1);
  assert.ok(population.decisions["economic-ceiling"].increaseShare > 0.1);
});
