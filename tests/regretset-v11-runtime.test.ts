import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import selectorJson from "../lib/mmm/artifacts/regretset-v11-selector.json";
import {
  REGRETSET_V11_CANDIDATE_COUNT,
  REGRETSET_V11_CANDIDATE_SHA256,
  REGRETSET_V11_SELECTOR_SHA256,
  regretSetV11CandidateSpecifications,
  regretSetV11Context,
  regretSetV11SelectorMetadata,
  scoreRegretSetV11,
} from "../lib/mmm/regretset-v11";
import type { Dataset } from "../lib/mmm/types";

const ROOT = new URL("../", import.meta.url);

function sha256(path: string): string {
  return createHash("sha256")
    .update(readFileSync(new URL(path, ROOT)))
    .digest("hex");
}

test("runtime artifacts are byte-identical to the frozen V11 contract", () => {
  assert.equal(
    sha256("lib/mmm/artifacts/regretset-v11-selector.json"),
    REGRETSET_V11_SELECTOR_SHA256,
  );
  assert.equal(
    sha256("lib/mmm/artifacts/regretset-v11-candidates.json"),
    REGRETSET_V11_CANDIDATE_SHA256,
  );
  const metadata = regretSetV11SelectorMetadata();
  assert.equal(metadata.modelCount, 5);
  assert.equal(metadata.featureNames.length, 232);
  assert.equal(metadata.contextNames.length, 63);
});

test("production candidate generator preserves the frozen 24 by 2 paired set", () => {
  const dataset = {
    mediaColumns: [
      "facebook_prospecting_spend",
      "nonbrand_search_spend",
      "tv_spend",
      "brand_search_spend",
    ],
  } as Dataset;
  const specifications = regretSetV11CandidateSpecifications(dataset, [], true);
  assert.equal(specifications.length, REGRETSET_V11_CANDIDATE_COUNT);
  const counts = new Map<string, number>();
  for (const specification of specifications) {
    const base = specification.id.split(" · ")[0];
    counts.set(base, (counts.get(base) ?? 0) + 1);
    assert.deepEqual(
      specification.evidencePriorChannels,
      specification.id.endsWith("benchmark-gap-fill")
        ? dataset.mediaColumns
        : [],
    );
  }
  assert.equal(counts.size, 24);
  assert.ok([...counts.values()].every((count) => count === 2));
  assert.ok(
    specifications.some((specification) =>
      Boolean(specification.config.channelResponses?.tv_spend),
    ),
  );
});

test("V11 scoring is finite, complete-set only, and permutation equivariant", () => {
  const selector = selectorJson as unknown as {
    models: { means: number[] }[];
  };
  const base = selector.models[0].means;
  const candidates = Array.from({ length: 24 }, (_, index) =>
    (["benchmark-gap-fill", "experiments-only"] as const).map((arm) => ({
      candidateId: `V6-C${String(index + 1).padStart(2, "0")} · ${arm}`,
      features: base.map(
        (value, feature) =>
          value + 0.01 * Math.sin((index + 1) * (feature + 1) * 0.017),
      ),
      valid: true,
      posteriorStatus: "labelled" as const,
    })),
  ).flat();
  assert.equal(regretSetV11Context(candidates).length, 63);
  const receipts = scoreRegretSetV11(candidates);
  assert.equal(receipts.length, 48);
  assert.equal(receipts.filter((receipt) => receipt.selected).length, 1);
  assert.deepEqual(
    receipts.map((receipt) => receipt.rank).sort((left, right) => left - right),
    Array.from({ length: 48 }, (_, index) => index + 1),
  );
  assert.ok(
    receipts.every((receipt) =>
      [
        receipt.predictedMean,
        receipt.predictedP90,
        receipt.predictedDanger,
        receipt.ensembleUncertainty,
        receipt.adjustedRisk,
      ].every(Number.isFinite),
    ),
  );

  const reversed = scoreRegretSetV11([...candidates].reverse());
  const originalWinner = receipts.find((receipt) => receipt.selected)!;
  const reversedWinner = reversed.find((receipt) => receipt.selected)!;
  assert.equal(reversedWinner.candidateId, originalWinner.candidateId);
  assert.ok(
    Math.abs(reversedWinner.adjustedRisk - originalWinner.adjustedRisk) < 1e-10,
  );

  assert.throws(
    () => scoreRegretSetV11(candidates.slice(0, 47)),
    /exactly 48 candidates/,
  );
});
