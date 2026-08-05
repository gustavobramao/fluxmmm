import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_ADVANCED_CONFIG,
  runAdvancedModel,
} from "../lib/mmm/advanced";
import { parseCsv } from "../lib/mmm/csv";
import { DEFAULT_CONFIG, runModel } from "../lib/mmm/models";
import { createDataset, validateDataset } from "../lib/mmm/schema";
import { runStructuralValidation } from "../lib/mmm/validation/structural";

type FixtureCadence = "daily" | "weekly" | "monthly";
type FixtureCase =
  | "high-correlation"
  | "duplicated"
  | "sparse"
  | "extreme-scale"
  | "zero-variance"
  | "overparameterized"
  | "too-short";

const CADENCES: FixtureCadence[] = ["daily", "weekly", "monthly"];
const STABLE_CASES: FixtureCase[] = [
  "high-correlation",
  "duplicated",
  "sparse",
  "extreme-scale",
];

function fixtureLength(cadence: FixtureCadence, short = false): number {
  if (cadence === "daily") return short ? 210 : 364;
  if (cadence === "weekly") return short ? 30 : 52;
  return short ? 18 : 24;
}

function fixtureDate(cadence: FixtureCadence, index: number): string {
  if (cadence === "monthly") {
    return new Date(Date.UTC(2019, index, 1)).toISOString().slice(0, 10);
  }
  const dayOffset = cadence === "daily" ? index : index * 7;
  return new Date(Date.UTC(2020, 0, 6 + dayOffset))
    .toISOString()
    .slice(0, 10);
}

function fixtureSpends(
  fixtureCase: FixtureCase,
  cadence: FixtureCadence,
  index: number,
): number[] {
  const base = 60 + (index % 11) * 9 + Math.floor(index / 11) % 7;
  if (fixtureCase === "high-correlation") {
    return [
      base,
      base * (1 + (index % 5) * 1e-6) + (index % 17 === 0 ? 0.01 : 0),
      35 + (index * 3) % 47,
      22 + (index * 5) % 31,
      18 + (index * 7) % 29,
    ];
  }
  if (fixtureCase === "duplicated") {
    return [
      base,
      base,
      35 + (index * 3) % 47,
      22 + (index * 5) % 31,
      18 + (index * 7) % 29,
    ];
  }
  if (fixtureCase === "sparse") {
    const period = cadence === "daily" ? 28 : 8;
    return [
      index % period === 0 ? 600 + index : 0,
      index % (period + 3) === 1 ? 420 + index : 0,
      index % (period + 5) === 2 ? 300 + index : 0,
      index % (period + 7) === 3 ? 240 + index : 0,
      index % (period + 9) === 4 ? 180 + index : 0,
    ];
  }
  if (fixtureCase === "extreme-scale") {
    return [
      1e10 + index * 1e7,
      1e-8 * (1 + (index % 13)),
      1e5 + (index % 17) * 2e4,
      0.01 + (index % 11) * 0.002,
      100 + (index % 19) * 11,
    ];
  }
  if (fixtureCase === "zero-variance") {
    return [
      100,
      40 + (index * 3) % 47,
      35 + (index * 5) % 43,
      22 + (index * 7) % 31,
      18 + (index * 11) % 29,
    ];
  }
  return [
    base,
    40 + (index * 3) % 47,
    35 + (index * 5) % 43,
    22 + (index * 7) % 31,
    18 + (index * 11) % 29,
  ];
}

function fixtureCsv(
  fixtureCase: FixtureCase,
  cadence: FixtureCadence,
): string {
  const count = fixtureLength(cadence, fixtureCase === "too-short");
  const rows = Array.from({ length: count }, (_, index) => {
    const spends = fixtureSpends(fixtureCase, cadence, index);
    const mediaSignal = spends.reduce(
      (total, value, mediaIndex) =>
        total + Math.log1p(Math.max(value, 0)) * (mediaIndex + 1) * 18,
      0,
    );
    const revenue =
      75_000 + index * 31 + mediaSignal + (index % 12) * 115;
    return [
      fixtureDate(cadence, index),
      revenue,
      ...spends,
      20 + (index % 9),
    ].join(",");
  });
  return [
    "date,revenue,search_spend,social_spend,video_spend,retail_spend,audio_spend,control",
    ...rows,
  ].join("\n");
}

async function fixtureDataset(
  fixtureCase: FixtureCase,
  cadence: FixtureCadence,
) {
  const csv = fixtureCsv(fixtureCase, cadence);
  const parsed = parseCsv(csv);
  return createDataset(
    `${fixtureCase}-${cadence}.csv`,
    csv,
    parsed.columns,
    parsed.rows,
  );
}

for (const cadence of CADENCES) {
  test(`${cadence} difficult designs remain finite and reproducible`, async () => {
    for (const fixtureCase of STABLE_CASES) {
      const dataset = await fixtureDataset(fixtureCase, cadence);
      const validation = validateDataset(dataset);
      assert.notEqual(
        validation.status,
        "blocked",
        `${fixtureCase} should remain modelable at ${cadence} cadence: ${validation.issues.map((issue) => issue.title).join(", ")}`,
      );
      const config = {
        ...DEFAULT_CONFIG,
        cyclePeriod: cadence === "monthly" ? 3 : 26,
      };
      const first = await runModel(
        dataset,
        config,
        [],
        "frequentist",
        `${fixtureCase}-${cadence}-first`,
      );
      const second = await runModel(
        dataset,
        config,
        [],
        "frequentist",
        `${fixtureCase}-${cadence}-second`,
      );

      assert.ok(first.predicted.every(Number.isFinite));
      assert.ok(first.channels.every((channel) => Number.isFinite(channel.roi)));
      assert.deepEqual(first.predicted, second.predicted);
      assert.deepEqual(first.channels, second.channels);
      assert.ok(first.numerical);
      if (fixtureCase === "duplicated") {
        assert.ok(
          first.numerical.dataRank < first.numerical.parameterCount,
          "duplicate media must remain visible in the observational data rank",
        );
        assert.equal(first.numerical.status, "review");
      }
    }
  });

  test(`${cadence} zero-variance media fails closed with a specific diagnostic`, async () => {
    const dataset = await fixtureDataset("zero-variance", cadence);
    const validation = validateDataset(dataset);
    assert.equal(validation.status, "blocked");
    assert.ok(
      validation.issues.some(
        (issue) =>
          issue.level === "error" && issue.title === "Media without variation",
      ),
    );
  });

  test(`${cadence} excessive parameterization is solved but explicitly identified`, async () => {
    const dataset = await fixtureDataset("overparameterized", cadence);
    const validation = validateDataset(dataset);
    assert.notEqual(validation.status, "blocked");
    const result = await runAdvancedModel(
      dataset,
      {
        ...DEFAULT_CONFIG,
        cyclePeriod: cadence === "monthly" ? 3 : 26,
      },
      {
        ...DEFAULT_ADVANCED_CONFIG,
        timeVarying: true,
        kernelKnots: 10,
      },
      [],
      `overparameterized-${cadence}`,
    );

    assert.ok(result.predicted.every(Number.isFinite));
    assert.ok(result.numerical);
    assert.ok(result.numerical.dataRank < result.numerical.parameterCount);
    assert.equal(result.numerical.status, "review");
  });

  test(`${cadence} insufficient history stops with an understandable diagnostic`, async () => {
    const dataset = await fixtureDataset("too-short", cadence);
    const validation = validateDataset(dataset);
    assert.equal(validation.status, "blocked");
    assert.ok(
      validation.issues.some(
        (issue) => issue.level === "error" && issue.title === "Too little history",
      ),
    );
  });
}

test("material non-negative clipping preserves the boundary and discloses approximation impact", async () => {
  const rows = Array.from({ length: 104 }, (_, index) => {
    const search = index % 4 < 2 ? 1_000 : 10;
    const social = 50 + (index * 7) % 83;
    const video = 30 + (index * 5) % 61;
    const retail = 25 + (index * 3) % 47;
    const audio = 15 + (index * 11) % 37;
    const revenue =
      220_000 - search * 80 + social * 35 + video * 18 + (index % 13) * 90;
    return [
      fixtureDate("weekly", index),
      revenue,
      search,
      social,
      video,
      retail,
      audio,
      20 + (index % 9),
    ].join(",");
  });
  const csv = [
    "date,revenue,search_spend,social_spend,video_spend,retail_spend,audio_spend,control",
    ...rows,
  ].join("\n");
  const parsed = parseCsv(csv);
  const dataset = await createDataset(
    "negative-media-signal-weekly.csv",
    csv,
    parsed.columns,
    parsed.rows,
  );
  const result = await runModel(
    dataset,
    DEFAULT_CONFIG,
    [],
    "frequentist",
    "material-clipping-fixture",
  );
  const clipping = result.numerical?.clipping;

  assert.ok(clipping?.applied);
  assert.equal(clipping.severity, "warning");
  assert.ok(clipping.material);
  assert.ok(clipping.channels.length >= 1);
  for (const channel of clipping.channels) {
    assert.ok(channel.unconstrainedCoefficient < 0);
    assert.equal(channel.constrainedCoefficient, 0);
    assert.ok(channel.contributionChange > 0);
    assert.ok(Number.isFinite(channel.unconstrainedRoi));
    assert.ok(Number.isFinite(channel.constrainedRoi));
  }

  const structure = runStructuralValidation(
    dataset,
    {
      kind: "frequentist",
      config: DEFAULT_CONFIG,
      advancedConfig: DEFAULT_ADVANCED_CONFIG,
      experiments: [],
      validationOptions: { anchorIndependenceConfirmed: false },
    },
    result,
  );
  const boundary = structure.tests.find(
    (diagnostic) => diagnostic.id === "non-negative-boundary",
  );
  assert.ok(boundary);
  assert.notEqual(boundary.status, "pass");
  assert.match(boundary.detail, /Gaussian approximation requires structural review/i);
});
