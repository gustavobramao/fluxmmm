import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  experimentCsvTemplate,
  parseExperimentCsv,
} from "../lib/mmm/experiment-csv";

const channels = [
  "tv_spend",
  "brand_search_spend",
  "facebook_prospecting_spend",
];
const dateRange = { minimum: "2021-01-01", maximum: "2025-12-31" };

test("imports the canonical Flux experiment schema atomically", async () => {
  const raw = await readFile(
    new URL("../public/data/robyn_experiments.csv", import.meta.url),
    "utf8",
  );
  const result = parseExperimentCsv(raw, ["facebook_S", "tv_S"], {
    minimum: "2015-11-23",
    maximum: "2019-11-11",
  });

  assert.equal(result.experiments.length, 2);
  assert.equal(result.experiments[0].channel, "facebook_S");
  assert.equal(result.experiments[0].incrementalOutcome, 40000);
  assert.equal(result.experiments[0].scope, "immediate");
});

test("recognizes common aliases and maps friendly channel names", () => {
  const raw = [
    "media_channel,campaign_start,campaign_end,carryover_end_date,roi,test_spend,roi_ci_low,roi_ci_high,confidence_level,study_name",
    "TV,2023/06/01,2023/07/01,2023/09/01,1.4,100000,1.1,1.7,95%,Geo lift",
  ].join("\n");
  const result = parseExperimentCsv(raw, channels, dateRange);
  const experiment = result.experiments[0];

  assert.equal(experiment.channel, "tv_spend");
  assert.equal(experiment.startDate, "2023-06-01");
  assert.equal(experiment.incrementalOutcome, 140000);
  assert.equal(experiment.incrementalSpend, 100000);
  assert.ok(Math.abs(experiment.standardError - 0.1530612245) < 1e-8);
  assert.equal(experiment.scope, "total");
  assert.ok(result.warnings.some((warning) => warning.includes("Mapped channel")));
});

test("reports the exact invalid row and does not silently invent uncertainty", () => {
  const raw = [
    "channel,start_date,end_date,incremental_outcome,incremental_spend",
    "TV,2023-06-01,2023-07-01,140000,100000",
  ].join("\n");

  assert.throws(
    () => parseExperimentCsv(raw, channels, dateRange),
    /Include standard_error, or both roi_low and roi_high/,
  );
});

test("rejects experiments outside the uploaded dataset", () => {
  const raw = [
    "channel,start_date,end_date,incremental_outcome,incremental_spend,standard_error",
    "brand_search,2020-01-01,2020-02-01,1000,5000,0.2",
  ].join("\n");

  assert.throws(
    () => parseExperimentCsv(raw, channels, dateRange),
    /Row 2: the experiment window must fall inside/,
  );
});

test("creates a canonical downloadable template", () => {
  const template = experimentCsvTemplate("tv_spend", {
    startDate: "2023-01-01",
    endDate: "2023-02-01",
    outcomeEndDate: "2023-03-01",
  });
  const result = parseExperimentCsv(template, channels, dateRange);

  assert.equal(result.experiments.length, 1);
  assert.equal(result.experiments[0].source, "Example lift study");
});
