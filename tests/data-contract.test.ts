import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { parseCsv } from "../lib/mmm/csv";
import {
  createDataset,
  updateColumnRole,
  validateDataset,
} from "../lib/mmm/schema";

function isoDate(timestamp: number): string {
  return new Date(timestamp).toISOString().slice(0, 10);
}

function slashDate(
  timestamp: number,
  order: "month-day" | "day-month",
): string {
  const date = new Date(timestamp);
  const month = date.getUTCMonth() + 1;
  const day = date.getUTCDate();
  const year = String(date.getUTCFullYear()).slice(-2);
  return order === "month-day"
    ? `${month}/${day}/${year}`
    : `${day}/${month}/${year}`;
}

function weeklyCsv(
  count = 104,
  mutate?: (row: string[], index: number) => string[],
  extraHeaders: string[] = [],
): string {
  const start = Date.UTC(2022, 0, 3);
  const headers = ["date", "revenue", "search_spend", "control", ...extraHeaders];
  const rows = Array.from({ length: count }, (_, index) => {
    const row = [
      isoDate(start + index * 7 * 86_400_000),
      String(10_000 + index * 31 + (index % 5) * 17),
      String(100 + (index % 9) * 23),
      String(50 + (index % 7)),
      ...extraHeaders.map((_, extraIndex) => String(index + extraIndex + 1)),
    ];
    return (mutate?.(row, index) ?? row).join(",");
  });
  return [headers.join(","), ...rows].join("\n");
}

async function datasetFromCsv(csv: string) {
  const parsed = parseCsv(csv);
  return createDataset("contract.csv", csv, parsed.columns, parsed.rows);
}

test("CSV parser rejects ambiguous headers and malformed rows", () => {
  assert.throws(
    () => parseCsv("date,revenue,Revenue\n2024-01-01,1,2"),
    /Duplicate CSV header/i,
  );
  assert.throws(
    () => parseCsv("date,revenue\n2024-01-01,1,extra"),
    /3 values; expected 2/i,
  );
  assert.throws(
    () => parseCsv('date,revenue\n2024-01-01,"100'),
    /unclosed quoted value/i,
  );
  assert.throws(
    () => parseCsv(",revenue\n2024-01-01,100"),
    /non-empty header/i,
  );
});

test("valid weekly uploads are sorted before any analysis", async () => {
  const ordered = weeklyCsv();
  const [header, ...rows] = ordered.split("\n");
  const reversed = [header, ...rows.reverse()].join("\n");
  const dataset = await datasetFromCsv(reversed);
  const validation = validateDataset(dataset);

  assert.equal(dataset.rows[0][dataset.dateColumn], "2022-01-03");
  assert.equal(dataset.chronologyReordered, true);
  assert.equal(validation.status, "ready");
  assert.equal(validation.cadence, "weekly");
  assert.equal(validation.intervalDays, 7);
  assert.equal(validation.gapCount, 0);
  assert.ok(
    validation.issues.some((issue) => issue.title === "Chronology normalized"),
  );
});

test("unambiguous slash dates are repaired automatically with an audit receipt", async () => {
  const start = Date.UTC(2022, 0, 3);
  for (const order of ["month-day", "day-month"] as const) {
    const slashCsv = weeklyCsv(104, (row, index) => {
      row[0] = slashDate(start + index * 7 * 86_400_000, order);
      return row;
    });
    const dataset = await datasetFromCsv(slashCsv);
    const validation = validateDataset(dataset);

    assert.equal(validation.status, "ready");
    assert.equal(validation.frequency, "Weekly");
    assert.equal(dataset.rows[0][dataset.dateColumn], "2022-01-03");
    assert.equal(validation.automaticRepairs.length, 1);
    assert.equal(validation.automaticRepairs[0].count, 104);
    assert.match(validation.automaticRepairs[0].detail, /source CSV remains unchanged/i);
  }
});

test("ambiguous slash dates still require an explicit user decision", async () => {
  const csv = [
    "date,revenue,search_spend",
    "1/2/24,1000,100",
    "1/9/24,1100,120",
  ].join("\n");
  const dataset = await datasetFromCsv(csv);
  const validation = validateDataset(dataset);

  assert.equal(validation.status, "blocked");
  assert.equal(validation.automaticRepairs.length, 0);
  assert.equal(validation.invalidDates, 2);
  assert.ok(validation.issues.some((issue) => issue.title === "Invalid dates"));
});

test("the public Robyn fixture satisfies the cadence-aware contract", async () => {
  const rawCsv = await readFile(
    new URL("../public/data/robyn_weekly.csv", import.meta.url),
    "utf8",
  );
  const dataset = await datasetFromCsv(rawCsv);
  const validation = validateDataset(dataset);

  assert.equal(validation.status, "ready");
  assert.equal(validation.frequency, "Weekly");
  assert.equal(validation.invalidDates, 0);
  assert.equal(validation.duplicateDates, 0);
  assert.equal(validation.modeledMissingCells, 0);
  assert.deepEqual(
    dataset.specs
      .filter((column) => column.role === "outcome")
      .map((column) => column.name),
    ["revenue"],
  );
});

test("daily uploads are aggregated into complete weekly modeling periods", async () => {
  const daily = weeklyCsv(728, (row, index) => {
    row[0] = isoDate(Date.UTC(2022, 0, 3) + index * 86_400_000);
    return row;
  });
  const dataset = await datasetFromCsv(daily);
  const validation = validateDataset(dataset);

  assert.equal(validation.status, "ready");
  assert.equal(validation.frequency, "Daily → Weekly");
  assert.equal(validation.sourceCadence, "daily");
  assert.equal(validation.modelCadence, "weekly");
  assert.equal(dataset.sourceRowCount, 728);
  assert.equal(dataset.rows.length, 104);
  assert.equal(dataset.periodsPerYear, 52.18);
  assert.equal(
    Number(dataset.rows[0].search_spend),
    Array.from({ length: 7 }, (_, index) => 100 + (index % 9) * 23).reduce(
      (sum, value) => sum + value,
      0,
    ),
  );
  assert.ok(
    validation.issues.some(
      (issue) => issue.title === "Daily data aggregated to weekly",
    ),
  );
});

test("monthly uploads remain native monthly modeling tables", async () => {
  const monthly = weeklyCsv(48, (row, index) => {
    row[0] = isoDate(Date.UTC(2015, index, 1));
    return row;
  });
  const dataset = await datasetFromCsv(monthly);
  const validation = validateDataset(dataset);

  assert.equal(validation.status, "ready");
  assert.equal(validation.frequency, "Monthly");
  assert.equal(validation.sourceCadence, "monthly");
  assert.equal(validation.modelCadence, "monthly");
  assert.equal(dataset.rows.length, 48);
  assert.equal(dataset.periodsPerYear, 12);
  assert.equal(dataset.periodUnit, "month");
});

test("irregular, duplicated, and gapped dates are blocked", async () => {
  const irregular = weeklyCsv(104, (row, index) => {
    if (index === 40) row[0] = isoDate(Date.UTC(2022, 0, 3) + (index * 7 + 2) * 86_400_000);
    return row;
  });
  const duplicated = weeklyCsv(104, (row, index) => {
    if (index === 40) row[0] = isoDate(Date.UTC(2022, 0, 3) + 39 * 7 * 86_400_000);
    return row;
  });
  const [gapHeader, ...gapRows] = weeklyCsv(105).split("\n");
  gapRows.splice(40, 1);
  const gapped = [gapHeader, ...gapRows].join("\n");

  for (const [csv, issue] of [
    [irregular, "Unsupported cadence"],
    [duplicated, "Duplicate dates"],
    [gapped, "Missing weekly periods"],
  ] as const) {
    const validation = validateDataset(await datasetFromCsv(csv));
    assert.equal(validation.status, "blocked");
    assert.ok(
      validation.issues.some((candidate) => candidate.title === issue),
      `${issue} should be reported`,
    );
  }
});

test("invalid, missing, and non-numeric modeled values are never coerced to zero", async () => {
  const invalidDate = weeklyCsv(104, (row, index) => {
    if (index === 12) row[0] = "2022-02-30";
    return row;
  });
  const missingOutcome = weeklyCsv(104, (row, index) => {
    if (index === 12) row[1] = "";
    return row;
  });
  const nonNumericSpend = weeklyCsv(104, (row, index) => {
    if (index === 12) row[2] = "unknown";
    return row;
  });

  const invalidValidation = validateDataset(await datasetFromCsv(invalidDate));
  const missingValidation = validateDataset(await datasetFromCsv(missingOutcome));
  const numericValidation = validateDataset(await datasetFromCsv(nonNumericSpend));

  assert.equal(invalidValidation.invalidDates, 1);
  assert.ok(invalidValidation.issues.some((issue) => issue.title === "Invalid dates"));
  assert.equal(missingValidation.modeledMissingCells, 1);
  assert.ok(
    missingValidation.issues.some((issue) => issue.title === "Missing modeled values"),
  );
  assert.ok(
    numericValidation.issues.some(
      (issue) => issue.title === "Non-numeric modeled values",
    ),
  );
});

test("semantic remapping creates a distinct dataset contract fingerprint", async () => {
  const dataset = await datasetFromCsv(weeklyCsv());
  const remapped = await updateColumnRole(dataset, "control", "ignore");

  assert.notEqual(remapped.hash, dataset.hash);
  assert.equal(remapped.sourceHash, dataset.sourceHash);
  assert.deepEqual(remapped.controlColumns, []);
  assert.equal(validateDataset(remapped).status, "ready");
});

test("controls beyond the four-column estimator limit are named explicitly", async () => {
  const dataset = await datasetFromCsv(
    weeklyCsv(104, undefined, ["control_2", "control_3", "control_4", "control_5"]),
  );
  const validation = validateDataset(dataset);

  assert.equal(validation.status, "review");
  assert.deepEqual(validation.excludedControlColumns, ["control_5"]);
  assert.ok(
    validation.issues.some((issue) => issue.title === "Control limit reached"),
  );
});
