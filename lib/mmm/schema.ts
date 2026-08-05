import { sha256 } from "./csv";
import type {
  ColumnRole,
  ColumnSpec,
  Dataset,
  DatasetCadence,
  DatasetRepair,
  DataRow,
  ModelCadence,
  ValidationIssue,
  ValidationResult,
} from "./types";

const DATE_NAMES = ["date", "day", "week", "month", "ds"];
const OUTCOME_NAMES = [
  "revenue",
  "sales",
  "conversions",
  "orders",
  "outcome",
  "kpi",
  "y",
];
const DAY_MS = 86_400_000;
const MAX_MODELED_CONTROLS = 4;
const DATA_CONTRACT_VERSION = "cadence-v1.2-daily-weekly-monthly";

function isMissing(value: string | number | undefined): boolean {
  return value === undefined || (typeof value === "string" && value.trim() === "");
}

export function parseDatasetDate(
  value: string | number | undefined,
): number {
  if (isMissing(value)) return Number.NaN;
  const raw = String(value).trim();
  if (!/^\d{4}-\d{2}-\d{2}$/.test(raw)) return Number.NaN;

  const timestamp = Date.parse(raw);
  if (!Number.isFinite(timestamp)) return Number.NaN;

  const [year, month, day] = raw
    .slice(0, 10)
    .split("-")
    .map(Number);
  const parsed = new Date(timestamp);
  if (
    parsed.getUTCFullYear() !== year ||
    parsed.getUTCMonth() + 1 !== month ||
    parsed.getUTCDate() !== day
  ) {
    return Number.NaN;
  }
  return timestamp;
}

function isoDateFromParts(year: number, month: number, day: number): string | null {
  const timestamp = Date.UTC(year, month - 1, day);
  const parsed = new Date(timestamp);
  if (
    parsed.getUTCFullYear() !== year ||
    parsed.getUTCMonth() + 1 !== month ||
    parsed.getUTCDate() !== day
  ) {
    return null;
  }
  return parsed.toISOString().slice(0, 10);
}

function expandTwoDigitYear(year: number): number {
  if (year >= 100) return year;
  return year <= 68 ? 2000 + year : 1900 + year;
}

function normalizeUnambiguousDateColumn(
  rows: DataRow[],
  dateColumn: string,
): { rows: DataRow[]; repairs: DatasetRepair[] } {
  const populated = rows
    .map((row) => row[dateColumn])
    .filter((value) => !isMissing(value));
  if (!populated.length || populated.every((value) => Number.isFinite(parseDatasetDate(value)))) {
    return { rows: [...rows], repairs: [] };
  }

  const parts = populated.map((value) =>
    String(value)
      .trim()
      .match(/^(\d{1,2})\/(\d{1,2})\/(\d{2}|\d{4})$/),
  );
  if (parts.some((match) => !match)) {
    return { rows: [...rows], repairs: [] };
  }

  const firstExceedsMonth = parts.some((match) => Number(match?.[1]) > 12);
  const secondExceedsMonth = parts.some((match) => Number(match?.[2]) > 12);
  if (firstExceedsMonth === secondExceedsMonth) {
    return { rows: [...rows], repairs: [] };
  }

  const order = secondExceedsMonth ? "month-day-year" : "day-month-year";
  let repairCount = 0;
  const normalized = rows.map((row) => {
    const value = row[dateColumn];
    if (isMissing(value)) return { ...row };
    const match = String(value)
      .trim()
      .match(/^(\d{1,2})\/(\d{1,2})\/(\d{2}|\d{4})$/);
    if (!match) return { ...row };
    const first = Number(match[1]);
    const second = Number(match[2]);
    const year = expandTwoDigitYear(Number(match[3]));
    const month = order === "month-day-year" ? first : second;
    const day = order === "month-day-year" ? second : first;
    const iso = isoDateFromParts(year, month, day);
    if (!iso) return { ...row };
    repairCount += 1;
    return { ...row, [dateColumn]: iso };
  });

  if (
    repairCount !== populated.length ||
    normalized.some((row) =>
      !isMissing(row[dateColumn]) && !Number.isFinite(parseDatasetDate(row[dateColumn])),
    )
  ) {
    return { rows: [...rows], repairs: [] };
  }

  const sourceFormat =
    order === "month-day-year" ? "month/day/year" : "day/month/year";
  return {
    rows: normalized,
    repairs: [
      {
        id: "date-format",
        column: dateColumn,
        count: repairCount,
        title: "Date format standardized",
        detail: `${repairCount} ${sourceFormat} value${repairCount === 1 ? " was" : "s were"} converted to the internal YYYY-MM-DD contract. The source CSV remains unchanged.`,
      },
    ],
  };
}

function aggregateValues(
  rows: DataRow[],
  spec: ColumnSpec,
): string | number {
  const values = rows.map((row) => row[spec.name]);
  if (values.some(isMissing)) return "";
  if (spec.role === "date") return values[0];

  if (
    spec.role === "outcome" ||
    spec.role === "media_spend" ||
    spec.role === "media_exposure"
  ) {
    if (values.some((value) => !Number.isFinite(Number(value)))) {
      return String(values.find((value) => !Number.isFinite(Number(value))) ?? "");
    }
    return values.reduce<number>((total, value) => total + Number(value), 0);
  }

  if (spec.role === "control" && spec.numeric) {
    if (values.some((value) => !Number.isFinite(Number(value)))) {
      return String(values.find((value) => !Number.isFinite(Number(value))) ?? "");
    }
    return (
      values.reduce<number>((total, value) => total + Number(value), 0) /
      Math.max(values.length, 1)
    );
  }

  if (spec.role === "categorical") {
    const counts = new Map<string, number>();
    values.forEach((value) => {
      const key = String(value);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    });
    return [...counts.entries()].sort(
      (left, right) => right[1] - left[1] || left[0].localeCompare(right[0]),
    )[0]?.[0] ?? "";
  }

  return values.at(-1) ?? "";
}

function aggregateDailyRows(
  rows: DataRow[],
  specs: ColumnSpec[],
  dateColumn: string,
): { rows: DataRow[]; repair: DatasetRepair } {
  const fullDayCount = Math.floor(rows.length / 7) * 7;
  const omittedDays = rows.length - fullDayCount;
  const completeRows = rows.slice(0, fullDayCount);
  const weeklyRows = Array.from(
    { length: Math.floor(completeRows.length / 7) },
    (_, weekIndex) => {
      const group = completeRows.slice(weekIndex * 7, weekIndex * 7 + 7);
      return Object.fromEntries(
        specs.map((spec) => [
          spec.name,
          spec.name === dateColumn
            ? group[0]?.[dateColumn] ?? ""
            : aggregateValues(group, spec),
        ]),
      );
    },
  );
  return {
    rows: weeklyRows,
    repair: {
      id: "daily-aggregation",
      column: dateColumn,
      count: rows.length,
      title: "Daily data aggregated to weekly",
      detail: `${rows.length} consecutive daily rows produced ${weeklyRows.length} complete seven-day modeling periods. Outcome, spend, and exposure were summed; numeric controls were averaged; categorical controls used their mode.${omittedDays ? ` ${omittedDays} trailing day${omittedDays === 1 ? " was" : "s were"} omitted because it did not form a complete period.` : ""} The source CSV remains unchanged.`,
    },
  };
}

function prepareRows(
  rows: DataRow[],
  dateColumn: string,
  specs: ColumnSpec[],
): {
  rows: DataRow[];
  reordered: boolean;
  repairs: DatasetRepair[];
  sourceCadence: DatasetCadence;
  modelCadence: ModelCadence;
  periodsPerYear: number;
  periodUnit: "week" | "month";
} {
  const normalized = normalizeUnambiguousDateColumn(rows, dateColumn);
  const chronology = normalizeChronology(normalized.rows, dateColumn);
  const timestamps = chronology.rows
    .map((row) => parseDatasetDate(row[dateColumn]))
    .filter(Number.isFinite);
  const sourceCadence = cadenceFromDates(timestamps).cadence;
  if (sourceCadence === "daily") {
    const aggregation = aggregateDailyRows(
      chronology.rows,
      specs,
      dateColumn,
    );
    return {
      rows: aggregation.rows,
      reordered: chronology.reordered,
      repairs: [...normalized.repairs, aggregation.repair],
      sourceCadence,
      modelCadence: "weekly",
      periodsPerYear: 52.18,
      periodUnit: "week",
    };
  }
  const modelCadence: ModelCadence =
    sourceCadence === "monthly" ? "monthly" : "weekly";
  return {
    rows: chronology.rows,
    reordered: chronology.reordered,
    repairs: normalized.repairs,
    sourceCadence,
    modelCadence,
    periodsPerYear: modelCadence === "monthly" ? 12 : 52.18,
    periodUnit: modelCadence === "monthly" ? "month" : "week",
  };
}

function normalizeChronology(
  rows: DataRow[],
  dateColumn: string,
): { rows: DataRow[]; reordered: boolean } {
  const indexed = rows.map((row, index) => ({
    row,
    index,
    timestamp: parseDatasetDate(row[dateColumn]),
  }));
  if (indexed.some((entry) => !Number.isFinite(entry.timestamp))) {
    return { rows: [...rows], reordered: false };
  }
  const sorted = [...indexed].sort(
    (left, right) => left.timestamp - right.timestamp || left.index - right.index,
  );
  return {
    rows: sorted.map((entry) => entry.row),
    reordered: sorted.some((entry, index) => entry.index !== index),
  };
}

function datasetContract(
  sourceHash: string,
  specs: ColumnSpec[],
  dateColumn: string,
  outcomeColumn: string,
): string {
  return JSON.stringify({
    version: DATA_CONTRACT_VERSION,
    sourceHash,
    dateColumn,
    outcomeColumn,
    columns: specs.map(({ name, role, channel, numeric }) => ({
      name,
      role,
      channel: channel ?? null,
      numeric,
    })),
  });
}

async function contractHash(
  sourceHash: string,
  specs: ColumnSpec[],
  dateColumn: string,
  outcomeColumn: string,
): Promise<string> {
  return sha256(datasetContract(sourceHash, specs, dateColumn, outcomeColumn));
}

function cadenceFromDates(timestamps: number[]): {
  cadence: DatasetCadence;
  intervalDays: number | null;
  gapCount: number;
} {
  if (timestamps.length < 2) {
    return { cadence: "unknown", intervalDays: null, gapCount: 0 };
  }
  const sorted = [...timestamps].sort((left, right) => left - right);
  const deltas = sorted.slice(1).map(
    (timestamp, index) => (timestamp - sorted[index]) / DAY_MS,
  );
  const ordered = [...deltas].sort((left, right) => left - right);
  const median = ordered[Math.floor(ordered.length / 2)] ?? null;
  const exact = (target: number) =>
    deltas.every((delta) => Math.abs(delta - target) < 1e-6);

  if (exact(7)) {
    return { cadence: "weekly", intervalDays: 7, gapCount: 0 };
  }
  if (
    deltas.every(
      (delta) =>
        delta >= 7 &&
        Math.abs(delta / 7 - Math.round(delta / 7)) < 1e-6,
    )
  ) {
    return {
      cadence: "weekly",
      intervalDays: 7,
      gapCount: deltas.reduce(
        (count, delta) => count + Math.max(0, Math.round(delta / 7) - 1),
        0,
      ),
    };
  }
  if (exact(1)) {
    return { cadence: "daily", intervalDays: 1, gapCount: 0 };
  }
  if (deltas.every((delta) => delta >= 28 && delta <= 31)) {
    return { cadence: "monthly", intervalDays: median, gapCount: 0 };
  }
  return { cadence: "irregular", intervalDays: median, gapCount: 0 };
}

function isNumericColumn(rows: DataRow[], column: string): boolean {
  const populated = rows
    .map((row) => row[column])
    .filter((value) => value !== "" && value !== undefined);
  if (populated.length === 0) return false;
  return (
    populated.filter((value) => Number.isFinite(Number(value))).length /
      populated.length >
    0.9
  );
}

function inferRole(name: string, numeric: boolean): ColumnRole {
  const lower = name.toLowerCase();
  const tokens = lower.split(/[^a-z0-9]+/).filter(Boolean);
  if (DATE_NAMES.some((candidate) => lower === candidate || tokens.includes(candidate))) {
    return "date";
  }
  if (
    !/competitor|market|category/.test(lower) &&
    OUTCOME_NAMES.some(
      (candidate) => lower === candidate || tokens.includes(candidate),
    )
  ) {
    return "outcome";
  }
  if (
    /(_s$|spend|cost|investment)/i.test(name) &&
    !/competitor/i.test(name)
  ) {
    return "media_spend";
  }
  if (/(impression|click|reach|_i$|_p$)/i.test(name)) {
    return "media_exposure";
  }
  if (!numeric) return "categorical";
  return "control";
}

function channelFromColumn(name: string): string {
  return name
    .replace(/_(S|I|P)$/i, "")
    .replace(/_(spend|cost|impressions?|clicks?)$/i, "")
    .replace(/_/g, " ");
}

export async function createDataset(
  name: string,
  rawCsv: string,
  columns: string[],
  rows: DataRow[],
): Promise<Dataset> {
  const specs: ColumnSpec[] = columns.map((column) => {
    const numeric = isNumericColumn(rows, column);
    const role = inferRole(column, numeric);
    return {
      name: column,
      role,
      numeric,
      ...(role === "media_spend" || role === "media_exposure"
        ? { channel: channelFromColumn(column) }
        : {}),
    };
  });

  const dateColumn =
    specs.find((column) => column.role === "date")?.name ?? columns[0];
  const outcomeColumn =
    specs.find((column) => column.role === "outcome")?.name ?? columns[1];
  const mediaColumns = specs
    .filter((column) => column.role === "media_spend")
    .map((column) => column.name);
  const controlColumns = specs
    .filter((column) => column.role === "control")
    .map((column) => column.name);

  const sourceHash = await sha256(rawCsv);
  const prepared = prepareRows(rows, dateColumn, specs);

  return {
    name,
    rawCsv,
    sourceRows: rows.map((row) => ({ ...row })),
    sourceRowCount: rows.length,
    rows: prepared.rows,
    columns,
    specs,
    dateColumn,
    outcomeColumn,
    mediaColumns,
    controlColumns,
    sourceHash,
    hash: await contractHash(sourceHash, specs, dateColumn, outcomeColumn),
    chronologyReordered: prepared.reordered,
    automaticRepairs: prepared.repairs,
    sourceCadence: prepared.sourceCadence,
    modelCadence: prepared.modelCadence,
    periodsPerYear: prepared.periodsPerYear,
    periodUnit: prepared.periodUnit,
  };
}

export function validateDataset(dataset: Dataset): ValidationResult {
  const issues: ValidationIssue[] = [];
  dataset.automaticRepairs.forEach((repair) => {
    issues.push({
      level: "info",
      title: repair.title,
      detail: repair.detail,
    });
  });
  const parsedDates = dataset.rows.map((row) =>
    parseDatasetDate(row[dataset.dateColumn]),
  );
  const validDates = parsedDates.filter(Number.isFinite);
  const duplicateDates = validDates.length - new Set(validDates).size;
  const invalidDates = parsedDates.length - validDates.length;
  const activeControlColumns = dataset.controlColumns.slice(
    0,
    MAX_MODELED_CONTROLS,
  );
  const excludedControlColumns = dataset.controlColumns.slice(
    MAX_MODELED_CONTROLS,
  );
  const modeledColumns = [
    dataset.dateColumn,
    dataset.outcomeColumn,
    ...dataset.mediaColumns,
    ...activeControlColumns,
  ].filter(Boolean);
  const missingCells = dataset.rows.reduce(
    (count, row) =>
      count +
      dataset.columns.filter(
        (column) => row[column] === "" || row[column] === undefined,
      ).length,
    0,
  );
  const modeledMissingCells = dataset.rows.reduce(
    (count, row) =>
      count + modeledColumns.filter((column) => isMissing(row[column])).length,
    0,
  );

  const dateRoles = dataset.specs.filter((column) => column.role === "date");
  const outcomeRoles = dataset.specs.filter(
    (column) => column.role === "outcome",
  );

  if (dateRoles.length !== 1) {
    issues.push({
      level: "error",
      title: "Map one date column",
      detail:
        dateRoles.length === 0
          ? "Exactly one column must be mapped to Date."
          : `${dateRoles.length} columns are mapped to Date; keep exactly one.`,
    });
  }

  if (outcomeRoles.length !== 1) {
    issues.push({
      level: "error",
      title: "Map one outcome column",
      detail:
        outcomeRoles.length === 0
          ? "Exactly one numeric column must be mapped to Outcome."
          : `${outcomeRoles.length} columns are mapped to Outcome; keep exactly one.`,
    });
  }

  const minimumPeriods = dataset.modelCadence === "monthly" ? 24 : 52;
  const preferredPeriods = dataset.modelCadence === "monthly" ? 36 : 104;
  if (dataset.rows.length < minimumPeriods) {
    issues.push({
      level: "error",
      title: "Too little history",
      detail: `MMM needs at least ${minimumPeriods} ${dataset.periodUnit === "week" ? "weekly" : "monthly"} periods; ${preferredPeriods}+ is preferred.`,
    });
  } else if (dataset.rows.length < preferredPeriods) {
    issues.push({
      level: "warning",
      title: "Limited history",
      detail: "Annual effects may be weakly identified with fewer than two years.",
    });
  }

  if (!dataset.mediaColumns.length) {
    issues.push({
      level: "error",
      title: "No spend columns found",
      detail: "Map at least one numeric column to media spend.",
    });
  }

  if (duplicateDates > 0) {
    issues.push({
      level: "error",
      title: "Duplicate dates",
      detail: `${duplicateDates} duplicated period${duplicateDates === 1 ? "" : "s"} detected.`,
    });
  }

  if (invalidDates > 0) {
    issues.push({
      level: "error",
      title: "Invalid dates",
      detail: `${invalidDates} date value${invalidDates === 1 ? " is" : "s are"} invalid. Use ISO dates formatted YYYY-MM-DD.`,
    });
  }

  if (modeledMissingCells > 0) {
    issues.push({
      level: "error",
      title: "Missing modeled values",
      detail: `${modeledMissingCells} empty cell${modeledMissingCells === 1 ? " appears" : "s appear"} in the date, outcome, media, or active-control contract. Flux will not silently replace them with zero.`,
    });
  } else if (missingCells > 0) {
    issues.push({
      level: "warning",
      title: "Missing optional values",
      detail: `${missingCells} empty cell${missingCells === 1 ? " appears" : "s appear"} only in optional or ignored columns and do not enter the model.`,
    });
  }

  const activeNumericColumns = [
    dataset.outcomeColumn,
    ...dataset.mediaColumns,
    ...activeControlColumns,
  ].filter(Boolean);
  const nonNumericModeledCells = dataset.rows.reduce(
    (count, row) =>
      count +
      activeNumericColumns.filter(
        (column) =>
          !isMissing(row[column]) && !Number.isFinite(Number(row[column])),
      ).length,
    0,
  );
  if (nonNumericModeledCells > 0) {
    issues.push({
      level: "error",
      title: "Non-numeric modeled values",
      detail: `${nonNumericModeledCells} populated cell${nonNumericModeledCells === 1 ? "" : "s"} in the outcome, media, or active controls cannot be parsed as numbers.`,
    });
  }

  const negativeMedia = dataset.mediaColumns.some((column) =>
    dataset.rows.some((row) => Number(row[column]) < 0),
  );
  if (negativeMedia) {
    issues.push({
      level: "error",
      title: "Negative media values",
      detail: "Spend and exposure must be non-negative.",
    });
  }

  const zeroVarianceMedia = dataset.mediaColumns.filter((column) => {
    const values = dataset.rows
      .map((row) => Number(row[column]))
      .filter(Number.isFinite);
    return values.length > 0 && new Set(values).size < 2;
  });
  if (zeroVarianceMedia.length > 0) {
    issues.push({
      level: "error",
      title: "Media without variation",
      detail: `${zeroVarianceMedia.join(", ")} ${zeroVarianceMedia.length === 1 ? "is" : "are"} constant and cannot identify a response curve.`,
    });
  }

  const outcomeValues = dataset.rows
    .map((row) => Number(row[dataset.outcomeColumn]))
    .filter(Number.isFinite);
  if (outcomeValues.length > 0 && new Set(outcomeValues).size < 2) {
    issues.push({
      level: "error",
      title: "Outcome has no variation",
      detail: "The outcome must vary over time for MMM estimation.",
    });
  }

  if (excludedControlColumns.length > 0) {
    issues.push({
      level: "warning",
      title: "Control limit reached",
      detail: `${activeControlColumns.join(", ")} enter this v1 model. ${excludedControlColumns.join(", ")} are mapped as controls but excluded because v1 supports four active controls; map unused controls to Ignore to make the contract explicit.`,
    });
  }

  const cadenceResult = cadenceFromDates(validDates);
  const modeledFrequency =
    cadenceResult.cadence === "weekly"
      ? cadenceResult.gapCount > 0
        ? "Weekly · gaps detected"
        : "Weekly"
      : cadenceResult.cadence === "unknown"
        ? "Unknown"
        : cadenceResult.cadence[0].toUpperCase() + cadenceResult.cadence.slice(1);
  const frequency =
    dataset.sourceCadence === "daily" && cadenceResult.cadence === "weekly"
      ? "Daily → Weekly"
      : modeledFrequency;

  if (invalidDates === 0 && duplicateDates === 0) {
    if (
      cadenceResult.cadence !== "weekly" &&
      cadenceResult.cadence !== "monthly"
    ) {
      issues.push({
        level: "error",
        title: "Unsupported cadence",
        detail:
          cadenceResult.cadence === "daily"
            ? "Daily data could not be converted into complete weekly periods. Review the date sequence and history length."
            : "Dates are irregular. Flux requires consecutive daily, weekly, or monthly observations before modeling.",
      });
    } else if (cadenceResult.cadence === "weekly" && cadenceResult.gapCount > 0) {
      issues.push({
        level: "error",
        title: "Missing weekly periods",
        detail: `${cadenceResult.gapCount} weekly period${cadenceResult.gapCount === 1 ? " is" : "s are"} missing. Add the absent weeks explicitly before modeling; do not let adstock jump across gaps.`,
      });
    }
  }

  if (dataset.chronologyReordered && invalidDates === 0) {
    issues.push({
      level: "info",
      title: "Chronology normalized",
      detail: "Uploaded rows were reordered from earliest to latest before any adstock, seasonality, EDA, or model calculation.",
    });
  }

  const errors = issues.filter((issue) => issue.level === "error").length;
  const warnings = issues.filter((issue) => issue.level === "warning").length;
  const score = Math.max(0, 100 - errors * 30 - warnings * 6);

  if (errors === 0 && warnings === 0) {
    issues.push({
      level: "info",
      title: "Schema validated",
      detail: "Dates, outcome, media spend, and controls are model-ready.",
    });
  }

  return {
    score,
    status: errors ? "blocked" : warnings ? "review" : "ready",
    issues,
    startDate:
      validDates.length > 0
        ? new Date(Math.min(...validDates)).toISOString().slice(0, 10)
        : "—",
    endDate:
      validDates.length > 0
        ? new Date(Math.max(...validDates)).toISOString().slice(0, 10)
        : "—",
    frequency,
    cadence: cadenceResult.cadence,
    sourceCadence: dataset.sourceCadence,
    modelCadence: dataset.modelCadence,
    intervalDays: cadenceResult.intervalDays,
    gapCount: cadenceResult.gapCount,
    invalidDates,
    modeledMissingCells,
    excludedControlColumns,
    automaticRepairs: dataset.automaticRepairs,
    missingCells,
    duplicateDates,
  };
}

export async function updateColumnRole(
  dataset: Dataset,
  columnName: string,
  role: ColumnRole,
): Promise<Dataset> {
  const specs = dataset.specs.map((column) =>
    column.name === columnName
      ? {
          ...column,
          role,
          ...(role === "media_spend" || role === "media_exposure"
            ? { channel: column.channel ?? channelFromColumn(column.name) }
            : {}),
        }
      : column,
  );

  const dateColumn =
    specs.find((column) => column.role === "date")?.name ?? dataset.dateColumn;
  const outcomeColumn =
    specs.find((column) => column.role === "outcome")?.name ??
    dataset.outcomeColumn;
  const prepared = prepareRows(dataset.sourceRows, dateColumn, specs);
  const dateColumnChanged = dateColumn !== dataset.dateColumn;

  return {
    ...dataset,
    rows: prepared.rows,
    specs,
    dateColumn,
    outcomeColumn,
    mediaColumns: specs
      .filter((column) => column.role === "media_spend")
      .map((column) => column.name),
    controlColumns: specs
      .filter((column) => column.role === "control")
      .map((column) => column.name),
    hash: await contractHash(
      dataset.sourceHash,
      specs,
      dateColumn,
      outcomeColumn,
    ),
    chronologyReordered:
      dataset.chronologyReordered || prepared.reordered,
    automaticRepairs: dateColumnChanged
      ? prepared.repairs
      : dataset.automaticRepairs,
    sourceCadence: prepared.sourceCadence,
    modelCadence: prepared.modelCadence,
    periodsPerYear: prepared.periodsPerYear,
    periodUnit: prepared.periodUnit,
  };
}
