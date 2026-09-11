import { parseCsv } from "./csv";
import type { DataRow, Experiment } from "./types";

const HEADER_ALIASES = {
  channel: ["channel", "channel_name", "media_channel", "media", "platform"],
  startDate: ["start_date", "campaign_start", "test_start", "experiment_start", "start"],
  endDate: ["end_date", "campaign_end", "test_end", "experiment_end", "end"],
  outcomeEndDate: [
    "outcome_end_date",
    "measurement_end_date",
    "carryover_end_date",
    "outcome_through",
  ],
  incrementalOutcome: [
    "incremental_outcome",
    "incremental_revenue",
    "incremental_sales",
    "incremental_value",
    "incremental_conversions",
    "lift",
  ],
  incrementalSpend: [
    "incremental_spend",
    "test_spend",
    "incremental_cost",
    "media_spend",
    "spend",
  ],
  roi: ["roi", "incremental_roi", "iroas", "roas"],
  standardError: [
    "standard_error",
    "roi_standard_error",
    "std_error",
    "stderr",
    "se",
  ],
  roiLow: ["roi_low", "roi_ci_low", "confidence_interval_low", "ci_low"],
  roiHigh: ["roi_high", "roi_ci_high", "confidence_interval_high", "ci_high"],
  confidence: ["confidence", "confidence_level", "coverage"],
  scope: ["scope", "effect_scope"],
  source: ["source", "experiment_name", "study_name", "label"],
} as const;

type CanonicalField = keyof typeof HEADER_ALIASES;

export interface ExperimentCsvImportResult {
  experiments: Experiment[];
  warnings: string[];
  mappedHeaders: Partial<Record<CanonicalField, string>>;
}

function normalizeIdentifier(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/^fb(?=[_\s-]|$)/, "facebook")
    .replace(/[^a-z0-9]+/g, "");
}

function channelBase(value: string): string {
  return normalizeIdentifier(value).replace(
    /(spend|cost|investment|impressions|impression|clicks|click)$/,
    "",
  );
}

function findHeader(columns: string[], field: CanonicalField): string | undefined {
  const aliases = new Set(HEADER_ALIASES[field].map(normalizeIdentifier));
  return columns.find((column) => aliases.has(normalizeIdentifier(column)));
}

function cell(row: DataRow, header?: string): string | number | undefined {
  return header ? row[header] : undefined;
}

function requiredText(
  row: DataRow,
  header: string | undefined,
  label: string,
  line: number,
): string {
  const value = cell(row, header);
  if (value === undefined || String(value).trim() === "") {
    throw new Error(`Row ${line}: ${label} is required.`);
  }
  return String(value).trim();
}

function optionalNumber(row: DataRow, header?: string): number | undefined {
  const value = cell(row, header);
  if (value === undefined || String(value).trim() === "") return undefined;
  const numeric = typeof value === "number" ? value : Number(String(value).replace(/,/g, ""));
  return Number.isFinite(numeric) ? numeric : undefined;
}

function requiredNumber(
  row: DataRow,
  header: string | undefined,
  label: string,
  line: number,
): number {
  const value = optionalNumber(row, header);
  if (value === undefined) throw new Error(`Row ${line}: ${label} must be a number.`);
  return value;
}

function normalizeDate(value: string, label: string, line: number): string {
  const trimmed = value.trim();
  const explicit = trimmed.match(/^(\d{4})[-/]([01]?\d)[-/]([0-3]?\d)$/);
  const date = explicit
    ? new Date(Date.UTC(Number(explicit[1]), Number(explicit[2]) - 1, Number(explicit[3])))
    : new Date(trimmed);
  if (!Number.isFinite(date.getTime())) {
    throw new Error(`Row ${line}: ${label} must be a valid date, preferably YYYY-MM-DD.`);
  }
  return date.toISOString().slice(0, 10);
}

function normalizeConfidence(value: string | number | undefined, line: number): number {
  if (value === undefined || String(value).trim() === "") return 0.95;
  const raw = String(value).trim();
  const parsed = Number(raw.replace("%", ""));
  if (!Number.isFinite(parsed)) throw new Error(`Row ${line}: confidence must be numeric.`);
  const confidence = raw.includes("%") || parsed > 1 ? parsed / 100 : parsed;
  if (confidence <= 0 || confidence >= 1) {
    throw new Error(`Row ${line}: confidence must be between 0 and 1, or a percentage such as 95%.`);
  }
  return confidence;
}

function normalCriticalValue(confidence: number): number {
  if (confidence >= 0.99) return 2.576;
  if (confidence >= 0.98) return 2.326;
  if (confidence >= 0.95) return 1.96;
  if (confidence >= 0.9) return 1.645;
  return 1.282;
}

function resolveChannel(value: string, mediaColumns: string[], line: number): string {
  const exact = mediaColumns.filter(
    (channel) => normalizeIdentifier(channel) === normalizeIdentifier(value),
  );
  if (exact.length === 1) return exact[0];
  const base = mediaColumns.filter(
    (channel) => channelBase(channel) === channelBase(value),
  );
  if (base.length === 1) return base[0];
  if (base.length > 1) {
    throw new Error(
      `Row ${line}: channel “${value}” is ambiguous. Use one of: ${base.join(", ")}.`,
    );
  }
  throw new Error(
    `Row ${line}: channel “${value}” does not match this dataset. Use one of: ${mediaColumns.join(", ")}.`,
  );
}

export function experimentSignature(experiment: Experiment): string {
  return [
    experiment.channel.toLowerCase(),
    experiment.startDate,
    experiment.endDate,
    experiment.outcomeEndDate ?? experiment.endDate,
    experiment.source.trim().toLowerCase(),
  ].join("|");
}

export function parseExperimentCsv(
  rawCsv: string,
  mediaColumns: string[],
  dateRange?: { minimum: string; maximum: string },
): ExperimentCsvImportResult {
  if (!mediaColumns.length) throw new Error("Load a dataset with media channels before importing experiments.");
  const parsed = parseCsv(rawCsv);
  const mappedHeaders = Object.fromEntries(
    (Object.keys(HEADER_ALIASES) as CanonicalField[])
      .map((field) => [field, findHeader(parsed.columns, field)])
      .filter((entry): entry is [CanonicalField, string] => Boolean(entry[1])),
  ) as Partial<Record<CanonicalField, string>>;

  for (const field of ["channel", "startDate", "endDate", "incrementalSpend"] as const) {
    if (!mappedHeaders[field]) {
      throw new Error(`Missing required column: ${HEADER_ALIASES[field][0]}.`);
    }
  }
  if (!mappedHeaders.incrementalOutcome && !mappedHeaders.roi) {
    throw new Error("Include incremental_outcome, or include roi so Flux can derive it from incremental_spend.");
  }
  if (!mappedHeaders.standardError && !(mappedHeaders.roiLow && mappedHeaders.roiHigh)) {
    throw new Error("Include standard_error, or both roi_low and roi_high so Flux can derive uncertainty.");
  }

  const warnings = new Set<string>();
  const seen = new Set<string>();
  const experiments = parsed.rows.map((row, index) => {
    const line = index + 2;
    const suppliedChannel = requiredText(row, mappedHeaders.channel, "channel", line);
    const channel = resolveChannel(suppliedChannel, mediaColumns, line);
    if (channel !== suppliedChannel) warnings.add(`Mapped channel “${suppliedChannel}” to “${channel}”.`);
    const startDate = normalizeDate(
      requiredText(row, mappedHeaders.startDate, "start_date", line),
      "start_date",
      line,
    );
    const endDate = normalizeDate(
      requiredText(row, mappedHeaders.endDate, "end_date", line),
      "end_date",
      line,
    );
    const rawOutcomeEnd = cell(row, mappedHeaders.outcomeEndDate);
    const outcomeEndDate = rawOutcomeEnd === undefined || String(rawOutcomeEnd).trim() === ""
      ? endDate
      : normalizeDate(String(rawOutcomeEnd), "outcome_end_date", line);
    if (startDate > endDate) throw new Error(`Row ${line}: start_date must not be after end_date.`);
    if (outcomeEndDate < endDate) {
      throw new Error(`Row ${line}: outcome_end_date must not be before end_date.`);
    }
    if (dateRange && (startDate < dateRange.minimum || endDate > dateRange.maximum)) {
      throw new Error(
        `Row ${line}: the experiment window must fall inside ${dateRange.minimum} to ${dateRange.maximum}.`,
      );
    }
    if (dateRange && outcomeEndDate > dateRange.maximum) {
      throw new Error(`Row ${line}: outcome_end_date exceeds the dataset end (${dateRange.maximum}).`);
    }

    const incrementalSpend = requiredNumber(
      row,
      mappedHeaders.incrementalSpend,
      "incremental_spend",
      line,
    );
    if (incrementalSpend <= 0) throw new Error(`Row ${line}: incremental_spend must be greater than zero.`);
    let incrementalOutcome = optionalNumber(row, mappedHeaders.incrementalOutcome);
    if (incrementalOutcome === undefined) {
      incrementalOutcome = requiredNumber(row, mappedHeaders.roi, "roi", line) * incrementalSpend;
      warnings.add("Derived incremental_outcome from ROI × incremental_spend.");
    }

    const confidence = normalizeConfidence(cell(row, mappedHeaders.confidence), line);
    let standardError = optionalNumber(row, mappedHeaders.standardError);
    if (standardError === undefined) {
      const low = requiredNumber(row, mappedHeaders.roiLow, "roi_low", line);
      const high = requiredNumber(row, mappedHeaders.roiHigh, "roi_high", line);
      if (high <= low) throw new Error(`Row ${line}: roi_high must be greater than roi_low.`);
      standardError = (high - low) / (2 * normalCriticalValue(confidence));
      warnings.add("Derived ROI standard_error from the supplied confidence interval.");
    }
    if (standardError <= 0) throw new Error(`Row ${line}: standard_error must be greater than zero.`);

    const rawScope = String(cell(row, mappedHeaders.scope) ?? "").trim().toLowerCase();
    const scope = rawScope || (outcomeEndDate > endDate ? "total" : "immediate");
    if (scope !== "immediate" && scope !== "total") {
      throw new Error(`Row ${line}: scope must be immediate or total.`);
    }
    const source = String(cell(row, mappedHeaders.source) ?? "").trim() || `Imported experiment ${index + 1}`;
    const experiment: Experiment = {
      channel,
      startDate,
      endDate,
      outcomeEndDate: outcomeEndDate === endDate ? undefined : outcomeEndDate,
      incrementalOutcome,
      incrementalSpend,
      standardError,
      confidence,
      scope,
      source,
    };
    const signature = experimentSignature(experiment);
    if (seen.has(signature)) throw new Error(`Row ${line}: this experiment is duplicated within the CSV.`);
    seen.add(signature);
    return experiment;
  });

  for (const [field, header] of Object.entries(mappedHeaders)) {
    const canonical = HEADER_ALIASES[field as CanonicalField][0];
    if (header && normalizeIdentifier(header) !== normalizeIdentifier(canonical)) {
      warnings.add(`Recognized “${header}” as “${canonical}”.`);
    }
  }
  return { experiments, warnings: [...warnings], mappedHeaders };
}

export function experimentCsvTemplate(
  channel: string,
  dates: { startDate: string; endDate: string; outcomeEndDate: string },
): string {
  const headers = [
    "channel",
    "start_date",
    "end_date",
    "outcome_end_date",
    "incremental_outcome",
    "incremental_spend",
    "standard_error",
    "confidence",
    "scope",
    "source",
  ];
  const values = [
    channel,
    dates.startDate,
    dates.endDate,
    dates.outcomeEndDate,
    50000,
    20000,
    0.35,
    0.95,
    "total",
    "Example lift study",
  ];
  return `${headers.join(",")}\n${values.join(",")}\n`;
}
