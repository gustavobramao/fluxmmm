import type { DataRow } from "./types";

function splitCsvLine(line: string, lineNumber: number): string[] {
  const cells: string[] = [];
  let value = "";
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (quoted && line[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      cells.push(value.trim());
      value = "";
    } else {
      value += character;
    }
  }

  if (quoted) {
    throw new Error(`Line ${lineNumber} contains an unclosed quoted value.`);
  }

  cells.push(value.trim());
  return cells;
}

export function parseCsv(csv: string): { columns: string[]; rows: DataRow[] } {
  const lines = csv
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);

  if (lines.length < 2) {
    throw new Error("The file needs a header and at least one data row.");
  }

  const columns = splitCsvLine(lines[0], 1).map((column) =>
    column.replace(/^"|"$/g, ""),
  );

  if (columns.some((column) => !column)) {
    throw new Error("Every CSV column needs a non-empty header.");
  }

  const normalizedColumns = columns.map((column) => column.toLowerCase());
  const duplicateColumn = normalizedColumns.find(
    (column, index) => normalizedColumns.indexOf(column) !== index,
  );
  if (duplicateColumn) {
    throw new Error(`Duplicate CSV header detected: ${duplicateColumn}.`);
  }

  const rows = lines.slice(1).map((line, rowIndex) => {
    const lineNumber = rowIndex + 2;
    const values = splitCsvLine(line, lineNumber);
    if (values.length !== columns.length) {
      throw new Error(
        `Line ${lineNumber} has ${values.length} values; expected ${columns.length}.`,
      );
    }
    const row: DataRow = {};
    columns.forEach((column, index) => {
      const raw = values[index] ?? "";
      const number = Number(raw);
      row[column] =
        raw !== "" && Number.isFinite(number) ? number : raw.replace(/^"|"$/g, "");
    });
    return row;
  });

  return { columns, rows };
}

export function toNumber(value: string | number | undefined): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export async function sha256(value: string): Promise<string> {
  const bytes = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export function formatCompact(value: number, currency = false): string {
  const formatter = new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
    ...(currency ? { style: "currency", currency: "USD" } : {}),
  });
  return formatter.format(value);
}

export function formatFull(value: number, currency = false): string {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
    ...(currency ? { style: "currency", currency: "USD" } : {}),
  }).format(value);
}
