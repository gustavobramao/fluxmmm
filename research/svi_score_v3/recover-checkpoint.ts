import { resolve } from "node:path";
import { recoverCheckpoint } from "./checkpoint";
import type { SviScoreV3Record } from "./types";

function argument(name: string): string {
  const prefix = `--${name}=`;
  const value = process.argv.find((candidate) => candidate.startsWith(prefix));
  if (!value) throw new Error(`Missing --${name}=...`);
  return resolve(value.slice(prefix.length));
}

const inputPath = argument("input");
const outputPath = argument("output");
const records = await recoverCheckpoint<SviScoreV3Record>(inputPath, outputPath);

process.stdout.write(
  `${JSON.stringify(
    {
      inputPath,
      outputPath,
      recoveredRecords: records.length,
      labelled: records.filter((record) => record.status === "labelled").length,
      review: records.filter((record) => record.status === "review").length,
      errors: records.filter((record) => record.status === "error").length,
    },
    null,
    2,
  )}\n`,
);
