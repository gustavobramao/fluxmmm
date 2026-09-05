import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { SVI_SCORE_V3_CONTRACT } from "./contract";
import type { SviScoreV3Record } from "./types";

const runArgument = process.argv.find((argument) => argument.startsWith("--run-id="));
const runId = (runArgument?.slice("--run-id=".length) || "design-pilot").replace(
  /[^a-z0-9-]/gi,
  "-",
);
const path = resolve(".flux-artifacts/svi-score-v3", `${runId}-records.json`);
const records = JSON.parse(await readFile(path, "utf8")) as SviScoreV3Record[];
const terminal = records.filter((record) => record.status !== "prepared");
const businessCounts = new Map<string, number>();
terminal.forEach((record) =>
  businessCounts.set(
    record.businessId,
    (businessCounts.get(record.businessId) ?? 0) + 1,
  ),
);
const businessesComplete = [...businessCounts.values()].filter(
  (count) => count === SVI_SCORE_V3_CONTRACT.candidatesPerBusiness,
).length;
const partialBusinesses = [...businessCounts.entries()]
  .filter(([, count]) => count < SVI_SCORE_V3_CONTRACT.candidatesPerBusiness)
  .map(([businessId, count]) => ({ businessId, completedCandidates: count }));
const runtimeSeconds = terminal.reduce(
  (total, record) => total + (record.runtimeSeconds ?? 0),
  0,
);
process.stdout.write(
  `${JSON.stringify(
    {
      runId,
      businessesStarted: businessCounts.size,
      businessesComplete,
      partialBusinesses,
      attempts: terminal.length,
      labelled: terminal.filter((record) => record.status === "labelled").length,
      review: terminal.filter((record) => record.status === "review").length,
      errors: terminal.filter((record) => record.status === "error").length,
      completionPercent:
        (100 * terminal.length) / SVI_SCORE_V3_CONTRACT.targetSviFits,
      accumulatedWorkerHours: runtimeSeconds / 3_600,
    },
    null,
    2,
  )}\n`,
);
