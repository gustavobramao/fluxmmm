import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { V10_AMSS_CONTRACT, V10_AMSS_VERSION } from "./contract";

const ROOT = resolve("research/svi_score_v10_amss");
const FREEZE_ROOT = resolve(ROOT, "frozen-protocol");
const ARTIFACT_ROOT = resolve(".flux-artifacts/svi-score-v10-amss");

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function invariant(condition: unknown, detail: string): asserts condition {
  if (!condition) throw new Error(`V10 AMSS protocol verification failed: ${detail}.`);
}

const manifest = JSON.parse(
  await readFile(resolve(FREEZE_ROOT, "manifest.json"), "utf8"),
) as {
  version: string;
  status: string;
  contractSha256: string;
  candidateSpecificationsSha256: string;
  candidateSpecificationsFileSha256: string;
  cohortDesignSha256: string;
  businesses: number;
  evidenceGroupCounts: Record<string, number>;
  moduleOrderCounts: Record<string, number>;
  evidenceByModuleOrderCounts: Record<string, Record<string, number>>;
  frozenV9ManifestSha256: string;
  frozenV9SelectorSha256: string;
  cloudImageDigest: string;
  sourceSha256: Record<string, string>;
  truthOpened: boolean;
};
const contract = await readFile(resolve(FREEZE_ROOT, "scientific-contract.json"));
const candidates = await readFile(resolve(FREEZE_ROOT, "candidate-specifications.json"));
const design = await readFile(resolve(FREEZE_ROOT, "cohort-design.csv"));

invariant(manifest.version === V10_AMSS_VERSION, "version changed");
invariant(manifest.status === "frozen-before-cohort-generation", "freeze status changed");
invariant(!manifest.truthOpened, "truth was marked open before selection");
invariant(sha256(contract) === manifest.contractSha256, "scientific contract changed");
invariant(
  sha256(candidates) === manifest.candidateSpecificationsFileSha256,
  "candidate specification file changed",
);
invariant(
  Object.values(manifest.evidenceByModuleOrderCounts).every((counts) =>
    Object.values(counts).every((count) => count === 4 || count === 5)
  ),
  "evidence assignment is confounded with media-module order",
);
invariant(sha256(design) === manifest.cohortDesignSha256, "cohort design changed");
invariant(manifest.businesses === 100, "business count is not 100");
invariant(
  Object.values(manifest.evidenceGroupCounts).every((count) => count === 25),
  "evidence groups are not balanced 25/25/25/25",
);
invariant(
  Object.values(manifest.moduleOrderCounts).reduce((sum, count) => sum + count, 0) === 100 &&
    Object.values(manifest.moduleOrderCounts).every((count) => count === 16 || count === 17),
  "module orders are not balanced",
);
invariant(
  manifest.cloudImageDigest === V10_AMSS_CONTRACT.orchestration.containerImageDigest,
  "cloud image digest changed",
);
invariant(
  sha256(await readFile(resolve(ROOT, "frozen-v9/manifest.json"))) ===
    manifest.frozenV9ManifestSha256,
  "frozen V9 manifest changed",
);
invariant(
  sha256(await readFile(resolve(ROOT, "frozen-v9/selector.json"))) ===
    manifest.frozenV9SelectorSha256,
  "frozen V9 selector changed",
);
const amendment001 = JSON.parse(
  await readFile(resolve(ARTIFACT_ROOT, "opened-truth/technical-amendment-001.json"), "utf8"),
) as {
  version: string;
  scientificContractChanged?: boolean;
  originalOpenTruthRsha256: string;
  amendedOpenTruthRsha256: string;
};
const amendment004 = JSON.parse(
  await readFile(resolve(ARTIFACT_ROOT, "opened-truth/technical-amendment-004.json"), "utf8"),
) as {
  version: string;
  status: string;
  scientificContractChanged: boolean;
  correctedSources: {
    auditAssemblerSha256: string;
    contractTestSha256: string;
  };
};
invariant(amendment001.version === V10_AMSS_VERSION, "technical amendment 001 version changed");
invariant(amendment004.version === V10_AMSS_VERSION, "technical amendment 004 version changed");
invariant(
  amendment004.status === "frozen-contract-enforcement-correction" &&
    !amendment004.scientificContractChanged,
  "technical amendment 004 is invalid",
);
for (const [path, expected] of Object.entries(manifest.sourceSha256)) {
  const current = sha256(await readFile(resolve(path)));
  if (current === expected) continue;
  if (path === "research/svi_score_v10_amss/open-truth.R") {
    invariant(
      amendment001.originalOpenTruthRsha256 === expected &&
        amendment001.amendedOpenTruthRsha256 === current,
      `${path} is not covered by technical amendment 001`,
    );
    continue;
  }
  if (path === "research/svi_score_v10_amss/build-external-audit.ts") {
    invariant(
      amendment004.correctedSources.auditAssemblerSha256 === current,
      `${path} is not covered by technical amendment 004`,
    );
    continue;
  }
  if (path === "tests/svi-score-v10-amss.test.ts") {
    invariant(
      amendment004.correctedSources.contractTestSha256 === current,
      `${path} is not covered by technical amendment 004`,
    );
    continue;
  }
  invariant(false, `${path} changed`);
}

const parsedCandidates = JSON.parse(candidates.toString("utf8")) as unknown[];
invariant(parsedCandidates.length === 24, "candidate count changed");
const lines = design.toString("utf8").trim().split("\n");
invariant(lines.length === 101, "cohort design row count changed");

console.log(JSON.stringify({
  verified: true,
  version: manifest.version,
  businesses: manifest.businesses,
  candidateSpecifications: parsedCandidates.length,
  candidateFits: manifest.businesses * parsedCandidates.length * 2,
  evidenceGroupCounts: manifest.evidenceGroupCounts,
  moduleOrderCounts: manifest.moduleOrderCounts,
  truthFirewallAtFreeze: "closed",
  currentStage: "truth-opened-after-frozen-selection",
  technicalAmendmentsVerified: [1, 4],
}, null, 2));
