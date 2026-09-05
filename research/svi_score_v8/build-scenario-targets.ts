import { spawn } from "node:child_process";
import { cpus } from "node:os";
import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import type { SamplingDecisionDraw } from "../../lib/mmm/sampling";
import {
  evaluateAlignedPosteriorDecisionTruth,
  evaluateAlignedPosteriorDecisionTruthExact,
} from "../score_v2/evaluate";
import { generateSyntheticBusiness } from "../score_v2/simulator";
import type { DecisionScenarioId } from "../score_v2/types";
import { sviDesignBusinesses } from "../svi_score_v3/design";
import type { SviScoreV3Record } from "../svi_score_v3/types";
import { SVI_SCORE_V8_CONTRACT, SVI_SCORE_V8_VERSION } from "./contract";
import type { V8ScenarioTargetRow } from "./types";

const ROOT = resolve(".flux-artifacts/svi-score-v8");
const PARTS = resolve(ROOT, "scenario-target-parts");
const RECORD_PATH = resolve(".flux-artifacts/svi-score-v3/design-pilot-records.json");
const MANIFEST_PATH = resolve(ROOT, "development-record-manifest.json");
const OUTPUT_PATH = resolve(ROOT, "development-scenario-targets.json");

interface DevelopmentRecord {
  businessId: string;
  family: string;
  split: "train" | "validation";
  candidateId: string;
  inferenceFingerprint: string;
  aggregateDecisionLoss: number;
}

interface DevelopmentManifest {
  version: string;
  sourceSha256: string;
  records: DevelopmentRecord[];
}

function integerArgument(name: string, fallback: number): number {
  const prefix = `--${name}=`;
  const raw = process.argv.find((argument) => argument.startsWith(prefix));
  return raw ? Number.parseInt(raw.slice(prefix.length), 10) : fallback;
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.tmp`;
  await writeFile(temporary, JSON.stringify(value));
  await rename(temporary, path);
}

async function buildDevelopmentManifest(): Promise<DevelopmentManifest> {
  const source = await readFile(RECORD_PATH);
  const records = JSON.parse(source.toString("utf8")) as SviScoreV3Record[];
  // The split check intentionally precedes every posterior-label access. Audit
  // records are never projected into the development manifest.
  const development = records.flatMap((record): DevelopmentRecord[] => {
    if (record.split === "audit") return [];
    if (
      (record.status !== "labelled" && record.status !== "review") ||
      record.sviDecisionLoss === undefined ||
      !Number.isFinite(record.sviDecisionLoss)
    ) {
      throw new Error(
        `V8 has no finite development label for ${record.businessId}/${record.candidateId}.`,
      );
    }
    return [{
      businessId: record.businessId,
      family: record.family,
      split: record.split,
      candidateId: record.candidateId,
      inferenceFingerprint: record.inferenceFingerprint,
      aggregateDecisionLoss: record.sviDecisionLoss,
    }];
  });
  if (development.length !== SVI_SCORE_V8_CONTRACT.cohort.developmentRows) {
    throw new Error(`V8 expected 20,160 development records; found ${development.length}.`);
  }
  const manifest = {
    version: SVI_SCORE_V8_VERSION,
    sourceSha256: sha256(source),
    records: development,
  } satisfies DevelopmentManifest;
  await mkdir(ROOT, { recursive: true });
  await atomicJson(MANIFEST_PATH, manifest);
  return manifest;
}

async function worker(shard: number, shards: number): Promise<void> {
  const manifest = JSON.parse(
    await readFile(MANIFEST_PATH, "utf8"),
  ) as DevelopmentManifest;
  if (manifest.version !== SVI_SCORE_V8_VERSION) {
    throw new Error("V8 development manifest version changed.");
  }
  const recordsByBusiness = new Map<string, DevelopmentRecord[]>();
  manifest.records.forEach((record) => {
    recordsByBusiness.set(record.businessId, [
      ...(recordsByBusiness.get(record.businessId) ?? []),
      record,
    ]);
  });
  const designs = sviDesignBusinesses()
    .filter(({ family }) => family.split !== "audit")
    .filter((_, index) => index % shards === shard);
  const rows: V8ScenarioTargetRow[] = [];
  for (let index = 0; index < designs.length; index += 1) {
    const { family, scenario } = designs[index];
    const records = recordsByBusiness.get(scenario.id) ?? [];
    if (records.length !== SVI_SCORE_V8_CONTRACT.cohort.candidatesPerBusiness) {
      throw new Error(`${scenario.id} has ${records.length}/48 V8 records.`);
    }
    const business = generateSyntheticBusiness(scenario);
    for (const record of records) {
      const resultPath = resolve(
        ".flux-artifacts/svi-score-v3",
        `${record.inferenceFingerprint}.result.json.gz`,
      );
      const result = JSON.parse(
        gunzipSync(await readFile(resultPath)).toString("utf8"),
      ) as {
        fingerprint: string;
        engine: string;
        contract: unknown;
        decisionDraws: SamplingDecisionDraw[];
      };
      if (
        result.fingerprint !== record.inferenceFingerprint ||
        result.engine !== SVI_SCORE_V8_CONTRACT.inference.engine.replace("FullRankADVI", "· FullRankADVI") ||
        JSON.stringify(result.contract) !==
          JSON.stringify(SVI_SCORE_V8_CONTRACT.inference.contract)
      ) {
        throw new Error(`SVI contract mismatch for ${scenario.id}/${record.candidateId}.`);
      }
      let evaluated = evaluateAlignedPosteriorDecisionTruth(
        business,
        result.decisionDraws,
        64,
      );
      if (Math.abs(evaluated.decisionLoss - record.aggregateDecisionLoss) > 1e-10) {
        evaluated = evaluateAlignedPosteriorDecisionTruthExact(
          business,
          result.decisionDraws,
          64,
        );
        if (Math.abs(evaluated.decisionLoss - record.aggregateDecisionLoss) > 1e-10) {
          throw new Error(
            `Scenario reconstruction changed the frozen SVI label for ${scenario.id}/${record.candidateId}.`,
          );
        }
      }
      rows.push({
        businessId: scenario.id,
        candidateId: record.candidateId,
        scenarioDecisionLoss: evaluated.scenarioDecisionLoss,
        scenarioProfitRegret: evaluated.scenarioProfitRegret,
        scenarioRecommendedSpend: evaluated.scenarioRecommendedSpend,
        aggregateDecisionLoss: evaluated.decisionLoss,
      });
    }
    if ((index + 1) % 5 === 0 || index + 1 === designs.length) {
      process.stdout.write(
        `V8 scenario shard ${shard + 1}/${shards}: ${index + 1}/${designs.length} businesses\n`,
      );
    }
  }
  await mkdir(PARTS, { recursive: true });
  await atomicJson(resolve(PARTS, `part-${shard}-of-${shards}.json`), {
    version: SVI_SCORE_V8_VERSION,
    sourceSha256: manifest.sourceSha256,
    shard,
    shards,
    rows,
  });
}

async function runChild(shard: number, shards: number): Promise<void> {
  await new Promise<void>((resolveProcess, rejectProcess) => {
    const child = spawn(
      process.execPath,
      [
        "--import",
        "tsx",
        resolve("research/svi_score_v8/build-scenario-targets.ts"),
        "--worker",
        `--shard=${shard}`,
        `--shards=${shards}`,
      ],
      { cwd: process.cwd(), env: process.env, stdio: "inherit" },
    );
    child.on("error", rejectProcess);
    child.on("close", (code) =>
      code === 0
        ? resolveProcess()
        : rejectProcess(new Error(`V8 scenario worker ${shard} exited ${code}.`))
    );
  });
}

async function orchestrate(): Promise<void> {
  const workers = Math.max(
    1,
    Math.min(integerArgument("workers", Math.min(8, cpus().length)), 16),
  );
  const manifest = await buildDevelopmentManifest();
  await mkdir(PARTS, { recursive: true });
  await Promise.all(Array.from({ length: workers }, (_, shard) =>
    runChild(shard, workers)
  ));
  const rows: V8ScenarioTargetRow[] = [];
  for (let shard = 0; shard < workers; shard += 1) {
    const part = JSON.parse(
      await readFile(resolve(PARTS, `part-${shard}-of-${workers}.json`), "utf8"),
    ) as {
      version: string;
      sourceSha256: string;
      rows: V8ScenarioTargetRow[];
    };
    if (
      part.version !== SVI_SCORE_V8_VERSION ||
      part.sourceSha256 !== manifest.sourceSha256
    ) {
      throw new Error(`V8 scenario part ${shard} has stale provenance.`);
    }
    rows.push(...part.rows);
  }
  const keys = new Set(rows.map((row) => `${row.businessId}\u0000${row.candidateId}`));
  if (
    rows.length !== SVI_SCORE_V8_CONTRACT.cohort.developmentRows ||
    keys.size !== rows.length
  ) {
    throw new Error(`V8 scenario target merge produced ${rows.length}/${keys.size} rows.`);
  }
  rows.sort((left, right) =>
    left.businessId.localeCompare(right.businessId) ||
    left.candidateId.localeCompare(right.candidateId)
  );
  const payload = {
    artifactId: "flux-svi-score-v8-development-scenario-targets-v1",
    version: SVI_SCORE_V8_VERSION,
    generatedAt: new Date().toISOString(),
    provenance: {
      developmentManifestSha256: sha256(await readFile(MANIFEST_PATH)),
      sourceRecordSha256: manifest.sourceSha256,
      sviEngine: SVI_SCORE_V8_CONTRACT.inference.engine,
      inferenceContract: SVI_SCORE_V8_CONTRACT.inference.contract,
      posteriorDecisionDrawsPerCandidate: 64,
      auditRowsDereferenced: 0,
      nutsUsed: false,
    },
    scenarios: SVI_SCORE_V8_CONTRACT.targets.scenarios,
    rows,
  };
  await atomicJson(OUTPUT_PATH, payload);
  process.stdout.write(`${JSON.stringify({
    path: OUTPUT_PATH,
    rows: rows.length,
    businesses: new Set(rows.map((row) => row.businessId)).size,
    auditRowsDereferenced: 0,
  }, null, 2)}\n`);
}

if (process.argv.includes("--worker")) {
  await worker(integerArgument("shard", 0), integerArgument("shards", 1));
} else {
  await orchestrate();
}
