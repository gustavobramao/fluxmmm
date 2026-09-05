import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { V10_AMSS_VERSION } from "./contract";

const ROOT = resolve(".flux-artifacts/svi-score-v10-amss");
const OUTPUT = resolve(ROOT, "opened-truth/technical-amendment-003.json");
const paths = {
  amendment002: resolve(ROOT, "opened-truth/technical-amendment-002.json"),
  candidateActions: resolve(ROOT, "observable-freeze/candidate-actions.csv"),
  frozenSelections: resolve(ROOT, "observable-freeze/frozen-selections.json"),
  dockerfile: resolve("research/svi_score_v10_amss/cloud/Dockerfile.truth"),
};

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

const source = Object.fromEntries(await Promise.all(
  Object.entries(paths).map(async ([key, path]) => [key, await readFile(path)] as const),
));
const amendment002 = JSON.parse(source.amendment002.toString("utf8")) as {
  version: string;
  status: string;
  immutableInputs: { candidateActionsSha256: string; frozenSelectionsSha256: string };
};
if (
  amendment002.version !== V10_AMSS_VERSION ||
  amendment002.status !== "frozen-cloud-execution-only-amendment" ||
  amendment002.immutableInputs.candidateActionsSha256 !== sha256(source.candidateActions) ||
  amendment002.immutableInputs.frozenSelectionsSha256 !== sha256(source.frozenSelections)
) throw new Error("Technical amendment 002 no longer matches frozen inputs.");

const amendment = {
  artifactId: "flux-v10-amss-truth-evaluator-technical-amendment-003",
  version: V10_AMSS_VERSION,
  status: "frozen-container-dependency-only-amendment",
  amendedAt: new Date().toISOString(),
  failedBuild: {
    id: "e28a14c7-7a9d-47ea-8916-20ff6ad8abd3",
    stage: "R package installation",
    cause: "AMSS dependency assertthat was absent from the container",
    imageProduced: false,
    truthTasksStarted: 0,
    truthPartsProduced: 0,
  },
  correction: "Install assertthat before installing the pinned AMSS source archive.",
  scientificContractChanged: false,
  frozenCandidatesChanged: false,
  frozenActionsChanged: false,
  frozenSelectionsChanged: false,
  randomSeedsChanged: false,
  truthReplicatesChanged: false,
  estimandChanged: false,
  immutableInputs: {
    technicalAmendment002Sha256: sha256(source.amendment002),
    candidateActionsSha256: sha256(source.candidateActions),
    frozenSelectionsSha256: sha256(source.frozenSelections),
  },
  correctedDockerfileSha256: sha256(source.dockerfile),
};
await writeFile(OUTPUT, `${JSON.stringify(amendment, null, 2)}\n`, { flag: "wx" });
console.log(JSON.stringify({ frozen: true, amendment: OUTPUT, truthTasksStarted: 0 }, null, 2));
