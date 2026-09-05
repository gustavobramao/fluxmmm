import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { V10_AMSS_VERSION } from "./contract";

const ROOT = resolve(".flux-artifacts/svi-score-v10-amss");
const OUTPUT = resolve(ROOT, "opened-truth/cloud-image-receipt.json");
const paths = {
  amendment003: resolve(ROOT, "opened-truth/technical-amendment-003.json"),
  candidateActions: resolve(ROOT, "observable-freeze/candidate-actions.csv"),
  frozenSelections: resolve(ROOT, "observable-freeze/frozen-selections.json"),
  evaluator: resolve("research/svi_score_v10_amss/open-truth.R"),
  cloudTask: resolve("research/svi_score_v10_amss/cloud/truth_task.py"),
  dockerfile: resolve("research/svi_score_v10_amss/cloud/Dockerfile.truth"),
  cloudRun: resolve("research/svi_score_v10_amss/cloud/truth-cloud-run.yaml"),
};

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

const source = Object.fromEntries(await Promise.all(
  Object.entries(paths).map(async ([key, path]) => [key, await readFile(path)] as const),
));
const amendment003 = JSON.parse(source.amendment003.toString("utf8")) as {
  version: string;
  status: string;
  correctedDockerfileSha256: string;
  immutableInputs: { candidateActionsSha256: string; frozenSelectionsSha256: string };
};
if (
  amendment003.version !== V10_AMSS_VERSION ||
  amendment003.status !== "frozen-container-dependency-only-amendment" ||
  amendment003.correctedDockerfileSha256 !== sha256(source.dockerfile) ||
  amendment003.immutableInputs.candidateActionsSha256 !== sha256(source.candidateActions) ||
  amendment003.immutableInputs.frozenSelectionsSha256 !== sha256(source.frozenSelections)
) throw new Error("The successful cloud image does not match the frozen build amendment.");

const imageDigest = "sha256:dea05248f57db3411cbd892668c6d523d62063d88126d59fa688ac00f8eea3eb";
if (!source.cloudRun.toString("utf8").includes(`amss-truth-v10@${imageDigest}`)) {
  throw new Error("The Cloud Run job template does not pin the successful image digest.");
}
const receipt = {
  artifactId: "flux-v10-amss-cloud-truth-image-receipt-v1",
  version: V10_AMSS_VERSION,
  status: "image-built-and-pinned-before-truth-tasks",
  buildId: "1250ccd9-2c87-47e7-b830-7cb9af086e2c",
  image:
    `europe-west1-docker.pkg.dev/fluxmmm-production/fluxmmm-research/amss-truth-v10@${imageDigest}`,
  imageDigest,
  truthTasksStartedAtFreeze: 0,
  scientificContractChanged: false,
  immutableInputs: {
    technicalAmendment003Sha256: sha256(source.amendment003),
    candidateActionsSha256: sha256(source.candidateActions),
    frozenSelectionsSha256: sha256(source.frozenSelections),
  },
  executionSources: {
    evaluatorSha256: sha256(source.evaluator),
    cloudTaskSha256: sha256(source.cloudTask),
    dockerfileSha256: sha256(source.dockerfile),
    pinnedCloudRunTemplateSha256: sha256(source.cloudRun),
  },
};
await writeFile(OUTPUT, `${JSON.stringify(receipt, null, 2)}\n`, { flag: "wx" });
console.log(JSON.stringify({ frozen: true, image: receipt.image, truthTasksStarted: 0 }, null, 2));
