import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { generateAuditPopulation } from "./population";

const artifact = generateAuditPopulation();
const directory = resolve("research/score_v3/artifacts");
await mkdir(directory, { recursive: true });
const publicArtifact = {
  artifactId: artifact.artifactId,
  simulatorVersion: artifact.simulatorVersion,
  evidenceRegistryVersion: artifact.evidenceRegistryVersion,
  businessCount: artifact.businessCount,
  splits: artifact.splits,
  families: artifact.families,
  coverage: artifact.coverage,
  decisions: artifact.decisions,
  sampleBusinesses: artifact.families.map((family) =>
    artifact.businesses.find((business) => business.family === family.id)!,
  ),
};
await writeFile(
  resolve(directory, "simulator-audit-v3.json"),
  JSON.stringify(artifact, null, 2),
);
await writeFile(
  resolve(directory, "simulator-audit-v3-summary.json"),
  JSON.stringify(publicArtifact, null, 2),
);
process.stdout.write(
  `Simulator Audit V3 generated ${artifact.businessCount} businesses: ` +
    `${artifact.splits.train} train, ${artifact.splits.validation} validation, ` +
    `${artifact.splits.audit} untouched audit.\n`,
);
