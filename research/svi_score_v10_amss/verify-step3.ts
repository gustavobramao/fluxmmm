import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const ROOT = resolve("research/svi_score_v10_amss");
const ARTIFACT_ROOT = resolve(ROOT, "step3-artifacts");

function invariant(condition: unknown, detail: string): asserts condition {
  if (!condition) throw new Error(`V10 AMSS Step 3 verification failed: ${detail}.`);
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

const manifest = JSON.parse(
  await readFile(resolve(ARTIFACT_ROOT, "manifest.json"), "utf8"),
) as {
  status: string;
  researchLabels: boolean;
  truthCalculated: boolean;
  upstreamRevision: string;
  observedInterface: {
    rowsAfterBurnIn: number;
    rawChannels: string[];
    frozenV9Roles: string[];
    perfectMarketRateExposed: boolean;
  };
  inputs: Record<string, string>;
  outputs: Record<string, string>;
};
const proxyContract = JSON.parse(
  await readFile(resolve(ROOT, "amss-paid-social-proxy-contract.json"), "utf8"),
) as {
  upstreamAmssRevision: string;
  mechanism: { rawChannelName: string; amssModule: string; moduleModified: boolean };
  frozenV9Mapping: Record<string, string>;
  researchBoundary: {
    tuneProxyAfterFluxOutcomes: boolean;
    dropBusinessesAfterFluxOutcomes: boolean;
    threeChannelSmokeProducesResearchLabels: boolean;
  };
};

invariant(manifest.status === "passed", "smoke status is not passed");
invariant(!manifest.researchLabels, "smoke produced research labels");
invariant(!manifest.truthCalculated, "smoke opened AMSS truth");
invariant(
  !manifest.observedInterface.perfectMarketRateExposed,
  "perfect market rate leaked into the Flux input",
);
invariant(manifest.observedInterface.rowsAfterBurnIn === 156, "row count changed");
invariant(
  JSON.stringify(manifest.observedInterface.rawChannels) ===
    JSON.stringify(["paid_social_proxy", "search", "tv"]),
  "raw channel order changed",
);
invariant(
  manifest.upstreamRevision === proxyContract.upstreamAmssRevision,
  "AMSS revision changed",
);
invariant(
  proxyContract.mechanism.rawChannelName === "paid_social_proxy" &&
    proxyContract.mechanism.amssModule === "DefaultTraditionalMediaModule" &&
    !proxyContract.mechanism.moduleModified,
  "paid-social proxy no longer uses the unchanged public AMSS module",
);
invariant(
  !proxyContract.researchBoundary.tuneProxyAfterFluxOutcomes &&
    !proxyContract.researchBoundary.dropBusinessesAfterFluxOutcomes &&
    !proxyContract.researchBoundary.threeChannelSmokeProducesResearchLabels,
  "research boundary changed",
);
invariant(
  JSON.stringify(Object.values(proxyContract.frozenV9Mapping)) ===
    JSON.stringify(manifest.observedInterface.frozenV9Roles),
  "V9 role mapping changed",
);

for (const [path, expected] of Object.entries(manifest.inputs)) {
  invariant(
    sha256(await readFile(resolve(ROOT, path))) === expected,
    `${path} checksum changed`,
  );
}
for (const [path, expected] of Object.entries(manifest.outputs)) {
  invariant(
    sha256(await readFile(resolve(ARTIFACT_ROOT, path))) === expected,
    `${path} checksum changed`,
  );
}

const input = await readFile(
  resolve(ARTIFACT_ROOT, "three-channel-flux-input-smoke.csv"),
  "utf8",
);
const [header, ...rows] = input.trim().split("\n");
invariant(rows.length === 156, "Flux input artifact is incomplete");
invariant(
  header ===
    "date,revenue,meta_acquisition_spend,google_search_nonbrand_spend,ctv_spend",
  "Flux input schema changed",
);
invariant(!input.includes("market.rate"), "latent market rate is visible");

console.log(JSON.stringify({
  verified: true,
  upstreamRevision: manifest.upstreamRevision,
  rawChannels: manifest.observedInterface.rawChannels,
  frozenV9Roles: manifest.observedInterface.frozenV9Roles,
  paidSocialMechanism: "AMSS DefaultTraditionalMediaModule (unchanged)",
  truthFirewall: "closed",
  researchLabelsCreated: false,
}, null, 2));
