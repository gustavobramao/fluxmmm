import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${pathname}`, {
      headers: { accept: "text/html", host: "localhost" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the Flux MMM workspace shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Flux MMM — Open measurement workspace<\/title>/i);
  assert.match(html, /Preparing your measurement workspace/);
  assert.match(html, /Loading the public weekly demo dataset/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
  assert.match(html, /og:image/);
  assert.match(html, /\/og\.png/);
});

test("ships the Robyn fixtures and local-only model contracts", async () => {
  const [
    sample,
    experiments,
    modelSource,
    workbenchSource,
    agenticSearchSource,
    samplingApiSource,
    samplingSource,
    budgetSource,
    samplingServiceSource,
    workerSource,
    viteConfig,
    scoreLabSource,
    auditArtifact,
    learnedScoreArtifact,
    scoreV6Artifact,
  ] =
    await Promise.all([
      readFile(new URL("../public/data/robyn_weekly.csv", import.meta.url), "utf8"),
      readFile(
        new URL("../public/data/robyn_experiments.csv", import.meta.url),
        "utf8",
      ),
      readFile(new URL("../lib/mmm/models.ts", import.meta.url), "utf8"),
      readFile(new URL("../app/workbench.tsx", import.meta.url), "utf8"),
      readFile(new URL("../lib/mmm/agentic-search.ts", import.meta.url), "utf8"),
      readFile(new URL("../lib/mmm/sampling-api.ts", import.meta.url), "utf8"),
      readFile(new URL("../lib/mmm/sampling.ts", import.meta.url), "utf8"),
      readFile(new URL("../lib/mmm/budget.ts", import.meta.url), "utf8"),
      readFile(new URL("../scripts/mcmc_server.py", import.meta.url), "utf8"),
      readFile(new URL("../worker/index.ts", import.meta.url), "utf8"),
      readFile(new URL("../vite.config.ts", import.meta.url), "utf8"),
      readFile(new URL("../app/components/score-lab-view.tsx", import.meta.url), "utf8"),
      readFile(new URL("../research/score_v3/artifacts/simulator-audit-v3-summary.json", import.meta.url), "utf8"),
      readFile(new URL("../research/score_v4/artifacts/learned-score-v4.json", import.meta.url), "utf8"),
      readFile(new URL("../research/score_v6/artifacts/learned-score-v6-pilot.json", import.meta.url), "utf8"),
    ]);

  assert.match(sample, /"DATE","revenue","tv_S"/);
  assert.ok(sample.split("\n").length >= 200);
  assert.match(experiments, /incremental_outcome,incremental_spend/);
  assert.match(modelSource, /function buildDesign/);
  assert.match(modelSource, /priorMean/);
  assert.match(modelSource, /dataset\.periodsPerYear/);
  assert.match(workbenchSource, /Calibration evidence/);
  assert.match(workbenchSource, /Prior or likelihood/);
  assert.doesNotMatch(workbenchSource, /Prior, not likelihood/);
  assert.match(workbenchSource, /Global channel search · active V6 score/);
  assert.match(workbenchSource, /Mandatory channel-response grid/);
  assert.match(workbenchSource, /Learned diagnostic decision-loss score · V6/);
  assert.match(workbenchSource, /Two-sided coherence gate · immutable/);
  assert.match(workbenchSource, /Mandatory Advanced challenge/);
  assert.match(workbenchSource, /restart-balanced refinements/);
  assert.match(workbenchSource, /Champion neighborhood/);
  assert.match(workbenchSource, /Search confidence/);
  assert.match(workbenchSource, /Exact winning configuration loaded/);
  assert.match(workbenchSource, /Winning base settings inherited/);
  assert.match(workbenchSource, /View all \{parameterCount\} parameters/);
  assert.match(workbenchSource, /Force promote strongest candidate/);
  assert.match(workbenchSource, /Confirm force promotion/);
  assert.match(workbenchSource, /Manual force-promotion override/);
  assert.match(workbenchSource, /Forced candidate configuration loaded/);
  assert.match(workbenchSource, /function SpecificationInspector/);
  assert.match(workbenchSource, /function DrawerShell/);
  assert.match(workbenchSource, /createPortal/);
  assert.match(workbenchSource, /drawer-scroll-region/);
  assert.match(workbenchSource, /Advanced search dimensions/);
  assert.match(workbenchSource, /Base estimator · Advanced knobs inactive/);
  assert.match(workbenchSource, /Production inference/);
  assert.match(workbenchSource, /Choose the computational depth/);
  assert.match(workbenchSource, /Apply recommended retry/);
  assert.match(workbenchSource, /Analytic screening versus MCMC channel ROI/);
  assert.match(workbenchSource, /Why ROI moved/);
  assert.match(workbenchSource, /Budget planner/);
  assert.match(workbenchSource, /Plan a budget/);
  assert.match(workbenchSource, /Hit a target/);
  assert.match(workbenchSource, /Find the ceiling/);
  assert.match(workbenchSource, /How optimization works/);
  assert.match(workbenchSource, /Show the mathematics/);
  assert.match(workbenchSource, /Channel evidence attribution/);
  assert.match(workbenchSource, /Share of local ROI precision/);
  assert.match(workbenchSource, /Conditional data separation/);
  assert.doesNotMatch(workbenchSource, />Prior-led</);
  assert.doesNotMatch(workbenchSource, />Data \+ prior</);
  assert.match(workbenchSource, /Non-negative boundary/);
  assert.match(workbenchSource, /Interval materially affected/);
  assert.match(workbenchSource, /origin: DatasetExperimentOrigin = "advertiser-upload"/);
  assert.match(workbenchSource, /Public fixture · read only/);
  assert.match(workbenchSource, /if \(!demoMode\) fileRef\.current\?\.click\(\)/);
  assert.match(workbenchSource, /defaultExperimentsForDataset\(origin\)/);
  assert.match(
    workbenchSource,
    /"Robyn Public weekly data demo\.csv",\s*"robyn-demo"/s,
  );
  assert.match(workbenchSource, /Shared contract · screening vs full posterior/);
  assert.match(workbenchSource, /Local approximation against sampled probability/);
  assert.match(workbenchSource, /How the two estimates are constructed/);
  assert.doesNotMatch(workbenchSource, /MCMC becomes truth when converged/);
  assert.match(agenticSearchSource, /function proposeAgenticCandidate/);
  assert.match(agenticSearchSource, /function generateAgenticLocalChallenge/);
  assert.match(agenticSearchSource, /function agenticSearchConfidence/);
  assert.match(agenticSearchSource, /function agenticStoppingDecision/);
  assert.match(samplingApiSource, /SAMPLING_SERVICE_PORT = 8790/);
  assert.match(samplingSource, /SAMPLING_PRESETS/);
  assert.match(samplingSource, /compileSamplingModel/);
  assert.match(samplingSource, /full-response-same-window-experiment-roi/);
  assert.match(samplingSource, /posteriorSamples/);
  const workspaceSamplingContract = samplingSource.match(
    /SAMPLING_ENGINE_VERSION\s*=\s*\n?\s*"([^"]+)"/,
  )?.[1];
  const serviceSamplingContract = samplingServiceSource.match(
    /SAMPLING_CONTRACT_VERSION\s*=\s*\(\s*"([^"]+)"/,
  )?.[1];
  assert.ok(workspaceSamplingContract, "workspace sampling contract is declared");
  assert.equal(
    serviceSamplingContract,
    workspaceSamplingContract,
    "browser compiler and local sampler must advertise the same contract",
  );
  assert.match(budgetSource, /function solveFixedBudget/);
  assert.match(budgetSource, /function futureTransformedResponse/);
  assert.match(budgetSource, /Multi-start constrained frontier search/);
  assert.match(samplingServiceSource, /pm\.sample/);
  assert.match(samplingServiceSource, /posterior_predictive_draws/);
  assert.match(samplingServiceSource, /hdi_bounds/);
  assert.match(samplingServiceSource, /adstock_decay/);
  assert.match(samplingServiceSource, /planning_deviation_weight/);
  assert.match(samplingServiceSource, /sharedRoiTransform/);
  assert.match(samplingServiceSource, /density_explanation/);
  assert.match(samplingServiceSource, /approximation_assessment/);
  assert.match(samplingServiceSource, /baseline_parameter_indexes/);
  assert.match(samplingServiceSource, /posteriorSamples/);
  assert.doesNotMatch(samplingServiceSource, /media\.get\("mapRoi"\)/);
  assert.match(samplingServiceSource, /R-hat ≤ 1\.01/);
  assert.match(workerSource, /CREATE TABLE IF NOT EXISTS model_runs/);
  assert.match(viteConfig, /binding: "DB"/);
  assert.match(viteConfig, /binding: "DATASETS"/);
  assert.doesNotMatch(viteConfig, /hostingConfig|sites\(\)/);
  assert.match(scoreLabSource, /Simulator Audit V3/);
  assert.match(scoreLabSource, /Split generators—not random rows/);
  assert.match(scoreLabSource, /V5D-SVI learns decision risk from posterior evidence\./);
  assert.match(scoreLabSource, /Five practitioner pillars summarize the 136-token model/);
  assert.match(scoreLabSource, /Named scalars, not embeddings or language-model tokens/);
  assert.match(scoreLabSource, /Research release, not silent production activation/);
  assert.match(auditArtifact, /"businessCount": 500/);
  assert.match(auditArtifact, /"audit": 80/);
  assert.match(learnedScoreArtifact, /"activation": "active"/);
  assert.match(learnedScoreArtifact, /"candidates": 3200/);
  assert.match(scoreV6Artifact, /"activation": "active"/);
  assert.match(scoreV6Artifact, /"runtimeContractChanged": true/);
  assert.match(scoreV6Artifact, /"ready": false/);
  assert.match(scoreV6Artifact, /"candidates": 14400/);
  await access(new URL("../public/og.png", import.meta.url));
  await assert.rejects(
    access(new URL("../app/chatgpt-auth.ts", import.meta.url)),
  );
});

test("interactive charts use complete canvas redraws", async () => {
  const chartSources = await Promise.all(
    [
      "../app/components/budget-chart.tsx",
      "../app/components/time-series-chart.tsx",
      "../app/components/sampling-chart.tsx",
      "../app/components/validation-evidence-chart.tsx",
    ].map((path) => readFile(new URL(path, import.meta.url), "utf8")),
  );

  for (const source of chartSources) {
    assert.match(source, /useDirtyRect:\s*false/);
    assert.doesNotMatch(source, /useDirtyRect:\s*true/);
  }
});
