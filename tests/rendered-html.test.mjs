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
    samplingSource,
    budgetSource,
    samplingServiceSource,
    workerSource,
    viteConfig,
    scoreLabSource,
    auditArtifact,
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
      readFile(new URL("../lib/mmm/sampling.ts", import.meta.url), "utf8"),
      readFile(new URL("../lib/mmm/budget.ts", import.meta.url), "utf8"),
      readFile(new URL("../scripts/mcmc_server.py", import.meta.url), "utf8"),
      readFile(new URL("../worker/index.ts", import.meta.url), "utf8"),
      readFile(new URL("../vite.config.ts", import.meta.url), "utf8"),
      readFile(new URL("../app/components/score-lab-view.tsx", import.meta.url), "utf8"),
      readFile(new URL("../research/score_v3/artifacts/simulator-audit-v3-summary.json", import.meta.url), "utf8"),
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
  assert.match(workbenchSource, /Confidence search · v4/);
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
  assert.match(workbenchSource, /Channel identification/);
  assert.match(workbenchSource, /Data \+ prior/);
  assert.match(workbenchSource, /Prior-led/);
  assert.match(workbenchSource, /Non-negative boundary/);
  assert.match(workbenchSource, /Interval materially affected/);
  assert.match(workbenchSource, /origin: DatasetExperimentOrigin = "advertiser-upload"/);
  assert.match(workbenchSource, /Public fixture · read only/);
  assert.match(workbenchSource, /if \(!demoMode\) fileRef\.current\?\.click\(\)/);
  assert.match(workbenchSource, /defaultExperimentsForDataset\(origin\)/);
  assert.match(workbenchSource, /"Robyn Public weekly data demo\.csv", "robyn-demo"/);
  assert.match(workbenchSource, /Shared contract · screening vs full posterior/);
  assert.match(workbenchSource, /Local approximation against sampled probability/);
  assert.match(workbenchSource, /How the two estimates are constructed/);
  assert.doesNotMatch(workbenchSource, /MCMC becomes truth when converged/);
  assert.match(agenticSearchSource, /function proposeAgenticCandidate/);
  assert.match(agenticSearchSource, /function generateAgenticLocalChallenge/);
  assert.match(agenticSearchSource, /function agenticSearchConfidence/);
  assert.match(agenticSearchSource, /function agenticStoppingDecision/);
  assert.match(samplingSource, /SAMPLING_PRESETS/);
  assert.match(samplingSource, /compileSamplingModel/);
  assert.match(samplingSource, /full-response-latent-planning-ppc-hdi/);
  assert.match(samplingSource, /posteriorSamples/);
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
  assert.match(scoreLabSource, /No learned score yet/);
  assert.match(auditArtifact, /"businessCount": 500/);
  assert.match(auditArtifact, /"audit": 80/);
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
