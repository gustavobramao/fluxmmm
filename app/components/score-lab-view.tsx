"use client";

import { useMemo, useState } from "react";
import {
  SCORE_LAB_SCENARIOS,
  type ScoreLabCandidate,
  type ScoreLabLayer,
} from "../score-lab-data";
import { SIMULATOR_AUDIT_V3 } from "../score-audit-v3-data";
import type { DistributionSummary } from "../../research/score_v3/types";
import scoreV6Artifact from "../../research/score_v6/artifacts/learned-score-v6-pilot.json";
import {
  ACTIVE_SCORE_CONTRACT,
  LEARNED_SCORE_ARTIFACT,
  SCORE_LAYER_ORDER,
} from "../../lib/mmm/score-contract";

const layerCopy: Record<ScoreLabLayer, { label: string; weight: number }> = {
  generalization: { label: "Prediction", weight: 20 },
  structure: { label: "Structure", weight: 15 },
  causal: { label: "Causal", weight: 25 },
  decision: { label: "ROI coherence", weight: 40 },
};

const channelLabels = {
  paid_social: "Paid social",
  search: "Search",
  tv: "TV",
};

function percentage(value: number) {
  return `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`;
}

function compactMoney(value: number) {
  const magnitude = Math.abs(value);
  if (magnitude >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
  if (magnitude >= 1_000) return `$${Math.round(value / 1_000)}K`;
  return `$${Math.round(value)}`;
}

function rangeLabel(
  summary: DistributionSummary,
  format: (value: number) => string,
) {
  return `${format(summary.p10)}–${format(summary.p90)}`;
}

const scoreLayerLabels = {
  generalization: "Prediction",
  structure: "Structure",
  causal: "Causal",
  decision: "ROI coherence",
};

function ScoreGovernancePanel() {
  const challenger = scoreV6Artifact;
  const validation = challenger.performance.validation;
  const audit = challenger.performance.audit;
  const gateEntries = Object.entries(challenger.gates);
  const passedGates = gateEntries.filter(([, passed]) => passed).length;
  const validationCoverage =
    challenger.pipelineReadiness.decisionGradeBusinessShare;
  const requiredCoverage = challenger.pipelineReadiness.requiredShare;
  return (
    <>
      <section className="card learned-score-hero active">
        <div>
          <span className="eyebrow">Active score contract</span>
          <h2>V6 is now the validation ranker used by Agentic.</h2>
          <p>
            V6 learned which of twenty validation diagnostics best predict lower-regret
            decisions. Immutable gates still decide whether a candidate is eligible;
            V6 only ranks candidates inside the highest available eligibility tier.
          </p>
        </div>
        <div className="learned-score-activation">
          <span>Activated</span>
          <strong>20</strong>
          <small>learned diagnostic weights</small>
        </div>
      </section>

      <section className="score-version-strip" aria-label="Flux score and sampler versions">
        <article className="card active">
          <span>Active runtime score</span>
          <strong>V6</strong>
          <small>{ACTIVE_SCORE_CONTRACT.version} · used by Agentic</small>
        </article>
        <article className="card baseline">
          <span>Previous runtime score</span>
          <strong>V4</strong>
          <small>Four aggregate layer weights · retained as reference</small>
        </article>
        <article className="card sampler">
          <span>Production inference</span>
          <strong>PyMC NUTS</strong>
          <small>Shared sampler v2.2 · runs after promotion</small>
        </article>
      </section>

      <section className="card learned-score-science">
        <div className="card-heading">
          <div><span className="eyebrow">Nested validation</span><h2>Two holdouts answer two different questions</h2></div>
          <span className="score-lab-inspect">No synthetic truth enters an MMM fit</span>
        </div>
        <div className="score-nested-validation" aria-label="Nested validation used by V6">
          <article>
            <span>Inside every synthetic business</span>
            <h3>Does this candidate MMM generalize?</h3>
            <div><b>Past periods</b><i>→</i><b>Unseen future periods and spend regimes</b><i>→</i><b>20 diagnostics + immutable gates</b></div>
            <p>This evaluates each MMM candidate. Causal truth stays hidden until fitting and diagnostics are complete.</p>
          </article>
          <article>
            <span>Across synthetic businesses</span>
            <h3>Do the learned V6 weights generalize?</h3>
            <div><b>180 training businesses</b><i>→</i><b>60 unseen validation businesses</b><i>→</i><b>60 sealed audit businesses</b></div>
            <p>This evaluates the ranker itself. The weights are frozen before validation and the adversarial audit.</p>
          </article>
        </div>
      </section>

      <section className="card learned-score-weights">
        <div className="card-heading">
          <div><span className="eyebrow">V6 learned group emphasis</span><h2>Twenty diagnostics, kept in four readable layers</h2></div>
          <span className="score-lab-inspect">Previous V4 marker shown for comparison</span>
        </div>
        <div className="learned-weight-grid simple">
          {SCORE_LAYER_ORDER.map((layer) => {
            const learned = challenger.model.groupWeights[layer];
            const previousWeight = LEARNED_SCORE_ARTIFACT.model.weights[layer];
            return (
              <article key={layer}>
                <span>{scoreLayerLabels[layer]}</span>
                <strong>{percentage(learned)}</strong>
                <div>
                  <i style={{ width: `${learned * 100}%` }} />
                  <em style={{ left: `${previousWeight * 100}%` }} />
                </div>
                <small>Previous V4: {percentage(previousWeight)}</small>
              </article>
            );
          })}
        </div>
        <p className="learned-score-plain-language">
          <b>What changed:</b> V6 learns non-negative weights across twenty granular diagnostics;
          the four percentages above are their readable group totals. Agentic now uses the
          underlying twenty-weight formula rather than approximating it with four layer weights.
        </p>
      </section>

      <section className="card learned-score-proof-simple challenger">
        <div>
          <span className="eyebrow">Did V6 improve decisions?</span>
          <h2>Yes—across unseen families and the sealed audit</h2>
          <p>V6 reduced mean economic decision loss on unseen and sealed businesses, while passing every tail and family safety check.</p>
        </div>
        <div className="learned-proof-metrics">
          <article><span>Unseen families</span><strong>{percentage(validation.relativeMeanLossReduction)}</strong><small>lower mean decision loss</small></article>
          <article><span>Sealed audit</span><strong>{percentage(audit.relativeMeanLossReduction)}</strong><small>lower mean decision loss</small></article>
          <article><span>Ranker activation gates</span><strong>{passedGates}/{gateEntries.length}</strong><small>all score-learning gates passed</small></article>
        </div>
      </section>

      <section className="score-research-gate">
        <div>
          <span className="eyebrow">Separate Agentic readiness signal</span>
          <b>{percentage(validationCoverage)} of validation businesses produced a decision-grade candidate.</b>
          <p>This remains below the {percentage(requiredCoverage)} search-readiness target, but it no longer blocks V6. The score can rank valid candidates better; improving how often Agentic finds one is a separate optimization problem.</p>
        </div>
        <div className="score-research-gate-meter" aria-label={`${percentage(validationCoverage)} coverage against a ${percentage(requiredCoverage)} target`}>
          <span><i style={{ width: `${validationCoverage * 100}%` }} /><em style={{ left: `${requiredCoverage * 100}%` }} /></span>
          <small><b>{percentage(validationCoverage)}</b> observed <b>{percentage(requiredCoverage)}</b> target</small>
        </div>
      </section>

      <section className="learned-score-safety">
        <i>◆</i>
        <div><b>The production sampler remains a separate shared stage</b><p>Agentic searches quickly with analytic MAP and the active V6 score. After promotion, PyMC NUTS v2.2 performs confirmatory inference without changing the winning specification.</p></div>
      </section>

      <details className="card learned-score-receipt">
        <summary><span><b>V6 research receipt</b><small>Formula, cohort, gates, and activation decision</small></span><i>＋</i></summary>
        <div>
          <p className="learned-score-formula">
            {challenger.model.formula}
          </p>
          <p>{challenger.activationReason} {challenger.pipelineReadiness.detail}</p>
          <div className="learned-receipt-comparison">
            <span>Validation mean loss <b>{validation.heuristic.meanLoss.toFixed(2)} → {validation.learned.meanLoss.toFixed(2)}</b></span>
            <span>Audit mean loss <b>{audit.heuristic.meanLoss.toFixed(2)} → {audit.learned.meanLoss.toFixed(2)}</b></span>
            <span>Cohort <b>{challenger.cohort.businesses} businesses · {challenger.cohort.candidates.toLocaleString()} candidates</b></span>
          </div>
          <div className="score-research-gate-list">
            {gateEntries.map(([gate, passed]) => (
              <span key={gate} className={passed ? "pass" : "fail"}><i>{passed ? "✓" : "!"}</i>{gate.replace(/([A-Z])/g, " $1")}</span>
            ))}
          </div>
          <small>{challenger.version} · {challenger.artifactId} · train {challenger.cohort.splits.train} / validation {challenger.cohort.splits.validation} / untouched audit {challenger.cohort.splits.audit}</small>
        </div>
      </details>
    </>
  );
}

function SimulatorAuditV3Panel() {
  const audit = SIMULATOR_AUDIT_V3;
  const channelOrder = ["paid_social", "search", "tv"] as const;
  const decisionOrder = [
    "budget-reduction",
    "fixed-budget-mix",
    "budget-growth",
    "economic-ceiling",
  ] as const;
  return (
    <>
      <section className="card audit-v3-hero">
        <div>
          <span className="eyebrow">Simulator Audit V3</span>
          <h2>Is the answer-key factory credible enough to train a score?</h2>
          <p>
            Five hundred deterministic businesses audit population coverage and
            decision labels before any replacement Flux score is learned.
          </p>
        </div>
        <div className="audit-v3-count">
          <strong>{audit.businessCount}</strong>
          <span>audited businesses</span>
          <small>{audit.artifactId}</small>
        </div>
      </section>

      <section className="audit-v3-splits" aria-label="Predeclared simulator splits">
        <article className="card train">
          <span>Train</span><strong>{audit.splits.train}</strong>
          <p>Three generator families may inform score fitting.</p>
        </article>
        <article className="card validation">
          <span>Validation</span><strong>{audit.splits.validation}</strong>
          <p>Two unseen parameter families select and calibrate the score.</p>
        </article>
        <article className="card audit">
          <span>Untouched audit</span><strong>{audit.splits.audit}</strong>
          <p>Wrong evidence and swapped mechanics stay sealed until final evaluation.</p>
        </article>
      </section>

      <section className="card audit-v3-families">
        <div className="card-heading">
          <div><span className="eyebrow">Family-level holdout</span><h2>Split generators—not random rows</h2></div>
          <span className="score-lab-inspect">Predeclared before score learning</span>
        </div>
        <div className="audit-v3-family-grid">
          {audit.families.map((family) => (
            <article key={family.id} className={family.split}>
              <span>{family.split}</span>
              <strong>{family.label}</strong>
              <b>{family.businessCount} businesses</b>
              <p>{family.description}</p>
              {family.heldOutReason && <small>{family.heldOutReason}</small>}
            </article>
          ))}
        </div>
      </section>

      <section className="card audit-v3-coverage">
        <div className="card-heading">
          <div><span className="eyebrow">Population coverage</span><h2>Truth varies by channel and business</h2></div>
          <span className="score-lab-inspect">P10–P90 · median shown</span>
        </div>
        <div className="audit-v3-channel-table">
          <div className="head"><span>Channel</span><span>Average iROAS</span><span>Marginal iROAS</span><span>Memory</span><span>Weibull share</span></div>
          {channelOrder.map((channel) => {
            const coverage = audit.coverage.channel[channel];
            return (
              <div key={channel}>
                <strong>{channelLabels[channel]}</strong>
                <span>{coverage.roi.median.toFixed(2)}×<small>{rangeLabel(coverage.roi, (value) => `${value.toFixed(2)}×`)}</small></span>
                <span>{coverage.marginalRoi.median.toFixed(2)}×<small>{rangeLabel(coverage.marginalRoi, (value) => `${value.toFixed(2)}×`)}</small></span>
                <span>{coverage.memory.median.toFixed(2)}<small>{rangeLabel(coverage.memory, (value) => value.toFixed(2))}</small></span>
                <span>{percentage(coverage.weibullShare)}<small>family varies independently</small></span>
              </div>
            );
          })}
        </div>
        <div className="audit-v3-coverage-foot">
          <span><b>{audit.coverage.historyWeeks.median}</b> median weeks <small>{audit.coverage.historyWeeks.minimum}–{audit.coverage.historyWeeks.maximum}</small></span>
          <span><b>{percentage(audit.coverage.noiseShare.median)}</b> median outcome noise <small>P90 {percentage(audit.coverage.noiseShare.p90)}</small></span>
          <span><b>{percentage(audit.coverage.experimentCoverage)}</b> with ≥1 experiment <small>availability still varies by channel</small></span>
          <span><b>{percentage(audit.coverage.benchmarkLogError.median)}</b> median benchmark log error <small>benchmarks remain fallible</small></span>
        </div>
      </section>

      <section className="card audit-v3-decisions">
        <div className="card-heading">
          <div><span className="eyebrow">Continuous decision label</span><h2>Four decisions replace one coarse allocation grid</h2></div>
          <span className="score-lab-inspect">Dense deterministic search</span>
        </div>
        <div className="audit-v3-decision-grid">
          {decisionOrder.map((id) => {
            const decision = audit.decisions[id];
            return (
              <article key={id}>
                <span>{decision.label}</span>
                <strong>{compactMoney(decision.profitOpportunity.median)}</strong>
                <small>median hidden profit opportunity</small>
                <div>
                  <i style={{ width: `${decision.decreaseShare * 100}%` }} />
                  <i style={{ width: `${decision.unchangedShare * 100}%` }} />
                  <i style={{ width: `${decision.increaseShare * 100}%` }} />
                </div>
                <p>{id === "fixed-budget-mix" ? `${percentage(decision.mixShiftShare.median)} median budget moved between channels` : `${percentage(decision.decreaseShare)} decrease · ${percentage(decision.unchangedShare)} hold · ${percentage(decision.increaseShare)} increase`}</p>
              </article>
            );
          })}
        </div>
        <p className="audit-v3-label-note">
          Candidate label = mean profit regret across all four decisions. Each
          component remains visible, so failures cannot be hidden by averaging.
        </p>
      </section>

      <section className="audit-v3-gate">
        <i>◇</i>
        <div><b>The simulator is audited independently of the ranker</b><p>This population contract supports score learning across versions. V6 remains conditional on the declared simulator, family-held-out validation, and sealed adversarial audit.</p></div>
      </section>
    </>
  );
}

function CandidateBars({
  candidates,
  selectedId,
  onSelect,
}: {
  candidates: ScoreLabCandidate[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="score-lab-candidate-list">
      <div className="score-lab-candidate-row head">
        <span>Candidate</span><span>Heuristic score</span><span>Simulator profit regret</span>
      </div>
      {candidates.map((candidate, index) => (
        <button
          key={candidate.id}
          className={`score-lab-candidate-row ${candidate.id === selectedId ? "selected" : ""}`}
          onClick={() => onSelect(candidate.id)}
        >
          <strong><i>{index + 1}</i><span>{candidate.label}<small>{candidate.evidenceArm === "benchmark-gap-fill" ? "Benchmark gap-fill" : "Experiments only"}</small></span></strong>
          <span className="score-lab-bar heuristic">
            <i style={{ width: `${candidate.score}%` }} />
            <b>{candidate.score.toFixed(1)}</b>
          </span>
          <span className={`score-lab-bar regret ${candidate.profitRegret <= 0.025 ? "good" : ""}`}>
            <i style={{ width: `${Math.min(candidate.profitRegret * 100, 100)}%` }} />
            <b>{percentage(candidate.profitRegret)}</b>
          </span>
        </button>
      ))}
    </div>
  );
}

export function ScoreLabView() {
  const [labMode, setLabMode] = useState<"pilot" | "audit" | "learned">("learned");
  const [scenarioId, setScenarioId] = useState("demand-confounded-search");
  const scenario = SCORE_LAB_SCENARIOS.find((item) => item.id === scenarioId) ?? SCORE_LAB_SCENARIOS[0];
  const heuristicRanked = useMemo(
    () => [...scenario.candidates].sort((left, right) => right.score - left.score),
    [scenario],
  );
  const heuristicWinner = heuristicRanked[0];
  const truthRanked = useMemo(
    () => [...scenario.candidates].sort(
      (left, right) => left.profitRegret - right.profitRegret || left.roiError - right.roiError,
    ),
    [scenario],
  );
  const truthWinner = truthRanked[0];
  const [selectedCandidateId, setSelectedCandidateId] = useState(heuristicWinner.id);
  const selectedCandidate = scenario.candidates.find(
    (candidate) => candidate.id === selectedCandidateId,
  ) ?? heuristicWinner;
  const selectScenario = (id: string) => {
    const next = SCORE_LAB_SCENARIOS.find((item) => item.id === id);
    if (!next) return;
    setScenarioId(id);
    setSelectedCandidateId(
      [...next.candidates].sort((left, right) => right.score - left.score)[0].id,
    );
  };
  const winnersDiffer = heuristicWinner.id !== truthWinner.id;

  return (
    <div className="view score-lab-view">
      <section className="page-heading compact-heading score-lab-heading">
        <div>
          <span className="kicker">Validation score governance · active V6</span>
          <h1>Score research</h1>
          <p>See how V6 learns twenty validation weights from known synthetic decision outcomes while keeping candidate eligibility gates immutable.</p>
        </div>
        <span className="score-lab-status active"><i /> Runtime · V6 active</span>
      </section>

      <section className="score-lab-mode-tabs compact" aria-label="Score research views">
        <button className={labMode === "learned" ? "active" : ""} onClick={() => setLabMode("learned")}>
          <span>1</span><b>Active V6 method</b><small>Weights, holdouts, and gates</small>
        </button>
        <button className={labMode === "audit" ? "active" : ""} onClick={() => setLabMode("audit")}>
          <span>2</span><b>Audit the simulator</b><small>Optional research detail</small>
        </button>
        <button className={labMode === "pilot" ? "active" : ""} onClick={() => setLabMode("pilot")}>
          <span>3</span><b>See a worked example</b><small>Optional model comparison</small>
        </button>
      </section>

      {labMode === "learned" ? <ScoreGovernancePanel /> : labMode === "audit" ? <SimulatorAuditV3Panel /> : <>

      <section className="score-lab-scenario-card card">
        <div className="score-lab-scenario-intro">
          <span className="eyebrow">Choose the identification challenge</span>
          <h2>{scenario.label}</h2>
          <p>{scenario.description}</p>
        </div>
        <div className="score-lab-scenario-tabs" role="tablist" aria-label="Synthetic scenarios">
          {SCORE_LAB_SCENARIOS.map((item) => (
            <button
              key={item.id}
              role="tab"
              aria-selected={item.id === scenario.id}
              className={item.id === scenario.id ? "active" : ""}
              onClick={() => selectScenario(item.id)}
            >
              <i />{item.shortLabel}
            </button>
          ))}
        </div>
        <div className="score-lab-complication"><span>!</span><p>{scenario.complication}</p><small>Seed {scenario.seed}</small></div>
      </section>

      <details className="card score-lab-contract" open>
        <summary>
          <span><span className="eyebrow">Audited simulator contract</span><b>What is actually different across channels</b></span>
          <small>Versioned evidence · delivery · response · oracle</small>
        </summary>
        <div className="score-lab-contract-grid">
          {scenario.truthChannels.map((channel) => (
            <article key={channel.channel}>
              <span>{channelLabels[channel.channel]}</span>
              <strong>{channel.targetRoi.toFixed(2)}× <small>average iROAS</small></strong>
              <p>{channel.marginalRoiAtObserved.toFixed(2)}× marginal iROAS at observed spend</p>
              <ul>
                <li>{channel.delivery.kind === "auction" ? "Demand-limited auction delivery" : channel.delivery.kind === "reach" ? "Reach and frequency delivery" : "Persistent flight / GRP delivery"}</li>
                <li>{channel.response.family === "weibull" ? `Weibull ${channel.response.shape?.toFixed(1)} / ${channel.response.scale?.toFixed(1)} weeks` : `Geometric decay ${channel.response.decay?.toFixed(2)}`}</li>
                <li>Hill {channel.response.hillShape.toFixed(2)} · half-saturation Q{Math.round(channel.response.halfSaturationQuantile * 100)}</li>
              </ul>
            </article>
          ))}
          <article className="oracle">
            <span>Decision oracle</span>
            <strong>{percentage(scenario.oracle.effectiveRevenueMargin)} <small>effective margin</small></strong>
            <p>{scenario.oracle.optimalIncrementalProfit > 0 ? `$${Math.round(scenario.oracle.optimalIncrementalProfit / 1000)}K hidden optimal profit` : "Zero spend is the hidden optimum"}</p>
            <ul>
              <li>Tests {scenario.oracle.evaluatedBudgetShares.map((share) => `${Math.round(share * 100)}%`).join(", ")} spend increments</li>
              <li>Channel concentration constraints enforced</li>
              <li>Outcome is replayed through hidden delivery curves</li>
            </ul>
          </article>
        </div>
        <div className="score-lab-evidence-strip">
          <b>Independent evidence generated after truth:</b>
          {scenario.experiments.map((experiment) => (
            <span key={experiment.channel}>{channelLabels[experiment.channel]} {experiment.observedRoi.toFixed(2)}× ± {experiment.standardError.toFixed(2)} <small>({experiment.design})</small></span>
          ))}
          <span>Search <small>benchmark gap-fill arm only</small></span>
        </div>
      </details>

      <section className={`score-lab-verdict ${winnersDiffer ? "disagrees" : "agrees"}`}>
        <article>
          <span className="eyebrow">What the current heuristic chooses</span>
          <div className="score-lab-big-number">{heuristicWinner.score.toFixed(1)}<small>/ 100</small></div>
          <h2>{heuristicWinner.label}</h2>
          <p>Ranks first from prediction, structure, causal robustness, and ROI coherence.</p>
        </article>
        <div className="score-lab-verdict-bridge">
          <span>{winnersDiffer ? "≠" : "="}</span>
          <b>{winnersDiffer ? "Different decision" : "Same decision"}</b>
          <small>Truth is revealed only after fitting</small>
        </div>
        <article>
          <span className="eyebrow">What synthetic truth reveals</span>
          <div className="score-lab-big-number truth">{percentage(truthWinner.profitRegret)}<small>mean decision regret</small></div>
          <h2>{truthWinner.label}</h2>
          <p>Loses the least hidden profit across cut, reallocation, growth, and ceiling decisions.</p>
        </article>
      </section>

      <section className="score-lab-metric-grid">
        <article><span>Heuristic winner regret</span><strong className={heuristicWinner.profitRegret > 0.1 ? "warning" : "good"}>{percentage(heuristicWinner.profitRegret)}</strong><small>Mean across four decisions</small></article>
        <article><span>Best available regret</span><strong className="good">{percentage(truthWinner.profitRegret)}</strong><small>Four-decision mean · fourteen fitted arms</small></article>
        <article><span>Winner ROI error</span><strong>{percentage(heuristicWinner.roiError)}</strong><small>Spend-weighted log error</small></article>
        <article><span>Benchmark vs truth</span><strong>{percentage(scenario.benchmarkError)}</strong><small>Benchmarks remain fallible</small></article>
      </section>

      <section className="score-lab-main-grid">
        <article className="card score-lab-ranking-card">
          <div className="card-heading">
            <div><span className="eyebrow">Same candidates · two rankings</span><h2>High score does not always mean low regret</h2></div>
            <span className="score-lab-inspect">Select a row to inspect</span>
          </div>
          <CandidateBars candidates={heuristicRanked} selectedId={selectedCandidate.id} onSelect={setSelectedCandidateId} />
          <div className="score-lab-axis-note"><span>Heuristic: higher is better</span><span>Regret: lower is better</span></div>
        </article>

        <article className="card score-lab-selected-card">
          <div className="card-heading">
            <div><span className="eyebrow">Selected candidate</span><h2>{selectedCandidate.label}</h2></div>
            <b>{selectedCandidate.score.toFixed(1)}</b>
          </div>
          <div className="score-lab-layer-list">
            {(Object.keys(layerCopy) as ScoreLabLayer[]).map((layer) => (
              <div key={layer}>
                <span>{layerCopy[layer].label}<small>{layerCopy[layer].weight}% heuristic weight</small></span>
                <i><em style={{ width: `${selectedCandidate.layers[layer]}%` }} /></i>
                <b>{selectedCandidate.layers[layer].toFixed(1)}</b>
              </div>
            ))}
          </div>
          <div className="score-lab-selected-truth">
            <div><span>ROI truth error</span><b>{percentage(selectedCandidate.roiError)}</b></div>
            <div><span>Contribution error</span><b>{percentage(selectedCandidate.contributionError)}</b></div>
            <div><span>Profit regret</span><b>{percentage(selectedCandidate.profitRegret)}</b></div>
          </div>
        </article>
      </section>

      <section className="card score-lab-roi-card">
        <div className="card-heading">
          <div><span className="eyebrow">Hidden answer key</span><h2>Did the candidates recover channel ROI?</h2></div>
          <span className="score-lab-inspect">Aggregate incremental ROI</span>
        </div>
        <div className="score-lab-roi-table">
          <div className="head"><span>Channel</span><span>Injected truth</span><span>Heuristic winner</span><span>Lowest-regret candidate</span></div>
          {(Object.keys(channelLabels) as (keyof typeof channelLabels)[]).map((channel) => {
            const heuristicGap = Math.abs(heuristicWinner.roi[channel] - scenario.truth[channel]) / scenario.truth[channel];
            const truthGap = Math.abs(truthWinner.roi[channel] - scenario.truth[channel]) / scenario.truth[channel];
            return (
              <div key={channel}>
                <strong>{channelLabels[channel]}</strong>
                <span className="truth-value">{scenario.truth[channel].toFixed(2)}×<small>Injected before noise</small></span>
                <span className={heuristicGap > 0.5 ? "warning" : ""}>{heuristicWinner.roi[channel].toFixed(2)}×<small>{Math.round(heuristicGap * 100)}% from truth</small></span>
                <span className={truthGap <= heuristicGap ? "good" : ""}>{truthWinner.roi[channel].toFixed(2)}×<small>{Math.round(truthGap * 100)}% from truth</small></span>
              </div>
            );
          })}
        </div>
      </section>

      <section className="score-lab-explainer">
        <article className="card">
          <span className="eyebrow">Current V1 heuristic</span>
          <h2>A transparent rule chosen by us</h2>
          <div className="score-lab-formula">100 × G<sup>.20</sup> × S<sup>.15</sup> × C<sup>.25</sup> × D<sup>.40</sup></div>
          <p>Useful and interpretable—but its weights were not learned from known decision outcomes.</p>
        </article>
        <div className="score-lab-arrow">→</div>
        <article className="card future">
          <span className="eyebrow">Proposed learned target</span>
          <h2>Predict whether the decision works</h2>
          <div className="score-lab-formula">P(regret &lt; 10% | validation evidence)</div>
          <p>Simulation supplies labels such as true ROI error and profit regret. No learned score is displayed until held-out tests justify it.</p>
        </article>
      </section>

      <details className="card score-lab-method">
        <summary><span><b>How the synthetic answer key is created</b><small>Open the method without leaving the product</small></span><i>＋</i></summary>
        <div>
          <article><span>1</span><p><b>Draw a plausible business</b><small>ROI is sampled hierarchically from a versioned DTC evidence registry, with explicit outcome, geography, and limitations.</small></p></article>
          <article><span>2</span><p><b>Generate delivery and confounding</b><small>Auctions, reach/frequency, TV flights, planning, commercial intensity, promotions, price variation, and multiplicative noise produce observational data.</small></p></article>
          <article><span>3</span><p><b>Generate imperfect evidence</b><small>Independent experiments are noisy measurements. A paired arm adds industry priors only where an experiment is absent.</small></p></article>
          <article><span>4</span><p><b>Reveal profit truth</b><small>After fitting, each recommendation is replayed through the hidden channel curves with margin, LTV, budget levels, and constraints.</small></p></article>
        </div>
      </details>

      <section className="score-lab-pilot-note">
        <i>◇</i><div><b>What this pilot establishes</b><p>This small audited matrix now separates model assumptions, evidence arms, channel delivery, and profit decisions. Its percentages are conditional on the declared simulator—not claims about your real business. Scale-up and held-out simulator families are still required before learning a replacement score.</p></div>
      </section>
      </>}
    </div>
  );
}
