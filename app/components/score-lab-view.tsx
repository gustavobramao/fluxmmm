"use client";

import { useMemo, useState } from "react";
import {
  SCORE_LAB_SCENARIOS,
  type ScoreLabCandidate,
  type ScoreLabLayer,
} from "../score-lab-data";
import { SIMULATOR_AUDIT_V3 } from "../score-audit-v3-data";
import type { DistributionSummary } from "../../research/score_v3/types";
import { RegretSetResearch } from "./regretset-research";
import v5dSviDevelopment from "../../research/svi_score_v5d_svi/artifacts/svi-score-v5d-svi-development.json";
import v5dSviImportance from "../../research/svi_score_v5d_svi/artifacts/svi-score-v5d-svi-feature-importance.json";

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

const V5D_SVI_PILLARS = [
  {
    name: "Posterior decision quality",
    groups: ["posterior-roi", "posterior-decision-safety", "posterior-reliability", "posterior-predictive"],
    question: "Are ROI levels, uncertainty, predictive coverage, and SVI stability usable for a decision?",
  },
  {
    name: "Model specification fit",
    groups: ["model-specification-and-interactions"],
    question: "Which adstock, saturation, likelihood, calibration, and dynamic assumptions fit this context?",
  },
  {
    name: "Temporal identification",
    groups: ["temporal-evidence"],
    question: "Can the data distinguish carryover and response across the campaign life cycle?",
  },
  {
    name: "Predictive and structural adequacy",
    groups: ["generalization", "structure"],
    question: "Does the model generalize while leaving defensible residual and coefficient behavior?",
  },
  {
    name: "Causal and decision coherence",
    groups: ["causal-robustness", "decision-coherence"],
    question: "Do ROI and budget decisions survive anchors, confounder stress, placebos, and evidence deletion?",
  },
].map((pillar) => ({
  ...pillar,
  importance: v5dSviImportance.risk.groups
    .filter((group) => pillar.groups.includes(group.group))
    .reduce((total, group) => total + group.importance, 0),
}));

function V5dSviOverview() {
  const baseline = v5dSviDevelopment.comparison.baseline;
  const current = v5dSviDevelopment.comparison.fullSvi;
  return (
    <>
      <section className="card v5d-score-hero">
        <div>
          <span className="eyebrow">Frozen public research release</span>
          <h2>V5D-SVI learns decision risk from posterior evidence.</h2>
          <p>
            Instead of assigning hand-written weights to four validation scores,
            Flux learns which observable signals predict excess economic loss across
            synthetic advertisers. The hidden answer key never enters an MMM fit or
            a deployment token.
          </p>
        </div>
        <div className="v5d-score-release">
          <span>V5D-SVI</span>
          <strong>136</strong>
          <small>auditable diagnostic tokens</small>
        </div>
      </section>

      <section className="v5d-score-contract" aria-label="V5D-SVI research contract">
        <article className="card"><span>Development cohort</span><strong>420</strong><p>advertiser worlds kept intact across folds</p></article>
        <article className="card"><span>Candidate evidence</span><strong>20,160</strong><p>doubly out-of-fold candidate assessments</p></article>
        <article className="card"><span>Loss heads</span><strong>Mean + P90</strong><p>average opportunity loss and downside protection</p></article>
        <article className="card"><span>Decision risk</span><strong>65 / 35</strong><p>declared mean-versus-tail business preference</p></article>
      </section>

      <section className="card v5d-score-pillars">
        <div className="card-heading">
          <div><span className="eyebrow">Readable interpretation</span><h2>Five practitioner pillars summarize the 136-token model</h2></div>
          <span className="score-lab-inspect">Importance explains the fitted selector; it is not a new score weight</span>
        </div>
        <div className="v5d-pillar-list">
          {V5D_SVI_PILLARS.map((pillar, index) => (
            <article key={pillar.name}>
              <span>{index + 1}</span>
              <div><b>{pillar.name}</b><p>{pillar.question}</p></div>
              <i><em style={{ width: `${pillar.importance * 100}%` }} /></i>
              <strong>{percentage(pillar.importance)}</strong>
            </article>
          ))}
        </div>
        <p className="v5d-score-caption">
          Importance is the normalized mean absolute standardized risk coefficient
          across 20 cross-fitted selectors. Correlated tokens can exchange importance,
          so these are predictive associations rather than causal effects.
        </p>
      </section>

      <section className="card v5d-score-evidence">
        <div className="card-heading">
          <div><span className="eyebrow">What the development evidence says</span><h2>Useful ranking signal, but automatic promotion is not yet justified</h2></div>
          <span className="score-lab-inspect">Lower loss is better</span>
        </div>
        <div className="v5d-score-evidence-grid">
          <article className="positive"><span>Costly-pair ordering</span><strong>{percentage(current.proxy.economicallyWeightedPairwiseAccuracy)}</strong><small>from {percentage(baseline.proxy.economicallyWeightedPairwiseAccuracy)}</small></article>
          <article className="positive"><span>Search objective</span><strong>{current.primary.objective.toFixed(2)}</strong><small>from {baseline.primary.objective.toFixed(2)}</small></article>
          <article className="warning"><span>Champion excess loss</span><strong>{current.proxy.riskChampionMeanExcessLoss.toFixed(2)}</strong><small>baseline {baseline.proxy.riskChampionMeanExcessLoss.toFixed(2)}</small></article>
          <article className="warning"><span>False champions</span><strong>{percentage(current.proxy.dangerousFalseChampionShare)}</strong><small>current safety pool is too permissive</small></article>
        </div>
        <div className="v5d-score-boundary">
          <i>!</i>
          <div><b>Research release, not silent production activation</b><p>V5D-SVI improves the ordering of economically costly mistakes, but its minimum predicted-risk candidate is still vulnerable to winner&apos;s curse. Agentic should continue to require immutable gates, local challenges, manual review, and confirmatory NUTS before production promotion.</p></div>
        </div>
      </section>

      <details className="card v5d-token-audit">
        <summary><span><b>Inspect the most influential raw tokens</b><small>Implementation names and conditional directions</small></span><i>＋</i></summary>
        <div>
          {v5dSviImportance.top50RiskFeatures.slice(0, 10).map((token) => (
            <article key={token.feature}>
              <span>{token.rank}</span>
              <div><b>{token.description}</b><small>{token.feature}</small></div>
              <em>{token.direction === "raises-predicted-loss" ? "Higher risk" : "Lower risk"}</em>
              <strong>{percentage(token.importance)}</strong>
            </article>
          ))}
          <p>All 50 leading raw tokens and the exact five-pillar mapping are documented in the paper appendix.</p>
        </div>
      </details>
    </>
  );
}

function V5dSviLearningPanel() {
  const blocks = [
    ["Validation diagnostics", "41", "Generalization, structure, causal robustness, and decision coherence"],
    ["Temporal evidence", "35", "Carryover support, flight behavior, kernel separation, and temporal interactions"],
    ["Model specification", "39", "Declared assumptions plus truth-blind interactions"],
    ["SVI posterior", "21", "Convergence, coverage, ROI centers and widths, plausibility, and seed agreement"],
  ];
  return (
    <>
      <section className="card v5d-learning-flow">
        <div className="card-heading">
          <div><span className="eyebrow">Training protocol</span><h2>One candidate becomes one truth-blind token vector</h2></div>
          <span className="score-lab-inspect">Truth opens only after the budget action is scored</span>
        </div>
        <div className="v5d-learning-steps">
          <article><span>1</span><b>Simulate a business</b><p>Draw channel mechanics, demand, confounding, evidence, and a hidden economic oracle.</p></article>
          <i>→</i>
          <article><span>2</span><b>Fit 48 candidates</b><p>Twenty-four model specifications under two evidence contracts use stored FullRankADVI posteriors.</p></article>
          <i>→</i>
          <article><span>3</span><b>Create 136 tokens</b><p>Only diagnostics, assumptions, and posterior observables available for a new advertiser are retained.</p></article>
          <i>→</i>
          <article><span>4</span><b>Reveal economic loss</b><p>The chosen budget is replayed through hidden truth and converted to excess opportunity loss.</p></article>
        </div>
      </section>

      <section className="card v5d-token-contract">
        <div className="card-heading">
          <div><span className="eyebrow">Token contract</span><h2>Named scalars, not embeddings or language-model tokens</h2></div>
          <span className="score-lab-inspect">115 deployment diagnostics + 21 posterior tokens</span>
        </div>
        <div className="v5d-token-blocks">
          {blocks.map(([label, count, description]) => (
            <article key={label}><strong>{count}</strong><div><b>{label}</b><p>{description}</p></div></article>
          ))}
        </div>
      </section>

      <section className="v5d-head-grid">
        <article className="card">
          <span className="eyebrow">Head 1 · expected loss</span>
          <h2>Robust mean prediction</h2>
          <div className="v5d-formula">Huber(mean prediction − excess loss) + L2</div>
          <p>Huber loss behaves like squared error for ordinary misses but limits the leverage of catastrophic outliers.</p>
        </article>
        <article className="card">
          <span className="eyebrow">Head 2 · downside loss</span>
          <h2>Conditional P90 prediction</h2>
          <div className="v5d-formula">Pinball<sub>.90</sub>(P90 prediction − excess loss) + L2</div>
          <p>Pinball loss teaches the second head to estimate a high-loss boundary rather than another average.</p>
        </article>
        <article className="card risk">
          <span className="eyebrow">Declared business preference</span>
          <h2>One ranking risk</h2>
          <div className="v5d-formula">Risk = 0.65 × mean + 0.35 × P90</div>
          <p>The 65/35 mixture is subjective risk appetite. It is disclosed separately from the learned coefficients.</p>
        </article>
      </section>

      <section className="card v5d-crossfit">
        <div>
          <span className="eyebrow">Leakage control</span>
          <h2>Two-axis grouped cross-fitting</h2>
          <p>Each assessment is produced by a selector that saw neither that advertiser nor its nearby parameter region during training.</p>
        </div>
        <div className="v5d-crossfit-map">
          <span><b>5</b> advertiser folds</span><i>×</i><span><b>4</b> candidate regions</span><i>=</i><span><b>20</b> fitted selectors</span>
        </div>
      </section>

      <details className="card score-lab-method">
        <summary><span><b>Objective and governance details</b><small>What the learner optimizes and what remains immutable</small></span><i>＋</i></summary>
        <div>
          <article><span>1</span><p><b>Mean and tail targets</b><small>Both heads predict uncapped within-business excess economic loss relative to the eligible oracle.</small></p></article>
          <article><span>2</span><p><b>Cost-sensitive ranking</b><small>Each epoch emphasizes the challenger that most dangerously outranks the oracle, weighted by its economic gap.</small></p></article>
          <article><span>3</span><p><b>Immutable eligibility</b><small>Temporal integrity, structural validity, evidence coherence, and posterior reliability cannot be traded for a lower predicted risk.</small></p></article>
          <article><span>4</span><p><b>Claim boundary</b><small>The sealed 80-business audit was not opened for this release. Fresh generalization and automatic production activation are not claimed.</small></p></article>
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
        <div><b>The simulator is audited independently of the selector</b><p>This population contract supports score learning across versions. V5D-SVI remains conditional on the declared generator, grouped cross-fitting, and a future independent audit.</p></div>
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
  const [labMode, setLabMode] = useState<
    "evidence" | "method" | "findings" | "example" | "legacy-release" | "legacy-method" | "legacy-audit"
  >("evidence");
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
          <span className="kicker">RegretSet-MMM · relative log-regret release</span>
          <h1>Score research</h1>
          <p>Learn how posterior and validation evidence predicts economic decision loss across advertisers.</p>
        </div>
        <span className="score-lab-status active"><i /> External synthetic audit complete</span>
      </section>

      <section className="score-lab-mode-tabs compact" aria-label="Score research views">
        <button className={labMode === "evidence" ? "active" : ""} onClick={() => setLabMode("evidence")}>
          <span>1</span><b>Out-of-sample evidence</b><small>Five-fold advertiser-grouped evaluation</small>
        </button>
        <button className={labMode === "method" ? "active" : ""} onClick={() => setLabMode("method")}>
          <span>2</span><b>How it learns</b><small>Truth firewall, tokens, and loss</small>
        </button>
        <button className={labMode === "findings" ? "active" : ""} onClick={() => setLabMode("findings")}>
          <span>3</span><b>Selector insights</b><small>Information, routing, and attribution</small>
        </button>
        <button className={labMode === "example" ? "active" : ""} onClick={() => setLabMode("example")}>
          <span>4</span><b>Worked example</b><small>Why score learning matters</small>
        </button>
      </section>

      {labMode === "evidence" ? <RegretSetResearch view="evidence" /> : labMode === "method" ? <RegretSetResearch view="method" /> : labMode === "findings" ? <RegretSetResearch view="findings" /> : labMode === "legacy-release" ? <V5dSviOverview /> : labMode === "legacy-method" ? <V5dSviLearningPanel /> : labMode === "legacy-audit" ? <SimulatorAuditV3Panel /> : <>

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
        <article><span>Best available regret</span><strong className="good">{percentage(truthWinner.profitRegret)}</strong><small>Four-decision mean · {scenario.candidates.length} fitted candidates</small></article>
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
          <span className="eyebrow">Earlier heuristic</span>
          <h2>A transparent rule chosen by hand</h2>
          <div className="score-lab-formula">100 × G<sup>.20</sup> × S<sup>.15</sup> × C<sup>.25</sup> × D<sup>.40</sup></div>
          <p>Useful and interpretable—but its weights were not learned from known decision outcomes.</p>
        </article>
        <div className="score-lab-arrow">→</div>
        <article className="card future">
          <span className="eyebrow">V5D-SVI learned target</span>
          <h2>Predict the economic loss distribution</h2>
          <div className="score-lab-formula">Risk = 0.65 × mean loss + 0.35 × P90 loss</div>
          <p>Simulation supplies excess-loss labels only after fitting; the selector sees validation, specification, and posterior tokens.</p>
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
        <i>◇</i><div><b>How to read this example</b><p>It shows why a plausible-looking validation score can select the wrong economic decision. It explains the learning problem; its percentages are conditional on one declared simulator world and are not claims about a real advertiser.</p></div>
      </section>
      </>}
    </div>
  );
}
