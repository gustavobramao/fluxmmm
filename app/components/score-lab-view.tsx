"use client";

import { useMemo, useState } from "react";
import {
  SCORE_LAB_SCENARIOS,
  type ScoreLabCandidate,
  type ScoreLabLayer,
} from "../score-lab-data";

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
        <span>Candidate</span><span>Heuristic score</span><span>True budget regret</span>
      </div>
      {candidates.map((candidate, index) => (
        <button
          key={candidate.id}
          className={`score-lab-candidate-row ${candidate.id === selectedId ? "selected" : ""}`}
          onClick={() => onSelect(candidate.id)}
        >
          <strong><i>{index + 1}</i>{candidate.label}</strong>
          <span className="score-lab-bar heuristic">
            <i style={{ width: `${candidate.score}%` }} />
            <b>{candidate.score.toFixed(1)}</b>
          </span>
          <span className={`score-lab-bar regret ${candidate.budgetRegret <= 0.025 ? "good" : ""}`}>
            <i style={{ width: `${Math.min(candidate.budgetRegret * 100, 100)}%` }} />
            <b>{percentage(candidate.budgetRegret)}</b>
          </span>
        </button>
      ))}
    </div>
  );
}

export function ScoreLabView() {
  const [scenarioId, setScenarioId] = useState("demand-confounded-search");
  const scenario = SCORE_LAB_SCENARIOS.find((item) => item.id === scenarioId) ?? SCORE_LAB_SCENARIOS[0];
  const heuristicRanked = useMemo(
    () => [...scenario.candidates].sort((left, right) => right.score - left.score),
    [scenario],
  );
  const heuristicWinner = heuristicRanked[0];
  const truthRanked = useMemo(
    () => [...scenario.candidates].sort(
      (left, right) => left.budgetRegret - right.budgetRegret || left.roiError - right.roiError,
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
          <span className="kicker">Simulation-calibrated validation · research V2</span>
          <h1>Score Lab</h1>
          <p>
            Give Flux a dataset without its answer key, fit competing models,
            then reveal the synthetic causal truth and measure the decision each model would make.
          </p>
        </div>
        <span className="score-lab-status"><i /> Pilot · not production scoring</span>
      </section>

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
          <div className="score-lab-big-number truth">{percentage(truthWinner.budgetRegret)}<small>budget regret</small></div>
          <h2>{truthWinner.label}</h2>
          <p>Loses the least true incremental outcome versus the hidden optimal allocation.</p>
        </article>
      </section>

      <section className="score-lab-metric-grid">
        <article><span>Heuristic winner regret</span><strong className={heuristicWinner.budgetRegret > 0.1 ? "warning" : "good"}>{percentage(heuristicWinner.budgetRegret)}</strong><small>True opportunity lost</small></article>
        <article><span>Best available regret</span><strong className="good">{percentage(truthWinner.budgetRegret)}</strong><small>Within five pilot candidates</small></article>
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
            <div><span>Budget regret</span><b>{percentage(selectedCandidate.budgetRegret)}</b></div>
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
          <p>Simulation supplies labels such as true ROI error and budget regret. No learned score is displayed until held-out tests justify it.</p>
        </article>
      </section>

      <details className="card score-lab-method">
        <summary><span><b>How the synthetic answer key is created</b><small>Open the method without leaving the product</small></span><i>＋</i></summary>
        <div>
          <article><span>1</span><p><b>Generate spend and confounding</b><small>Latent demand, planning intensity, seasonality, and noise create realistic observational difficulty.</small></p></article>
          <article><span>2</span><p><b>Inject exact causal ROI</b><small>Each response coefficient is scaled so total contribution divided by total spend equals the declared truth.</small></p></article>
          <article><span>3</span><p><b>Hide the answer key</b><small>Flux receives only dates, revenue, media spend, and allowed controls—not latent demand or true contribution.</small></p></article>
          <article><span>4</span><p><b>Evaluate the actual decision</b><small>The recommended allocation is replayed through the true nonlinear curves to calculate lost opportunity.</small></p></article>
        </div>
      </details>

      <section className="score-lab-pilot-note">
        <i>◇</i><div><b>What this pilot establishes</b><p>Across 30 candidate fits, the current heuristic selected a lowest-regret model in 1 of 6 scenarios; its winners averaged 36.7% regret. This supports a larger simulation study—not an immediate production score replacement.</p></div>
      </section>
    </div>
  );
}
