"use client";

import research from "../data/regretset-v11-public.json";

type RegretSetResearchProps = {
  view: "evidence" | "method" | "findings";
};

function percent(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function EvidenceView() {
  const validation = research.crossValidation;
  return (
    <>
      <section className="card regretset-hero">
        <div>
          <span className="eyebrow">Frozen research method · grouped out-of-sample evaluation</span>
          <h2>RegretSet-MMM selects models by their downstream economic consequences.</h2>
          <p>
            The frozen selector compares a complete candidate set, predicts economic
            loss from information available for a new advertiser, and ranks the
            candidates without access to that advertiser&apos;s causal truth.
          </p>
        </div>
        <div className="regretset-release">
          <span>{research.researchVersion}</span>
          <strong>{research.method}</strong>
          <small>{research.status}</small>
        </div>
      </section>

      <section className="regretset-metrics" aria-label="RegretSet research scale">
        <article className="card"><span>Advertiser worlds</span><strong>{research.development.businesses}</strong><p>kept together across {validation.folds} outer folds</p></article>
        <article className="card"><span>Posterior fits reused</span><strong>{research.development.candidateFits.toLocaleString()}</strong><p>{research.development.candidatesPerBusiness} candidates per advertiser</p></article>
        <article className="card"><span>Candidate tokens</span><strong>{research.development.totalCandidateTokens}</strong><p>{research.development.rawCandidateTokens} raw + {research.development.relativeCandidateTokens} relative</p></article>
        <article className="card"><span>Information firewall</span><strong>Truth-blind</strong><p>no loss, oracle rank, or advertiser ID enters deployment tokens</p></article>
      </section>

      <section className="card regretset-comparison">
        <div className="card-heading">
          <div><span className="eyebrow">Advertiser-grouped cross-validation</span><h2>Out-of-sample economic loss on unseen advertisers</h2></div>
          <span className="score-lab-inspect">Lower is better</span>
        </div>
        <div className="regretset-table">
          <div className="head"><span>Selection rule</span><span>Primary objective</span><span>Uncapped objective</span><span>Uncapped P95</span><span>Danger rate</span></div>
          {validation.selectors.map((selector) => (
            <article key={selector.id} className={selector.id}>
              <div><b>{selector.label}</b><small>{selector.description}</small></div>
              <span><i><em style={{ width: `${Math.min(selector.objective / validation.maximumObjective, 1) * 100}%` }} /></i><b>{selector.objective.toFixed(3)}</b></span>
              <strong>{selector.uncappedObjective.toFixed(3)}</strong>
              <strong>{selector.uncappedP95.toFixed(3)}</strong>
              <strong>{percent(selector.dangerRate)}</strong>
            </article>
          ))}
        </div>
        <div className="regretset-result-callout">
          <strong>{percent(validation.objectiveReductionVsPrediction)}</strong>
          <p><b>lower cross-validated objective than prediction-only selection.</b><br />The posterior-geometry-only ablation also performed materially worse, showing that posterior geometry was important but insufficient on its own.</p>
        </div>
        <p className="regretset-boundary">{research.claimBoundary}</p>
      </section>
    </>
  );
}

function MethodView() {
  const development = research.development;
  return (
    <>
      <section className="card regretset-flow-card">
        <div className="card-heading">
          <div><span className="eyebrow">Truth firewall</span><h2>The selector learns after the MMMs make their decisions</h2></div>
          <span className="score-lab-inspect">No hidden truth enters fitting or deployment tokens</span>
        </div>
        <div className="regretset-flow">
          <article><span>1</span><b>Simulate an advertiser</b><p>Create observed media history, confounding, evidence, and a hidden response surface.</p></article>
          <i>→</i>
          <article><span>2</span><b>Fit 48 Bayesian MMMs</b><p>Every declared specification receives the same FullRankADVI inference contract.</p></article>
          <i>→</i>
          <article><span>3</span><b>Compare candidates</b><p>Build {development.rawCandidateTokens} raw, {development.relativeCandidateTokens} relative, and {development.contextTokens} context tokens.</p></article>
          <i>→</i>
          <article className="truth"><span>4</span><b>Learn from decision loss</b><p>Budget actions are replayed through hidden truth only to create training labels.</p></article>
        </div>
      </section>

      <section className="regretset-contract-grid">
        <article className="card"><span>Candidate set</span><strong>{development.candidatesPerBusiness}</strong><p>models compared jointly, not as unrelated rows</p></article>
        <article className="card"><span>Candidate contract</span><strong>{development.totalCandidateTokens}</strong><p>truth-blind tokens per fitted candidate</p></article>
        <article className="card"><span>Economic labels</span><strong>{development.decisionScenarios}</strong><p>reduce, reallocate, grow, and economic ceiling</p></article>
        <article className="card"><span>Posterior engine</span><strong>SVI</strong><p>{development.posteriorEngine}</p></article>
      </section>

      <section className="card regretset-lineage">
        <div className="card-heading">
          <div><span className="eyebrow">Final architecture</span><h2>Raw diagnostics become meaningful inside the candidate set.</h2></div>
          <span className="score-lab-inspect">Permutation-equivariant DeepSets</span>
        </div>
        <div className="regretset-lineage-grid">
          <article><span>Observable evidence</span><b>232 raw tokens</b><p>Generalization, structure, causal stress, decision coherence, posterior geometry, and specification context.</p></article>
          <i>＋</i>
          <article className="adaptive"><span>Relative position</span><b>Rank + robust distance</b><p>Each raw token is expressed relative to the other candidates for the same advertiser.</p></article>
          <i>＝</i>
          <article className="unchanged"><span>Joint selection</span><b>One comparable risk</b><p>Three ensemble members predict mean and P90 loss; no hand-set post-prediction penalty changes the ranking.</p></article>
        </div>
      </section>

      <section className="regretset-heads">
        <article className="card"><span className="eyebrow">Training target</span><h2>log(1 + uncapped regret)</h2><p>The transform preserves ordering among severe mistakes while limiting their numerical dominance.</p></article>
        <article className="card"><span className="eyebrow">Set-wise learning</span><h2>Context changes the ranking</h2><p>A permutation-equivariant learner compares each candidate with the full alternative set for the same advertiser.</p></article>
        <article className="card risk"><span className="eyebrow">Business risk preference</span><h2>65% mean + 35% P90</h2><p>The risk mixture is a disclosed business choice—not a learned scientific constant.</p></article>
      </section>

      <section className="regretset-runtime-note">
        <i>i</i>
        <div><b>Research inference and production confirmation</b><p>RegretSet-MMM ranks the complete SVI candidate pool. After promotion, NUTS can confirm the same frozen specification and reveal approximation shifts without changing the selector&apos;s training claim.</p></div>
      </section>
    </>
  );
}

function FindingsView() {
  const attribution = research.evidenceAttribution;
  const sources = [
    ["Data", attribution.observational, "observational"],
    ["Experiment", attribution.experiment, "experiment"],
    ["Benchmark", attribution.benchmark, "benchmark"],
    ["Regularization", attribution.regularization, "regularization"],
  ] as const;
  const maxEffect = Math.max(...research.selectorInsights.pillars.map((item) => Math.abs(item.relativeLossChange)));
  return (
    <>
      <section className="card regretset-finding-hero">
        <div>
          <span className="eyebrow">Cross-fitted interpretation</span>
          <h2>Posterior geometry matters most—and is not enough by itself.</h2>
          <p>Removing posterior-geometry tokens increased held-out loss by 42.5%, while a model using only posterior geometry was 28.1% worse than the full selector. The useful signal is therefore joint: posterior behavior interpreted with prediction, structure, evidence, and specification context.</p>
        </div>
        <div><strong>6 pillars</strong><small>drop-column refit · five outer folds</small></div>
      </section>

      <section className="card selector-global-card">
        <div className="card-heading">
          <div><span className="eyebrow">Outer-fold drop-column refit</span><h2>What changed held-out economic loss?</h2></div>
          <span className="score-lab-inspect">Positive means loss rose after removal</span>
        </div>
        <p className="selector-method-copy">Each raw pillar and its candidate-relative counterparts are removed together; the selector is then relearned on outer-training advertisers and evaluated on the untouched fold.</p>
        <div className="selector-effect-axis-labels"><span>Inconclusive or replaceable</span><b>0</b><span>Loss rose after removal</span></div>
        <div className="selector-effect-list">
          {research.selectorInsights.pillars.map((pillar) => {
            const width = Math.min(Math.abs(pillar.relativeLossChange) / maxEffect, 1) * 50;
            const positiveFolds = Math.round(pillar.positiveFoldShare * 5);
            return <article key={pillar.id}>
              <div><b>{pillar.label}</b><small>{pillar.tokenCount} final tokens · {pillar.question}</small></div>
              <div className="selector-effect-track"><i className={pillar.relativeLossChange >= 0 ? "positive" : "negative"} style={{ width: `${width}%`, left: pillar.relativeLossChange >= 0 ? "50%" : `${50 - width}%` }} /></div>
              <strong className={pillar.relativeLossChange >= 0 ? "positive" : "negative"}>{pillar.relativeLossChange >= 0 ? "+" : ""}{percent(pillar.relativeLossChange)}</strong>
              <div className="selector-folds">{[0, 1, 2, 3, 4].map((fold) => <i key={fold} className={fold < positiveFolds ? "active" : ""} />)}<small>{positiveFolds}/5 folds</small></div>
            </article>;
          })}
        </div>
        <div className="selector-interpretation-callout"><i>!</i><p><b>Read nonpositive estimates as inconclusive.</b> Correlated token groups can substitute for one another; this analysis does not justify removing a scientific safety check.</p></div>
      </section>

      <section className="card regretset-attribution">
        <div className="card-heading">
          <div><span className="eyebrow">Continuous posterior evidence attribution</span><h2>Where did local ROI precision come from?</h2></div>
          <span className="score-lab-inspect">{attribution.basis}</span>
        </div>
        <div className="regretset-attribution-bar">{sources.map(([label, value, id]) => <i key={id} className={id} style={{ width: `${value * 100}%` }} title={`${label}: ${percent(value)}`} />)}</div>
        <div className="regretset-attribution-legend">{sources.map(([label, value, id]) => <span key={id} className={id}><i />{label}<b>{percent(value)}</b></span>)}</div>
        <p>{attribution.note}</p>
      </section>

      <section className="regretset-runtime-note safe">
        <i>✓</i>
        <div><b>Interpretation remains scoped</b><p>These are cross-fitted predictive associations in the declared synthetic population. They do not turn neural-network behavior into a causal claim about which validation check creates better business outcomes.</p></div>
      </section>
    </>
  );
}

export function RegretSetResearch({ view }: RegretSetResearchProps) {
  if (view === "method") return <MethodView />;
  if (view === "findings") return <FindingsView />;
  return <EvidenceView />;
}
