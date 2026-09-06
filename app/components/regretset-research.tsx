import research from "../data/regretset-v11-public.json";

type RegretSetResearchProps = {
  view: "evidence" | "method" | "findings";
};

function percent(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function EvidenceView() {
  const external = research.externalAudit;
  return (
    <>
      <section className="card regretset-hero">
        <div>
          <span className="eyebrow">Frozen research method · external synthetic audit complete</span>
          <h2>RegretSet-MMM selects models by their downstream economic consequences.</h2>
          <p>
            V11 compares the complete candidate set and adapts how it reads validation
            evidence to the advertiser&apos;s evidence environment. It predicts which posterior
            is least likely to lose economic value in budget decisions—without seeing causal
            truth at deployment.
          </p>
        </div>
        <div className="regretset-release">
          <span>{research.researchVersion}</span>
          <strong>{research.method}</strong>
          <small>{research.status}</small>
        </div>
      </section>

      <section className="regretset-metrics" aria-label="RegretSet research scale">
        <article className="card"><span>Development</span><strong>{research.development.businesses}</strong><p>advertiser worlds · {research.development.candidateFits.toLocaleString()} SVI fits</p></article>
        <article className="card"><span>Evidence regimes</span><strong>{research.development.evidenceRegimes}</strong><p>matched views per development advertiser</p></article>
        <article className="card"><span>External AMSS audit</span><strong>{external.businesses}</strong><p>independent worlds · {external.candidateFits.toLocaleString()} SVI fits</p></article>
        <article className="card"><span>Frozen decisions</span><strong>{external.frozenActions.toLocaleString()}</strong><p>actions scored only after AMSS truth opened</p></article>
      </section>

      <section className="card regretset-comparison">
        <div className="card-heading">
          <div><span className="eyebrow">External synthetic transport</span><h2>Lower decision risk on an independently developed simulator</h2></div>
          <span className="score-lab-inspect">Capped loss · lower is better</span>
        </div>
        <div className="regretset-table">
          <div className="head"><span>Selection rule</span><span>Mean risk</span><span>P90</span><span>P95</span><span>Objective</span></div>
          {external.selectors.map((selector) => (
            <article key={selector.id} className={selector.id}>
              <div><b>{selector.label}</b><small>{selector.description}</small></div>
              <span><i><em style={{ width: `${selector.mean * 100}%` }} /></i><b>{selector.mean.toFixed(3)}</b></span>
              <strong>{selector.p90.toFixed(3)}</strong>
              <strong>{selector.p95.toFixed(3)}</strong>
              <strong>{selector.objective.toFixed(3)}</strong>
            </article>
          ))}
        </div>
        <div className="regretset-result-callout">
          <strong>{percent(external.objectiveReductionVsPrediction)}</strong>
          <p><b>lower mean-tail objective than prediction-only.</b><br />The fresh audit also reduced the objective by {percent(external.objectiveReductionVsV9)} versus original V9 and passed all {external.checksTotal} predeclared confirmatory checks.</p>
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
          <span className="score-lab-inspect">No hidden truth enters fitting or tokens</span>
        </div>
        <div className="regretset-flow">
          <article><span>1</span><b>Simulate an advertiser</b><p>Create observed media history, confounding, evidence, and a sealed response surface.</p></article>
          <i>→</i>
          <article><span>2</span><b>Fit 48 Bayesian MMMs</b><p>Every declared specification receives the same FullRankADVI inference contract.</p></article>
          <i>→</i>
          <article><span>3</span><b>Build observable context</b><p>{development.candidateTokens} candidate tokens plus {development.contextTokens} business-level evidence tokens.</p></article>
          <i>→</i>
          <article className="truth"><span>4</span><b>Reveal loss after action</b><p>Four frozen budget actions are replayed through truth and labeled by forgone value.</p></article>
        </div>
      </section>

      <section className="regretset-contract-grid">
        <article className="card"><span>Candidate set</span><strong>{development.candidatesPerBusiness}</strong><p>models compared jointly, not as unrelated rows</p></article>
        <article className="card"><span>Candidate contract</span><strong>{development.candidateTokens}</strong><p>named, truth-blind tokens per fitted candidate</p></article>
        <article className="card"><span>Economic labels</span><strong>{development.decisionScenarios}</strong><p>reduce, reallocate, grow, and economic ceiling</p></article>
        <article className="card"><span>Posterior engine</span><strong>SVI</strong><p>{development.posteriorEngine}</p></article>
      </section>

      <section className="card regretset-lineage">
        <div className="card-heading">
          <div><span className="eyebrow">Method evolution</span><h2>V11 adapts to evidence without discarding V9</h2></div>
          <span className="score-lab-inspect">Joint path + evidence-conditioned residual path</span>
        </div>
        <div className="regretset-lineage-grid">
          <article><span>Original V9</span><b>One joint set encoder</b><p>{research.lineage.v9}</p></article>
          <i>＋</i>
          <article className="adaptive"><span>Adaptive V11</span><b>Context-gated experts</b><p>{research.lineage.v11}</p></article>
          <i>＝</i>
          <article className="unchanged"><span>Scientific contract</span><b>Targets stay fixed</b><p>{research.lineage.unchanged}</p></article>
        </div>
      </section>

      <section className="regretset-heads">
        <article className="card"><span className="eyebrow">Distributional outputs</span><h2>Predict more than an average</h2><p>Separate heads estimate mean loss, upper-tail loss, scenario loss, and the probability of a dangerous false champion.</p></article>
        <article className="card"><span className="eyebrow">Set-wise learning</span><h2>Context changes the ranking</h2><p>A permutation-invariant DeepSets learner compares each candidate with the full alternative set for the same advertiser.</p></article>
        <article className="card risk"><span className="eyebrow">Business risk preference</span><h2>65% mean + 35% P90</h2><p>The risk mixture is a disclosed business choice—not a learned scientific constant.</p></article>
      </section>

      <section className="regretset-runtime-note">
        <i>i</i>
        <div><b>Research selector versus fast product search</b><p>Adaptive RegretSet evaluates a complete pool of SVI-fitted candidates. Broad Agentic search remains the responsive analytic coverage stage; MAP screening is never relabeled as the frozen V11 selector.</p></div>
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
  return (
    <>
      <section className="card regretset-finding-hero">
        <div>
          <span className="eyebrow">Evidence-adaptive routing</span>
          <h2>The selector changes emphasis with the available evidence.</h2>
          <p>Four residual experts complement the joint all-token pathway. Their softmax shares describe routing—not universal feature importance or source validity.</p>
        </div>
        <div><strong>{research.development.contextTokens}</strong><small>candidate-independent evidence-context tokens</small></div>
      </section>

      <section className="regretset-findings-grid adaptive-experts">
        {research.adaptiveExperts.map((expert) => (
          <article className={`card ${expert.id}`} key={expert.id}><span>{expert.label}</span><strong>{percent(expert.weight)}</strong><p>{expert.description}</p></article>
        ))}
      </section>

      <section className="card regretset-attribution">
        <div className="card-heading">
          <div><span className="eyebrow">Continuous evidence attribution</span><h2>Where did local ROI precision come from?</h2></div>
          <span className="score-lab-inspect">{attribution.basis}</span>
        </div>
        <div className="regretset-attribution-bar" aria-label="Spend-weighted evidence attribution">
          {sources.map(([label, value, id]) => <i key={id} className={id} style={{ width: `${value * 100}%` }} title={`${label}: ${percent(value)}`} />)}
        </div>
        <div className="regretset-attribution-legend">
          {sources.map(([label, value, id]) => <span key={id} className={id}><i />{label}<b>{percent(value)}</b></span>)}
        </div>
        <p>{attribution.note} Observational separation after conditioning on correlated regressors averaged {percent(attribution.conditionalDataShareMean)}.</p>
      </section>

      <section className="regretset-runtime-note safe">
        <i>✓</i>
        <div><b>Continuous attribution replaces arbitrary labels</b><p>Flux reports source influence continuously. Influence, source quality, conflict, conditional identification, and decision dependence remain separate quantities.</p></div>
      </section>
    </>
  );
}

export function RegretSetResearch({ view }: RegretSetResearchProps) {
  if (view === "method") return <MethodView />;
  if (view === "findings") return <FindingsView />;
  return <EvidenceView />;
}
