import research from "../data/regretset-v9-public.json";

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
            V9 compares the complete set of candidate Bayesian MMMs for an advertiser.
            It predicts which specification is least likely to lose economic value when
            its posterior is used for budget decisions—without seeing causal truth at deployment.
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
        <article className="card"><span>Internal sealed audit</span><strong>{research.internalAudit.businesses}</strong><p>new worlds · {research.internalAudit.checksPassed}/{research.internalAudit.checksTotal} checks passed</p></article>
        <article className="card"><span>External AMSS audit</span><strong>{external.businesses}</strong><p>independent worlds · {external.candidateFits.toLocaleString()} SVI fits</p></article>
        <article className="card"><span>Frozen decisions</span><strong>{external.frozenActions.toLocaleString()}</strong><p>actions scored only after AMSS truth opened</p></article>
      </section>

      <section className="card regretset-comparison">
        <div className="card-heading">
          <div><span className="eyebrow">External synthetic transport</span><h2>Lower decision risk on an independently developed simulator</h2></div>
          <span className="score-lab-inspect">Capped loss · lower is better</span>
        </div>
        <div className="regretset-table">
          <div className="head"><span>Selector</span><span>Mean risk</span><span>P90</span><span>P95</span></div>
          {external.selectors.map((selector) => (
            <article key={selector.id} className={selector.id}>
              <div><b>{selector.label}</b><small>{selector.description}</small></div>
              <span><i><em style={{ width: `${selector.mean * 100}%` }} /></i><b>{selector.mean.toFixed(3)}</b></span>
              <strong>{selector.p90.toFixed(3)}</strong>
              <strong>{selector.p95.toFixed(3)}</strong>
            </article>
          ))}
        </div>
        <div className="regretset-result-callout">
          <strong>{percent(external.meanReductionVsRandom)}</strong>
          <p><b>lower mean decision risk than uniform random valid selection.</b><br />RegretSet had lower business-level loss than the expected random policy in {external.pairedLowerLossBusinesses} of {external.pairedBusinesses} AMSS businesses.</p>
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
          <article><span>3</span><b>Build 232 observable tokens</b><p>Diagnostics, assumptions, evidence, and posterior decision geometry only.</p></article>
          <i>→</i>
          <article className="truth"><span>4</span><b>Reveal loss after action</b><p>Four frozen budget actions are replayed through truth and labeled by forgone value.</p></article>
        </div>
      </section>

      <section className="regretset-contract-grid">
        <article className="card"><span>Candidate set</span><strong>{development.candidatesPerBusiness}</strong><p>models compared jointly, not as unrelated rows</p></article>
        <article className="card"><span>Observable contract</span><strong>{development.tokensPerCandidate}</strong><p>named and reproducible tokens per candidate</p></article>
        <article className="card"><span>Economic labels</span><strong>{development.decisionScenarios}</strong><p>reduce, reallocate, grow, and economic ceiling</p></article>
        <article className="card"><span>Posterior engine</span><strong>SVI</strong><p>{development.posteriorEngine}</p></article>
      </section>

      <section className="regretset-heads">
        <article className="card"><span className="eyebrow">Distributional outputs</span><h2>Predict more than an average</h2><p>Separate heads estimate mean loss, upper-tail loss, scenario loss, and the probability of a dangerous false champion.</p></article>
        <article className="card"><span className="eyebrow">Set-wise learning</span><h2>Context changes the ranking</h2><p>A permutation-invariant DeepSets learner compares each candidate with the full alternative set for the same advertiser.</p></article>
        <article className="card risk"><span className="eyebrow">Business risk preference</span><h2>65% mean + 35% P90</h2><p>The risk mixture is a disclosed business choice—not a learned scientific constant.</p></article>
      </section>

      <section className="regretset-runtime-note">
        <i>i</i>
        <div><b>Research selector versus fast product search</b><p>RegretSet-MMM evaluates a fixed pool of SVI-fitted candidates. The current broad Agentic search remains an analytic coverage stage; the UI does not pretend that MAP ranking and the frozen V9 SVI selector are the same object.</p></div>
      </section>
    </>
  );
}

function FindingsView() {
  const interpretation = research.developmentInterpretation;
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
          <span className="eyebrow">Development-set ablation</span>
          <h2>The posterior allocation geometry carried the strongest transferable signal.</h2>
          <p>When this complete token pillar was removed and the selector relearned, held-out economic loss increased in four of five advertiser folds.</p>
        </div>
        <div><strong>+{percent(interpretation.posteriorGeometryRelativeIncrease)}</strong><small>held-out loss without posterior geometry</small></div>
      </section>

      <section className="regretset-findings-grid">
        <article className="card"><span>Tail shape & approximation shift</span><strong>+{percent(interpretation.tailShapeRelativeIncrease)}</strong><p>Skew, tail mass, and approximation movement added reusable signal.</p></article>
        <article className="card"><span>ROI location & uncertainty</span><strong>+{percent(interpretation.roiLocationRelativeIncrease)}</strong><p>Posterior centers and interval widths mattered beyond point diagnostics.</p></article>
        <article className="card"><span>Classical validation</span><strong>Still visible</strong><p>Prediction, structure, causal checks, and evidence coherence remain gates and model-criticism receipts.</p></article>
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
        <div><b>Arbitrary “data-led / prior-led” labels are retired</b><p>V9 reports continuous source influence. Source quality, conflict, conditional identification, and decision dependence remain separate so a large experiment share is never mistaken for proof that the experiment transported correctly.</p></div>
      </section>
    </>
  );
}

export function RegretSetResearch({ view }: RegretSetResearchProps) {
  if (view === "method") return <MethodView />;
  if (view === "findings") return <FindingsView />;
  return <EvidenceView />;
}
