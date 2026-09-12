#!/usr/bin/env python3
"""Build the cross-validated arXiv manuscript for RegretSet-MMM."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.svi_score_v9 import build_journal_paper as old  # noqa: E402


OUT = ROOT / "output/pdf/regretset_mmm_arxiv.pdf"
ASSETS = ROOT / "tmp/pdfs/regretset_mmm_arxiv_cross_validated_assets"
COMPARISON = ROOT / "research/regretset_rank_repair/artifacts/comprehensive-development-comparison.json"
RELATIVE = ROOT / "research/regretset_relative_features/artifacts/development-comparison.json"
IMPORTANCE = ROOT / "research/regretset_relative_features/artifacts/final-selector-importance.json"
EVIDENCE = ROOT / "research/svi_score_v9/artifacts/svi-score-v9-evidence-attribution.json"
DATASET = ROOT / ".flux-artifacts/svi-score-v11-evidence-adaptive/development-dataset.json"

old.ASSETS = ASSETS
old.EVIDENCE_ATTRIBUTION = EVIDENCE

Flowable = old.Flowable
Spacer = old.Spacer
PageBreak = old.PageBreak
HRFlowable = old.HRFlowable
P = old.P
H1 = old.H1
H2 = old.H2
H3 = old.H3
bullet = old.bullet
definition = old.definition
proof = old.proof
make_table = old.make_table
caption = old.caption
equation = old.equation
multiline_equation = old.multiline_equation
image_flow = old.image_flow
plt = old.plt
mm = old.mm
BLACK = old.BLACK

TITLE = "RegretSet-MMM: Learning to Select Marketing Mix Models by Their Downstream Economic Consequences"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_receipts() -> tuple[dict, dict, dict]:
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    relative = json.loads(RELATIVE.read_text(encoding="utf-8"))
    importance = json.loads(IMPORTANCE.read_text(encoding="utf-8"))
    required = {
        "raw-plus-relative-log1p", "context-no-gate-log1p", "relative-only-log1p",
        "pointwise-neural", "posterior-geometry-deepsets", "linear-ridge",
        "boosted-stumps", "prediction-only", "classic-pareto", "uniform-random-valid",
        "in-pool-oracle",
    }
    found = {row["selector"] for row in comparison["ranking"]}
    if not required.issubset(found):
        raise RuntimeError("Cross-validation comparison is incomplete")
    required_checks = (
        "allBusinessesCrossFitted", "pillarRegistryExhaustive",
        "allDerivedCounterpartsRemoved", "noPosteriorRefits",
    )
    if not all(importance["checks"][name] for name in required_checks):
        raise RuntimeError("Final-selector importance checks failed")
    return comparison, relative, importance


def save_research_design() -> Path:
    path = ASSETS / "figure-1-cross-validated-design.png"
    if path.exists():
        return path
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 3.15), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, title, note, fill="white"):
        ax.add_patch(plt.Rectangle((x, 1.55), 1.62, 1.28, facecolor=fill, edgecolor="#222222", lw=0.8))
        ax.text(x + 0.81, 2.35, title, ha="center", va="center", fontsize=7.2,
                fontweight="bold", family="serif")
        ax.text(x + 0.81, 1.86, note, ha="center", va="center", fontsize=6.05,
                color="#555555", family="serif", linespacing=1.12)

    items = [
        (0.05, "Advertiser world", "Observed history +\nhidden causal truth"),
        (2.03, "48 Bayesian MMMs", "FullRankADVI;\ntruth remains sealed"),
        (4.01, "Candidate evidence", "232 raw + 464\nrelative tokens"),
        (5.99, "Set-wise learner", "Jointly scores the\nunordered candidate set"),
        (7.97, "Budget actions", "Reduce, reallocate,\ngrow, ceiling"),
        (9.95, "Economic label", "Truth opened only\nafter actions are fixed"),
    ]
    for index, item in enumerate(items):
        box(*item, fill="#F2F5FA" if index == 3 else "white")
        if index:
            ax.annotate("", xy=(item[0] - 0.04, 2.19), xytext=(items[index - 1][0] + 1.66, 2.19),
                        arrowprops=dict(arrowstyle="->", lw=0.8, color="#222222"))
    ax.text(6.0, 3.72, "Training label path: hidden truth is used only after every candidate has acted",
            ha="center", fontsize=7.7, style="italic", family="serif", color="#9A3D32")
    ax.add_patch(plt.Rectangle((2.05, 0.34), 7.85, 0.62, facecolor="#F2F5FA", edgecolor="#355C9A", lw=0.7))
    ax.text(5.98, 0.65,
            "Deployment: new advertiser -> fitted candidate set -> truth-blind tokens -> frozen selector -> lowest predicted risk",
            ha="center", va="center", fontsize=6.65, family="serif", color="#355C9A")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_architecture() -> Path:
    path = ASSETS / "figure-3-final-architecture.png"
    if path.exists():
        return path
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 4.05), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, title, note, fill="white"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor="#222222", lw=0.8))
        ax.text(x + w / 2, y + h * .68, title, ha="center", va="center", fontsize=7.15,
                fontweight="bold", family="serif")
        ax.text(x + w / 2, y + h * .29, note, ha="center", va="center", fontsize=5.95,
                color="#555555", family="serif", linespacing=1.10)

    box(.05, 1.65, 1.55, 1.42, "232 raw tokens", "Diagnostics, posterior,\nevidence, assumptions")
    box(1.93, 3.50, 1.65, 1.10, "Percentile position", "Average-tie rank within\nthe valid candidate set", "#F2F5FA")
    box(1.93, 1.98, 1.65, 1.10, "Robust distance", "(value - median) / IQR\nclipped to [-8, 8]", "#F2F5FA")
    box(1.93, .46, 1.65, 1.10, "Absolute evidence", "Original 232-token\nmeasurement scale", "#F2F5FA")
    box(3.96, 1.65, 1.48, 1.42, "Candidate input", "696 candidate tokens\n+ 63 business context")
    box(5.82, 1.65, 1.30, 1.42, "Shared encoder", "759 -> 12\nSiLU")
    box(7.49, 3.34, 1.42, 1.04, "Mean pool", "h-bar across\nvalid candidates", "#F2F5FA")
    box(7.49, 1.65, 1.42, 1.42, "Equivariant join", "h, h-bar, h-h-bar,\nh times h-bar")
    box(9.28, 1.65, 1.22, 1.42, "Decoder", "48 -> 12\nSiLU")
    box(10.86, 1.36, 1.08, 2.00, "10 heads", "mean\nmedian\nP90 / P95\nCVaR90\n4 scenarios\ndanger")
    arrows = [((1.60, 2.36), (1.90, 2.36)), ((3.58, 2.36), (3.94, 2.36)),
              ((5.44, 2.36), (5.80, 2.36)), ((7.12, 2.36), (7.47, 2.36)),
              ((8.91, 2.36), (9.26, 2.36)), ((10.50, 2.36), (10.84, 2.36))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=.8))
    ax.annotate("", xy=(8.20, 3.31), xytext=(6.47, 3.10), arrowprops=dict(arrowstyle="->", lw=.75))
    ax.annotate("", xy=(8.20, 3.08), xytext=(8.20, 3.32), arrowprops=dict(arrowstyle="->", lw=.75))
    ax.text(6.0, 4.98, "Candidate order is arbitrary: pooling is invariant and per-candidate outputs are equivariant",
            ha="center", fontsize=7.8, style="italic", family="serif")
    ax.text(6.0, .12, "Deployment selects the minimum 65% predicted mean + 35% predicted P90; no added danger or disagreement penalty",
            ha="center", fontsize=7.0, family="serif", color="#355C9A")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_cv_design() -> Path:
    path = ASSETS / "figure-5-grouped-cv.png"
    if path.exists():
        return path
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.1, 3.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    colors = ["#355C9A", "#6E9E73", "#C89552", "#8A78A8", "#A85B50"]
    for fold in range(5):
        y = 4.55 - fold * .83
        ax.text(.08, y + .25, f"Outer fold {fold + 1}", fontsize=7.2, family="serif", va="center")
        for block in range(5):
            x = 1.55 + block * 1.55
            held = block == fold
            ax.add_patch(plt.Rectangle((x, y), 1.33, .5,
                                       facecolor="white" if held else colors[block],
                                       edgecolor=colors[block], lw=.8,
                                       hatch="///" if held else None))
            ax.text(x + .665, y + .25, "ASSESS" if held else "TRAIN", ha="center", va="center",
                    fontsize=5.7, color="#222222" if held else "white", family="serif", fontweight="bold")
    ax.text(4.9, .18,
            "All candidate rows and all matched evidence regimes from one advertiser remain in one fold;\nstandardization and network fitting use outer-training advertisers only.",
            ha="center", fontsize=7.15, family="serif")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_comparison() -> Path:
    path = ASSETS / "figure-6-selector-comparison.png"
    if path.exists():
        return path
    comparison, _, _ = load_receipts()
    lookup = {row["selector"]: row for row in comparison["ranking"]}
    order = [
        "raw-plus-relative-log1p", "context-no-gate-log1p", "relative-only-log1p",
        "pointwise-neural", "posterior-geometry-deepsets", "linear-ridge",
        "boosted-stumps", "prediction-only", "classic-pareto", "uniform-random-valid",
    ]
    labels = [
        "Final RegretSet-MMM", "Raw-token set learner", "Relative-token-only learner",
        "Pointwise neural", "Posterior-geometry-only set learner", "Linear ridge",
        "Boosted stumps", "Prediction-only", "Classic Pareto", "Uniform random valid",
    ]
    values = [lookup[key]["log1pObjective"] for key in order]
    fig, ax = plt.subplots(figsize=(8.9, 4.45), dpi=300)
    positions = list(range(len(order)))[::-1]
    colors = ["#355C9A"] + ["#A8ADB6"] * (len(order) - 1)
    ax.barh(positions, values, color=colors, height=.62)
    ax.set_yticks(positions, labels)
    ax.set_xlabel("cross-fitted 65% mean + 35% P90 log-regret (lower is better)", fontsize=7.5, family="serif")
    ax.set_xlim(0, 1.48)
    ax.grid(axis="x", color="#E1E1E1", lw=.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7)
    for pos, value in zip(positions, values):
        ax.text(value + .018, pos, f"{value:.3f}", va="center", fontsize=6.8, family="serif")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_importance() -> Path:
    path = ASSETS / "figure-7-final-importance.png"
    if path.exists():
        return path
    _, _, importance = load_receipts()
    rows = list(reversed(importance["pillars"]))
    labels = [row["pillar"] for row in rows]
    values = [100 * row["relativeIncrease"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.9, 3.95), dpi=300)
    positions = list(range(len(rows)))
    colors = ["#355C9A" if value > 0 else "#B9BDC5" for value in values]
    ax.barh(positions, values, color=colors, height=.58)
    ax.set_yticks(positions, labels)
    ax.axvline(0, color="#222222", lw=.75)
    ax.grid(axis="x", color="#E1E1E1", lw=.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("held-out log-regret objective change after removal and refit (%)", fontsize=7.4, family="serif")
    ax.tick_params(axis="both", labelsize=7)
    for pos, value in zip(positions, values):
        ax.text(value + (.35 if value >= 0 else -.35), pos, f"{value:+.1f}%",
                ha="left" if value >= 0 else "right", va="center", fontsize=6.8,
                family="serif", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def add_title(story: list[Flowable]) -> None:
    story.extend([
        Spacer(1, 3 * mm), P(TITLE, "title"),
        P("Truth-blind set-wise ranking with posterior evidence, candidate-relative context, and economic-regret supervision", "subtitle"),
        P("Gustavo Bramao", "author"),
        P("Independent Researcher | September 2026 | arXiv preprint", "date"),
        HRFlowable(width="100%", thickness=.5, color=BLACK, spaceBefore=1, spaceAfter=8),
        P("Abstract", "abstract_heading"),
        P(
            "Marketing mix models (MMMs) are often selected by predictive fit, calibration, or fixed diagnostic scores even though their operational purpose is budget decision support. We formulate candidate selection as learning to rank fitted Bayesian MMMs by downstream economic regret. For each of 420 synthetic advertisers, 48 FullRankADVI candidates are fitted without access to causal truth and represented by 232 deployment-observable diagnostic and posterior tokens. The final selector augments every token with its percentile position and robust distance within the advertiser's candidate set, then jointly scores the unordered set with a permutation-equivariant DeepSets network. Supervision combines loss-distribution heads with direct set-wise log-regret and regret-weighted pairwise objectives. Five-fold advertiser-grouped cross-validation yields a mean-tail log-regret objective of 0.805, compared with 1.031 for a posterior-geometry-only set learner, 1.188 for prediction-only selection, 1.243 for classic Pareto selection, and 1.377 for uniform random valid selection. Drop-column refits assess conditional information value rather than neural-weight magnitude. The results show that posterior geometry is important but insufficient: absolute diagnostics, relative candidate position, and joint set context work together. Every assessment prediction is produced by a model that did not train on that advertiser, and all deployment inputs are truth blind. Nevertheless, the evidence is synthetic and development-stage; repeated use of the same simulated population may induce method-selection optimism, and real-advertiser decision validity is not established.",
            "abstract",
        ),
        P("<b>Keywords:</b> marketing mix modeling; decision-focused learning; learning to rank; DeepSets; economic regret; Bayesian inference; posterior geometry", "keywords"),
        HRFlowable(width="100%", thickness=.5, color=BLACK, spaceBefore=0, spaceAfter=8),
    ])


def add_introduction(story: list[Flowable]) -> None:
    story.extend([
        H1("1", "Introduction"),
        P("Marketing mix models estimate how marketing activity contributes to business outcomes and translate those estimates into budget decisions. Spending is planned rather than randomized, channels co-move with latent demand, advertising is delayed and nonlinear, and external experiments or benchmarks are sparse and imperfectly transported. Several observationally plausible models can therefore fit one history yet imply materially different ROI, marginal response curves, and allocations [1-6]."),
        P("The model-selection problem is usually treated indirectly. Predictive error, information criteria, residual diagnostics, experiment calibration, or analyst-defined scores measure useful properties, but none is the opportunity cost of acting on the selected model. A model can forecast well while decomposing correlated channels incorrectly; conversely, a noisy but causally better-calibrated model may support a better allocation. The unresolved question is whether evidence available from fitted models can predict their downstream decision consequences across advertisers."),
        P("RegretSet-MMM converts that question into supervised set-wise ranking. Synthetic causal truth labels the loss of each candidate's budget actions, but remains unavailable during candidate fitting and ranking. For a new advertiser, the frozen selector sees only fitted candidate posteriors, validation diagnostics, model assumptions, external-evidence metadata, and each candidate's position relative to its alternatives."),
        image_flow(save_research_design(), 162),
        caption("Figure 1", "Training and deployment information paths. Hidden response truth creates labels only after candidate actions are fixed. Deployment reproduces the truth-blind path."),
        H2("1.1", "Research questions"),
        bullet("Can deployment-observable evidence rank candidate MMMs by downstream economic regret for an advertiser excluded from selector training?"),
        bullet("Does a set-wise nonlinear learner add value beyond prediction-only, conventional Pareto, pointwise neural, linear, and tree-based selection?"),
        bullet("Is posterior decision geometry sufficient, and which broader information pillars retain conditional value after correlated alternatives can substitute?"),
        H2("1.2", "Contributions"),
        bullet("A decision-focused formulation in which fitted Bayesian MMMs are ranked by the economic consequence of their actions rather than a hand-weighted quality score."),
        bullet("A deployment-safe representation combining 232 named raw tokens with 464 within-advertiser relative tokens, without candidate IDs, simulator metadata, or hidden truth."),
        bullet("A permutation-equivariant, multi-head set learner trained on uncapped log-regret, avoiding the information loss created when many poor candidates share a capped label."),
        bullet("A controlled architecture comparison and drop-column refit analysis that distinguish posterior geometry's importance from the stronger and unsupported claim that posterior geometry alone is sufficient."),
        bullet("A source-resolved local decomposition of ROI information among observational data, experiments, benchmarks, and regularization."),
        H2("1.3", "Claim boundary"),
        P("The empirical claim is deliberately narrow: within the declared synthetic population and 48-candidate library, truth-blind information predicts economically relevant candidate differences for advertisers held out from selector fitting. If the simulation spans mechanisms encountered in practice, the learned mapping may transfer to comparable advertisers. Cross-validation cannot establish that premise. The paper does not claim real-world causal validity, universal superiority, inclusion of the true model in the library, or exact posterior representation by variational inference."),
    ])


def add_related_work(story: list[Flowable]) -> None:
    story.extend([
        H1("2", "Related work and research gap"),
        H2("2.1", "Bayesian MMM and causal validation"),
        P("Bayesian MMMs combine controls, seasonality, carryover, saturation, and constrained media effects [1,2]. Recent open methods add time-varying coefficients, experiment calibration, and posterior optimization [4-8]. Yet nonlinear and time-varying effects can remain observationally indistinguishable, and demand-harvesting channels are vulnerable to endogeneity [3,5,6,9]. Predictive validation and information criteria assess generalization of outcomes [10,22]; they do not identify which causal decomposition will produce the best counterfactual allocation."),
        H2("2.2", "Decision-focused learning and ranking"),
        P("Decision-focused learning optimizes predictive systems for downstream decisions rather than parameter error alone [11,12]. Learning-to-rank methods exploit comparisons within a candidate list [13]. The present setting combines both: the items are complete fitted Bayesian MMMs, relevance is economic opportunity loss, and the top-ranked item supplies four budget actions. The target is therefore neither posterior accuracy alone nor generic ranking accuracy."),
        H2("2.3", "Set representation and relative context"),
        P("DeepSets characterizes permutation-invariant functions over unordered collections and motivates shared encoders with symmetric pooling [14]. Candidate identity and ordering are arbitrary, while a candidate's meaning is inherently relative: an interval width, forecast error, or calibration distance can be attractive in one opportunity set and poor in another. The final representation consequently retains both absolute evidence and two truth-blind within-set coordinates—percentile position and median/IQR distance. This is list-wise context, not a label-derived rank."),
        H2("2.4", "Posterior-informed amortized selection"),
        P("Prior-data fitted networks amortize Bayesian prediction over datasets sampled from a prior [23-27]. RegretSet-MMM instead fits explicit candidate posteriors and amortizes the downstream choice of which posterior to trust. It is thus a posterior-informed selection layer rather than an amortized posterior engine. Its validity remains conditional on the training distribution, a limitation shared by simulation-trained systems."),
        H2("2.5", "Gap"),
        P("Existing MMM workflows validate individual models, and decision-focused methods typically optimize actions from one prediction model. Missing is a reproducible method that learns, across advertiser worlds, how an unordered set of complete Bayesian MMMs should be ranked by the economic loss of their resulting decisions while respecting the information available at deployment. This paper addresses that gap and tests which architectural elements are necessary."),
    ])


def add_decision_problem(story: list[Flowable]) -> None:
    story.extend([
        H1("3", "Decision problem"),
        H2("3.1", "Candidates, actions, and loss"),
        P("Let b index advertisers and S_b the finite set of 48 candidate MMMs. Candidate c observes history D_b, declared evidence E_b, and a common business contract, then recommends action a_bcq in each of four budget scenarios q. Synthetic truth theta_b is inaccessible until all actions are fixed."),
        multiline_equation([
            r"p_c(\theta\mid D_b,E_b),\qquad a_{bcq}=\mathcal{A}_q[p_c(\theta\mid D_b,E_b)]",
            r"\theta_b^{true}\notin\{D_b,E_b,x_{bc}\}",
        ], "1"),
        P("Let V_bq(a; theta_b) be true incremental economic value under the same feasibility constraints for every candidate. Opportunity loss is the value forgone relative to the best feasible action, and candidate loss aggregates the four scenarios."),
        multiline_equation([
            r"a^*_{bq}=\arg\max_{a\in\mathcal{A}_{bq}}V_{bq}(a;\theta_b^{true})",
            r"\ell_{bcq}=V_{bq}(a^*_{bq};\theta_b^{true})-V_{bq}(a_{bcq};\theta_b^{true})\geq0",
            r"L_{bc}=\sum_{q=1}^{4}\omega_q\ell_{bcq}",
        ], "2"),
        H2("3.2", "Excess loss and primary evaluation scale"),
        P("The in-pool oracle is the eligible fitted candidate with minimum aggregate loss. Let Delta_bc be excess loss and s_b a positive advertiser-specific economic scale independent of candidate c. The uncapped normalized regret u_bc preserves severity, while g_bc = log(1+u_bc) compresses catastrophic values without making them identical. The logarithmic transform is the primary training and evaluation scale."),
        multiline_equation([
            r"c_b^*=\arg\min_{c\in S_b}L_{bc},\qquad \Delta_{bc}=L_{bc}-L_{bc_b^*}",
            r"u_{bc}=\Delta_{bc}/s_b,\qquad g_{bc}=\log(1+u_{bc})",
        ], "3"),
        H2("3.3", "Business risk preference"),
        P("For any loss scale, cohort risk combines the mean and the across-advertiser P90. The default 65/35 mixture is a declared business preference, not a learned scientific constant. Uncapped mean, P90, and P95 are reported beside the primary log-regret objective."),
        equation(r"J=0.65\,\operatorname{mean}_b(g_{b\widehat c_b})+0.35\,Q_{.90,b}(g_{b\widehat c_b})", "4"),
        P("An elementary finite-candidate inequality motivates regret-weighted oracle comparisons. Because it follows directly from the selected candidate appearing among the oracle misrankings, it is stated and proved in Appendix A rather than presented as a principal theoretical contribution."),
    ])


def add_population(story: list[Flowable]) -> None:
    story.extend([
        H1("4", "Synthetic advertiser population"),
        H2("4.1", "Why simulation is required"),
        P("Observed advertiser data do not reveal the causal value of every feasible counterfactual allocation. Simulation supplies that otherwise unavailable label. It also creates a risk: the selector may learn simulator artifacts. The population therefore varies duration, demand persistence, media planning, commercial shocks, controls, carryover, saturation, noise, evidence quality, and decision economics. The generator is separate from candidate transformation and selector code."),
        image_flow(old.save_dgp_dag(), 158),
        caption("Figure 2", "Advertiser data-generating structure. Spend is endogenous because latent demand and planning affect media and outcome."),
        H2("4.2", "Observed outcome and response"),
        multiline_equation([
            r"Y_{bt}=B_{bt}+\gamma_b^\top Z_{bt}+\sum_{j=1}^{J_b}m_{bjt}+\varepsilon_{bt}",
            r"m_{bjt}=\beta_{bj}h((w_{bj}*X_{bj})_t;\kappa_{bj},\eta_{bj})",
        ], "5"),
        P("Histories span 104-260 weeks. Carryover is geometric or Weibull; response is Hill-saturated; coefficients may vary through time; planning and latent demand can confound spend. Three archetypal roles—paid social, nonbrand search, and television—were chosen for contrasting delivery and endogeneity mechanisms. They are training archetypes, not a restriction that deployed MMMs contain only those channel names."),
        H2("4.3", "Mechanism families and evidence"),
        make_table([
            ["Family", "Primary stress", "Selection challenge"],
            ["Balanced direct-to-consumer", "mixed digital/offline signal", "reference environment"],
            ["Search demand harvesting", "search follows latent demand", "endogeneity"],
            ["Social frequency pressure", "reach/frequency saturation", "nonlinear marginal return"],
            ["Delayed television", "long, delayed carryover", "temporal support"],
            ["Correlated planning", "shared budget/promotion process", "conditional identification"],
            ["Wrong external evidence", "biased or transported anchors", "quality and conflict"],
            ["Swapped mechanics", "unexpected response families", "channel-name transfer"],
        ], widths=[45 * mm, 55 * mm, 62 * mm]),
        caption("Table 1", "Simulator families vary causal mechanisms rather than only random seeds."),
        P("Experiments are noisy measurements of ROI over declared test and outcome windows and vary in power, bias, replication, design, and transportability. Benchmarks are noisy priors with varying channel and business match. Candidate evidence use is declared before truth is opened."),
        H2("4.4", "Decision environments"),
        P("Each candidate is evaluated under budget reduction, fixed-budget reallocation, budget growth, and economic-ceiling search. The allocation engine uses 21 budget levels and 240 feasible allocations per level with channel-specific cut and growth constraints. Labels therefore depend on marginal response and substitution, not average ROI alone."),
    ])


def add_candidates(story: list[Flowable]) -> None:
    story.extend([
        H1("5", "Candidate Bayesian MMMs"),
        H2("5.1", "Common candidate library"),
        P("Every advertiser receives the same 48-candidate opportunity set: 24 structural specifications crossed with two evidence settings. Specifications vary adstock family and memory, Hill saturation, baseline Fourier terms, ridge strength, likelihood, prior family, time-varying coefficients, planning controls, and calibration route. Misspecification is deliberate; otherwise selection would be trivial."),
        make_table([
            ["Component", "Candidate variation", "Ambiguity represented"],
            ["Carryover", "geometric or Weibull; short to long", "immediate decay versus delayed peak"],
            ["Saturation", "multiple Hill shapes and half-saturation", "weak versus strong diminishing returns"],
            ["Baseline", "Fourier order, cycle, controls", "recurring demand allocation"],
            ["Coefficients", "constant or time-varying", "stable versus evolving response"],
            ["Distributions", "positive priors; Gaussian, Student-t, log-normal", "tails, skew, robustness"],
            ["Evidence", "experiment or benchmark route", "external identification and transport"],
        ], widths=[33 * mm, 59 * mm, 70 * mm]),
        caption("Table 2", "Principal candidate dimensions."),
        H2("5.2", "Full-rank variational posterior"),
        P("All 20,160 candidate fits use PyMC 6.2 FullRankADVI under a common contract. A multivariate Gaussian variational family retains covariance needed for posterior geometry. Multiple convergence and reproducibility diagnostics enter the observable representation. Full-rank ADVI is used for scale, not because it is more accurate than MCMC; it can smooth multimodality and distort tails [15,16]."),
        equation(r"\phi^*=\arg\max_\phi\{\mathbb{E}_{q_\phi(\theta)}[\log p(D,E,\theta)]-\mathbb{E}_{q_\phi(\theta)}[\log q_\phi(\theta)]\}", "6"),
        H2("5.3", "Truth firewall"),
        P("Business identity, generator family, fold, candidate identity, proposal metadata, hidden parameters, decision loss, and truth errors are prohibited inputs. Tests scan token names and verify invariance to candidate and channel permutation. The only path from truth to the learner is through labels for outer-training advertisers after actions are fixed."),
    ])


def add_representation(story: list[Flowable]) -> None:
    story.extend([
        H1("6", "Deployment-observable representation"),
        H2("6.1", "Named raw tokens"),
        P("A token is a named numerical input with a declared construction. The 232-token raw representation contains 136 validation/specification tokens and 96 posterior decision tokens. It records predictive generalization, residual structure, causal and placebo stress, ROI coherence, temporal evidence, evidence quality and conflict, declared model assumptions, posterior ROI geometry, response uncertainty, contribution stability, and cross-channel dependence."),
        make_table([
            ["Pillar", "Examples", "Question"],
            ["Predictive generalization", "rolling holdout; coverage; spend regimes", "does fit survive time and spend shifts?"],
            ["Structural adequacy", "residuals; variance; collinearity; boundaries", "what misspecification remains?"],
            ["Causal and temporal robustness", "placebos; anchors; carryover support", "are timing and causal claims defensible?"],
            ["Decision and evidence coherence", "ROI stability; source quality/conflict", "are decisions plausible and source-robust?"],
            ["Posterior decision geometry", "ROI tails; contribution CV; covariance", "where are uncertainty and substitution risk?"],
            ["Specification context", "adstock; saturation; likelihood; dynamics", "which assumptions produced the evidence?"],
        ], widths=[42 * mm, 67 * mm, 53 * mm]),
        caption("Table 3", "Exhaustive practitioner taxonomy. Pillars are interpretation groups, not fixed score weights."),
        H2("6.2", "Channel-order invariance"),
        P("Twenty-three channel posterior metrics are reduced by mean, standard deviation, minimum, and maximum, producing 92 tokens. Four summaries measure mean and maximum absolute posterior correlation among channel log-ROI and contribution draws. This representation accepts differing channel counts and is invariant to channel order."),
        H2("6.3", "Candidate-relative coordinates"),
        P("Absolute values alone ignore the opportunity set. For every raw token k, the selector also observes the candidate's average-tie percentile rank and robust median/IQR distance among valid candidates for the same advertiser and evidence regime. These 464 derived values use no loss, oracle, synthetic truth, or cross-advertiser information."),
        multiline_equation([
            r"r_{bck}=\frac{\operatorname{rank}_{S_b}(x_{bck})-1}{|S_b|-1}",
            r"z_{bck}=\operatorname{clip}_{[-8,8]}\frac{x_{bck}-\operatorname{median}_{d\in S_b}x_{bdk}}{Q_{.75}-Q_{.25}}",
            r"\widetilde x_{bc}=[x_{bc},r_{bc},z_{bc}]\in\mathbb{R}^{696}",
        ], "7"),
        P("The relative coordinates are deployment-safe because a new advertiser supplies the complete candidate set before scoring. They do not say which candidate is truly good; they say how each candidate differs from currently available alternatives. Absolute and relative evidence are both retained because identical ranks can correspond to very different absolute quality."),
        H2("6.4", "Advertiser context"),
        P("Sixty-three candidate-independent context tokens record evidence availability and coverage, explicit missingness masks, candidate-set regime, and set-level diagnostic summaries. Context is concatenated to every candidate before the shared encoder. It contains no advertiser identifier or truth-derived statistic."),
    ])


def add_architecture(story: list[Flowable]) -> None:
    story.extend([
        H1("7", "Final RegretSet-MMM architecture and objective"),
        H2("7.1", "Permutation-equivariant set learner"),
        P("Each 696-token candidate vector is concatenated with 63 business-context tokens. A shared affine-SiLU encoder maps the 759 inputs to 12 hidden units. Mean pooling over valid candidates forms an invariant advertiser representation. The decoder receives the candidate embedding, pooled mean, their difference, and elementwise product; shared decoding therefore yields outputs that permute exactly with candidate order."),
        multiline_equation([
            r"h_{bc}=\operatorname{SiLU}(W_1[\widetilde x_{bc},q_b]+a_1),\qquad \bar h_b=|S_b|^{-1}\sum_{c\in S_b}h_{bc}",
            r"v_{bc}=[h_{bc},\bar h_b,h_{bc}-\bar h_b,h_{bc}\odot\bar h_b]",
            r"d_{bc}=\operatorname{SiLU}(W_2v_{bc}+a_2),\qquad \widehat y_{bc}=W_3d_{bc}+a_3",
        ], "8"),
        image_flow(save_architecture(), 165),
        caption("Figure 3", "Final architecture. Relative tokens augment rather than replace absolute evidence; no evidence gate or post-prediction safety penalty is used."),
        H2("7.2", "Ten supervised heads"),
        P("The decoder predicts mean, median, P90, P95, and CVaR90 of the four scenario losses; four scenario-specific absolute losses; and dangerous-false-champion probability. Positive softplus increments enforce median <= P90 <= P95 <= CVaR90. With four scenarios, P90 and P95 are interpolated upper-scenario summaries, not claims of frequentist coverage for future advertisers."),
        H2("7.3", "Joint loss"),
        P("Location targets use Huber loss, upper summaries use pinball loss, danger uses binary cross-entropy, and scenario heads preserve decision-specific supervision [17-19]. The ranking terms use g_bc = log(1+u_bc), not the capped regret target. For temperature T = 0.35, the set loss charges expected log-regret under a differentiable soft selection distribution; the pair loss penalizes every candidate that can outrank the oracle in proportion to its log-regret."),
        multiline_equation([
            r"p_{bc}=\frac{\exp(-f_{bc}/T)}{\sum_{d\in S_b}\exp(-f_{bd}/T)},\qquad \mathcal{L}_{set}=\sum_b\sum_{c\in S_b}p_{bc}g_{bc}",
            r"\mathcal{L}_{pair}=\sum_b\sum_{c\neq c_b^*}g_{bc}\operatorname{softplus}(f_{bc_b^*}-f_{bc})",
            r"\mathcal{L}=\mathcal{L}_{heads}+\mathcal{L}_{set}+0.25\mathcal{L}_{pair}+10^{-3}\|\Theta\|_2^2",
        ], "9"),
        P("The head coefficients balance heterogeneous supervised gradients; they are not feature importance. Training uses Adam, batch size 32 advertiser-regime sets, gradient-norm clipping at 5, and 18 epochs. Three business-bootstrap models form the fold ensemble. At selection time f_bc is simply 65% predicted mean plus 35% predicted P90. Earlier hand-added danger and disagreement penalties are absent because they did not improve the cross-validated objective."),
        H2("7.4", "Why this architecture"),
        P("The design encodes four requirements. First, candidate order must not matter, motivating shared maps and symmetric pooling [14]. Second, the decision is comparative, motivating within-set coordinates. Third, absolute evidence must remain available because a best-in-set candidate can still be poor, motivating the joint raw-plus-relative input. Fourth, economic mistakes vary in severity, motivating uncapped log-regret rather than a binary winner or saturated cap. The benchmark ablations in Section 10 test each argument rather than relying on architectural intuition."),
    ])


def add_evidence(story: list[Flowable]) -> None:
    story.extend([
        H1("8", "Source-resolved posterior evidence"),
        H2("8.1", "Local information decomposition"),
        P("A channel posterior may be informed by observational data D, experiments E, benchmarks B, or regularization R. At a local Laplace approximation, posterior curvature is decomposed into positive-semidefinite source components. For ROI gradient a_j and posterior covariance Sigma, source contributions propagate through the common covariance and normalize to continuous shares."),
        multiline_equation([
            r"H_{post}=H_D+H_E+H_B+H_R,\qquad \Sigma=H_{post}^{-1}",
            r"v_{s,j}=a_j^\top\Sigma H_s\Sigma a_j,\qquad \alpha_{s,j}=v_{s,j}/\sum_r v_{r,j}",
        ], "10"),
        image_flow(old.save_evidence_attribution(), 156),
        caption("Figure 4", "Local source attribution for channel posteriors from cross-fitted selected development candidates. Shares measure local influence, not source validity."),
        P("Across 1,260 channel posteriors, equal-channel mean attribution was 42.1% observational data, 36.9% experiments, 7.0% benchmarks, and 14.0% regularization. Spend-weighted shares were similar. Conditional observational identification was much lower—8.5% on average—because correlated design columns can contribute joint precision without separately identifying a channel."),
        H2("8.2", "Influence is not validity"),
        P("A high experiment share means the ROI posterior is locally sensitive to experiment information; it does not prove independence, power, or transport. RegretSet-MMM therefore keeps influence separate from quality, standardized source conflict, conditional identification, and the decision change under source deletion. If two transformed media columns are identical, the likelihood depends on their coefficients only through the sum, so separate effects require external information or restrictions."),
    ])


def add_validation(story: list[Flowable]) -> None:
    story.extend([
        H1("9", "Leakage-safe cross-validated evaluation"),
        H2("9.1", "Unit of generalization"),
        P("The development corpus contains 420 advertiser worlds and 20,160 fitted candidates. The unit of generalization is the advertiser, not the candidate row. Five outer folds assign complete advertisers: all 48 candidates and all matched evidence-regime sets for one advertiser stay together. For each assessment advertiser, feature standardization, network parameters, and bootstrap ensemble are fitted using only the other advertisers."),
        image_flow(save_cv_design(), 155),
        caption("Figure 5", "Advertiser-grouped cross-fitting. Every reported assessment prediction is out of fold at the advertiser level."),
        H2("9.2", "Information-time protocol"),
        make_table([
            ["Stage", "Permitted information", "Prohibited information"],
            ["Candidate fit", "observed history, controls, declared evidence, specification", "synthetic response truth, economic loss"],
            ["Token construction", "diagnostics, posterior draws, evidence metadata, candidate set", "business/family/fold/candidate IDs, oracle, truth errors"],
            ["Selector training", "outer-training tokens and post-action labels", "outer-assessment labels"],
            ["Assessment selection", "outer-assessment truth-blind tokens; fitted fold model", "assessment truth and losses"],
            ["Evaluation", "selected action joined to hidden truth", "no model update after truth reveal"],
        ], widths=[34 * mm, 66 * mm, 62 * mm]),
        caption("Table 4", "Information available at each stage."),
        H2("9.3", "Matched evidence regimes"),
        P("Each advertiser contributes full, experiments-only, and benchmark-gap-fill candidate sets during training. They share observed history and hidden truth and remain in one outer fold. This expands evidence environments without allowing one advertiser to appear in both training and assessment. Primary assessment uses the complete 48-candidate set."),
        H2("9.4", "Endpoints and comparators"),
        P("The primary endpoint is the 65/35 mean-tail objective on log1p uncapped normalized excess loss. Secondary endpoints report uncapped normalized mean, P90, P95, dangerous-false-champion rate, and oracle recall. Comparators isolate prediction-only selection, classic Pareto compromise, pointwise and set-wise learning, linear and tree models, relative-only evidence, and posterior-geometry-only evidence. Uniform random valid and the truth-informed in-pool oracle provide lower-information and unattainable reference points."),
        H2("9.5", "Inferential status"),
        P("These are developmental cross-validated estimates. They prevent direct record leakage and ensure each assessment advertiser is unseen by the fitted selector, but the same 420-advertiser corpus informed repeated methodological choices. Confidence intervals for selected comparisons are therefore descriptive and may understate architecture-selection uncertainty [20]. No confirmatory evaluation is reported in this manuscript."),
    ])


def add_results(story: list[Flowable]) -> None:
    comparison, relative, importance = load_receipts()
    lookup = {row["selector"]: row for row in comparison["ranking"]}
    winner = lookup["raw-plus-relative-log1p"]
    rel_pair = relative["pairedComparisonsVersusRawLogRegret"]["raw-plus-relative-log1p"]["objectives"]
    order = [
        ("raw-plus-relative-log1p", "RegretSet-MMM (raw + relative)"),
        ("context-no-gate-log1p", "Raw-token set learner"),
        ("relative-only-log1p", "Relative-token-only set learner"),
        ("pointwise-neural", "Pointwise neural learner"),
        ("posterior-geometry-deepsets", "Posterior-geometry-only set learner"),
        ("linear-ridge", "Linear ridge"),
        ("boosted-stumps", "Boosted stumps"),
        ("prediction-only", "Prediction-only"),
        ("classic-pareto", "Classic Pareto"),
        ("uniform-random-valid", "Uniform random valid"),
        ("in-pool-oracle", "In-pool oracle"),
    ]
    story.extend([
        H1("10", "Results"),
        H2("10.1", "Primary cross-fitted comparison"),
        P(f"RegretSet-MMM achieved a log-regret objective of {winner['log1pObjective']:.3f}: mean {comparison['metrics']['raw-plus-relative-log1p']['log1p']['mean']:.3f}, P90 {comparison['metrics']['raw-plus-relative-log1p']['log1p']['p90']:.3f}, and P95 {comparison['metrics']['raw-plus-relative-log1p']['log1p']['p95']:.3f}. On the uncapped normalized scale, the corresponding values were mean {winner['uncappedMean']:.3f}, P90 {winner['uncappedP90']:.3f}, and P95 {winner['uncappedP95']:.3f}. The dangerous-false-champion rate was {100 * winner['dangerousRate']:.1f}% and exact in-pool-oracle recall was {100 * winner['oracleRecall']:.1f}%."),
        image_flow(save_comparison(), 164),
        caption("Figure 6", "Advertiser-grouped cross-fitted comparison under the primary log-regret objective. The in-pool oracle equals zero and is omitted from the plot."),
        make_table([
            ["Selector", "Log objective", "Uncapped mean", "Uncapped P90", "Uncapped P95", "Oracle recall"],
            *[[label, f"{lookup[key]['log1pObjective']:.3f}", f"{lookup[key]['uncappedMean']:.3f}",
               f"{lookup[key]['uncappedP90']:.3f}", f"{lookup[key]['uncappedP95']:.3f}",
               f"{100 * lookup[key]['oracleRecall']:.1f}%"] for key, label in order],
        ], widths=[49 * mm, 23 * mm, 23 * mm, 23 * mm, 23 * mm, 23 * mm]),
        caption("Table 5", "Cross-fitted selector performance for 420 advertisers. Lower loss is better; the in-pool oracle is truth informed and not deployable."),
        P("Relative to prediction-only selection, classic Pareto, and uniform random valid, the final objective was lower by 32.3%, 35.3%, and 41.6%, respectively. These comparisons show economic signal beyond forecast fit and conventional multi-diagnostic compromise within this population; they do not establish universal superiority."),
        H2("10.2", "Why raw and relative evidence are complementary"),
        P(f"Adding candidate-relative coordinates to the raw set learner reduced the log objective from {lookup['context-no-gate-log1p']['log1pObjective']:.3f} to {winner['log1pObjective']:.3f}, a 3.9% point-estimate improvement. The paired difference was {rel_pair['log1p']['candidateMinusFrozenV11']:.3f}, with a family-stratified 95% bootstrap interval [{rel_pair['log1p']['twoSided95Interval'][0]:.3f}, {rel_pair['log1p']['twoSided95Interval'][1]:.3f}]. The interval includes zero, so the incremental benefit is promising rather than conclusive. Relative tokens alone scored {lookup['relative-only-log1p']['log1pObjective']:.3f}; ranks without absolute quality are insufficient."),
        H2("10.3", "Why posterior geometry matters but is not sufficient"),
        P(f"A DeepSets learner restricted to posterior decision geometry scored {lookup['posterior-geometry-deepsets']['log1pObjective']:.3f}, 28.1% above the final model. The pointwise neural model scored {lookup['pointwise-neural']['log1pObjective']:.3f}, and linear ridge and boosted stumps scored {lookup['linear-ridge']['log1pObjective']:.3f} and {lookup['boosted-stumps']['log1pObjective']:.3f}. Thus neither posterior geometry alone, nonlinearity without set context, nor simple regression reproduced the final result. This establishes that posterior geometry is not sufficient; it does not establish that every nonposterior token is individually necessary. The drop-column refits below address conditional pillar necessity in the presence of correlated substitutes."),
        H2("10.4", "Conditional information value"),
        P("Raw network coefficients are not stable scientific importance measures: nonlinear encoders distribute signal across correlated inputs. Following concerns about permutation extrapolation and model reliance [28,29], the primary interpretation removes an entire practitioner pillar—including raw values and both relative counterparts—then refits the identical learner on outer-training advertisers. A positive change means the remaining pillars could not replace the removed information at this sample size; it is not a causal effect of the diagnostic."),
        image_flow(save_importance(), 158),
        caption("Figure 7", "Final-selector drop-column refits. Bars are changes in held-out log-regret after removal and relearning; they are conditional algorithmic information values, not hand-assigned weights."),
        make_table([
            ["Pillar", "Raw / final tokens", "Objective without pillar", "Change", "Adverse folds"],
            *[[row["pillar"], f"{row['rawTokenCount']} / {row['finalTokenCount']}", f"{row['dropColumnObjective']:.3f}",
               f"{100 * row['relativeIncrease']:+.1f}%", f"{100 * row['positiveFoldShare']:.0f}%"]
              for row in importance["pillars"]],
        ], widths=[52 * mm, 30 * mm, 32 * mm, 23 * mm, 25 * mm]),
        caption("Table 6", "Conditional information value for the final raw-plus-relative selector. Final-token counts include raw, percentile-rank, and robust-z versions."),
    ])


def add_discussion(story: list[Flowable]) -> None:
    _, _, importance = load_receipts()
    positive = [row for row in importance["pillars"] if row["relativeIncrease"] > 0]
    negative = [row for row in importance["pillars"] if row["relativeIncrease"] <= 0]
    positive_text = ", ".join(f"{row['pillar']} ({100 * row['relativeIncrease']:+.1f}%)" for row in positive) or "none"
    negative_text = ", ".join(f"{row['pillar']} ({100 * row['relativeIncrease']:+.1f}%)" for row in negative) or "none"
    story.extend([
        H1("11", "Discussion and implications"),
        H2("11.1", "What was learned"),
        P("The principal finding is not that one diagnostic dominates. Economic selection improved when the learner saw each candidate as an element of an opportunity set, retained absolute quality, and optimized a severity-preserving regret target. Exact oracle recovery remained low, yet the selected candidates' mean-tail loss was materially lower than conventional baselines. For budget decisions, avoiding costly false champions can matter more than reproducing the exact oracle identity."),
        P(f"In final-architecture drop-column refits, pillars with positive point estimates were {positive_text}. The remaining point estimates were {negative_text}. Because the representation is correlated and the learner is refitted after removal, nonpositive estimates are inconclusive about the underlying diagnostics: they neither establish absence of information nor justify removing a scientific check."),
        H2("11.2", "Posterior geometry in context"),
        P("Posterior geometry summarizes what a decision maker actually faces: ROI location and width, implausible or near-zero mass, asymmetric tails, contribution stability, response uncertainty, and cross-channel covariance. Its importance is consistent with Bayesian decision theory because allocations depend on joint response uncertainty, not only point estimates. But the posterior-geometry-only failure shows that uncertainty cannot interpret itself. Predictive behavior, residual structure, temporal challenges, evidence quality, and specification context help distinguish a legitimately broad posterior from an unstable or misspecified one."),
        H2("11.3", "Managerial meaning"),
        P("For an advertiser, the selector does not import simulated ROI coefficients. It fits the advertiser's own candidate MMMs, computes their posteriors and diagnostics, asks how each candidate compares with its current alternatives, and applies a cross-advertiser mapping learned from simulated decisions. The output is a ranking under a declared risk preference. Managers should treat it as structured evidence for promotion or review, not as an automatic certification of causality."),
        H2("11.4", "Methodological implication"),
        P("Prediction-only and classic Pareto selection left substantial economic value on the table in this simulation because their objectives are proxies for the action loss. The set architecture also outperformed pointwise learning, supporting the view that candidate quality is context dependent. In ranking language, an MMM is not simply good or bad; it is more or less defensible relative to the alternatives and decision contract available for the same advertiser."),
    ])


def add_limitations(story: list[Flowable]) -> None:
    story.extend([
        H1("12", "Limitations, reproducibility, and future research"),
        H2("12.1", "Synthetic validity"),
        P("The strongest limitation is population validity. Ground truth makes economic labels possible, but the selector can only learn mechanisms present in the generator. Three channel archetypes, declared response families, evidence processes, and allocation constraints do not span all industries, platforms, creative dynamics, auctions, retail media, brand formation, or organizational behavior. Real-world transfer is plausible only to the extent that these mechanisms and token relationships reconstruct the target setting."),
        H2("12.2", "Development-selection optimism"),
        P("Advertiser-grouped cross-fitting prevents a candidate or matched regime from leaking into the training fold of its own assessment advertiser. It does not erase researcher adaptation. Architectures, targets, and representations were developed using summaries from the same 420-advertiser corpus. Reported differences are therefore model-development evidence and may be optimistic [20]. A future study should freeze this specification and evaluate it on real-world advertisers for which credible incrementality experiments identify marginal ROI and response curves over comparable decision horizons."),
        H2("12.3", "Posterior and candidate-library dependence"),
        P("FullRankADVI may understate tails or miss multimodality. The in-pool oracle is limited to the 48 fitted candidates, so zero excess loss need not equal structural truth or globally optimal action. Results may change with candidate diversity, posterior engine, decision grid, economic scale, or the 65/35 preference. The framework can accept NUTS draws, but equivalence is not assumed."),
        H2("12.4", "Interpretation limits"),
        P("Drop-column refits estimate conditional algorithmic reliance, not causal importance or a universal validation weight. Correlated groups can substitute; negative changes can reflect sample size, optimization variance, or regularization. Source attribution is local curvature sensitivity and must remain separate from evidence validity."),
        H2("12.5", "Reproducibility"),
        P("Code, frozen token registries, simulator contracts, candidate fingerprints, cross-fitted comparison receipts, and artifact hashes are available in the public repository. The final comparison and importance analyses reuse 20,160 stored posterior fits. Reproduction should verify hashes before rebuilding tables and figures."),
        bullet("Repository: https://github.com/gustavobramao/fluxmmm"),
        H2("12.6", "Priority extensions"),
        bullet("Pre-registered evaluations on real advertisers with credible experiments that identify marginal ROI and response curves over decision-relevant horizons."),
        bullet("Prospective advertiser studies in which later experiments and realized outcomes can test whether the frozen ranking improves budget decisions."),
        bullet("Hierarchical or domain-adaptive selectors that quantify distance from the simulated training population."),
        bullet("Inference-engine comparisons using identical token contracts from FullRankADVI and NUTS."),
        bullet("Candidate libraries with richer brand, retail-media, offline, and cross-channel dynamics."),
    ])


def add_conclusion(story: list[Flowable]) -> None:
    story.extend([
        H1("13", "Conclusion"),
        P("RegretSet-MMM reframes MMM validation as a decision-focused ranking problem. It learns from the economic consequences of candidate actions while preserving a strict separation between truth-derived training labels and deployment-observable inputs. The final architecture combines absolute diagnostics and posterior summaries, within-advertiser relative coordinates, and permutation-equivariant set context. In advertiser-grouped cross-validation it produced lower mean-tail log-regret than prediction-only, classic Pareto, pointwise neural, posterior-only, linear, tree, and random-valid selection."),
        P("The result is informative but conditional. It shows that posterior and validation evidence can predict economic differences among fitted models in a heterogeneous synthetic population; it does not prove real-advertiser causal validity. The most defensible next step is a frozen, independent test of the present specification in real-world advertiser settings with credible experimental evidence on marginal ROI and response curves. The paper supplies the method, information-time contract, comparative evidence, and reproducibility artifacts needed for that test."),
    ])


REFERENCES = [
    "[1] Jin, Y., Wang, Y., Sun, Y., Chan, D., and Koehler, J. (2017). Bayesian methods for media mix modeling with carryover and shape effects. Google Research.",
    "[2] Chan, D. and Perry, M. (2017). Challenges and opportunities in media mix modeling. Google Research.",
    "[3] Chen, A. et al. (2018). Bias correction for paid search in media mix modeling. arXiv:1807.03292.",
    "[4] Ng, E., Wang, Z., and Dai, A. (2021). Bayesian time varying coefficient model with applications to marketing mix modeling. arXiv:2106.03322.",
    "[5] Zhang, Y. et al. (2024). Media mix model calibration with Bayesian priors. Google Research.",
    "[6] Dew, R., Padilla, N., and Shchetkina, A. (2024). Your MMM is broken: identification of nonlinear and time-varying effects in marketing mix models. arXiv:2408.07678.",
    "[7] Runge, J., Skokan, I., Zhou, G., and Pauwels, K. (2024). Packaging up media mix modeling: an introduction to Robyn's open-source approach. arXiv:2403.14674.",
    "[8] Google Meridian Team (2026). Meridian marketing mix modeling. Software and methodological documentation.",
    "[9] Gordon, B. R. et al. (2019). A comparison of approaches to advertising measurement: evidence from big field experiments at Facebook. Marketing Science, 38(2), 193-225.",
    "[10] Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and widely applicable information criterion in singular learning theory. JMLR, 11, 3571-3594.",
    "[11] Elmachtoub, A. N. and Grigas, P. (2022). Smart predict, then optimize. Management Science, 68(1), 9-26.",
    "[12] Wilder, B., Dilkina, B., and Tambe, M. (2019). Melding the data-decisions pipeline. AAAI, 33, 1658-1665.",
    "[13] Burges, C. J. C. et al. (2005). Learning to rank using gradient descent. ICML, 89-96.",
    "[14] Zaheer, M. et al. (2017). Deep Sets. Advances in Neural Information Processing Systems, 30.",
    "[15] Kucukelbir, A. et al. (2017). Automatic differentiation variational inference. JMLR, 18(14), 1-45.",
    "[16] Blei, D. M., Kucukelbir, A., and McAuliffe, D. M. (2017). Variational inference: a review for statisticians. JASA, 112(518), 859-877.",
    "[17] Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica, 46(1), 33-50.",
    "[18] Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional value-at-risk. Journal of Risk, 2, 21-42.",
    "[19] Huber, P. J. (1964). Robust estimation of a location parameter. Annals of Mathematical Statistics, 35(1), 73-101.",
    "[20] Cawley, G. C. and Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. JMLR, 11, 2079-2107.",
    "[21] Lewis, R. A. and Rao, J. M. (2015). The unfavorable economics of measuring the returns to advertising. QJE, 130(4), 1941-1973.",
    "[22] Vehtari, A., Gelman, A., and Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. Statistics and Computing, 27, 1413-1432.",
    "[23] Muller, S. et al. (2022). Transformers can do Bayesian inference. ICLR.",
    "[24] Nagler, T. (2023). Statistical foundations of prior-data fitted networks. ICML, 25660-25676.",
    "[25] Hollmann, N. et al. (2025). Accurate predictions on small data with a tabular foundation model. Nature, 637, 319-326.",
    "[26] Ma, Y. et al. (2026). Foundation models for causal inference via prior-data fitted networks. ICLR.",
    "[27] Melnychuk, V. et al. (2026). Frequentist consistency of prior-data fitted networks for causal inference. arXiv:2603.12037.",
    "[28] Fisher, A., Rudin, C., and Dominici, F. (2019). All models are wrong, but many are useful: learning a variable's importance by studying an entire class of prediction models simultaneously. JMLR, 20(177), 1-81.",
    "[29] Hooker, G., Mentch, L., and Zhou, S. (2021). Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance. Statistics and Computing, 31, 82.",
]


def add_references(story: list[Flowable]) -> None:
    story.append(H1("", "References"))
    for ref in REFERENCES:
        story.append(P(ref, "reference"))


def add_appendix(story: list[Flowable]) -> None:
    comparison, _, importance = load_receipts()
    story.extend([
        PageBreak(), H1("A", "Appendix: elementary finite-candidate bound"),
        P("Let w_bc be any nonnegative regret weight satisfying w_bc* = 0, and let the selector choose the candidate with minimum predicted risk f_bc. The implementation uses w_bc = log(1 + Delta_bc/s_b)."),
        definition("Proposition A.1", "For any nonempty finite candidate set with deterministic tie breaking,"),
        equation(r"w_{b\widehat c_b}\leq\sum_{c\neq c_b^*}w_{bc}\,\mathrm{I}[f_{bc}\leq f_{bc_b^*}]", "A.1"),
        proof("If the selector chooses the oracle, the left side is zero. Otherwise the selected candidate is a non-oracle candidate with predicted risk no greater than the oracle's. Its term therefore appears on the right with weight equal to the left side, and all other terms are nonnegative."),
        P("Because softplus(u) >= log 2 for u >= 0, each indicator has the differentiable upper bound softplus(f_bc* - f_bc)/log 2. The implemented pairwise term omits the constant 1/log 2 because it only rescales the loss. This elementary observation motivates severity-weighted comparisons; it does not prove out-of-population generalization or candidate-library adequacy."),
        H1("B", "Appendix: final research contract"),
        make_table([
            ["Object", "Value"],
            ["Advertisers / candidates", "420 / 20,160"],
            ["Outer grouping", "5 advertiser-level folds"],
            ["Posterior", "PyMC 6.2 FullRankADVI"],
            ["Raw / relative candidate tokens", "232 / 464"],
            ["Business context", "63 tokens with explicit missingness"],
            ["Encoder / decoder", "12 / 12 SiLU units"],
            ["Outputs", "10 loss-distribution, scenario, and danger heads"],
            ["Training regimes", "full; experiments-only; benchmark gap-fill"],
            ["Primary target", "log1p uncapped normalized excess economic loss"],
            ["Set temperature / pair weight", "0.35 / 0.25"],
            ["Deployment risk", "65% predicted mean + 35% predicted P90"],
            ["Post-prediction penalties", "none"],
            ["Fold ensemble", "3 business-bootstrap fits"],
        ], widths=[66 * mm, 96 * mm]),
        caption("Table B.1", "Final selector and evaluation contract."),
        H1("C", "Appendix: reproducibility receipt"),
        make_table([
            ["Artifact", "SHA-256"],
            ["Development dataset", sha256(DATASET)],
            ["Comprehensive comparison", sha256(COMPARISON)],
            ["Relative-feature analysis", sha256(RELATIVE)],
            ["Final-selector importance", sha256(IMPORTANCE)],
        ], widths=[55 * mm, 107 * mm]),
        caption("Table C.1", "Machine-verifiable artifacts used by this manuscript."),
        P(f"All {importance['data']['businesses']} assessment advertisers are cross-fitted, and the final importance analysis reports no posterior refits. The comparison artifact status is '{comparison['status']}'."),
    ])


def build() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    old.TITLE = TITLE
    story: list[Flowable] = []
    add_title(story)
    add_introduction(story)
    add_related_work(story)
    add_decision_problem(story)
    add_population(story)
    add_candidates(story)
    add_representation(story)
    add_architecture(story)
    add_evidence(story)
    add_validation(story)
    add_results(story)
    add_discussion(story)
    add_limitations(story)
    add_conclusion(story)
    add_references(story)
    add_appendix(story)
    old.RegretSetDoc(str(OUT)).build(story)
    print(OUT)


if __name__ == "__main__":
    build()
