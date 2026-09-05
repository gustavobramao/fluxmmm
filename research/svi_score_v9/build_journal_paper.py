#!/usr/bin/env python3
"""Build the RegretSet-MMM journal working paper.

This manuscript describes the frozen method and reports the completed,
one-time confirmatory synthetic audit without presenting internal release
history as a scientific contribution.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.svi_score_v5d_svi import build_journal_paper as base  # noqa: E402


OUT = ROOT / "output" / "pdf" / "regretset_mmm_working_paper.pdf"
ASSETS = ROOT / "tmp" / "pdfs" / "regretset_mmm_journal_assets"
base.ASSETS = ASSETS
INTERPRETABILITY = ROOT / "research" / "svi_score_v9" / "artifacts" / "svi-score-v9-interpretability.json"
POSTERIOR_GEOMETRY = ROOT / "research" / "svi_score_v9" / "artifacts" / "svi-score-v9-posterior-geometry.json"
EVIDENCE_ATTRIBUTION = ROOT / "research" / "svi_score_v9" / "artifacts" / "svi-score-v9-evidence-attribution.json"
AUDIT_RESULT = ROOT / "research" / "svi_score_v9" / "artifacts" / "svi-score-v9-sealed-audit.json"
AUDIT_RECOVERY = ROOT / ".flux-artifacts" / "svi-score-v9" / "audit" / "sealed-audit-dataset-recovery-receipt.json"
AMSS_RESULT = ROOT / ".flux-artifacts" / "svi-score-v10-amss" / "external-audit" / "result.json"
AMSS_ACCEPTANCE = ROOT / ".flux-artifacts" / "svi-score-v10-amss" / "external-audit" / "acceptance-receipt.json"
AMSS_CANDIDATE_LOSSES = ROOT / ".flux-artifacts" / "svi-score-v10-amss" / "external-audit" / "candidate-losses.csv"
AMSS_CONTRACT = ROOT / "research" / "svi_score_v10_amss" / "frozen-protocol" / "scientific-contract.json"

colors = base.colors
plt = base.plt
mm = base.mm
PILImage = base.PILImage
Flowable = base.Flowable
Paragraph = base.Paragraph
Spacer = base.Spacer
PageBreak = base.PageBreak
HRFlowable = base.HRFlowable
BaseDocTemplate = base.BaseDocTemplate
Frame = base.Frame
PageTemplate = base.PageTemplate
Table = base.Table
TableStyle = base.TableStyle

PAGE_W = base.PAGE_W
PAGE_H = base.PAGE_H
LEFT = base.LEFT
RIGHT = base.RIGHT
TOP = base.TOP
BOTTOM = base.BOTTOM
TEXT_W = base.TEXT_W
BLACK = base.BLACK
MID = base.MID
RULE = base.RULE
PALE = base.PALE
BLUE = base.BLUE
RED = base.RED
FONT = base.FONT
FONT_BOLD = base.FONT_BOLD
FONT_ITALIC = base.FONT_ITALIC
S = base.S

P = base.P
H1 = base.H1
H2 = base.H2
H3 = base.H3
bullet = base.bullet
definition = base.definition
proof = base.proof
make_table = base.make_table
caption = base.caption
equation = base.equation
multiline_equation = base.multiline_equation
image_flow = base.image_flow


TITLE = "RegretSet-MMM: Learning to Select Marketing Mix Models by Their Downstream Economic Consequences"
SHORT_TITLE = "RegretSet-MMM"


class RegretSetDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=base.A4,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title=TITLE,
            author="Gustavo Bramao",
            subject="RegretSet-MMM methodological working paper",
            keywords=(
                "marketing mix modeling, decision-focused learning, Bayesian inference, "
                "economic regret, DeepSets, evidence attribution, synthetic validation"
            ),
        )
        frame = Frame(
            LEFT,
            BOTTOM,
            TEXT_W,
            PAGE_H - TOP - BOTTOM,
            id="journal-body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="journal", frames=[frame], onPage=page_header))
        self.section = ""

    def afterFlowable(self, flowable: Flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name == "Heading1":
            self.section = flowable.getPlainText()
            key = f"section-{self.seq.nextf('section')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(self.section, key, level=0, closed=False)


def page_header(canvas, doc: RegretSetDoc) -> None:
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.35)
        canvas.line(LEFT, PAGE_H - 13.5 * mm, PAGE_W - RIGHT, PAGE_H - 13.5 * mm)
        canvas.setFont(FONT_ITALIC, 7.5)
        canvas.setFillColor(MID)
        canvas.drawString(LEFT, PAGE_H - 10.7 * mm, "Bramao: RegretSet-MMM")
        canvas.drawRightString(PAGE_W - RIGHT, PAGE_H - 10.7 * mm, "Working paper - external synthetic validation complete")
        canvas.setFont(FONT, 8)
        canvas.setFillColor(BLACK)
        canvas.drawCentredString(PAGE_W / 2, 10.5 * mm, str(doc.page))
    canvas.restoreState()


def save_research_design() -> Path:
    path = ASSETS / "figure-1-research-design-v2.png"
    if path.exists():
        return path
    fig, ax = plt.subplots(figsize=(9.2, 3.0), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, note: str, fill: str = "white") -> None:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor="#222222", lw=0.8))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=7.25,
                fontweight="bold", family="serif")
        ax.text(x + w / 2, y + h * 0.29, note, ha="center", va="center", fontsize=6.15,
                color="#555555", family="serif", linespacing=1.15)

    boxes = [
        (0.1, 1.65, 1.75, 1.25, "Advertiser world", "Observed history +\nhidden response truth"),
        (2.15, 1.65, 1.75, 1.25, "48 Bayesian MMMs", "FullRankADVI fits;\ntruth prohibited"),
        (4.2, 1.65, 1.75, 1.25, "232 tokens", "Diagnostics, posterior,\nevidence, assumptions"),
        (6.25, 1.65, 1.75, 1.25, "Set-wise learner", "Jointly compares the\nunordered candidate set"),
        (8.3, 1.65, 1.75, 1.25, "Four decisions", "Reduce, reallocate,\ngrow, ceiling"),
        (10.35, 1.65, 1.55, 1.25, "Loss label", "Reveal truth only\nafter action"),
    ]
    for args in boxes:
        box(*args)
    for i in range(len(boxes) - 1):
        x = boxes[i][0] + boxes[i][2]
        nx = boxes[i + 1][0]
        ax.annotate("", xy=(nx - 0.04, 2.28), xytext=(x + 0.04, 2.28),
                    arrowprops=dict(arrowstyle="->", lw=0.85, color="#222222"))
    ax.add_patch(plt.Rectangle((1.95, 0.37), 6.3, 0.66, facecolor="#F2F5FA", edgecolor="#355C9A", lw=0.7))
    ax.text(5.1, 0.70, "DEPLOYMENT PATH", ha="center", va="center", fontsize=7.5,
            fontweight="bold", color="#355C9A", family="serif")
    ax.text(5.1, 0.51, "new advertiser → candidate posteriors → 232 observable tokens → frozen selector → candidate risk",
            ha="center", va="center", fontsize=6.8, family="serif")
    ax.annotate("", xy=(6.95, 1.61), xytext=(6.95, 1.04),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#355C9A"))
    ax.text(10.95, 3.42, "TRAINING ONLY", ha="center", fontsize=7.3, color="#A8483C",
            fontweight="bold", family="serif")
    ax.text(6.0, 3.63, "Truth firewall: response truth and realized economic loss never enter candidate fitting or deployment tokens",
            ha="center", fontsize=7.7, style="italic", family="serif")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_dgp_dag() -> Path:
    path = ASSETS / "figure-2-dgp-dag-v2.png"
    if path.exists():
        return path
    fig, ax = plt.subplots(figsize=(8.8, 4.2), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")
    nodes = {
        "U": (1.1, 4.9, "Latent demand\nand market state"),
        "P": (4.1, 4.9, "Media planning\nintensity"),
        "C": (7.0, 4.9, "Observed controls,\nprice and\npromotion"),
        "X": (2.5, 2.8, "Spend and\ndelivery"),
        "R": (5.3, 2.8, "Channel-specific\nresponse transforms"),
        "B": (8.4, 2.8, "Baseline\ndemand"),
        "Y": (8.4, 1.0, "Observed\noutcome"),
        "Z": (2.0, 0.65, "External evidence\nwith quality and bias"),
    }
    for key, (x, y, label) in nodes.items():
        ax.add_patch(plt.Rectangle((x - 0.88, y - 0.38), 1.76, 0.76, facecolor="white",
                                   edgecolor="#222222", lw=0.85))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.35, family="serif", linespacing=1.0)
    arrows = [
        ("U", "P"), ("U", "X"), ("U", "B"), ("P", "X"), ("P", "C"),
        ("C", "B"), ("X", "R"), ("R", "Y"), ("B", "Y"), ("Z", "R"),
    ]
    for source, target in arrows:
        x1, y1, _ = nodes[source]
        x2, y2, _ = nodes[target]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=0.85, color="#333333", shrinkA=28, shrinkB=28))
    ax.text(5.5, 5.75, "Endogeneity arises because latent demand and planning affect both media and outcome",
            ha="center", fontsize=8.0, style="italic", family="serif")
    ax.text(5.5, 0.06, "Hidden truth includes the channel response surface, causal contribution and counterfactual value of each allocation.",
            ha="center", fontsize=7.7, family="serif")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_selector_architecture() -> Path:
    path = ASSETS / "figure-3-selector-architecture-v2.png"
    if path.exists():
        return path
    fig, ax = plt.subplots(figsize=(9.2, 3.35), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, note: str, fill: str = "white") -> None:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor="#222222", lw=0.8))
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=7.25,
                fontweight="bold", family="serif")
        ax.text(x + w / 2, y + h * 0.28, note, ha="center", va="center", fontsize=6.1,
                color="#555555", family="serif", linespacing=1.12)

    box(0.10, 1.35, 1.55, 1.30, "Candidate tokens", "48 candidates\n× 232 inputs")
    box(2.00, 1.35, 1.45, 1.30, "Shared encoder", "φ(xc)\n12 or 20 units")
    box(3.82, 2.95, 1.62, 0.92, "Mean pool", "h = average φ(xc)", "#F2F5FA")
    box(3.82, 1.35, 1.62, 1.30, "Equivariant join", "φ(xc), h, φ−h,\nφ⊙h")
    box(5.82, 1.35, 1.38, 1.30, "Decoder", "shared nonlinear\ncandidate map")
    box(7.58, 0.85, 2.15, 2.30, "Ten supervised heads", "mean + median\nP90 + P95 + CVaR90\nfour scenario losses\ndanger probability")
    box(10.12, 1.35, 1.75, 1.30, "Adjusted risk", "65% mean + 35% P90\n+ danger + disagreement", "#F2F5FA")
    for a, b in [((1.65, 2.00), (1.98, 2.00)), ((3.45, 2.00), (3.80, 2.00)),
                 ((5.44, 2.00), (5.80, 2.00)), ((7.20, 2.00), (7.56, 2.00)),
                 ((9.73, 2.00), (10.10, 2.00))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=0.85))
    ax.annotate("", xy=(4.63, 2.92), xytext=(2.72, 2.67), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(4.63, 2.67), xytext=(4.63, 2.93), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(6.0, 4.18, "Permutation-invariant advertiser context; permutation-equivariant candidate predictions",
            ha="center", fontsize=8.0, style="italic", family="serif")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_evidence_attribution() -> Path:
    path = ASSETS / "figure-4-development-evidence-attribution.png"
    if path.exists():
        return path
    receipt = json.loads(EVIDENCE_ATTRIBUTION.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.0), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.55]})
    ax = axes[0]
    ax.axis("off")
    ax.text(0.5, 0.92, r"$H_{post}=H_D+H_E+H_B+H_R$", ha="center", fontsize=14,
            math_fontfamily="stix")
    sources = [("Observational data", r"$H_D$"), ("Experiments", r"$H_E$"),
               ("Benchmarks", r"$H_B$"), ("Regularization", r"$H_R$")]
    for index, (label, symbol) in enumerate(sources):
        y = 0.73 - 0.17 * index
        ax.add_patch(plt.Rectangle((0.07, y - 0.052), 0.86, 0.105, facecolor="#F5F5F5",
                                   edgecolor="#555555", lw=0.55))
        ax.text(0.13, y, label, va="center", fontsize=8.1, family="serif")
        ax.text(0.86, y, symbol, ha="right", va="center", fontsize=10.5, math_fontfamily="stix")
    ax = axes[1]
    labels = ["All channels", "Paid social", "Nonbrand search", "CTV"]
    groups = [
        receipt["overall"],
        receipt["byChannel"]["meta_acquisition_spend"],
        receipt["byChannel"]["google_search_nonbrand_spend"],
        receipt["byChannel"]["ctv_spend"],
    ]
    keys = ["observational", "experiment", "benchmark", "regularization"]
    shares = [[group["equalChannelMean"][key] for key in keys] for group in groups]
    palette = ["#355C9A", "#6E9E73", "#C89552", "#8A8A8A"]
    left = [0.0] * len(labels)
    for idx, source in enumerate(["Data", "Experiment", "Benchmark", "Regularization"]):
        values = [row[idx] for row in shares]
        ax.barh(labels, values, left=left, color=palette[idx], label=source, height=0.48)
        left = [left[i] + values[i] for i in range(len(labels))]
    ax.set_xlim(0, 1)
    ax.set_xlabel("local source share", fontsize=8, family="serif")
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=7.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_development_importance() -> Path:
    path = ASSETS / "figure-7-development-importance.png"
    pillar_receipt = json.loads(INTERPRETABILITY.read_text())
    lens_receipt = json.loads(POSTERIOR_GEOMETRY.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.7), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.0]})

    display_labels = {
        "Predictive generalization": "Predictive\ngeneralization",
        "Structural adequacy": "Structural\nadequacy",
        "Causal and temporal robustness": "Causal and temporal\nrobustness",
        "Decision and evidence coherence": "Decision and evidence\ncoherence",
        "Posterior decision geometry": "Posterior decision\ngeometry",
        "Specification context": "Specification\ncontext",
        "Inference reliability": "Inference\nreliability",
        "Predictive and plausibility safety": "Predictive and\nplausibility safety",
        "ROI location and uncertainty": "ROI location and\nuncertainty",
        "Tail shape and approximation shift": "Tail shape and\napproximation shift",
        "Response-mechanics uncertainty": "Response-mechanics\nuncertainty",
        "Contribution and channel dependence": "Contribution and\nchannel dependence",
    }

    panels = [
        (
            axes[0],
            list(reversed(pillar_receipt["pillars"])),
            "pillar",
            "All 232 tokens",
        ),
        (
            axes[1],
            list(reversed([row for row in lens_receipt["lenses"] if row["tokenCount"] > 0])),
            "lens",
            "Inside posterior decision geometry",
        ),
    ]
    for ax, rows, label_key, title in panels:
        labels = [display_labels[row[label_key]] for row in rows]
        values = [100 * row["dropColumnRelativeIncrease"] for row in rows]
        bar_colors = ["#355C9A" if value > 0 else "#B9BDC5" for value in values]
        ax.barh(labels, values, color=bar_colors, height=0.58)
        ax.axvline(0, color="#222222", lw=0.75)
        ax.grid(axis="x", color="#E0E0E0", lw=0.5)
        ax.set_title(title, fontsize=8.5, fontweight="bold", family="serif", pad=7)
        ax.tick_params(axis="both", labelsize=7.0)
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        for label in ax.get_yticklabels():
            label.set_family("serif")
        for value, bar in zip(values, ax.patches):
            if abs(value) >= 3.0:
                x = value / 2
                align = "center"
                text_color = "white"
            else:
                x = value + (0.35 if value >= 0 else -0.35)
                align = "left" if value >= 0 else "right"
                text_color = "#222222"
            ax.text(
                x,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.1f}%",
                va="center",
                ha=align,
                fontsize=6.8,
                family="serif",
                color=text_color,
                fontweight="bold",
            )
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_xlabel("held-out economic-loss change after removal and refit", fontsize=7.1, family="serif")
    axes[1].set_xlabel("held-out economic-loss change after removal and refit", fontsize=7.1, family="serif")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_validation_design() -> Path:
    path = ASSETS / "figure-5-validation-design-final.png"
    if path.exists():
        return path
    fig, ax = plt.subplots(figsize=(9.0, 3.65), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, note: str, fill: str = "white") -> None:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor="#222222", lw=0.8))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=8.2,
                fontweight="bold", family="serif")
        ax.text(x + w / 2, y + h * 0.28, note, ha="center", va="center", fontsize=6.8,
                color="#555555", family="serif", linespacing=1.1)

    box(0.2, 3.15, 2.35, 1.25, "Development population", "420 advertiser worlds\n20,160 fitted candidates")
    box(3.1, 3.15, 2.2, 1.25, "Outer grouped CV", "5 unseen-advertiser folds\nperformance estimation")
    box(5.85, 3.15, 2.2, 1.25, "Inner grouped CV", "4 advertiser folds\narchitecture and policy")
    box(8.6, 3.15, 3.0, 1.25, "Cross-fitted development", "every reported prediction made\nwithout its advertiser")
    for a, b in [((2.55, 3.77), (3.08, 3.77)), ((5.3, 3.77), (5.83, 3.77)), ((8.05, 3.77), (8.58, 3.77))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=0.85))
    box(0.2, 0.55, 3.1, 1.35, "Mechanism challenge", "hold out one complete simulator\nfamily during development")
    box(4.45, 0.55, 3.1, 1.35, "Freeze", "tokens, model, risk policy,\nendpoints, code and hashes")
    box(8.7, 0.55, 2.9, 1.35, "Sealed audit", "140 new worlds; 6,720 fits\none opening; frozen analysis")
    ax.annotate("", xy=(4.42, 1.22), xytext=(3.32, 1.22), arrowprops=dict(arrowstyle="->", lw=0.85))
    ax.annotate("", xy=(8.67, 1.22), xytext=(7.57, 1.22), arrowprops=dict(arrowstyle="->", lw=0.85))
    ax.annotate("", xy=(6.0, 1.92), xytext=(9.7, 3.13), arrowprops=dict(arrowstyle="->", lw=0.65, color="#555555"))
    ax.text(6.0, 2.48, "Grouped folds estimate transfer to new advertisers; the sealed cohort tests the frozen research claim",
            ha="center", fontsize=7.8, style="italic", family="serif")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_confirmatory_audit() -> Path:
    path = ASSETS / "figure-6-confirmatory-risk-coverage.png"
    result = json.loads(AUDIT_RESULT.read_text())
    points = result["v9"]["riskCoverageCurve"]["points"]
    coverage = [100 * row["promotionCoverage"] for row in points]
    mean_loss = [row["meanPromotedExcessLoss"] for row in points]
    p90_loss = [row["p90PromotedExcessLoss"] for row in points]
    objective = [row["selectionObjective"] for row in points]

    fig, ax = plt.subplots(figsize=(8.7, 3.35), dpi=300)
    ax.plot(coverage, p90_loss, color="#A8483C", lw=1.6, marker="o", ms=3.2,
            label="P90 excess loss")
    ax.plot(coverage, objective, color="#355C9A", lw=2.0, marker="o", ms=3.5,
            label="65% mean + 35% P90 objective")
    ax.plot(coverage, mean_loss, color="#6E9E73", lw=1.6, marker="o", ms=3.2,
            label="Mean excess loss")
    ax.axvline(70, color="#777777", lw=0.8, ls="--")
    ax.text(70.5, max(p90_loss) * 0.92, "declared 70% operating point", fontsize=7.1,
            color="#666666", family="serif", va="top")
    ax.set_xlim(59, 101)
    ax.set_ylim(0, max(p90_loss) * 1.12)
    ax.set_xticks([60, 70, 80, 90, 100])
    ax.set_xlabel("promotion coverage (%)", fontsize=8.2, family="serif")
    ax.set_ylabel("selected-model excess loss", fontsize=8.2, family="serif")
    ax.grid(color="#E2E2E2", lw=0.6)
    ax.tick_params(labelsize=7.6)
    ax.legend(loc="upper left", frameon=False, fontsize=7.2, ncol=3)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.65)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def percentile(values: Sequence[float], probability: float) -> float:
    """Linear-interpolated empirical quantile without an extra dependency."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def read_amss_candidate_losses() -> list[dict[str, str]]:
    with AMSS_CANDIDATE_LOSSES.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    valid = [row for row in rows if row["valid"].lower() == "true"]
    if len(valid) != 4_800:
        raise RuntimeError(f"Expected 4,800 valid AMSS candidates, found {len(valid):,}.")
    return valid


def amss_random_comparator() -> dict[str, float]:
    """Exact distribution induced by uniform valid-candidate selection.

    Every AMSS business has 48 valid candidates, so pooling the 4,800 candidate
    risks gives each business equal weight and integrates over the random draw
    exactly.  This is distinct from taking quantiles of the 100 business-level
    conditional expectations stored in the audit receipt.
    """
    risks = [float(row["risk"]) for row in read_amss_candidate_losses()]
    return {
        "mean": statistics.fmean(risks),
        "p90": percentile(risks, 0.90),
        "p95": percentile(risks, 0.95),
    }


def amss_paired_random_inference() -> dict[str, float | list[float] | int]:
    result = json.loads(AMSS_RESULT.read_text())
    by_group: dict[str, list[float]] = {}
    for row in result["selections"]:
        difference = row["v9Risk"] - row["randomValidExpectedRisk"]
        by_group.setdefault(row["evidenceGroup"], []).append(difference)
    if sorted(map(len, by_group.values())) != [25, 25, 25, 25]:
        raise RuntimeError("AMSS paired bootstrap expects four evidence groups of 25 businesses.")
    rng = random.Random(10_031_772)
    bootstrap_means: list[float] = []
    for _ in range(10_000):
        draw: list[float] = []
        for differences in by_group.values():
            draw.extend(rng.choice(differences) for _ in differences)
        bootstrap_means.append(statistics.fmean(draw))
    observed = [
        row["v9Risk"] - row["randomValidExpectedRisk"]
        for row in result["selections"]
    ]
    return {
        "point": statistics.fmean(observed),
        "interval": [percentile(bootstrap_means, 0.025), percentile(bootstrap_means, 0.975)],
        "lower": sum(value < 0 for value in observed),
        "equal": sum(abs(value) < 1e-12 for value in observed),
        "higher": sum(value > 0 for value in observed),
        "resamples": 10_000,
        "seed": 10_031_772,
    }


def save_amss_external_validation() -> Path:
    path = ASSETS / "figure-7-amss-external-validation.png"
    result = json.loads(AMSS_RESULT.read_text())
    valid = read_amss_candidate_losses()
    selections = result["selections"]

    regretset = sorted(row["v9Risk"] for row in selections)
    random_valid = sorted(float(row["risk"]) for row in valid)
    oracle = sorted(row["oracleRisk"] for row in selections)
    paired = sorted(
        row["v9Risk"] - row["randomValidExpectedRisk"]
        for row in selections
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=300, gridspec_kw={"width_ratios": [1.12, 1.0]})
    ax = axes[0]
    for values, color, label, width in [
        (regretset, "#355C9A", "RegretSet-MMM", 2.0),
        (random_valid, "#A8483C", "Uniform random valid", 1.6),
        (oracle, "#6E9E73", "In-pool oracle", 1.6),
    ]:
        y = [(index + 1) / len(values) for index in range(len(values))]
        ax.step(values, y, where="post", color=color, lw=width, label=label)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("capped business decision risk", fontsize=8.0, family="serif")
    ax.set_ylabel("cumulative share", fontsize=8.0, family="serif")
    ax.set_title("A. Exact risk distributions", fontsize=8.5, fontweight="bold", family="serif")
    ax.grid(color="#E2E2E2", lw=0.5)
    ax.tick_params(labelsize=7.4)
    ax.legend(loc="lower right", frameon=False, fontsize=7.0)

    ax = axes[1]
    positions = list(range(1, len(paired) + 1))
    colors_for_points = ["#6E9E73" if value < 0 else "#A8483C" for value in paired]
    ax.scatter(positions, paired, c=colors_for_points, s=10, alpha=0.9, linewidths=0)
    ax.axhline(0, color="#222222", lw=0.8)
    ax.axhline(statistics.fmean(paired), color="#355C9A", lw=1.2, ls="--")
    ax.text(4, -0.46, "RegretSet-MMM lower risk", fontsize=7.0, color="#4E7B54", family="serif")
    ax.text(4, 0.57, "Random-valid expectation lower", fontsize=7.0, color="#8E3E34", family="serif")
    ax.set_xlim(0, 101)
    ax.set_ylim(min(-0.52, min(paired) * 1.08), max(0.62, max(paired) * 1.08))
    ax.set_xlabel("businesses ordered by paired difference", fontsize=8.0, family="serif")
    ax.set_ylabel("RegretSet minus expected random risk", fontsize=8.0, family="serif")
    ax.set_title("B. Paired external transport result", fontsize=8.5, fontweight="bold", family="serif")
    ax.grid(axis="y", color="#E2E2E2", lw=0.5)
    ax.tick_params(labelsize=7.4)
    for panel in axes:
        for spine in panel.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.6)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def add_title(story: list[Flowable]) -> None:
    story.extend([
        Spacer(1, 3 * mm),
        P(TITLE, "title"),
        P("Set-wise posterior representation, economic-regret supervision, and source-resolved evidence attribution", "subtitle"),
        P("Gustavo Bramao", "author"),
        P("Independent Researcher | September 2026 | Working paper - external synthetic validation complete", "date"),
        HRFlowable(width="100%", thickness=0.5, color=BLACK, spaceBefore=1, spaceAfter=8),
        P("Abstract", "abstract_heading"),
        P(
            "Marketing mix modeling (MMM) is commonly selected by predictive fit, residual diagnostics, calibration agreement, or fixed analyst scores, although its operational purpose is budget decision support. RegretSet-MMM instead learns across advertisers which complete Bayesian MMM specification is least likely to produce costly budget decisions. Forty-eight truth-blind candidate MMMs per advertiser are fitted with full-rank variational inference and represented by 232 deployment-observable tokens covering diagnostics, posterior decision geometry, temporal identification, evidence attribution, and assumptions. A permutation-invariant multi-head learner predicts loss across four predeclared budget decisions. We establish a finite-candidate bound connecting selected capped regret to regret-weighted oracle misrankings and decompose local ROI information among observational data, experiments, benchmarks, and regularization. After grouped development and an internal sealed audit, the frozen selector was transported without retraining to 100 businesses generated by Google's independently developed Aggregate Marketing System Simulator. Across 4,800 candidate fits and 19,200 frozen actions, RegretSet-MMM achieved mean capped decision risk of 0.230, versus 0.400 under uniform random selection among valid candidates and 0.012 for an unattainable in-pool oracle. Its exact randomization-distribution P90 and P95 risks were also lower. The results establish external synthetic transport under the declared candidate and utility contracts, not real-advertiser effectiveness or universal state-of-the-art performance.",
            "abstract",
        ),
        P("<b>Keywords:</b> marketing mix modeling; decision-focused learning; Bayesian inference; DeepSets; economic regret; source attribution; budget optimization; synthetic validation", "keywords"),
        HRFlowable(width="100%", thickness=0.5, color=BLACK, spaceBefore=0, spaceAfter=8),
    ])


def add_introduction(story: list[Flowable]) -> None:
    story.extend([
        H1("1", "Introduction"),
        P(
            "Marketing mix models estimate how marketing activity contributes to business outcomes and translate those estimates into resource-allocation decisions. Their scientific difficulty is familiar: spending is planned rather than randomized, channels co-move with latent demand, advertising has delayed and nonlinear effects, and external experiments or benchmarks are sparse and imperfectly transported. Consequently, several observationally plausible models can fit the same historical series yet imply substantially different returns on investment (ROI), marginal response curves, and optimal budgets [1–6]."
        ),
        P(
            "Most model-selection procedures remain one step removed from the decision. They favor out-of-sample prediction, likelihood-based criteria, calibration distance, structural diagnostics, or a fixed weighted score. These quantities are useful evidence, but none is identical to the economic opportunity lost when a selected model reallocates budget incorrectly. Nor is there a universal reason that a diagnostic should have the same importance for a low-signal search advertiser, a heavily flighted television advertiser, and a business with a strong independent experiment."
        ),
        P(
            "RegretSet-MMM treats model selection as a supervised decision problem across advertisers. In a synthetic world, causal response surfaces are known. Candidate MMMs are fitted without access to that truth, asked to choose budgets under declared scenarios, and then labeled by the value forgone relative to the best feasible action under the hidden response surface. A set-wise learner maps only deployment-observable diagnostics and posterior summaries to the distribution of this loss. For a new advertiser, the learned rule ranks candidate specifications without observing causal truth."
        ),
        image_flow(save_research_design(), 162),
        caption("Figure 1", "Research and deployment paths. Hidden causal truth is opened only after a candidate has acted in simulation. Deployment uses the frozen mapping from observable candidate sets to predicted decision risk."),
        H2("1.1", "Contributions"),
        bullet("A decision-focused formulation of cross-advertiser MMM selection in which the supervised target is downstream economic loss rather than a hand-weighted model-quality score."),
        bullet("A permutation-invariant learner that evaluates all 48 candidates jointly and emits location, tail, scenario, and dangerous-false-champion predictions for every candidate."),
        bullet("A 232-token, truth-blind representation combining classical diagnostics with SVI posterior geometry, channel-response uncertainty, cross-channel posterior dependence, evidence provenance, and model assumptions."),
        bullet("A finite-candidate regret bound and a theorem-aligned, gap-weighted ranking surrogate that penalize economically costly misrankings more strongly than harmless ones."),
        bullet("A continuous local decomposition of ROI information into observational data, experiments, industry benchmarks, and regularization, together with separate notions of evidence quality, conflict, and decision dependence."),
        bullet("An auditable validation design using advertiser-grouped nested cross-validation, complete-mechanism holdouts, a one-time internal sealed cohort, and a frozen external transport evaluation in Google's independently developed Aggregate Marketing System Simulator."),
        Spacer(1, 2 * mm),
        H2("1.2", "Claim boundary"),
        P(
            "The contribution is a method for learning and auditing an MMM selection rule. The completed internal sealed audit supports transfer to unseen advertisers sampled from the declared Flux population; the AMSS evaluation additionally supports transport to a separately authored synthetic system. Neither result implies real-advertiser effectiveness, universal state-of-the-art performance, inclusion of the true model in the candidate library, or exact posterior representation by variational inference."
        ),
    ])


def add_related_work(story: list[Flowable]) -> None:
    story.extend([
        H1("2", "Related work"),
        H2("2.1", "Bayesian marketing mix modeling"),
        P(
            "Modern Bayesian MMMs combine baseline controls, seasonality, adstock, saturation, and constrained or regularized media effects [1,2]. Open implementations have expanded access to geometric and Weibull carryover, response curves, time-varying coefficients, experiment calibration, and posterior budget optimization [4–8]. Recent work emphasizes that experiments should calibrate compatible estimands and time windows, while paid-search and other demand-harvesting channels remain especially vulnerable to endogeneity [3,5,9]. RegretSet-MMM does not replace those estimators; it learns which complete specification is least risky for a declared decision."
        ),
        H2("2.2", "Model assessment and causal validation"),
        P(
            "Predictive validation detects instability and distribution shift but need not recover a causal decomposition. Residual tests diagnose misspecification but cannot establish exogeneity from observed residuals alone. Experiment agreement tests external coherence, although experiment overlap, low power, and poor transport may mislead. Information criteria and cross-validation estimate predictive generalization [10]; they do not directly score the counterfactual value of a budget allocation. The proposed token registry retains these signals while allowing their relevance to be learned from economic labels."
        ),
        H2("2.3", "Decision-focused learning and learning to rank"),
        P(
            "Predict-then-optimize methods train predictive systems using downstream optimization loss rather than parameter error alone [11,12]. Ranking methods similarly exploit relative ordering within structured candidate sets [13]. RegretSet-MMM adapts these ideas to causal model selection: the objects being ranked are fitted Bayesian MMMs, each candidate generates four economic actions, and the training label is excess decision loss under a hidden causal response surface."
        ),
        H2("2.4", "Set representation and variational posterior inference"),
        P(
            "DeepSets provides a general architecture for permutation-invariant functions over unordered collections [14]. Here candidate order is arbitrary, so a shared encoder and symmetric pooling operator supply advertiser-specific context without creating a candidate-ID signal. Candidate posteriors are fitted with full-rank automatic differentiation variational inference (ADVI), which is scalable across tens of thousands of fits but may understate tails or miss multimodality [15,16]. The selection framework is inference-engine agnostic in principle: the same posterior tokens can be computed from more accurate draws when resources permit."
        ),
        H2("2.5", "Positioning of the contribution"),
        P(
            "The paper is not organized as a chronology of internal scores. The relevant scientific question is whether deployment-observable evidence predicts the economic consequences of selecting among plausible MMMs for unseen advertisers. The one-time audit uses a frozen, predeclared reduced-information selector as a falsification comparator while the contribution remains the decision-focused method, its evidence representation, and its auditable evaluation protocol."
        ),
        P(
            "Ground-truth simulation has an established role in marketing measurement research. AMASS generates aggregate marketing systems with known ROI and marginal ROI [23], while a recent independently released benchmark explicitly models endogenous budget feedback, promotional anticipation, scheduled television flights, and performance chasing [24]. These systems are especially relevant external challenges because their equations and implementation were not selected to favor RegretSet-MMM."
        ),
    ])


def add_decision_problem(story: list[Flowable]) -> None:
    story.extend([
        H1("3", "Decision problem"),
        H2("3.1", "Advertisers, candidates, and actions"),
        P(
            "Let b index advertisers and let S_b be a finite set of C candidate MMMs fitted to advertiser b. In the implemented design C = 48. Candidate c observes history D_b, declared external evidence E_b, and the business decision contract. It produces a posterior over parameters and a recommended allocation a_bcq for each scenario q. Synthetic causal truth θ_b is inaccessible until every candidate action is fixed."
        ),
        multiline_equation([
            r"p_c(\theta\mid D_b,E_b),\qquad a_{bcq}=\mathcal{A}_q\!\left[p_c(\theta\mid D_b,E_b)\right]",
            r"\theta_b^{\mathrm{true}}\notin\{D_b,E_b,x_{bc}\}",
        ], "1"),
        H2("3.2", "Economic value and loss"),
        P(
            "For action a, let V_bq(a; θ_b) denote its true incremental economic value in scenario q. The value function applies each allocation to the simulator's channel-specific response surfaces and converts incremental outcome to contribution using the advertiser's effective revenue margin before subtracting media cost. The feasible oracle action a*bq is the highest-value action in the same declared allocation grid and constraints seen by all candidates."
        ),
        multiline_equation([
            r"a^*_{bq}=\arg\max_{a\in\mathcal{A}_{bq}}V_{bq}(a;\theta_b^{\mathrm{true}})",
            r"\ell_{bcq}=V_{bq}(a^*_{bq};\theta_b^{\mathrm{true}})-V_{bq}(a_{bcq};\theta_b^{\mathrm{true}})\geq0",
        ], "2"),
        P(
            "The candidate-level label aggregates four predeclared decisions: reducing total spend, reallocating a fixed budget, growing spend, and finding an economic ceiling. The implementation retains all four scenario losses rather than training only a single average. A nonnegative weighted aggregate L_bc defines the principal decision loss, while scenario heads expose where the risk originates."
        ),
        equation(r"L_{bc}=\sum_{q=1}^{4}\omega_q\,\ell_{bcq},\qquad \omega_q\geq0,\quad\sum_q\omega_q=1", "3"),
        H2("3.3", "Excess loss and deployment risk"),
        P(
            "The in-pool oracle c*b is the candidate with minimum loss among eligible candidates, not the true structural model. Candidate excess loss Δ_bc measures the opportunity cost of selecting c instead of that oracle. A business-specific, candidate-independent scale s_b converts losses to a capped normalized target used by the ranking theorem and training regularizer. Uncapped monetary and normalized losses remain reportable safety endpoints."
        ),
        multiline_equation([
            r"c_b^*=\arg\min_{c\in S_b}L_{bc},\qquad \Delta_{bc}=L_{bc}-L_{bc_b^*}",
            r"\widetilde\Delta_{bc}=\min\!\left(\frac{\Delta_{bc}}{s_b},1\right)",
        ], "4"),
        H2("3.4", "Declared risk preference"),
        P(
            "The selector predicts both the conditional mean loss and the P90 loss. The default business protection rule assigns 65% weight to the mean and 35% to P90. This is a declared utility preference, not a learned scientific constant. A more risk-averse organization may increase the tail weight; a risk-neutral organization may reduce it. Sensitivity to this mixture must be reported without selecting the weight on sealed-audit performance."
        ),
        equation(r"R_{bc}=0.65\,\widehat{\mathbb{E}}[L_{bc}\mid X_b]+0.35\,\widehat Q_{0.90}[L_{bc}\mid X_b]", "5"),
    ])


def add_regret_theory(story: list[Flowable]) -> None:
    story.extend([
        H1("4", "Regret theory for finite candidate selection"),
        definition("Assumption 1 (finite eligible set)", "For each advertiser b, the eligible candidate set S_b is nonempty and finite; ties are resolved by a predeclared deterministic rule."),
        definition("Assumption 2 (candidate-independent scale)", "The positive economic scale s_b may depend on the advertiser and decision contract but not on which candidate is evaluated."),
        definition("Definition 1 (selector)", "Let f_bc be a scalar predicted risk for candidate c, with lower values preferred, and let the selected candidate be ĉ_b = arg min over c in S_b of f_bc."),
        H2("4.1", "Selected-regret bound"),
        P(
            "The following inequality motivates an all-pairs, regret-weighted ranking loss. It is deliberately modest: it connects a finite candidate selector to the oracle comparisons it gets wrong; it does not prove out-of-population generalization."
        ),
        definition("Proposition 1 (finite-candidate misranking bound)", "For every advertiser satisfying Assumptions 1–2,"),
        multiline_equation([
            r"\widetilde\Delta_{b\widehat c_b}\leq",
            r"\sum_{c\in S_b\setminus\{c_b^*\}}\widetilde\Delta_{bc}\,\mathrm{I}\!\left[f_{bc}\leq f_{bc_b^*}\right].",
        ], "6"),
        proof(
            "If the selector chooses the oracle, the left side is zero and the inequality is immediate. Otherwise the selected candidate ĉ_b is a non-oracle candidate satisfying f_bĉ ≤ f_bc*. Its term therefore appears in the sum with weight equal to the left side. Every remaining term is nonnegative."
        ),
        P(
            "The result says that a costly selected mistake is necessarily covered by the total weight of non-oracle candidates that the score ranks at least as favorably as the oracle. It does not require all ordering errors to be equally costly: a near-oracle swap receives a small gap, whereas an economically destructive false champion receives a large gap."
        ),
        H2("4.2", "Differentiable surrogate"),
        P(
            "For temperature T > 0 and u = (f_bc* − f_bc)/T, the event that c outranks the oracle implies u ≥ 0. Since softplus(u) is at least log 2 whenever u ≥ 0, the indicator is upper-bounded by softplus(u)/log 2."
        ),
        multiline_equation([
            r"\mathrm{I}[u\geq0]\leq\frac{\log(1+e^u)}{\log2}",
            r"\widetilde\Delta_{b\widehat c_b}\leq\sum_{c\neq c_b^*}\widetilde\Delta_{bc}\,\frac{\operatorname{softplus}\!\left((f_{bc_b^*}-f_{bc})/T\right)}{\log2}.",
        ], "7"),
        P(
            "Training therefore includes every oracle-versus-candidate comparison with its exact predeclared normalized gap. The implementation additionally optimizes a direct set-wise objective: a softmax distribution over negative predicted risk is charged the expected capped normalized regret of selecting from the complete candidate set. The theorem-aligned pair loss is a regularizer; the direct selection loss is the closer approximation to deployment."
        ),
        H2("4.3", "What the theorem does and does not prove"),
        bullet("It is deterministic and applies to the realized finite pool, regardless of how the score was trained."),
        bullet("It justifies gap-weighted oracle comparisons and supplies a testable receipt: selected capped regret must not exceed the reported hard or surrogate bound."),
        bullet("It does not show that S_b contains the true response model, that f generalizes, or that capped loss protects uncapped catastrophic tails; P90, P95, CVaR, and maximum uncapped loss remain necessary."),
    ])


def add_simulator(story: list[Flowable]) -> None:
    story.extend([
        H1("5", "Synthetic advertiser population"),
        H2("5.1", "Data-generating process"),
        P(
            "A synthetic benchmark is valuable because causal contributions and counterfactual budget value are known. It is also dangerous because a selector can learn artifacts of one generator. The population therefore varies business duration, seasonality, demand persistence, media planning, commercial shocks, channel delivery, response curves, controls, noise, external evidence, and decision economics. The generator is separate from the candidate transformation code."
        ),
        image_flow(save_dgp_dag(), 158),
        caption("Figure 2", "Causal structure of the advertiser generator. Spend is endogenous because latent demand and planning affect both media delivery and outcome. External evidence is generated separately and may be noisy, biased, or poorly transported."),
        H2("5.2", "Observed outcome"),
        P(
            "For advertiser b at time t, observed outcome is the sum of baseline demand, control effects, channel contributions, and stochastic disturbance. Each channel contribution applies a delivery process, a channel-specific carryover kernel, and a Hill-type saturation function. Channel parameters are not forced to be common: television can have delayed Weibull carryover while paid search can have short geometric memory, and saturation levels differ by channel and advertiser."
        ),
        multiline_equation([
            r"Y_{bt}=B_{bt}+\gamma_b^{\top}Z_{bt}+\sum_{j=1}^{J_b}m_{bjt}+\varepsilon_{bt}",
            r"m_{bjt}=\beta_{bj}\,h\!\left((w_{bj}*X_{bj})_t;\kappa_{bj},\eta_{bj}\right)",
        ], "8"),
        P(
            "The histories span 104–260 weeks. Demand, planning, and commercial-state persistence vary broadly; event shocks, promotions, observed demand proxies, measurement noise, gross margin, lifetime-value multipliers, and maximum budget changes are heterogeneous. The principal implementation contains three archetypal channels—paid social, paid search, and television—because their delivery and endogeneity mechanisms differ sharply. This simplification is a limitation, not a claim that real MMMs contain only three channels."
        ),
        H2("5.3", "Seven mechanism families"),
        make_table([
            ["Family", "Primary stress", "Why it matters"],
            ["Balanced DTC", "moderate persistence and signal", "reference business with mixed digital and offline media"],
            ["Search demand harvesting", "search follows latent demand", "tests endogeneity and inflated observational search ROI"],
            ["Social frequency pressure", "reach/frequency saturation", "tests nonlinear marginal returns"],
            ["Delayed television", "long, delayed carryover", "tests temporal support and post-flight attribution"],
            ["Correlated planning", "shared budget and promotion process", "tests multicollinearity and conditional identification"],
            ["Wrong external evidence", "biased or transported anchors", "tests conflict, quality, and source dependence"],
            ["Swapped mechanics", "unexpected channel response families", "tests transfer beyond channel-name stereotypes"],
        ], widths=[39 * mm, 50 * mm, 73 * mm]),
        caption("Table 1", "Predeclared simulator families. Each family changes a causal mechanism rather than merely a random seed."),
        H2("5.4", "Carryover and saturation ranges"),
        P(
            "Carryover is generated by either normalized geometric or Weibull convolution. Geometric decay ranges from approximately 0.01 to 0.78, with the upper tail concentrated in television. Weibull shape spans approximately 1.15–4.80; scale spans 1.5–8 weeks for digital channels and 3.5–13 weeks for television. Hill shapes range approximately 0.90–3.10 for paid social and 0.72–2.35 for other channels. Half-saturation points are tied to advertiser-specific spend quantiles between roughly 0.30 and 0.82. These ranges create short memory, delayed peaks, weak saturation, and frequency pressure without assuming one channel response globally."
        ),
        H2("5.5", "External evidence"),
        P(
            "Experiments are available probabilistically and represent noisy measurements of ROI over their tested spend and outcome window, including a declared post-test carryover period when appropriate. They have heterogeneous power, bias, replication, platform or geo design, and transportability. Benchmarks are noisy channel-level priors; a dedicated wrong-evidence family shifts them downward or upward. Candidate evidence arms determine whether available qualified evidence is used, but hidden truth never chooses an arm."
        ),
        multiline_equation([
            r"ROI^{\mathrm{exp}}_{bjW}=ROI^{\mathrm{true}}_{bjW}(1+b_{bjW})+\epsilon_{bjW}",
            r"ROI^{\mathrm{true}}_{bjW}=\frac{\sum_{t\in W^+}m_{bjt}^{\mathrm{incremental}}}{\sum_{t\in W}X_{bjt}^{\mathrm{tested}}}",
        ], "9"),
        H2("5.6", "Decision environments"),
        P(
            "The allocation engine evaluates 21 budget levels and 240 candidate allocations per level under channel-specific cut and growth constraints. Economic value is incremental outcome multiplied by an effective revenue margin minus incremental media cost. The four scenarios cover budget reduction (up to 35%), fixed-budget reallocation, budget growth (up to 50%), and an economic-ceiling search (up to 80% growth). This makes the label depend on marginal response and channel substitution, not only average ROI recovery."
        ),
    ])


def add_candidate_models(story: list[Flowable]) -> None:
    story.extend([
        H1("6", "Candidate Bayesian MMMs and posterior inference"),
        H2("6.1", "Candidate library"),
        P(
            "Each advertiser receives the same 48-candidate opportunity set: 24 model specifications crossed with two predeclared evidence settings. Specifications vary adstock family and parameters, Hill saturation, baseline Fourier order and cycle, ridge strength, likelihood family, prior family, optional time-varying coefficients, planning-intensity controls, and calibration route. The library is deliberately misspecified for some worlds; otherwise model selection would be trivial."
        ),
        make_table([
            ["Component", "Candidate variation", "Scientific ambiguity represented"],
            ["Carryover", "geometric or Weibull; short to long memory", "immediate decay versus delayed peak and long tail"],
            ["Saturation", "multiple Hill shapes and half-saturation values", "weak versus strong diminishing returns"],
            ["Baseline", "Fourier order, cycle, controls", "seasonality and recurring demand allocation"],
            ["Coefficients", "constant or time-varying", "stable effect versus evolving response"],
            ["Planning", "latent-intensity factor on/off", "shared demand and media-planning process"],
            ["Distributions", "positive priors; Gaussian, Student-t, or log-normal outcomes", "tail behavior, skew, and robustness"],
            ["Evidence", "qualified experiment or benchmark use", "external identification and transport"],
        ], widths=[31 * mm, 56 * mm, 75 * mm]),
        caption("Table 2", "Major candidate dimensions. Exact grids and fingerprints are part of the open research contract."),
        H2("6.2", "Full-rank variational inference"),
        P(
            "All research candidates use PyMC 6.2 FullRankADVI under a frozen contract. A multivariate Gaussian variational family q_φ(θ) is optimized by maximizing the evidence lower bound (ELBO), retaining covariance that a mean-field approximation would discard. Posterior draws from q_φ feed response, uncertainty, and decision tokens. Multiple convergence and reproducibility checks guard against unstable envelopes."
        ),
        equation(r"\phi^*=\arg\max_{\phi}\left\{\mathbb{E}_{q_{\phi}(\theta)}[\log p(D,E,\theta)]-\mathbb{E}_{q_{\phi}(\theta)}[\log q_{\phi}(\theta)]\right\}", "10"),
        P(
            "Variational inference is used because tens of thousands of full posterior fits are required. It is not assumed superior to Markov chain Monte Carlo. ADVI can smooth multimodal posteriors and distort tail mass, so the inference engine is part of the reported research contract. The learned selector is compatible with posterior draws from NUTS in principle, provided the 232-token contract is computed identically."
        ),
        H2("6.3", "Truth firewall"),
        P(
            "Business identity, generator family, split labels, candidate identity, search metadata, synthetic parameters, decision loss, ROI truth error, and contribution truth error are forbidden inputs. Tests permute candidate and channel order, scan token names, and verify that model predictions are unchanged by those permutations. The only path from hidden truth to the learner is through training labels after actions have been fixed."
        ),
    ])


def add_tokens(story: list[Flowable]) -> None:
    story.extend([
        H1("7", "Deployment-observable token representation"),
        H2("7.1", "Tokens rather than opaque embeddings"),
        P(
            "A token is a named numerical input with a declared construction and interpretation. The term emphasizes that the model receives a structured vocabulary of evidence rather than raw outcome and spend histories. Tokens may be individual diagnostics, response assumptions, posterior summaries, or predeclared interactions. The learner creates hidden embeddings from these tokens, but every input remains auditable and reproducible."
        ),
        P(
            "The representation contains 136 base tokens and 96 posterior decision tokens, for 232 per candidate. Base tokens cover predictive generalization, residual structure, causal and placebo stress, ROI coherence, temporal evidence, evidence quality and conflict, model assumptions, and their allowed interactions. Posterior tokens summarize channel-level ROI geometry, response uncertainty, evidence provenance, contribution stability, and cross-channel dependence from SVI draws."
        ),
        make_table([
            ["Token pillar", "Examples", "Question answered"],
            ["Predictive generalization", "rolling holdout error; high/low-spend regimes", "does the candidate forecast outside its fitting regime?"],
            ["Structural adequacy", "residual pattern; variance; collinearity; rank", "does the fitted structure leave diagnostic failures?"],
            ["Causal and temporal robustness", "future-media placebo; anchor recovery; carryover support", "are timing and causal claims compatible with available evidence?"],
            ["Decision coherence", "ROI plausibility; marginal-return and source-deletion changes", "does the posterior support stable, plausible allocations?"],
            ["Posterior decision geometry", "ROI tail mass; interval overlap; contribution CV; correlations", "where are location, tail, and substitution risks concentrated?"],
            ["Specification context", "adstock, saturation, likelihood, dynamics, calibration", "which assumptions generated those diagnostics?"],
        ], widths=[37 * mm, 61 * mm, 64 * mm]),
        caption("Table 3", "Practitioner-facing token taxonomy. Pillars organize interpretation only; they are not fixed score weights."),
        H2("7.2", "Channel-order invariance"),
        P(
            "Twenty-three channel metrics are reduced by four symmetric operators—mean, standard deviation, minimum, and maximum—yielding 92 tokens. Four additional summaries measure mean and maximum absolute posterior correlation among channel log-ROI and contribution draws. The representation is invariant to channel order and does not require the same number of channels."
        ),
        multiline_equation([
            r"T_{mk}(c)=\operatorname{Agg}_{j\in\mathcal{J}_b}\,t_{m}(q_c(\theta_j)),\quad \operatorname{Agg}\in\{\mathrm{mean,sd,min,max}\}",
            r"92=23\times4,\qquad 96=92+4\ \mathrm{pairwise\ dependence\ summaries}",
        ], "11"),
        H2("7.3", "Posterior token content"),
        make_table([
            ["Posterior family", "Representative quantities"],
            ["ROI location and interval", "log median, mean, lower and upper interval endpoints, relative width, draw SD"],
            ["Decision safety", "implausible probability, near-zero probability, skewness, analytic-to-SVI location shift, interval overlap"],
            ["Response mechanics", "adstock moments, Weibull scale and shape, saturation moments, half-saturation, response-family shares"],
            ["Contribution reliability", "log contribution coefficient of variation"],
            ["Evidence provenance", "experiment and benchmark source indicators"],
            ["Substitution geometry", "absolute pairwise log-ROI and contribution correlation summaries"],
        ], widths=[47 * mm, 115 * mm]),
        caption("Table 4", "Ninety-six SVI posterior decision tokens. All are computed without synthetic truth."),
        H2("7.4", "Why interactions matter"),
        P(
            "A diagnostic rarely has one global meaning. Wide ROI intervals may be acceptable for a small channel but dangerous near a large reallocation threshold. Residual drift may matter differently under long-memory television than under short-memory search. The base registry therefore includes only predeclared interactions between interpretable diagnostics and model assumptions; the nonlinear set learner can additionally form bounded interactions after fold-specific standardization."
        ),
    ])


def add_learning(story: list[Flowable]) -> None:
    story.extend([
        H1("8", "RegretSet-MMM learner"),
        H2("8.1", "Permutation-invariant candidate-set architecture"),
        P(
            "Candidates are not independent rows: whether a model is attractive depends on the alternatives available for the same advertiser. RegretSet-MMM therefore consumes the entire unordered set X_b = {x_bc}. A shared encoder φ maps each 232-token vector to a candidate embedding. Mean pooling creates advertiser context h_b. The decoder receives the candidate, context, difference, and elementwise interaction embeddings, preserving permutation-equivariant predictions."
        ),
        multiline_equation([
            r"z_{bc}=\phi(x_{bc}),\qquad h_b=|S_b|^{-1}\sum_{c\in S_b}z_{bc}",
            r"u_{bc}=\psi\!\left(z_{bc},h_b,z_{bc}-h_b,z_{bc}\odot h_b\right)",
        ], "12"),
        image_flow(save_selector_architecture(), 162),
        caption("Figure 3", "Set-wise, multi-head architecture. Reordering candidates or channels cannot change the candidate predictions. The ten heads expose different dimensions of decision loss rather than collapsing training to one scalar target."),
        H2("8.2", "Ten prediction heads"),
        P(
            "The decoder produces ten outputs per candidate: mean, median, P90, P95, and CVaR90 across the candidate's four predeclared scenario-loss values; the four scenario-specific absolute losses; and the probability of being a dangerous false champion. These summaries describe the declared scenario distribution, not posterior quantiles of an unknown causal truth. Quantile outputs use positive softplus increments so that median ≤ P90 ≤ P95 ≤ CVaR90 by construction. A candidate is dangerous when it appears sufficiently attractive to win while its true normalized excess loss exceeds the predeclared threshold."
        ),
        H2("8.3", "Supervised objective"),
        P(
            "The objective combines robust location loss, quantile loss, tail expectation, scenario decomposition, dangerous-champion classification, direct set-wise selection regret, and the theorem-aligned oracle-pair regularizer. Let y_bc denote mean loss, qτ,bc the empirical quantile of the four scenario losses, d_bc the danger indicator, and p_bc the model-implied soft selection probability."
        ),
        multiline_equation([
            r"\mathcal{L}_{heads}=\mathcal{L}_{Huber}(\widehat\mu,y)+0.15\,\mathcal{L}_{Huber}(\widehat q_{.5},q_{.5})",
            r"\quad+0.35\,\rho_{.90}(q_{.90}-\widehat q_{.90})+0.20\,\rho_{.95}(q_{.95}-\widehat q_{.95})",
            r"\quad+0.30\,\mathcal{L}_{Huber}(\widehat{CVaR}_{.90},CVaR_{.90})+0.12\sum_q\mathcal{L}_{Huber}(\widehat\ell_q,\ell_q)+0.45\,BCE(\widehat d,d),",
        ], "13"),
        multiline_equation([
            r"p_{bc}=\frac{\exp(-f_{bc}/T)}{\sum_{r\in S_b}\exp(-f_{br}/T)},\qquad \mathcal{L}_{set}=\sum_b\sum_{c\in S_b}p_{bc}\widetilde\Delta_{bc}",
            r"\mathcal{L}_{pair}=\sum_b\sum_{c\neq c_b^*}\widetilde\Delta_{bc}\operatorname{softplus}(f_{bc_b^*}-f_{bc}),",
        ], "14"),
        equation(r"\mathcal{L}=\mathcal{L}_{heads}+\mathcal{L}_{set}+0.25\,\mathcal{L}_{pair}+\lambda\|\Theta\|_2^2", "15"),
        P(
            "The numerical coefficients in the head loss are frozen optimization choices, not claims about the scientific importance of the corresponding diagnostic pillars. They balance gradients on distinct supervised targets. Candidate selection at deployment follows equation (16), whose 65/35 mean-tail mixture is explicitly a business risk preference."
        ),
        H2("8.4", "Ensemble policy and abstention"),
        P(
            "Five bootstrap selector fits estimate model disagreement. Predicted danger and disagreement increase adjusted risk. Confidence combines predicted danger, relative ensemble uncertainty, and ambiguity between the best and second-best candidates. Selective promotion can abstain on low-confidence advertisers; all primary results nevertheless report 100% coverage to prevent apparent improvement through selective omission."
        ),
        multiline_equation([
            r"f_{bc}=R_{bc}+\lambda_d\widehat p(d_{bc}=1)+0.25\,\operatorname{SD}_{m}(R^{(m)}_{bc})",
            r"\operatorname{conf}_b=1-\left(0.4\widehat p_d+0.3\,u_b^{rel}+0.3\,a_b\right),",
        ], "16"),
        H2("8.5", "Nested model and policy selection"),
        P(
            "Two compact architectures and three danger penalties are compared only inside four-fold inner grouped cross-validation. Five outer grouped folds estimate transfer to unseen advertisers. All rows from one advertiser remain in one fold. Fold-specific standardization is fitted only on the training advertisers, and cross-fitted predictions are retained for every development business."
        ),
    ])


def add_evidence_theory(story: list[Flowable]) -> None:
    story.extend([
        H1("9", "Source-resolved evidence attribution"),
        H2("9.1", "Posterior information decomposition"),
        P(
            "A channel estimate may be precise because the observational design identifies it, because an experiment anchors it, because an industry benchmark supplies prior information, or because generic regularization shrinks it. RegretSet-MMM reports these sources continuously. At the local Laplace approximation, posterior curvature is decomposed into positive semidefinite components for observational data D, experiments E, benchmarks B, and regularization R."
        ),
        equation(r"H_{post}=H_D+H_E+H_B+H_R,\qquad \Sigma=H_{post}^{-1}", "17"),
        P(
            "For channel-j ROI gradient g_j, the implemented variance attribution propagates each curvature source through the common posterior covariance. The raw components sum to the local posterior ROI variance up to numerical error; normalization yields nonnegative source shares."
        ),
        multiline_equation([
            r"v_{s,j}=g_j^{\top}\Sigma H_s\Sigma g_j,\qquad v_j=g_j^{\top}\Sigma g_j",
            r"a_{s,j}=\frac{v_{s,j}}{\sum_{r\in\{D,E,B,R\}}v_{r,j}},\qquad a_{s,j}\geq0,\quad\sum_s a_{s,j}=1.",
        ], "18"),
        image_flow(save_evidence_attribution(), 156),
        caption("Figure 4", "Observed local evidence attribution for the 1,260 channel posteriors in the 420 outer-fold selected development candidates. Bars report equal-channel mean Laplace curvature shares. Overall, observational data supplied 42.1%, experiments 36.9%, benchmarks 7.0%, and regularization 14.0% of local ROI information. These are development-cohort influence estimates, not scores of source validity."),
        P(
            "The empirical decomposition shows that the selected candidates were neither simply data-led nor prior-led. Observational information supplied the largest average share, but experiments supplied more than one third of local ROI information and were the largest component for nonbrand search. CTV had the strongest observational share. Benchmark information was deliberately narrower because it entered only as gap-filling evidence when qualified experiments were absent; weak regularization supplied the remaining local curvature. Spend weighting produced a similar split—45.0% data, 34.0% experiments, 6.1% benchmarks, and 14.9% regularization—so the result is not an artifact of treating small and large channels equally."
        ),
        P(
            "Conditional observational identification was far lower than the raw data share: the mean residualized data share was 8.5%, with a 3.9% median. The distinction is substantive. The data can contribute posterior precision jointly while offering little channel-specific variation after the other regressors are projected out. This is precisely why source attribution and conditional identification are reported separately."
        ),
        H2("9.2", "Conditional observational identification"),
        P(
            "Raw column magnitude overstates identification when channels are correlated. The implementation projects a channel's normalized ROI direction on every other design column and measures the weighted residual information. This is a Schur-complement interpretation: only variation that cannot be reproduced by the remaining regressors receives conditional observational credit. Perfectly duplicated channels have no separate likelihood identification even if both columns are large."
        ),
        definition("Proposition 2 (duplicated-channel non-identification)", "If two transformed media columns satisfy X1 = X2, the likelihood depends on β1 and β2 only through β1 + β2. Individual channel effects are unidentified without external information or additional restrictions."),
        proof("Substituting X2 = X1 gives X1β1 + X2β2 = X1(β1 + β2). Every coefficient pair with the same sum has the same likelihood."),
        H2("9.3", "Local source sensitivity"),
        P(
            "In a scalar conjugate Gaussian example, posterior mean is a precision-weighted average of source means. The derivative with respect to source s is exactly its normalized precision. This supplies a simple interpretation of local evidence share as sensitivity, while equation (18) extends the idea to correlated multivariate coefficients and an ROI functional."
        ),
        multiline_equation([
            r"\mu_{post}=\frac{\sum_s\tau_s\mu_s}{\sum_s\tau_s}",
            r"\frac{\partial\mu_{post}}{\partial\mu_s}=\frac{\tau_s}{\sum_r\tau_r}.",
        ], "19"),
        H2("9.4", "Bias transmission, conflict, and decision dependence"),
        P(
            "Influence is not validity. If source s has bias δ_s in the conjugate setting, posterior bias is the influence-weighted sum of source biases. RegretSet-MMM therefore keeps source influence distinct from source quality, standardized conflict, and the downstream decision change when a source is removed."
        ),
        multiline_equation([
            r"\operatorname{Bias}(\mu_{post})=\sum_s\frac{\tau_s}{\sum_r\tau_r}\,\delta_s",
            r"C_{s,j}=\frac{|\mu_{D,j}-\mu_{s,j}|}{\sqrt{\sigma_{D,j}^2+\sigma_{s,j}^2}}",
            r"D_s=\left\|\mathcal{A}[p(\theta\mid D,E,B)]-\mathcal{A}[p(\theta\mid D,E_{-s},B_{-s})]\right\|.",
        ], "20"),
        P(
            "Experiment quality records independence, MMM-data overlap, test power, tested spend range, campaign and outcome windows, post-test carryover, and temporal or audience transportability. Benchmark quality records channel, tactic, objective, and business-model match, recency, sample size, uncertainty, and whether the source is specific or a broad fallback. Descriptive phrases such as data-led or experiment-led may summarize equation (18), but they receive no hand-designed validation points and are never interpreted as evidence validity."
        ),
    ])


def add_validation(story: list[Flowable]) -> None:
    story.extend([
        H1("10", "Validation and governance"),
        H2("10.1", "Development cohort"),
        P(
            "The development dataset contains 420 advertiser worlds and 20,160 posterior-fitted candidates. Advertiser-grouped nested cross-validation separates hyperparameter selection from outer-fold assessment. A complete leave-one-generator-family-out challenge asks whether the learner transfers to a mechanism absent from its training families. Neither procedure is a substitute for a sealed audit, because researchers can still make method-level choices after seeing development summaries."
        ),
        image_flow(save_validation_design(), 160),
        caption("Figure 5", "Validation governance. Nested cross-validation estimates advertiser transfer during development. The separately generated sealed cohort is opened exactly once after the method, posterior contract, endpoints, and source hashes are frozen."),
        H2("10.2", "One-time sealed audit"),
        P(
            "The confirmatory cohort contains 140 new advertiser worlds - 20 from each of seven mechanism families - and 6,720 new FullRankADVI candidate fits. Its seed window is disjoint from development. Before cohort generation, the audit contract froze the 232-token registry, selector, risk policy, source code, inference engine, bootstrap procedure, family composition, and endpoints. Technical failures could be rerun only under the identical payload and posterior contract. No score, token, threshold, or endpoint could be repaired after truth was opened."
        ),
        H2("10.3", "Predeclared endpoints"),
        make_table([
            ["Endpoint", "Definition", "Role"],
            ["Mean excess loss", "average loss above the in-pool oracle", "central decision performance"],
            ["P90 and P95", "upper quantiles of selected excess loss", "downside protection"],
            ["Dangerous false champions", "selected candidates above the declared regret threshold", "unsafe top-one promotion"],
            ["Oracle recall", "fraction selecting the lowest-loss in-pool candidate", "ranking sharpness"],
            ["Risk–coverage area", "loss integrated from 60% to 100% promotion coverage", "quality of abstention"],
            ["Theorem violations", "selected capped regret above equation (6) receipt", "implementation consistency"],
        ], widths=[39 * mm, 74 * mm, 49 * mm]),
        caption("Table 5", "Absolute confirmatory endpoints. No internal-version comparison is needed to interpret these quantities."),
        H2("10.4", "Inference and multiplicity"),
        P(
            "Family-stratified bootstrap resampling preserves the audit's equal family design. The primary comparison uses 10,000 resamples and a fixed seed. Point estimates are reported for every predeclared absolute endpoint, all seven mechanism families, paired business outcomes, and the complete risk-coverage curve. Analyses not named in the frozen protocol are descriptive and are not used to alter the confirmatory conclusion."
        ),
    ])


def add_confirmatory_results(story: list[Flowable]) -> None:
    result = json.loads(AUDIT_RESULT.read_text())
    recovery = json.loads(AUDIT_RECOVERY.read_text())
    if not all(result["checks"].values()) or not result["conclusion"]["allConfirmatoryChecksPassed"]:
        raise RuntimeError("The manuscript expects a verified audit with every confirmatory check passed.")
    if any(recovery[key] for key in ("scientificMethodChanged", "selectorChanged", "posteriorChanged")):
        raise RuntimeError("The operational recovery receipt records a scientific change.")
    if recovery["hiddenOutcomesInspectedBeforeRecovery"]:
        raise RuntimeError("The recovery receipt records hidden-outcome inspection before completion.")
    full = result["v9"]["at100PercentCoverage"]
    selective = result["v9"]["at70PercentCoverage"]
    comparison = result["primaryInference"]
    paired = result["pairedBusinessOutcomes"]
    families = result["v9"]["byFamily"]
    family_labels = {
        "balanced-dtc": "Balanced DTC",
        "correlated-planning": "Correlated planning",
        "delayed-tv": "Delayed TV",
        "search-demand-harvesting": "Search demand harvesting",
        "social-frequency-pressure": "Social frequency pressure",
        "swapped-channel-mechanics": "Swapped channel mechanics",
        "wrong-evidence": "Wrong external evidence",
    }
    story.extend([
        H1("11", "Confirmatory sealed-audit results"),
        P(
            "The one-time audit evaluated the frozen selector on 140 previously unseen synthetic advertisers and 6,720 newly fitted candidate MMMs. All 6,720 candidate rows were finite, valid, and unique. The audit was scored once after posterior inference and truth-blind token construction were complete. All nine predeclared confirmatory checks passed."
        ),
        make_table([
            ["Confirmatory quantity", "100% coverage", "70% coverage", "Interpretation"],
            ["Mean excess loss", f"{full['meanPromotedExcessLoss']:.2f}", f"{selective['meanPromotedExcessLoss']:.2f}", "average opportunity lost relative to the in-pool oracle"],
            ["Median excess loss", f"{full['medianPromotedExcessLoss']:.2f}", f"{selective['medianPromotedExcessLoss']:.2f}", "typical selected-model loss"],
            ["P90 excess loss", f"{full['p90PromotedExcessLoss']:.2f}", f"{selective['p90PromotedExcessLoss']:.2f}", "loss not exceeded by 90% of promoted advertisers"],
            ["P95 excess loss", f"{full['p95PromotedExcessLoss']:.2f}", f"{selective['p95PromotedExcessLoss']:.2f}", "more extreme downside"],
            ["Decision-risk objective", f"{full['selectionObjective']:.2f}", f"{selective['selectionObjective']:.2f}", "65% mean plus 35% P90 excess loss"],
            ["Dangerous false champions", f"{100 * full['dangerousFalseChampionShare']:.1f}%", f"{100 * selective['dangerousFalseChampionShare']:.1f}%", "materially harmful top-one selections"],
            ["Oracle recall", f"{100 * full['oracleRecall']:.1f}%", f"{100 * selective['oracleRecall']:.1f}%", "exact recovery of the lowest-loss candidate"],
            ["Regret-bound violations", str(full["theoremViolations"]), str(selective["theoremViolations"]), "implementation check for equation (6)"],
        ], widths=[45 * mm, 31 * mm, 31 * mm, 55 * mm]),
        caption("Table 6", "Absolute sealed-audit performance. Coverage is the fraction of advertisers for which the selector promotes a candidate rather than abstaining."),
        H2("11.1", "Primary confirmatory inference"),
        P(
            f"The predeclared primary estimand was the full-coverage decision-risk objective for RegretSet-MMM minus the same objective for a frozen reduced-information comparator. The observed difference was {comparison['pointDifference']:.3f} loss units. The family-stratified two-sided 95% bootstrap interval was [{comparison['twoSided95Interval'][0]:.3f}, {comparison['twoSided95Interval'][1]:.3f}], and the predeclared one-sided 95% upper bound was {comparison['oneSided95UpperBound']:.3f}. All values favor RegretSet-MMM because lower loss is better. The interval excludes zero in the favorable direction, satisfying the superiority rule. Across individual businesses, RegretSet-MMM had lower loss in {100 * paired['v9LowerLossShare']:.1f}%, equal loss in {100 * paired['equalLossShare']:.1f}%, and higher loss in {100 * paired['v9HigherLossShare']:.1f}%."
        ),
        image_flow(save_confirmatory_audit(), 158),
        caption("Figure 6", "Absolute risk-coverage behavior on the sealed cohort. Abstaining on the least-confident advertisers reduces mean, tail, and composite decision loss. The 70% operating point was declared before the audit was opened."),
        PageBreak(),
        H2("11.2", "Mechanism-family stress results"),
        P(
            "The overall result does not imply uniform difficulty. Social-frequency-pressure advertisers produced the largest mean and P90 losses; swapped-channel-mechanics produced the largest P95 loss despite a relatively low median. Delayed-TV advertisers had the highest dangerous-false-champion share. This heterogeneity supports reporting the full loss distribution rather than a single average."
        ),
        make_table([
            ["Synthetic mechanism family", "Mean", "P90", "P95", "Danger / oracle"],
            *[
                [
                    family_labels[family],
                    f"{metrics['meanPromotedExcessLoss']:.2f}",
                    f"{metrics['p90PromotedExcessLoss']:.2f}",
                    f"{metrics['p95PromotedExcessLoss']:.2f}",
                    f"{100 * metrics['dangerousFalseChampionShare']:.0f}% / {100 * metrics['oracleRecall']:.0f}%",
                ]
                for family, metrics in families.items()
            ],
        ], widths=[56 * mm, 22 * mm, 22 * mm, 22 * mm, 40 * mm]),
        caption("Table 7", "Full-coverage results by predeclared mechanism family, with 20 unseen advertisers per family. The final column reports dangerous-false-champion share and exact in-pool oracle recall."),
        H2("11.3", "Confirmatory checks and claim boundary"),
        P(
            "The primary superiority check, full-coverage mean and P90 improvements, P95 non-inferiority, dangerous-false-champion improvement, selective-risk-area improvement, matched 70% objective improvement, theorem verification, and SVI-only inference check all passed. The finite-candidate bound had zero violations. The result therefore confirms synthetic generalization under the frozen cohort contract. It does not establish real-world generalization, global optimality, or realized advertiser profit improvement. Any future selector tuning requires a new sealed cohort."
        ),
        H2("11.4", "Operational provenance"),
        P(
            f"All 6,720 posterior fits completed under the frozen inference contract. During local label assembly, the orchestration parser mapped zero-based shard 0 to shard 1; consequently, seven label shards completed while 18 businesses in shard 0 were never executed. The missing 864 candidate labels were generated locally with a temporary parser correction that affected shard indexing only. The scientific method, posterior artifacts, selector, endpoints, and hidden outcomes remained unchanged; hidden outcomes were not inspected before recovery. All eight shards were then hash-checked and assembled once before the frozen scorer ran once. The machine-readable recovery receipt independently records that the scientific method, selector, and posterior did not change."
        ),
    ])


def add_external_validation(story: list[Flowable]) -> None:
    result = json.loads(AMSS_RESULT.read_text())
    acceptance = json.loads(AMSS_ACCEPTANCE.read_text())
    contract = json.loads(AMSS_CONTRACT.read_text())
    if result["status"] != "frozen-external-audit-complete-no-post-open-tuning":
        raise RuntimeError("The manuscript expects the completed frozen AMSS external audit.")
    if result["provenance"]["frozenV9Retrained"] or not result["provenance"]["candidateActionsFrozenBeforeTruth"]:
        raise RuntimeError("AMSS provenance does not preserve the frozen-selector contract.")
    if not acceptance["verification"]["deterministicReassembly"]:
        raise RuntimeError("AMSS acceptance receipt did not verify deterministic reassembly.")

    regretset = result["endpoints"]["frozenV9"]
    oracle = result["endpoints"]["inPoolOracle"]
    random_valid = amss_random_comparator()
    paired = amss_paired_random_inference()
    mean_reduction = 100 * (1 - regretset["mean"] / random_valid["mean"])
    p90_reduction = 100 * (1 - regretset["p90"] / random_valid["p90"])
    p95_reduction = 100 * (1 - regretset["p95"] / random_valid["p95"])

    story.extend([
        H1("12", "External transport validation in AMSS"),
        H2("12.1", "A frozen test in a separately authored simulator"),
        P(
            "The Aggregate Marketing System Simulator (AMSS) was developed independently of FluxMMM to generate aggregate marketing time series and associated ground-truth return and marginal-return measures [23]. It therefore provides a stronger transport test than drawing additional seeds from the Flux generator. Before any AMSS business was generated, the selector developed above, its 232-token registry, SVI contract, candidate library, evidence rules, four decision scenarios, loss definition, and 65/35 mean-tail utility were frozen. This frozen artifact is named <b>RegretSet-MMM</b>. It was not retrained, recalibrated, or reselected after AMSS truth became available."
        ),
        P(
            f"The external cohort contains {result['cohort']['businesses']} AMSS businesses, balanced across paid-social-experiment, search-experiment, television-experiment, and no-experiment evidence conditions. Each business contributes 24 structural specifications under two evidence arms, producing {result['cohort']['candidateModels']:,} FullRankADVI fits. Each posterior fixed actions for budget reduction, fixed-budget reallocation, budget growth, and an economic-ceiling decision before truth was opened, for {result['cohort']['decisionActions']:,} evaluated actions. The unit of evaluation is the business."
        ),
        P(
            "The design uses a deterministic 100-point marginal space-filling cohort and rotates all six orders of the three media modules. Candidate fitting observes 156 weekly periods after a 52-week burn-in. Frozen actions cover a 52-week decision horizon plus eight post-action carryover weeks. Every candidate chooses the feasible allocation that maximizes posterior-expected incremental contribution less incremental media cost. AMSS then evaluates the fixed action using four common-random-number truth replicates. Period-matched experiments, when assigned, compare treatment with a 20%-lower-budget path from the same pretest state and include a declared post-test outcome window."
        ),
        H2("12.2", "Three selection rules and what they mean"),
        definition("RegretSet-MMM", "The deployable learned rule. It sees only the same truth-blind candidate tokens available for a new advertiser and selects one of the 48 valid candidates. Lower realized decision risk is better."),
        definition("Uniform random valid candidate", "A chance benchmark that selects each of the 48 valid candidates with equal probability within a business. Its distribution is calculated exactly over all 4,800 business-candidate pairs, rather than approximated by one random draw."),
        definition("In-pool oracle", "A nondeployable lower benchmark that observes AMSS truth and chooses the lowest-risk action among the same 48 frozen candidates. It is not the true global model, a globally optimal budget, or an information set available to an advertiser."),
        make_table([
            ["Selection rule", "Role", "Mean", "P90", "P95"],
            ["RegretSet-MMM", "frozen deployable selector", f"{regretset['mean']:.3f}", f"{regretset['p90']:.3f}", f"{regretset['p95']:.3f}"],
            ["Uniform random valid", "chance selection among valid MMMs", f"{random_valid['mean']:.3f}", f"{random_valid['p90']:.3f}", f"{random_valid['p95']:.3f}"],
            ["In-pool oracle", "truth-informed lower benchmark", f"{oracle['mean']:.3f}", f"{oracle['p90']:.3f}", f"{oracle['p95']:.3f}"],
        ], widths=[44 * mm, 62 * mm, 18 * mm, 18 * mm, 18 * mm]),
        caption("Table 8", "External AMSS decision risk under the predeclared capped-loss contract. For uniform random selection, P90 and P95 integrate exactly over both businesses and the candidate draw. The in-pool oracle is a diagnostic lower benchmark, not a feasible selector."),
        H2("12.3", "External result"),
        P(
            f"RegretSet-MMM achieved mean capped decision risk of {regretset['mean']:.3f}, compared with {random_valid['mean']:.3f} under uniform random valid selection: a {mean_reduction:.1f}% reduction. Its P90 and P95 risks were {regretset['p90']:.3f} and {regretset['p95']:.3f}, respectively {p90_reduction:.1f}% and {p95_reduction:.1f}% below the exact random-selection distribution. Thus the external result is not limited to the average; the learned selector also reduced downside risk under the same utility and candidate pool."
        ),
        P(
            f"A descriptive evidence-group-stratified paired bootstrap compares each RegretSet-MMM risk with that business's expected risk under uniform valid selection. The mean paired difference was {paired['point']:.3f}; the 95% interval from {paired['resamples']:,} resamples was [{paired['interval'][0]:.3f}, {paired['interval'][1]:.3f}]. RegretSet-MMM had lower risk than the random-valid expectation for {paired['lower']} of 100 businesses and higher risk for {paired['higher']}. Because this comparator-specific interval was added for manuscript interpretation after the frozen audit, it is reported as descriptive rather than relabeled as a predeclared confirmatory test."
        ),
        image_flow(save_amss_external_validation(), 164),
        caption("Figure 7", "External AMSS transport. Panel A shows the exact capped-risk distributions for the frozen RegretSet-MMM selector, uniform random selection among valid candidates, and the truth-informed in-pool oracle. Panel B shows paired RegretSet-MMM risk minus each business's expected random-valid risk; values below zero favor RegretSet-MMM."),
        H2("12.4", "Interpretation and remaining headroom"),
        P(
            f"The in-pool oracle achieved mean risk of {oracle['mean']:.3f}. The gap between {regretset['mean']:.3f} and {oracle['mean']:.3f} demonstrates that substantial selection headroom remains even without expanding the candidate library. RegretSet-MMM therefore transfers useful ranking information to AMSS, but it does not recover the best candidate reliably enough to justify a claim of global optimality. The scientific contribution is the frozen external transport of a learned decision rule, not equivalence to the oracle."
        ),
        P(
            "The channel adapter maps AMSS native search to a nonbrand-search archetype, traditional television to the long-memory CTV role, and an unchanged AMSS traditional-media module to a paid-social stress proxy. The proxy preserves AMSS delivery, reach, frequency, state-transition, and stochastic mechanisms but is not a literal platform auction. This evaluation therefore tests transport across independently implemented marketing mechanisms and endogenous delivery, not exact platform-label fidelity."
        ),
        P(
            f"All {acceptance['verification']['truthParts']} truth partitions and {acceptance['verification']['truthRows']} action outcomes were verified, {acceptance['verification']['contractTests']} contract checks passed, and deterministic reassembly reproduced the accepted result. The protocol forbade post-truth retraining, reselection, and business exclusion. These controls support an external synthetic transport claim; they do not establish realized effects in advertiser data or universal superiority to every published MMM workflow."
        ),
    ])


def add_interpretation(story: list[Flowable]) -> None:
    story.extend([
        H1("13", "Interpretation for researchers and practitioners"),
        H2("13.1", "What the selector learns"),
        P(
            "The learner estimates a conditional mapping from an advertiser's joint candidate evidence to economic risk. It does not learn universal causal coefficients, and its hidden embedding is not an advertiser identity. At deployment it asks: among these candidates, which posterior and diagnostic configuration resembles configurations that led to low decision loss in held-out synthetic advertisers?"
        ),
        H2("13.2", "Development evidence: what predicted lower economic loss"),
        P(
            "Raw neural-network weights are not stable scientific importance measures because nonlinear encoders distribute information across correlated tokens. We therefore organize all 232 deployment-observable tokens into six exhaustive practitioner pillars. For each pillar g, the entire group is removed, the selector is relearned on the outer-training advertisers under the originally selected architecture and policy, and economic loss is measured on the untouched outer fold. The primary importance estimand is"
        ),
        equation(r"I_g=100\,\frac{J_{-g}^{OOF}-J_{all}^{OOF}}{J_{all}^{OOF}},", "21"),
        P(
            "where J is the declared 65% mean / 35% P90 selected-model excess-loss objective. Positive I_g means the remaining tokens could not replace the removed information; negative I_g means the group was conditionally redundant or harmful at the available sample size. The sign is not a causal statement about the diagnostic and does not license removing a scientific safety check. A twenty-repeat within-advertiser permutation analysis is retained as a sensitivity check, but drop-column refitting is primary because it permits the learner to adapt to the information that remains."
        ),
        image_flow(save_development_importance(), 164),
        caption("Figure 8", "Advertiser-grouped development importance. Bars show the percentage change in held-out economic loss when a complete token group is removed and the selector is relearned. Positive values indicate incremental predictive information; gray negative values indicate conditional redundancy or finite-sample overfit, not that the underlying scientific check is unnecessary. The sealed cohort was not opened for this analysis."),
        P(
            "The central result is unusually concentrated. Removing posterior decision geometry increased held-out economic loss by 38.7%, and the direction was adverse in four of five advertiser folds. Removing any other broad pillar reduced the aggregate objective: causal and temporal robustness −5.0%, predictive generalization −7.2%, decision and evidence coherence −9.7%, structural adequacy −11.6%, and specification context −14.5%. The permutation sensitivity analysis also disrupted posterior geometry in all 20 repetitions; specification context appeared important under permutation but not after refitting, a pattern consistent with interaction information that is used by the fitted network yet can be relearned from correlated posterior summaries."
        ),
        P(
            "A second ablation partitions the 105 posterior-geometry tokens. Removing tail shape and approximation-shift summaries increased held-out loss by 4.3% and did so in four of five folds; removing ROI location and uncertainty increased loss by 3.9% in three of five folds. The other posterior subgroups were replaceable after refitting. These subgroup effects do not add to the 38.7% whole-pillar effect: the network is nonlinear, and the full bundle carries interactions and jointly redundant signals that are destroyed together but recoverable when only one subgroup is absent."
        ),
        P(
            "Within this synthetic development population, the actionable lesson is therefore not that classical diagnostics cease to matter. It is that the strongest incremental signal for economic transfer came from the posterior state presented to the decision maker: the location and width of channel ROI distributions, implausible and near-zero tail mass, asymmetry and approximation shift, contribution stability, response-parameter uncertainty, and cross-channel covariance. Point fit scores or a named adstock family were not sufficient on their own. A candidate was more likely to transfer economically when its full posterior described a stable and decision-usable allocation problem."
        ),
        make_table([
            ["Practitioner pillar", "Tokens represented", "Development finding and use"],
            ["Posterior decision geometry", "ROI levels and intervals; tail mass; contribution stability; response uncertainty; channel covariance; SVI reliability", "+38.7%. Inspect the entire posterior allocation geometry before promoting a model."],
            ["Model specification context", "adstock, saturation, dynamics, likelihood, calibration and their declared interactions", "−14.5%. A named specification is context, not a transferable quality score by itself."],
            ["Causal and temporal robustness", "placebos, anchor recovery, confounder stress, whole-flight and carryover support", "−5.0%. Preserve as causal challenge evidence; current tokens added no robust incremental ranking information."],
            ["Predictive generalization", "rolling forecasts, coverage, fold stability and high/low-spend regimes", "−7.2%. Keep as transparent validation evidence; avoid equating forecast skill with decision quality."],
            ["Structural adequacy", "residual, variance, likelihood-shape, influence, rank and boundary checks", "−11.6%. Use for model criticism and failure diagnosis, not as a stand-alone promotion score."],
            ["Decision and evidence coherence", "ROI plausibility, source quality and conflict, identification and evidence-deletion decisions", "−9.7%. Retain source-specific receipts; simplify correlated inputs before retraining the selector."],
        ], widths=[39 * mm, 70 * mm, 53 * mm]),
        caption("Table 9", "Practitioner interpretation of outer-fold development ablation. Percentages are conditional predictive importance in the declared simulator population, not universal validation weights and not confirmatory sealed-audit outcomes."),
        H2("13.3", "A disciplined interpretation of the null pillars"),
        P(
            "A negative drop-column result does not imply that residual tests, temporal placebos, experiments, or business plausibility are dispensable. First, several appear again inside posterior geometry: weak identification expands intervals, collinearity induces channel covariance, and conflicting evidence moves posterior location or tail mass. Second, a diagnostic can be essential as a hard safety gate even when it adds no marginal ranking accuracy. Third, correlated token groups divide predictive credit. The result instead identifies a model-development priority: compress overlapping diagnostics, preserve raw validation receipts for human review, and require future selector versions to demonstrate positive outer-fold ablation before claiming a pillar improves economic ranking."
        ),
        H2("13.4", "What happens for a new advertiser"),
        P(
            "The system validates data cadence and schema, generates a declared candidate set, fits SVI posterior approximations, computes the 232 truth-blind tokens, and applies the frozen set-wise selector. It reports predicted mean and tail loss, confidence, scenario risk, ROI and contribution uncertainty, source attribution, source conflict, and source-deletion decision dependence. Low-confidence cases are reviewed or abstained. The final production posterior may be refitted with a higher-fidelity sampler; any posterior shift is disclosed rather than silently overwriting the validation receipt."
        ),
        H2("13.5", "Why plausible ROI is necessary but not sufficient"),
        P(
            "An industry-compatible average ROI can coexist with the wrong carryover, saturation, or channel covariance and therefore generate a poor marginal allocation. Conversely, a posterior may conflict with a broad benchmark because the advertiser genuinely differs. RegretSet-MMM uses plausibility, evidence quality, conflict, identification, and decision dependence as separate tokens. It never repairs a fitted ROI after seeing the economic label."
        ),
    ])


def add_limitations(story: list[Flowable]) -> None:
    story.extend([
        H1("14", "Limitations"),
        bullet("Synthetic transport: AMSS is independently authored and materially strengthens external validity, but both development and external evaluation remain simulated rather than realized advertiser interventions."),
        bullet("Candidate-pool dependence: the oracle is best among 48 fitted candidates under one decision contract, not the true global model. A stronger search can change both labels and selector behavior."),
        bullet("Approximate posterior inference: FullRankADVI may distort skew, multimodality, and tail dependence. All reported research inference is consistently SVI, so the conclusions are conditional on that approximation."),
        bullet("Restricted channel universe: three archetypal channels cannot represent the full granularity, geography, funnel stages, retail media, creative, or measurement systems of large advertisers."),
        bullet("Simplified experiments: external evidence is period-matched and quality-annotated but does not reproduce every auction, compliance, interference, platform-measurement, or geo-spillover mechanism."),
        bullet("Economic utility: conversion from outcome to contribution, allocation constraints, four scenario weights, and the 65/35 mean-tail preference are business choices; different utilities can produce different rankings."),
        bullet("Evidence attribution is local: curvature shares describe sensitivity near one posterior approximation, not global causal validity or a Shapley decomposition of all modeling choices."),
        bullet("Neural-selector uncertainty: bootstrap disagreement is an operational proxy rather than a full Bayesian posterior over the meta-model."),
        bullet("No real-world causal endpoint: the current design cannot establish that the learned rule improves realized advertiser profit."),
        H2("14.1", "Interpretive boundaries"),
        P(
            "The completed internal and AMSS audits establish transfer only under their declared synthetic populations, candidate pool, evidence rules, and decision utility. They do not establish universal state-of-the-art MMM performance, global optimality, superiority to expert modelers, or realized external validity across industries. Nor do they imply that evidence influence proves evidence quality, residual diagnostics prove exogeneity, or the finite-candidate regret bound proves population generalization."
        ),
    ])


def add_open_science(story: list[Flowable]) -> None:
    story.extend([
        H1("15", "Open science and reproducibility"),
        P(
            "FluxMMM is the open-source implementation of RegretSet-MMM. Scientific artifacts are content-addressed: simulator contract, candidate fingerprints, posterior settings, token registry, grouped folds, selector configuration, source hashes, and audit endpoints are versioned. Truth-leakage tests prohibit hidden labels from candidate fitting and deployment tokens. Permutation tests verify candidate- and channel-order invariance. Numerical tests verify the finite-candidate bound and evidence-share exhaustiveness."
        ),
        make_table([
            ["Stage", "Permitted information", "Frozen receipt"],
            ["World generation", "observed history and hidden truth", "seed, generator family, parameters, decision contract"],
            ["Candidate fitting", "observed data and declared evidence", "candidate and SVI fingerprint"],
            ["Token construction", "fit, diagnostics, metadata allowed by registry", "232-token names, hashes, leakage and permutation tests"],
            ["Decision labeling", "hidden truth only after action", "four actions, oracle actions, absolute and excess losses"],
            ["Selector training", "development labels only", "nested folds, hyperparameters, ensemble seeds"],
            ["Internal sealed audit", "truth opened once after freeze", "all endpoints, intervals, failures, provenance"],
            ["External AMSS audit", "frozen actions; AMSS truth opened once", "adapter contract, actions, hashes, exact random comparator, acceptance receipt"],
            ["Deployment", "no causal truth", "predicted risk, confidence, evidence profile, candidate fingerprint"],
        ], widths=[35 * mm, 57 * mm, 70 * mm]),
        caption("Table 10", "Information and artifact contract."),
        P(
            "Open sourcing allows researchers to challenge the simulator, contribute new channel mechanisms, add decision utilities, reproduce posterior tokens with different inference engines, and submit selectors under the same truth firewall. Revisions after an opened audit must receive a new method and audit identifier. Negative results are part of the public scientific record."
        ),
        H2("15.1", "Code and data availability"),
        P(
            'The canonical project website is <link href="https://fluxmmm-web.web.app/" color="#355C9A">https://fluxmmm-web.web.app/</link>. Source code, model contracts, tests, and reproducibility materials are available in the FluxMMM repository at <link href="https://github.com/gustavobramao/fluxmmm" color="#355C9A">https://github.com/gustavobramao/fluxmmm</link>. The public distribution identifies which artifacts are executable source, cached posterior output, or externally generated simulation evidence.'
        ),
    ])


def add_conclusion(story: list[Flowable]) -> None:
    story.extend([
        H1("16", "Conclusion"),
        P(
            "RegretSet-MMM reframes the selection of a marketing mix model as a learned, cross-advertiser decision problem. Candidate models are fitted without causal truth, summarized by named posterior and validation tokens, compared jointly as an unordered set, and supervised by the economic consequences of their budget actions. The method predicts the distribution—not only the mean—of loss and can abstain when top-one choice is ambiguous."
        ),
        P(
            "Two theoretical components make the design auditable. The finite-candidate inequality shows why economically costly oracle misrankings bound selected capped regret and motivates gap-weighted ranking. The source decomposition reveals whether local ROI precision is supplied by observational data, experiments, benchmarks, or generic regularization, without confusing influence with validity."
        ),
        P(
            "The internal sealed audit supports transfer within the declared Flux synthetic population. More importantly, the frozen selector reduced mean and upper-tail decision risk relative to uniform random valid-model selection after transport to 100 businesses from the independently developed AMSS system. Because no AMSS truth was available during selector training, candidate fitting, or action selection, this result demonstrates that deployment-observable posterior and validation evidence can carry useful economic-ranking information across synthetic data-generating systems."
        ),
        P(
            "The sizeable gap to the in-pool oracle, the restricted channel archetypes, the SVI approximation, and the absence of real-advertiser causal outcomes define the next research frontier. RegretSet-MMM should therefore be read as an externally transportable and falsifiable decision-focused selection protocol, not a claim of universal model recovery. Its open implementation allows researchers and advertisers to reproduce the evidence, test new simulators and candidate libraries, and challenge the learned rule under new decision utilities."
        ),
    ])


REFERENCES = [
    "[1] Jin, Y., Wang, Y., Sun, Y., Chan, D., and Koehler, J. (2017). Bayesian methods for media mix modeling with carryover and shape effects. Google Research.",
    "[2] Chan, D. and Perry, M. (2017). Challenges and opportunities in media mix modeling. Google Research.",
    "[3] Chen, A., Chan, D., Perry, M., Jin, Y., Sun, Y., Wang, Y., and Koehler, J. (2018). Bias correction for paid search in media mix modeling. arXiv:1807.03292.",
    "[4] Ng, E., Wang, Z., and Dai, A. (2021). Bayesian time varying coefficient model with applications to marketing mix modeling. arXiv:2106.03322.",
    "[5] Zhang, Y., Wurm, M., Li, E., Wakim, A., Kelly, J., Price, B., and Liu, Y. (2024). Media mix model calibration with Bayesian priors. Google Research.",
    "[6] Dew, R., Padilla, N., and Shchetkina, A. (2024). Your MMM is broken: identification of nonlinear and time-varying effects in marketing mix models. arXiv:2408.07678.",
    "[7] Runge, J., Skokan, I., Zhou, G., and Pauwels, K. (2024). Packaging up media mix modeling: an introduction to Robyn's open-source approach. arXiv:2403.14674.",
    "[8] Google Meridian Team (2026). Meridian marketing mix modeling. Software and methodological documentation.",
    "[9] Gordon, B. R., Zettelmeyer, F., Bhargava, N., and Chapsky, D. (2019). A comparison of approaches to advertising measurement: evidence from big field experiments at Facebook. Marketing Science, 38(2), 193–225.",
    "[10] Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and widely applicable information criterion in singular learning theory. Journal of Machine Learning Research, 11, 3571–3594.",
    "[11] Elmachtoub, A. N. and Grigas, P. (2022). Smart predict, then optimize. Management Science, 68(1), 9–26.",
    "[12] Wilder, B., Dilkina, B., and Tambe, M. (2019). Melding the data-decisions pipeline: decision-focused learning for combinatorial optimization. AAAI Conference on Artificial Intelligence, 33, 1658-1665.",
    "[13] Burges, C. J. C. et al. (2005). Learning to rank using gradient descent. International Conference on Machine Learning, 89-96.",
    "[14] Zaheer, M. et al. (2017). Deep Sets. Advances in Neural Information Processing Systems, 30.",
    "[15] Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., and Blei, D. M. (2017). Automatic differentiation variational inference. Journal of Machine Learning Research, 18(14), 1-45.",
    "[16] Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. (2017). Variational inference: a review for statisticians. Journal of the American Statistical Association, 112(518), 859–877.",
    "[17] Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica, 46(1), 33–50.",
    "[18] Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional value-at-risk. Journal of Risk, 2, 21–42.",
    "[19] Huber, P. J. (1964). Robust estimation of a location parameter. Annals of Mathematical Statistics, 35(1), 73–101.",
    "[20] Cawley, G. C. and Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research, 11, 2079–2107.",
    "[21] Lewis, R. A. and Rao, J. M. (2015). The unfavorable economics of measuring the returns to advertising. Quarterly Journal of Economics, 130(4), 1941–1973.",
    "[22] Vehtari, A., Gelman, A., and Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. Statistics and Computing, 27, 1413–1432.",
    "[23] Vaver, J. and Zhang, S. S.-H. (2017). Introduction to the Aggregate Marketing System Simulator. Google Research.",
    "[24] Heusch, N. (2026). A synthetic benchmark dataset with endogenous marketing spend for validating marketing mix models. arXiv:2608.21130.",
]


def add_references(story: list[Flowable]) -> None:
    story.append(H1("", "References"))
    for ref in REFERENCES:
        story.append(P(ref, "reference"))


def add_appendices(story: list[Flowable]) -> None:
    audit_hash = hashlib.sha256(AUDIT_RESULT.read_bytes()).hexdigest()
    audit = json.loads(AUDIT_RESULT.read_text())
    amss_result_hash = hashlib.sha256(AMSS_RESULT.read_bytes()).hexdigest()
    amss_candidate_hash = hashlib.sha256(AMSS_CANDIDATE_LOSSES.read_bytes()).hexdigest()
    amss_result = json.loads(AMSS_RESULT.read_text())
    amss_acceptance = json.loads(AMSS_ACCEPTANCE.read_text())
    amss_contract = json.loads(AMSS_CONTRACT.read_text())
    amss_random = amss_random_comparator()
    amss_paired = amss_paired_random_inference()
    story.extend([
        PageBreak(),
        H1("A", "Appendix: frozen research contract"),
        H2("A.1", "Study dimensions"),
        make_table([
            ["Object", "Frozen value"],
            ["Development advertisers", "420"],
            ["Candidates per advertiser", "48"],
            ["Development candidate fits", "20,160"],
            ["Outer / inner grouped folds", "5 / 4"],
            ["Posterior engine", "PyMC 6.2 FullRankADVI"],
            ["Tokens per candidate", "232 (136 base + 96 posterior decision)"],
            ["Selector", "small permutation-invariant DeepSets, 10 outputs"],
            ["Bootstrap ensemble", "5 fits"],
            ["Sealed audit", "140 advertisers × 48 candidates = 6,720 new fits"],
            ["Sealed families", "7, with 20 advertisers per family"],
            ["External AMSS audit", "100 businesses × 48 candidates = 4,800 new fits"],
            ["External frozen actions", "19,200 across four budget decisions"],
        ], widths=[67 * mm, 95 * mm]),
        caption("Table A.1", "Core study dimensions."),
        H2("A.2", "Architecture and policy grid"),
        make_table([
            ["Choice", "Values considered inside inner grouped folds"],
            ["Candidate encoder width", "12 or 20"],
            ["Decoder width", "12 or 16"],
            ["Epochs", "16 or 20"],
            ["Learning rate", "0.0025 or 0.0020"],
            ["Weight decay", "0.001"],
            ["Set-selection temperature", "0.35"],
            ["Danger penalty", "0, 0.25, or 0.50"],
            ["Ensemble disagreement penalty", "0.25"],
            ["Mean / P90 risk preference", "0.65 / 0.35"],
        ], widths=[67 * mm, 95 * mm]),
        caption("Table A.2", "Development choices. Outer-fold outcomes do not select these values."),
        H1("B", "Appendix: posterior decision token registry"),
        H2("B.1", "Twenty-three channel metrics"),
        make_table([
            ["Group", "Metrics before four symmetric aggregations"],
            ["ROI", "log median; log mean; log lower; log upper; log relative width; log draw SD"],
            ["Safety and shape", "implausible probability; near-zero probability; skewness; location shift; interval overlap"],
            ["Evidence", "experiment indicator; benchmark indicator"],
            ["Adstock", "mean; SD; log Weibull scale mean; Weibull shape mean; Weibull share; sum-normalization share"],
            ["Saturation", "mean; SD; half-saturation mean"],
            ["Contribution", "log coefficient of variation"],
        ], widths=[39 * mm, 123 * mm]),
        caption("Table B.1", "Channel metrics. Mean, SD, minimum, and maximum over channels produce 92 tokens."),
        H2("B.2", "Four pairwise posterior summaries"),
        bullet("Mean absolute pairwise correlation of channel log-ROI draws."),
        bullet("Maximum absolute pairwise correlation of channel log-ROI draws."),
        bullet("Mean absolute pairwise correlation of channel contribution draws."),
        bullet("Maximum absolute pairwise correlation of channel contribution draws."),
        H1("C", "Appendix: algorithmic protocol"),
        H2("C.1", "Development"),
        P("1. Generate advertiser worlds and seal hidden response truth.", "body_left"),
        P("2. Fit the common 48-candidate library with the immutable FullRankADVI contract.", "body_left"),
        P("3. Compute 232 truth-blind tokens and verify candidate- and channel-order invariance.", "body_left"),
        P("4. Optimize four actions from each candidate posterior under common feasibility constraints.", "body_left"),
        P("5. Reveal truth only to label action loss, in-pool oracle loss, excess loss, and danger.", "body_left"),
        P("6. Use nested advertiser-grouped cross-validation to select architecture and policy and obtain outer-fold predictions.", "body_left"),
        P("7. Run complete-mechanism holdouts, freeze method and endpoints, then generate a disjoint sealed cohort.", "body_left"),
        H2("C.2", "Deployment"),
        P("1. Validate schema, cadence, evidence, and decision utility for a new advertiser.", "body_left"),
        P("2. Generate and fit the declared candidate set without causal truth.", "body_left"),
        P("3. Compute the identical 232-token representation and continuous evidence reports.", "body_left"),
        P("4. Apply the frozen set-wise ensemble to predict loss distribution and adjusted risk.", "body_left"),
        P("5. Promote, abstain, or request review under the declared coverage and risk policy.", "body_left"),
        H1("D", "Appendix: sealed-audit provenance receipt"),
        make_table([
            ["Audit item", "Frozen completion value"],
            ["Audit identifier", "003; one-time confirmatory synthetic audit"],
            ["Completion timestamp", audit["generatedAt"]],
            ["Successful / failed candidate fits", "6,720 / 0"],
            ["Advertisers / families", "140 / 7; 20 advertisers per family"],
            ["Result artifact SHA-256", audit_hash],
            ["Dataset SHA-256", audit["provenance"]["auditDatasetSha256"]],
            ["Protocol freeze SHA-256", audit["provenance"]["protocolFreezeSha256"]],
            ["Primary difference", "-1.456; two-sided 95% interval [-2.837, -0.479]"],
            ["Confirmatory checks", "9 of 9 passed"],
            ["Theorem-bound verification", "0 violations across 140 selected candidates"],
            ["Inference contract", "PyMC 6.2 FullRankADVI; NUTS excluded"],
            ["Operational deviation", "zero-based shard 0 was skipped by an argument-parser clamp; 18 missing businesses were recovered locally under an indexing-only patch before one-time scoring"],
            ["Scientific changes during recovery", "none: posterior, selector, tokens, endpoints, and hidden outcomes remained unchanged"],
        ], widths=[65 * mm, 97 * mm]),
        caption("Table D.1", "Immutable completion receipt for the sealed audit. Full machine-readable provenance is distributed with the research artifacts."),
        H1("E", "Appendix: external AMSS transport receipt"),
        H2("E.1", "Frozen external contract"),
        make_table([
            ["External-audit item", "Frozen or verified value"],
            ["Protocol", amss_result["version"]],
            ["Simulator", "AMSS 1.0.1; separately authored aggregate marketing system"],
            ["Businesses / candidate fits", f"{amss_result['cohort']['businesses']} / {amss_result['cohort']['candidateModels']:,}"],
            ["Decision actions", f"{amss_result['cohort']['decisionActions']:,}; four per candidate"],
            ["Evidence allocation", "25 businesses each: paid-social, search, television, or no experiment"],
            ["Visible / future horizon", f"{amss_contract['time']['visibleWeeks']} / {amss_contract['time']['futureActionWeeks']} weeks; {amss_contract['time']['postActionCarryoverWeeks']} carryover weeks"],
            ["Frozen selector", "RegretSet-MMM; 232 tokens; no AMSS retraining or reselection"],
            ["Posterior", "PyMC 6.2 FullRankADVI; 5,000 iterations; 256 draws; two primary seeds"],
            ["Primary loss", "scenario excess loss capped at 1; business risk = 65% mean + 35% P90"],
            ["Truth verification", f"{amss_acceptance['verification']['truthParts']} partitions; {amss_acceptance['verification']['truthRows']} outcomes"],
            ["Contract verification", f"{amss_acceptance['verification']['contractTests']}; deterministic reassembly"],
            ["Result SHA-256", amss_result_hash],
            ["Candidate-loss SHA-256", amss_candidate_hash],
        ], widths=[58 * mm, 104 * mm]),
        caption("Table E.1", "External AMSS transport contract and reproducibility receipt."),
        H2("E.2", "Random-valid aggregation and paired uncertainty"),
        P(
            "Every external business has 48 valid candidates. Uniform random valid selection is therefore evaluated exactly by assigning probability 1/48 to every candidate within every business. Pooling all 4,800 candidate risks gives equal weight to businesses and integrates over the candidate draw without Monte Carlo error. This yields mean, P90, and P95 capped risks of "
            f"{amss_random['mean']:.3f}, {amss_random['p90']:.3f}, and {amss_random['p95']:.3f}. The frozen audit receipt also stores the P90 and P95 across the 100 conditional expected random risks; those answer a different question and are not used for the random-selection tail in Table 8."
        ),
        P(
            f"The descriptive paired interval resamples businesses within each of the four 25-business evidence groups. With {amss_paired['resamples']:,} resamples and seed {amss_paired['seed']}, the RegretSet-MMM-minus-expected-random mean was {amss_paired['point']:.3f}, with 95% interval [{amss_paired['interval'][0]:.3f}, {amss_paired['interval'][1]:.3f}]. The statistic was added for interpretation after the frozen audit and is not presented as a predeclared confirmatory endpoint."
        ),
        H2("E.3", "Channel-adapter boundary"),
        P(
            "AMSS native search is mapped to the frozen nonbrand-search role and AMSS traditional television to the long-memory CTV role. The paid-social stress proxy is generated with AMSS's unchanged DefaultTraditionalMediaModule and mapped to the frozen paid-social role. Raw proxy naming and module provenance are preserved. The adapter makes no claim that AMSS implements a native social-platform auction or that traditional television is literally connected television."
        ),
    ])


def build() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    story: list[Flowable] = []
    add_title(story)
    add_introduction(story)
    add_related_work(story)
    add_decision_problem(story)
    add_regret_theory(story)
    add_simulator(story)
    add_candidate_models(story)
    add_tokens(story)
    add_learning(story)
    add_evidence_theory(story)
    add_validation(story)
    add_confirmatory_results(story)
    add_external_validation(story)
    add_interpretation(story)
    add_limitations(story)
    add_open_science(story)
    add_conclusion(story)
    add_references(story)
    add_appendices(story)
    RegretSetDoc(str(OUT)).build(story)
    print(OUT)


if __name__ == "__main__":
    build()
