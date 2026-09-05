#!/usr/bin/env python3
"""Build the FluxMMM V5D-SVI manuscript in conventional journal style.

The manuscript is generated from frozen research artifacts and the implemented
model contracts. Mathematical displays are rendered with Matplotlib's STIX
math engine and embedded at publication resolution.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(".flux-artifacts/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf" / "fluxmmm_v5d_svi_working_paper.pdf"
ASSETS = ROOT / "tmp" / "pdfs" / "fluxmmm_v5d_svi_journal_assets"
DEVELOPMENT_PATH = ROOT / "research" / "svi_score_v5d_svi" / "artifacts" / "svi-score-v5d-svi-development.json"
IMPORTANCE_PATH = ROOT / "research" / "svi_score_v5d_svi" / "artifacts" / "svi-score-v5d-svi-feature-importance.json"
SIMULATOR_PATH = ROOT / "research" / "score_v3" / "artifacts" / "simulator-audit-v3-summary.json"

PAGE_W, PAGE_H = A4
LEFT = 24 * mm
RIGHT = 24 * mm
TOP = 19 * mm
BOTTOM = 20 * mm
TEXT_W = PAGE_W - LEFT - RIGHT

BLACK = colors.HexColor("#111111")
MID = colors.HexColor("#4A4A4A")
LIGHT = colors.HexColor("#D6D6D6")
RULE = colors.HexColor("#8A8A8A")
PALE = colors.HexColor("#F4F4F4")
BLUE = colors.HexColor("#355C9A")
RED = colors.HexColor("#A8483C")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


DEVELOPMENT = read_json(DEVELOPMENT_PATH)
IMPORTANCE = read_json(IMPORTANCE_PATH)
SIMULATOR = read_json(SIMULATOR_PATH)

PILLAR_CONTRACT = [
    {
        "name": "Posterior decision quality",
        "groups": [
            "posterior-roi",
            "posterior-decision-safety",
            "posterior-reliability",
            "posterior-predictive",
        ],
        "question": "Are ROI levels, uncertainty, predictive coverage, and SVI stability usable for a decision?",
        "reproduce": "Compare seeds, ELBO drift, posterior coverage, ROI interval width, and implausible posterior mass.",
    },
    {
        "name": "Model specification fit",
        "groups": ["model-specification-and-interactions"],
        "question": "Which adstock, saturation, likelihood, calibration, and dynamic assumptions work in this context?",
        "reproduce": "Encode declared assumptions and interact them only with predeclared truth-blind diagnostics.",
    },
    {
        "name": "Temporal identification",
        "groups": ["temporal-evidence"],
        "question": "Can the data distinguish carryover and response across the campaign life cycle?",
        "reproduce": "Test whole-flight prediction, post-flight residuals, kernel distinguishability, and carryover support.",
    },
    {
        "name": "Predictive and structural adequacy",
        "groups": ["generalization", "structure"],
        "question": "Does the model generalize across time and spend while leaving defensible residual structure?",
        "reproduce": "Use rolling folds, low/high-spend regimes, residual patterns, variance shape, collinearity, and functional-form checks.",
    },
    {
        "name": "Causal and decision coherence",
        "groups": ["causal-robustness", "decision-coherence"],
        "question": "Do ROI and budget decisions survive anchors, confounder stress, placebos, and evidence deletion?",
        "reproduce": "Run anchor recovery when qualified, future-media placebos, confounder sensitivity, source conflict, and leave-source-out decisions.",
    },
]


def pillar_importance() -> list[dict]:
    by_group = {
        item["group"]: float(item["importance"])
        for item in IMPORTANCE["risk"]["groups"]
    }
    return [
        {
            **pillar,
            "importance": sum(by_group.get(group, 0.0) for group in pillar["groups"]),
        }
        for pillar in PILLAR_CONTRACT
    ]


def register_fonts() -> None:
    fonts = {
        "PaperSerif": "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "PaperSerifBold": "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "PaperSerifItalic": "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
        "PaperSerifBoldItalic": "/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf",
    }
    for name, path in fonts.items():
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily(
        "PaperSerif",
        normal="PaperSerif",
        bold="PaperSerifBold",
        italic="PaperSerifItalic",
        boldItalic="PaperSerifBoldItalic",
    )


register_fonts()
FONT = "PaperSerif"
FONT_BOLD = "PaperSerifBold"
FONT_ITALIC = "PaperSerifItalic"


def styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=sample["Title"],
            fontName=FONT_BOLD,
            fontSize=18,
            leading=21.5,
            alignment=TA_CENTER,
            textColor=BLACK,
            spaceAfter=7,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName=FONT_ITALIC,
            fontSize=10.2,
            leading=13,
            alignment=TA_CENTER,
            textColor=MID,
            spaceAfter=10,
        ),
        "author": ParagraphStyle(
            "Author",
            fontName=FONT,
            fontSize=10.5,
            leading=13,
            alignment=TA_CENTER,
            textColor=BLACK,
            spaceAfter=2,
        ),
        "date": ParagraphStyle(
            "Date",
            fontName=FONT,
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            textColor=MID,
            spaceAfter=12,
        ),
        "abstract_heading": ParagraphStyle(
            "AbstractHeading",
            fontName=FONT_BOLD,
            fontSize=10,
            leading=12,
            alignment=TA_CENTER,
            textColor=BLACK,
            spaceBefore=2,
            spaceAfter=4,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            fontName=FONT,
            fontSize=9.2,
            leading=11.8,
            alignment=TA_JUSTIFY,
            leftIndent=13 * mm,
            rightIndent=13 * mm,
            textColor=BLACK,
            spaceAfter=7,
        ),
        "keywords": ParagraphStyle(
            "Keywords",
            fontName=FONT,
            fontSize=8.7,
            leading=11,
            alignment=TA_LEFT,
            leftIndent=13 * mm,
            rightIndent=13 * mm,
            textColor=BLACK,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=sample["Heading1"],
            fontName=FONT_BOLD,
            fontSize=12.2,
            leading=14.5,
            textColor=BLACK,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=sample["Heading2"],
            fontName=FONT_BOLD,
            fontSize=10.4,
            leading=12.5,
            textColor=BLACK,
            spaceBefore=8,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3",
            parent=sample["Heading3"],
            fontName=FONT_ITALIC,
            fontSize=9.8,
            leading=12,
            textColor=BLACK,
            spaceBefore=6,
            spaceAfter=2,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName=FONT,
            fontSize=9.45,
            leading=12.25,
            alignment=TA_JUSTIFY,
            textColor=BLACK,
            spaceAfter=5.1,
            allowWidows=0,
            allowOrphans=0,
        ),
        "body_left": ParagraphStyle(
            "BodyLeft",
            parent=sample["BodyText"],
            fontName=FONT,
            fontSize=9.45,
            leading=12.25,
            alignment=TA_LEFT,
            textColor=BLACK,
            spaceAfter=5.1,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=sample["BodyText"],
            fontName=FONT,
            fontSize=9.25,
            leading=12,
            leftIndent=5.5 * mm,
            firstLineIndent=-3.2 * mm,
            bulletIndent=0,
            textColor=BLACK,
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=sample["BodyText"],
            fontName=FONT,
            fontSize=8,
            leading=10.2,
            alignment=TA_LEFT,
            textColor=BLACK,
            spaceBefore=3,
            spaceAfter=7,
        ),
        "table": ParagraphStyle(
            "TableCell",
            fontName=FONT,
            fontSize=7.7,
            leading=9.5,
            textColor=BLACK,
            alignment=TA_LEFT,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            fontName=FONT_BOLD,
            fontSize=7.7,
            leading=9.5,
            textColor=BLACK,
            alignment=TA_LEFT,
        ),
        "definition": ParagraphStyle(
            "Definition",
            fontName=FONT,
            fontSize=9.25,
            leading=12.1,
            alignment=TA_JUSTIFY,
            leftIndent=5 * mm,
            rightIndent=5 * mm,
            textColor=BLACK,
            spaceBefore=4,
            spaceAfter=5,
        ),
        "proof": ParagraphStyle(
            "Proof",
            fontName=FONT,
            fontSize=9.1,
            leading=11.9,
            alignment=TA_JUSTIFY,
            leftIndent=5 * mm,
            rightIndent=5 * mm,
            textColor=BLACK,
            spaceAfter=6,
        ),
        "reference": ParagraphStyle(
            "Reference",
            fontName=FONT,
            fontSize=8.1,
            leading=10.2,
            leftIndent=6 * mm,
            firstLineIndent=-6 * mm,
            alignment=TA_LEFT,
            textColor=BLACK,
            spaceAfter=3,
        ),
        "note": ParagraphStyle(
            "Note",
            fontName=FONT,
            fontSize=8.4,
            leading=10.7,
            alignment=TA_JUSTIFY,
            textColor=MID,
            leftIndent=5 * mm,
            rightIndent=5 * mm,
            spaceAfter=5,
        ),
    }


S = styles()


class JournalDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title="Learning to Select Decision-Useful Marketing Mix Models",
            author="Gustavo Bramao",
            subject="FluxMMM V5D-SVI methodological working paper",
            keywords="marketing mix modeling, Bayesian inference, causal model selection, decision-focused learning, economic regret, evidence attribution",
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


def page_header(canvas, doc: JournalDoc) -> None:
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.35)
        canvas.line(LEFT, PAGE_H - 13.5 * mm, PAGE_W - RIGHT, PAGE_H - 13.5 * mm)
        canvas.setFont(FONT_ITALIC, 7.5)
        canvas.setFillColor(MID)
        canvas.drawString(LEFT, PAGE_H - 10.7 * mm, "Bramao: Learning to Select Decision-Useful MMMs")
        canvas.drawRightString(PAGE_W - RIGHT, PAGE_H - 10.7 * mm, "Working paper")
        canvas.setFont(FONT, 8)
        canvas.setFillColor(BLACK)
        canvas.drawCentredString(PAGE_W / 2, 10.5 * mm, str(doc.page))
    canvas.restoreState()


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def H1(number: str, text: str) -> Paragraph:
    return Paragraph(f"{number}&nbsp;&nbsp;{text}", S["h1"])


def H2(number: str, text: str) -> Paragraph:
    return Paragraph(f"{number}&nbsp;&nbsp;{text}", S["h2"])


def H3(text: str) -> Paragraph:
    return Paragraph(text, S["h3"])


def bullet(text: str) -> Paragraph:
    return Paragraph(f"<bullet>&bull;</bullet>{text}", S["bullet"])


def definition(label: str, text: str) -> Paragraph:
    return Paragraph(f"<b>{label}.</b> {text}", S["definition"])


def proof(text: str) -> Paragraph:
    return Paragraph(f"<i>Proof.</i> {text} &#9633;", S["proof"])


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def make_table(
    rows: Sequence[Sequence[object]],
    widths: Sequence[float] | None = None,
    repeat_rows: int = 1,
    right_columns: Iterable[int] = (),
) -> Table:
    right = set(right_columns)
    converted = []
    for r, row in enumerate(rows):
        converted_row = []
        for c, value in enumerate(row):
            style = S["table_head"] if r == 0 else S["table"]
            if c in right:
                style = ParagraphStyle(
                    f"TableRight-{r}-{c}",
                    parent=style,
                    alignment=TA_RIGHT,
                )
            converted_row.append(Paragraph(str(value), style))
        converted.append(converted_row)
    if widths is None:
        widths = [TEXT_W / len(rows[0])] * len(rows[0])
    table = Table(converted, colWidths=list(widths), repeatRows=repeat_rows, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, BLACK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.45, BLACK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, BLACK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def caption(label: str, text: str) -> Paragraph:
    return P(f"<b>{label}.</b> {text}", "caption")


def math_asset(lines: Sequence[str], number: str, width_mm: float = 150) -> Image:
    key = hashlib.sha256(("|".join(lines) + number).encode()).hexdigest()[:14]
    path = ASSETS / f"equation-{key}.png"
    if not path.exists():
        height = max(0.46, 0.31 * len(lines) + 0.12)
        fig = plt.figure(figsize=(7.2, height), dpi=320)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        for index, line in enumerate(lines):
            y = 1 - (index + 0.5) / len(lines)
            ax.text(
                0.49,
                y,
                f"${line}$",
                ha="center",
                va="center",
                fontsize=12.2,
                color="#111111",
                math_fontfamily="stix",
            )
        ax.text(0.985, 0.5, f"({number})", ha="right", va="center", fontsize=9.5, family="serif")
        fig.savefig(path, dpi=320, transparent=True, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    with PILImage.open(path) as im:
        ratio = im.height / im.width
    width = width_mm * mm
    return Image(str(path), width=width, height=width * ratio, hAlign="CENTER")


def equation(expr: str, number: str, width_mm: float = 150) -> Image:
    return math_asset([expr], number, width_mm)


def multiline_equation(lines: Sequence[str], number: str, width_mm: float = 150) -> Image:
    return math_asset(lines, number, width_mm)


def save_pipeline_figure() -> Path:
    path = ASSETS / "figure-1-pipeline.png"
    fig, ax = plt.subplots(figsize=(9.2, 2.75), dpi=260)
    ax.set_xlim(-0.05, 6.05)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = [
        ("Synthetic\nadvertiser", "Hidden truth"),
        ("48 Bayesian\nMMMs", "Truth-blind fits"),
        ("Posterior\nevidence", "Diagnostics"),
        ("Budget\nactions", "Four scenarios"),
        ("Economic\nlabels", "Truth after action"),
        ("Risk\nselector", "New advertisers"),
    ]
    for i, (top, bottom) in enumerate(labels):
        x = i + 0.5
        ax.add_patch(plt.Rectangle((x - 0.43, 0.27), 0.86, 0.49, facecolor="white", edgecolor="#222222", lw=0.8))
        ax.text(x, 0.59, top, ha="center", va="center", fontsize=7.8, linespacing=1.05, fontweight="bold", family="serif")
        ax.text(x, 0.38, bottom, ha="center", va="center", fontsize=6.8, color="#555555", family="serif")
        if i < len(labels) - 1:
            ax.annotate("", xy=(i + 1.065, 0.515), xytext=(i + 0.935, 0.515), arrowprops=dict(arrowstyle="->", lw=0.8, color="#222222"))
    ax.text(3.0, 0.075, "Hidden truth never enters candidate fitting or deployment tokens", ha="center", fontsize=7.8, style="italic", family="serif")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_dag_figure() -> Path:
    path = ASSETS / "figure-2-dag.png"
    if path.exists():
        return path
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=260)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    nodes = {
        "U": (1.2, 4.8, "Latent demand\nand market state"),
        "P": (4.1, 4.8, "Commercial planning\nand promotions"),
        "X": (2.5, 2.8, "Channel spend\nand delivery"),
        "A": (5.2, 2.8, "Adstock and\nsaturation"),
        "B": (7.7, 4.8, "Baseline demand\nand controls"),
        "Y": (7.7, 1.25, "Observed outcome"),
        "E": (4.2, 0.55, "Experiment or\nindustry evidence"),
    }
    for key, (x, y, label) in nodes.items():
        ax.add_patch(plt.Rectangle((x - 0.82, y - 0.37), 1.64, 0.74, facecolor="white", edgecolor="#222222", lw=0.9))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.3, family="serif")
    arrows = [
        ("U", "P"), ("U", "X"), ("U", "B"), ("P", "X"), ("P", "B"),
        ("X", "A"), ("A", "Y"), ("B", "Y"), ("E", "A"),
    ]
    for source, target in arrows:
        x1, y1, _ = nodes[source]
        x2, y2, _ = nodes[target]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333", shrinkA=26, shrinkB=26))
    ax.text(0.5, 0.05, "The causal challenge: latent demand and planning jointly affect media and outcome.", fontsize=8.4, style="italic", family="serif")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_evidence_figure() -> Path:
    path = ASSETS / "figure-3-evidence-attribution.png"
    if path.exists():
        return path
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 2.85), dpi=260, gridspec_kw={"width_ratios": [1.05, 1.45]})
    ax = axes[0]
    ax.axis("off")
    ax.text(0.5, 0.94, r"$H_{post}=H_D+H_E+H_B+H_R$", ha="center", va="center", fontsize=14, math_fontfamily="stix")
    labels = [("Data", "$H_D$"), ("Experiments", "$H_E$"), ("Benchmarks", "$H_B$"), ("Regularization", "$H_R$")]
    for i, (label, symbol) in enumerate(labels):
        y = 0.74 - i * 0.18
        ax.add_patch(plt.Rectangle((0.10, y - 0.055), 0.80, 0.11, facecolor="#F5F5F5", edgecolor="#555555", lw=0.6))
        ax.text(0.18, y, label, ha="left", va="center", fontsize=8.5, family="serif")
        ax.text(0.82, y, symbol, ha="right", va="center", fontsize=10.5, math_fontfamily="stix")
    ax = axes[1]
    channels = ["Paid social", "Search", "TV"]
    shares = [[0.22, 0.61, 0.13, 0.04], [0.55, 0.00, 0.38, 0.07], [0.17, 0.70, 0.08, 0.05]]
    colors_ = ["#3B5B92", "#6E9E73", "#C89552", "#8A8A8A"]
    left = [0, 0, 0]
    for k, source in enumerate(["Data", "Experiment", "Benchmark", "Regularization"]):
        values = [row[k] for row in shares]
        ax.barh(channels, values, left=left, color=colors_[k], label=source, height=0.48)
        left = [left[i] + values[i] for i in range(3)]
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of local ROI uncertainty reduction", fontsize=8, family="serif")
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=7.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_design_figure() -> Path:
    path = ASSETS / "figure-4-design.png"
    if path.exists():
        return path
    fig, ax = plt.subplots(figsize=(8.5, 2.8), dpi=260)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    blocks = [
        (0.3, 2.4, 5.5, 0.9, "500 advertiser worlds", "Five mechanism families"),
        (0.3, 0.8, 3.0, 0.9, "420 development", "20,160 fitted candidates"),
        (3.6, 0.8, 2.2, 0.9, "80 sealed audit", "3,840 untouched labels"),
        (6.3, 2.4, 3.3, 0.9, "48 candidates per business", "24 specifications x 2 evidence arms"),
        (6.3, 0.8, 3.3, 0.9, "Two-axis cross-fitting", "unseen business and parameter region"),
    ]
    for x, y, w, h, title, note in blocks:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="white", edgecolor="#222222", lw=0.8))
        ax.text(x + w / 2, y + 0.57, title, ha="center", fontsize=9, fontweight="bold", family="serif")
        ax.text(x + w / 2, y + 0.27, note, ha="center", fontsize=7.7, color="#555555", family="serif")
    ax.annotate("", xy=(3.3, 1.25), xytext=(2.1, 2.4), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(4.7, 1.7), xytext=(4.0, 2.4), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(6.3, 2.85), xytext=(5.8, 2.85), arrowprops=dict(arrowstyle="->", lw=0.8))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_importance_figure() -> Path:
    path = ASSETS / "figure-5-importance.png"
    if path.exists():
        return path
    rows = IMPORTANCE["top50RiskFeatures"][:15]
    labels = [row["feature"].replace("diagnostic:", "").replace("interaction:", "int: ").replace("svi:", "SVI: ") for row in rows][::-1]
    values = [row["importance"] for row in rows][::-1]
    signs = [row["meanCoefficient"] for row in rows][::-1]
    bar_colors = ["#355C9A" if sign < 0 else "#A8483C" for sign in signs]
    fig, ax = plt.subplots(figsize=(8.5, 5.3), dpi=260)
    ax.barh(range(len(values)), values, color=bar_colors, height=0.66)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=7.4, family="serif")
    ax.set_xlabel("normalized mean absolute standardized coefficient", fontsize=8.2, family="serif")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="x", color="#DDDDDD", lw=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.99, 1.03, "blue: predicts lower loss   red: predicts higher loss", transform=ax.transAxes, ha="right", fontsize=7.5, family="serif")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_results_figure() -> Path:
    path = ASSETS / "figure-6-results.png"
    if path.exists():
        return path
    comparison = DEVELOPMENT["comparison"]
    cells = ["Baseline", "Posterior features", "Posterior safety", "Full method"]
    keys = ["baseline", "featureOnly", "safetyOnly", "fullSvi"]
    objective = [comparison[key]["primary"]["objective"] for key in keys]
    champion = [comparison[key]["proxy"]["riskChampionMeanExcessLoss"] for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1), dpi=260)
    for ax, values, title in zip(axes, [objective, champion], ["Composite development objective", "Champion mean excess loss"]):
        ax.bar(range(4), values, color=["#777777", "#355C9A", "#777777", "#355C9A"], width=0.62)
        ax.set_xticks(range(4), labels=cells, rotation=23, ha="right", fontsize=7.2, family="serif")
        ax.set_title(title, fontsize=9, family="serif")
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.tick_params(axis="y", labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7.2, family="serif")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_selector_figure() -> Path:
    path = ASSETS / "figure-5-two-head-selector-v2.png"
    if path.exists():
        return path
    fig, ax = plt.subplots(figsize=(8.8, 3.45), dpi=260)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, note: str, fill: str = "white") -> None:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor="#222222", lw=0.8))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=8.3, fontweight="bold", family="serif")
        ax.text(x + w / 2, y + h * 0.30, note, ha="center", va="center", fontsize=7.0, color="#555555", family="serif")

    box(0.2, 1.45, 2.0, 1.35, "136 named tokens", "115 diagnostic/specification\n+ 21 SVI posterior")
    box(2.8, 1.45, 1.65, 1.35, "Fold transform", "standardize within fold\nclip to [-8, 8]")
    box(5.15, 2.45, 1.65, 1.05, "Mean-loss head", "Huber + ranking")
    box(5.15, 0.75, 1.65, 1.05, "P90-loss head", "pinball loss")
    box(7.55, 1.45, 2.0, 1.35, "Decision risk", "65% mean + 35% P90\nthen rank safe candidates")
    ax.annotate("", xy=(2.75, 2.12), xytext=(2.22, 2.12), arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("", xy=(5.08, 2.95), xytext=(4.47, 2.25), arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("", xy=(5.08, 1.27), xytext=(4.47, 1.98), arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("", xy=(7.48, 2.25), xytext=(6.82, 2.82), arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("", xy=(7.48, 1.85), xytext=(6.82, 1.28), arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.text(5.0, 0.18, "Both heads are linear and auditable; no hidden truth enters the token vector.", ha="center", fontsize=8, style="italic", family="serif")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_pillar_figure() -> Path:
    path = ASSETS / "figure-6-five-pillars-v2.png"
    if path.exists():
        return path
    rows = pillar_importance()[::-1]
    labels = [row["name"] for row in rows]
    values = [100 * row["importance"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 3.25), dpi=260)
    ax.barh(range(len(values)), values, color="#355C9A", height=0.58)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=8.2, family="serif")
    ax.set_xlabel("share of development-set coefficient importance (%)", fontsize=8.2, family="serif")
    ax.set_xlim(0, max(values) * 1.22)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="x", color="#DDDDDD", lw=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for index, value in enumerate(values):
        ax.text(value + 0.45, index, f"{value:.1f}%", va="center", fontsize=8, family="serif")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def image_flow(path: Path, width_mm: float = 158) -> Image:
    with PILImage.open(path) as im:
        ratio = im.height / im.width
    width = width_mm * mm
    return Image(str(path), width=width, height=width * ratio, hAlign="CENTER")


def add_title(story: list[Flowable]) -> None:
    story.extend([
        Spacer(1, 4 * mm),
        P("Learning to Select Decision-Useful Marketing Mix Models", "title"),
        P("A cross-advertiser, decision-focused framework with Bayesian posterior diagnostics and explicit evidence attribution", "subtitle"),
        P("Gustavo Bramao", "author"),
        P("Independent Researcher | September 2026 | Working paper", "date"),
        HRFlowable(width="100%", thickness=0.5, color=BLACK, spaceBefore=1, spaceAfter=8),
        P("Abstract", "abstract_heading"),
        P(
            "Marketing mix modeling (MMM) is usually evaluated through predictive fit, residual diagnostics, calibration agreement, or analyst judgment, even though its purpose is to support budget decisions. This paper formulates MMM specification selection as a supervised decision-risk problem across advertisers. We generate 500 heterogeneous synthetic direct-to-consumer business worlds with known causal response surfaces, endogenous media planning, channel-specific carryover and saturation, and imperfect external evidence. For each business, 48 truth-blind Bayesian MMM candidates are fitted, producing 24,000 posterior fits. Each candidate chooses allocations under four predeclared decision scenarios; hidden causal truth is revealed only after the action, yielding scenario-specific economic loss. A cross-fitted selector represents every fitted candidate by 136 named, truth-blind diagnostic tokens and uses two auditable prediction heads to estimate conditional mean and P90 excess loss. We call the frozen implementation V5D-SVI because candidate posteriors and posterior diagnostics are obtained with FullRank automatic differentiation variational inference. The inference engine is not the contribution. The contribution is an open protocol for learning what makes a causal MMM decision-useful and for transferring that rule to advertisers whose causal truth is unobserved. We prove a finite-candidate inequality linking selected-model excess loss to regret-weighted oracle misrankings and formalize continuous attribution of local ROI information to observational data, experiments, benchmarks, and regularization. The empirical evaluation finds useful economically weighted ordering signal but rejects automatic top-one promotion under the current safety policy. Consequently, the study establishes a research method and falsifiable benchmark, not a production or state-of-the-art claim.",
            "abstract",
        ),
        P("<b>Keywords:</b> marketing mix modeling; causal model selection; Bayesian inference; decision-focused learning; economic regret; evidence attribution; budget optimization; synthetic benchmark", "keywords"),
        HRFlowable(width="100%", thickness=0.5, color=BLACK, spaceBefore=0, spaceAfter=8),
    ])


def add_introduction(story: list[Flowable]) -> None:
    story.extend([
        H1("1", "Introduction"),
        P(
            "Marketing mix models estimate how marketing activity contributes to business outcomes and translate those estimates into budget recommendations. Their central scientific problem is not merely prediction. Media spending is planned rather than randomized, channels are correlated, advertising effects are delayed and nonlinear, and evidence from experiments or industry benchmarks is incomplete and imperfectly transported. Multiple models can therefore fit the same historical outcome while implying materially different return on investment (ROI), marginal returns, and allocations [1-5].",
        ),
        P(
            "Most MMM workflows select a preferred model by combining predictive error, decomposition plausibility, calibration, residual tests, and analyst review. Those ingredients are valuable, but their relative importance is usually fixed by convention. A model can predict well while assigning demand to the wrong channel; it can agree with a benchmark because the benchmark dominated its prior; or it can recover average ROI while producing a poor marginal-return surface. A universal hand-crafted score is therefore difficult to justify.",
        ),
        P(
            "FluxMMM begins from the decision. In simulation, one may ask what profit would have been obtained had a candidate model's recommendation been followed. This converts model selection into a supervised learning problem: learn which truth-blind diagnostics and assumptions predict low economic loss across a population of advertisers. When a new advertiser arrives, causal truth remains unavailable, but the learned mapping can rank candidate MMMs using only quantities available at deployment.",
        ),
        image_flow(save_pipeline_figure()),
        caption("Figure 1", "End-to-end research design. Candidate fitting and deployment tokens are truth-blind. Hidden truth labels only the economic consequence of a selected action during simulation."),
        H2("1.1", "Contributions"),
        bullet("A cross-advertiser formulation of MMM selection as prediction and ranking of downstream economic loss rather than a fixed validation score."),
        bullet("A reproducible population of heterogeneous advertiser worlds containing endogenous spend, channel-specific carryover and saturation, external evidence, and four declared budget decisions."),
        bullet("A truth firewall: synthetic truth is prohibited from candidate fitting, diagnostic construction, safety screening, and deployment scoring; it is opened only to label decisions."),
        bullet("A finite-candidate inequality that relates the excess loss of the selected MMM to regret-weighted misrankings of the in-pool oracle."),
        bullet("A continuous local information decomposition that attributes ROI uncertainty reduction to observational data, experiments, benchmarks, and weak regularization, while correcting observational credit for channel collinearity."),
        bullet("An open-source implementation, versioned contracts, deterministic artifacts, leakage tests, and explicit negative results intended for independent extension."),
        H2("1.2", "Scope of the claim"),
        P(
            "To our knowledge, the reviewed MMM literature does not train a cross-advertiser model selector directly on the realized economic consequences of complete fitted MMMs while exposing source-specific evidence attribution as model-selection tokens. This is a qualified novelty statement, not proof of priority. The current evidence is developmental and synthetic. It supports the research program; it does not establish external validity, production readiness, or global optimality.",
        ),
    ])


def add_related_work(story: list[Flowable]) -> None:
    story.extend([
        H1("2", "Related work and positioning"),
        H2("2.1", "Bayesian marketing mix modeling"),
        P(
            "Bayesian MMM provides a coherent language for carryover, nonlinear response, prior information, hierarchical structure, and uncertainty [1]. Time-varying coefficients can represent gradual changes in media productivity [4], while ROI reparameterization permits experiment evidence to enter through interpretable priors [5]. Robyn and Meridian have broadened access to automated candidate generation, calibration, diagnostics, and optimization [6-8]. These methods primarily concern how to specify, estimate, and evaluate an MMM. FluxMMM focuses on the meta-problem of selecting among complete fitted specifications when familiar diagnostics disagree.",
        ),
        H2("2.2", "Causal risk in observational media data"),
        P(
            "Paid search illustrates the identification problem: spend increases when latent demand increases, so a naive coefficient can attribute pre-existing demand to advertising [3]. More generally, media timing, promotions, brand demand, creative quality, and commercial planning can enter both spend and the outcome error. Residual plots can detect some misspecification but cannot prove exogeneity because unobserved components are not directly testable. Experiments provide independent information, but power, overlap, noncompliance, attribution windows, and transportability remain consequential [19,20].",
        ),
        H2("2.3", "Decision-focused learning and model selection"),
        P(
            "Decision-focused learning evaluates predictions through the optimization decisions they induce [10,11]. Learning-to-rank and structured prediction similarly prioritize correct ordering near the selected action [12]. FluxMMM applies this logic one level higher: each object being ranked is an entire causal MMM, and its label is the economic loss of the budget decision implied by that model. Cross-fitting by advertiser and parameter region follows the broader principle that model-selection performance must be estimated outside the data used to tune the selection rule [17].",
        ),
        H2("2.4", "Variational inference"),
        P(
            "FullRank automatic differentiation variational inference (ADVI) supplies posterior summaries for the large candidate library [13,14]. It approximates the posterior with a transformed multivariate Gaussian and optimizes an evidence lower bound. This is computationally practical and preserves covariance information unavailable from a point estimate, but it can underrepresent tails or distort multimodal geometry. NUTS remains the confirmatory reference for finalists [15]. The paper does not claim that SVI is intrinsically superior to MAP; posterior inference is infrastructure for the larger decision-risk experiment.",
        ),
    ])


def add_problem(story: list[Flowable]) -> None:
    story.extend([
        H1("3", "Formal problem"),
        H2("3.1", "Advertisers, candidate models, and observables"),
        P(
            "Let b index advertisers, c index candidate MMM specifications, j index media channels, t index time, and k index declared decision scenarios. Advertiser b supplies observed time-series data D_b, optional experiment evidence E_b, benchmark evidence B_b, and a feasible decision contract A_bk. Candidate c defines a response transformation, baseline, likelihood, prior route, and optional advanced structure. Its parameter posterior is",
        ),
        equation(r"p(\theta_{bc}\mid D_b,E_b,B_b,c)\ \propto\ p(D_b\mid\theta_{bc},c)\,p(E_b\mid\theta_{bc},c)\,p(\theta_{bc}\mid B_b,c)", "1"),
        P(
            "where an experiment may instead enter the prior, depending on the candidate contract. The fitted model yields a deployment-visible token vector z_bc containing validation diagnostics, posterior summaries, evidence characteristics, and specification indicators. Business identifier, generator family, split label, hidden truth, realized economic loss, and truth errors are forbidden inputs.",
        ),
        H2("3.2", "Posterior decision"),
        P(
            "For a feasible allocation a in A_bk, candidate c estimates incremental outcome and profit. FluxMMM selects one action from the posterior expected surface:",
        ),
        equation(r"\widehat a_{bck}=\arg\max_{a\in\mathcal{A}_{bk}}\;\mathbb{E}_{q_{bc}}\!\left[m_b\,\Delta Y_{bc}(a,\theta)-\mathrm{Spend}_b(a)\right]", "2"),
        P(
            "with gross margin m_b and approximate posterior q_bc. A single action is optimized from posterior expectations; the procedure does not optimize separately inside each draw and then average draw-specific optima, which would create an artificial information advantage.",
        ),
        H2("3.3", "Economic loss"),
        P(
            "The simulator evaluates the selected action on the hidden causal response surface. Let a*_bk be the true profit-maximizing action under the same constraints. Scenario loss is the foregone true profit normalized by a predeclared business scale s_bk:",
        ),
        multiline_equation([
            r"g_{bck}=\max\{0,\ \Pi^{\star}_{bk}-\Pi^{\mathrm{true}}_{bk}(\widehat a_{bck})\}",
            r"\ell_{bck}=g_{bck}\,/\,s_{bk}",
            r"L_{bc}=0.70\,K^{-1}\sum_{k=1}^{K}\ell_{bck}+0.30\,\max_k\ell_{bck}",
        ], "3"),
        P(
            "The 70/30 aggregation represents a declared business preference: 70% routine performance and 30% worst-scenario protection. It is not a statistical theorem. All scenario losses are retained so this utility can be changed or learned in future work.",
        ),
        H2("3.4", "Safe oracle and excess loss"),
        P(
            "Let S_b be the candidates permitted by a fixed safety policy. The in-pool economic oracle and candidate excess loss are",
        ),
        multiline_equation([
            r"c_b^{\star}=\arg\min_{c\in\mathcal{S}_b}L_{bc}",
            r"\Delta_{bc}=\max\{0,L_{bc}-L_{bc_b^{\star}}\}/s_{\mathrm{econ}}",
        ], "4"),
        P(
            "The frozen V5D-SVI release sets <i>s</i><sub>econ</sub> = 1 and leaves &#916; uncapped. This oracle is relative to the candidate library and safety policy. It is neither the true data-generating specification nor a claim of globally optimal causal structure.",
        ),
        H2("3.5", "Learned decision risk"),
        P(
            "The selector predicts conditional mean excess loss and its conditional 90th percentile. The deployment risk is",
        ),
        equation(r"\widehat R_{bc}=(1-\rho)\widehat\mu_{bc}+\rho\widehat q_{0.90,bc},\qquad \rho=0.35", "5"),
        P(
            "Lower risk is better. The token coefficients are learned; &#961; is a subjective business-protection parameter. At &#961; = 0.35, expected loss receives 65% weight and P90 loss receives 35% weight.",
        ),
    ])


def add_regret_theory(story: list[Flowable]) -> None:
    story.extend([
        H1("4", "Economic-regret theory"),
        definition("Definition 1 (normalized excess regret)", "For a candidate-independent scale <i>s</i><sub>b</sub> &gt; 0, define capped normalized regret as &#948;&#771;<sub>bc</sub> = min{&#916;<sub>bc</sub>/<i>s</i><sub>b</sub>, 1}. The cap gives a bounded theoretical endpoint; the empirical study continues to report uncapped loss separately."),
        definition("Definition 2 (risk selection)", "For a finite safe set <i>S</i><sub>b</sub> and learned risk function <i>f</i>, let c&#770;<sub>b</sub> be the candidate with minimum predicted risk. A competitor is oracle-misranked when its predicted risk is no greater than the predicted risk of c*<sub>b</sub>."),
        H2("4.1", "Finite-candidate regret bound"),
        P("The following elementary result explains why ranking errors should be weighted by their economic consequence rather than counted equally."),
        definition("Proposition 1", "For any advertiser b, finite nonempty safe set S_b, nonnegative normalized regrets, and deterministic risk ranking,"),
        equation(r"\widetilde\Delta_{b\widehat c_b}\ \leq\ \sum_{c\in\mathcal{S}_b\setminus\{c_b^{\star}\}}\widetilde\Delta_{bc}\,\mathbf{1}\!\left[\widehat R_{bc}\leq\widehat R_{bc_b^{\star}}\right]", "6"),
        proof(
            "If the selected candidate equals the oracle, the left side is zero and the inequality holds. Otherwise, selection implies that the selected candidate's predicted risk is no greater than the oracle's predicted risk. The right side therefore contains the selected candidate's own nonnegative regret as one summand, and every additional summand is nonnegative."
        ),
        P(
            "The bound is exact but may be loose because it sums all candidates ranked ahead of the oracle. It does not prove cross-advertiser generalization. It states a narrower and useful fact: avoiding economically costly oracle misrankings controls an upper bound on the regret of the selected model.",
        ),
        H2("4.2", "Smooth surrogate"),
        P("Let <i>u</i> be the oracle's predicted risk minus a competitor's predicted risk. The indicator of a misranking is bounded by a scaled softplus:"),
        multiline_equation([
            r"\mathbf{1}[u\geq0]\ \leq\ \log(1+e^u)/\log 2",
            r"\widetilde\Delta_{b\widehat c_b}\ \leq\ \sum_{c\neq c_b^{\star}}\widetilde\Delta_{bc}\,\frac{\log\!\left(1+e^{\widehat R_{bc_b^{\star}}-\widehat R_{bc}}\right)}{\log 2}",
        ], "7"),
        P(
            "This provides a differentiable, regret-weighted ranking objective. The repository contains an exact all-oracle-pairs implementation and a numerical audit of (6) and (7). The frozen V5D-SVI learner, however, is not trained on the complete capped expression. It uses uncapped excess-loss regression and one hardest oracle-versus-challenger comparison per advertiser per epoch. Proposition 1 therefore motivates and audits the selector but must not be described as the exact V5D-SVI training loss.",
        ),
        H2("4.3", "Why a wrong winner is economically different from a small ranking error"),
        P(
            "Suppose candidates A, B, and C have excess losses 0, 0.02, and 0.80. Misranking B ahead of A is a minor economic mistake; misranking C ahead of A is severe. Ordinary pairwise accuracy assigns both one error. Equation (6) assigns gaps 0.02 and 0.80. The theory therefore aligns ranking pressure with the objective of MMM: avoid model-selection mistakes that materially change decisions.",
        ),
    ])


def add_evidence_theory(story: list[Flowable]) -> None:
    story.extend([
        H1("5", "Evidence attribution and identification"),
        H2("5.1", "Continuous source decomposition"),
        P(
            "A posterior may be precise because the observational time series is informative, because an experiment or benchmark is strong, or because regularization is strong. These are not equivalent causal claims. FluxMMM decomposes the local posterior precision into four positive semidefinite curvature components:",
        ),
        equation(r"H_{\mathrm{post}}=H_D+H_E+H_B+H_R", "8"),
        P(
            "Here <i>H</i><sub>D</sub> is observational-likelihood curvature, <i>H</i><sub>E</sub> experiment curvature, <i>H</i><sub>B</sub> benchmark-prior curvature, and <i>H</i><sub>R</sub> weak-regularization curvature. Let <i>h</i><sub>j</sub>(&#952;) be channel <i>j</i>'s ROI, <i>g</i><sub>j</sub> its local gradient, and &#931; = <i>H</i><sub>post</sub><super>-1</super>. The source-attribution share is",
        ),
        equation(r"A_{s,j}=\frac{g_j^{\mathrm{T}}\Sigma H_s\Sigma g_j}{g_j^{\mathrm{T}}\Sigma g_j},\qquad s\in\{D,E,B,R\}", "9"),
        definition("Proposition 2 (exhaustive local attribution)", "If <i>H</i><sub>post</sub> is positive definite and every <i>H</i><sub>s</sub> is positive semidefinite, then <i>A</i><sub>s,j</sub> is nonnegative and the four shares sum to one."),
        proof(
            "Each numerator is a quadratic form in a positive semidefinite matrix and is therefore nonnegative. Summing the four numerators replaces the source-specific curvature by their sum, the posterior curvature. Because the posterior covariance is its inverse, the summed numerator reduces exactly to the denominator."
        ),
        image_flow(save_evidence_figure(), 154),
        caption("Figure 2", "Evidence attribution. The stacked bars are illustrative; production output reports continuous shares for each fitted channel. A descriptive label may summarize the largest share, but it receives no independent score."),
        H2("5.2", "Conditional observational information"),
        P(
            "Raw column magnitude is not channel identification. If two media columns move together, each can have a large sum of squares while the data contain little information about their separate coefficients. FluxMMM projects the channel ROI direction on all remaining regressors and measures the residual information. This is equivalent to the relevant Schur-complement logic. Observational attribution <i>A</i><sub>D,j</sub> is multiplied by this conditional-separation share before receiving identification credit.",
        ),
        definition("Proposition 3 (collinearity non-identification)", "If two transformed media columns satisfy <i>X</i><sub>1</sub> = <i>X</i><sub>2</sub>, then the likelihood depends on &#946;<sub>1</sub> and &#946;<sub>2</sub> only through &#946;<sub>1</sub> + &#946;<sub>2</sub>. Separate channel ROIs are not identified by the observational likelihood."),
        proof(
            "Substituting the equality of the two media columns into the linear predictor shows that their contribution depends only on the sum of the two coefficients. Any coefficient pair with the same sum produces the same likelihood. A channel-specific split must therefore come from external evidence, regularization, or another identifying assumption."
        ),
        H2("5.3", "Evidence quality, conflict, and dependence"),
        P(
            "Influence is not validity. FluxMMM separately records evidence quality, standardized conflict, and decision dependence. Experiment quality combines independence, precision, relevance, and transportability. Benchmark quality combines source confidence, channel and tactic match, precision, and transportability. Conflict compares the external source with a leave-source-out observational estimate:",
        ),
        equation(r"C_{s,j}=\frac{|\mu_{D,-s,j}-\mu_{s,j}|}{\sqrt{\sigma^2_{D,-s,j}+\sigma^2_{s,j}}}", "10"),
        P(
            "Decision dependence is measured by refitting without the source and recording changes in ROI location, interval width, contribution, probability of incremental allocation, and marginal-profit index. A source can dominate an ROI estimate yet leave the budget decision unchanged; conversely, a small ROI shift near the break-even threshold can change the allocation substantially.",
        ),
        H2("5.4", "Source sensitivity and bias transmission"),
        definition("Proposition 4 (local source sensitivity)", "In the Gaussian local approximation with source-centered information H_s mu_s, the posterior mean is mu_post=H_post^{-1} sum_s H_s mu_s and its sensitivity to source center mu_s is H_post^{-1}H_s."),
        equation(r"\frac{\partial\mu_{\mathrm{post}}}{\partial\mu_s}=H_{\mathrm{post}}^{-1}H_s", "11"),
        definition("Corollary 1 (source-bias transmission)", "If source s is biased by vector delta_s, the induced first-order posterior-mean bias is H_post^{-1}H_s delta_s; channel-j ROI bias is g_j^T H_post^{-1}H_s delta_s."),
        equation(r"\mathrm{Bias}(h_j)\ \approx\ \sum_s g_j^{\mathrm{T}}H_{\mathrm{post}}^{-1}H_s\,\delta_s", "12"),
        H2("5.5", "From labels to evidence profiles"),
        P(
            "The scientifically meaningful output is the continuous vector (A_Dj, A_Ej, A_Bj, A_Rj), not a hand-assigned 100/82/72/55/50 score. Human-readable source summaries must remain descriptive and must not imply that observational dominance is superior to qualified experimental evidence. The decision-validation calculation uses continuous attribution and quality-adjusted identification. Deprecated 30% and 70% prior-share classifications remain only as internal backward-compatible fields and are not displayed or scored.",
        ),
    ])


def add_simulator(story: list[Flowable]) -> None:
    story.extend([
        H1("6", "Synthetic advertiser population"),
        H2("6.1", "Why many business worlds are required"),
        P(
            "A single synthetic dataset can determine whether one estimator recovers one truth, but it cannot teach which diagnostics transfer across advertisers. FluxMMM generates 500 weekly direct-to-consumer business worlds spanning histories of 104 to 260 weeks, heterogeneous margins, noise, confounding, media delivery, response mechanics, and evidence quality. Each world stores both observed data and hidden potential-outcome components.",
        ),
        image_flow(save_design_figure(), 154),
        caption("Figure 3", "Population and candidate design. The 80 audit businesses remain outside the frozen V5D-SVI empirical result."),
        H2("6.2", "Causal data-generating process"),
        image_flow(save_dag_figure(), 150),
        caption("Figure 4", "Simplified causal graph. Latent demand and planning make media endogenous by affecting both spend and the non-media outcome component."),
        P(
            "For advertiser b, latent demand U_bt, commercial planning P_bt, promotions, seasonality, price, and event shocks generate a baseline outcome. Channel spend X_bjt is a stochastic function of budgets, demand, planning, delivery constraints, auction noise, and flighting. The outcome is",
        ),
        equation(r"Y_{bt}=B_{bt}+\sum_{j=1}^{J}\beta_{bj}(t)\,S_{bj}\!\left(A_{bj}(X_{bj,1:t})\right)+\varepsilon_{bt}", "13"),
        P(
            "where A_bj is geometric or Weibull adstock, S_bj is a Hill response, and beta_bj(t) may be constant or smooth. The data generator varies channel mechanisms separately: search can harvest latent demand, paid social can face frequency pressure, and TV can have delayed long-memory response. Common commercial intensity can simultaneously affect campaign timing and baseline demand.",
        ),
        H2("6.3", "Channel response"),
        multiline_equation([
            r"A^{\mathrm{geo}}_{bjt}=X_{bjt}+\theta_{bj}A^{\mathrm{geo}}_{bj,t-1}",
            r"A^{\mathrm{wei}}_{bjt}=\sum_{\ell=0}^{L}w_{bj\ell}(k_{bj},\lambda_{bj})X_{bj,t-\ell}",
            r"S_{bj}(a)=\frac{a^{\alpha_{bj}}}{a^{\alpha_{bj}}+\gamma_{bj}^{\alpha_{bj}}}",
        ], "14"),
        P(
            "ROI targets are imposed through scaling of the full causal contribution, not by simply multiplying raw spend. The simulator covers paid-social median ROI near 2.46, nonbrand-search median near 1.22, and TV/CTV median near 2.13, with wide business-level variation. Marginal ROI is lower than average ROI under saturation.",
        ),
        H2("6.4", "Experiment and benchmark evidence"),
        P(
            "Synthetic experiment estimates are noisy measurements of the true incremental contribution over the declared experiment and post-test carryover window, divided by tested spend over that window. They are not measurements of full-history ROI. Experiments vary in standard error, bias, overlap, and transportability. Benchmark arms fill channels lacking qualified experiments using tactic-matched or broad paid-media distributions; these priors are observable inputs, not hidden truth.",
        ),
        equation(r"\widehat{ROI}^{\mathrm{exp}}_{bjW}=ROI^{\mathrm{true}}_{bjW}(1+b_{bjW})+\epsilon_{bjW}", "15"),
        H2("6.5", "Population families"),
    ])
    rows = [["Family", "Split", "Businesses", "Dominant challenge"]]
    for family in SIMULATOR["families"]:
        rows.append([family["label"], family["split"], family["businessCount"], family["description"]])
    story.extend([
        make_table(rows, widths=[35 * mm, 22 * mm, 20 * mm, 85 * mm], right_columns=[2]),
        caption("Table 1", "Predeclared simulator families and dominant mechanisms."),
    ])


def add_models(story: list[Flowable]) -> None:
    story.extend([
        H1("7", "Candidate MMM library and inference"),
        H2("7.1", "Forty-eight candidates per advertiser"),
        P(
            "Each advertiser receives 24 base specifications crossed with two evidence arms: experiments-only and benchmark-gap-fill. The base library includes 12 Bayesian and 12 advanced candidates. Specifications vary adstock family and parameters, Hill shape, ridge strength, Fourier order and cycle, calibration route, coefficient prior distribution, outcome likelihood, time-varying coefficients, kernel smoothness, and latent planning intensity. The candidate library is deliberately finite and heterogeneous. FluxMMM claims the best discovered in-pool candidate, never a global optimum.",
        ),
        make_table([
            ["Dimension", "Candidate values or role"],
            ["Adstock", "Geometric or Weibull PDF with channel-specific memory"],
            ["Saturation", "Hill shape and half-saturation scale"],
            ["Baseline", "Intercept, trend, Fourier seasonality, cycle, observed controls"],
            ["Likelihood", "Gaussian, Student-t, or log-normal"],
            ["Calibration", "Informative prior or noisy measurement likelihood"],
            ["Advanced effects", "Smooth time-varying coefficients and optional planning factor"],
            ["Evidence arm", "Experiments only or benchmark gap-fill"],
        ], widths=[43 * mm, 119 * mm]),
        caption("Table 2", "Major candidate dimensions. Exact parameter contracts are stored with each fingerprinted fit."),
        H2("7.2", "Posterior approximation"),
        P(
            "All 24,000 candidates use PyMC FullRankADVI with two primary seeds, 5,000 optimization iterations per seed, and 256 retained posterior draws for an accepted fit. A third seed adjudicates disagreement. Stored diagnostics include finite values, evidence-lower-bound drift, seed agreement, posterior predictive coverage, channel ROI intervals, and posterior mass beyond evidence-informed bounds. FullRank covariance is important because channel ROIs and baseline components can trade off.",
        ),
        H2("7.3", "Safety policy"),
        P(
            "A V5D-SVI candidate is posterior-safe when it is finite, has stable optimization, agrees across seeds, covers at least 80% of observed outcomes with its predictive interval, assigns at most 20% posterior mass beyond any material channel's evidence-informed bound, and has maximum relative ROI interval width no greater than 3. If no candidate passes, selection is for review only. These thresholds are governance choices rather than learned scientific constants.",
        ),
        H2("7.4", "Confirmatory sampling"),
        P(
            "NUTS sampling is reserved for promoted finalists and stratified audits. The final decision should be re-evaluated when the confirmatory posterior materially changes ROI covariance, tail probability, marginal returns, or the selected allocation. Approximate and sampling-based posteriors must share the same structural and evidence contract for the comparison to be meaningful.",
        ),
    ])


def add_features(story: list[Flowable]) -> None:
    story.extend([
        H1("8", "Tokenized deployment evidence"),
        P(
            "The selector receives a structured description of each candidate fit. We call each named scalar a diagnostic token. This is not a language-model token or a learned embedding: the implemented object is an auditable name-value pair whose colon-delimited namespace records its source and meaning. The tokens are flattened in a fixed registry order only after their definitions have been frozen.",
        ),
        H2("8.1", "The 136-token contract"),
        make_table([
            ["Token block", "Count", "Construction and role"],
            ["Validation diagnostics", "41", "Generalization, structure, causal robustness, and decision coherence"],
            ["Temporal evidence", "35", "Carryover support, whole-flight behavior, kernel distinction, and temporal interactions"],
            ["Model assumptions and interactions", "39", "One-hot specification choices and predeclared diagnostic interactions"],
            ["SVI posterior", "21", "Convergence, seed agreement, predictive coverage, ROI centers, width, and implausible mass"],
            ["Total", "136", "115 deployment diagnostics and specifications plus 21 SVI-posterior tokens"],
        ], widths=[48 * mm, 18 * mm, 96 * mm], right_columns=[1]),
        caption("Table 3", "Frozen V5D-SVI token registry. Every token is observable before synthetic truth is opened."),
        P(
            "Base validation scores are scaled to the unit interval, bounded away from zero, and log transformed. For selected diagnostics, shortfall tokens encode how far a score lies below the predeclared 0.50 and 0.70 reference levels. Categorical assumptions are one-hot encoded. Context and interaction tokens multiply only quantities already visible at deployment.",
        ),
        multiline_equation([
            r"z_{bcj}=\operatorname{clip}(s_{bcj}/100,\,0.01,\,1),\qquad t^{\mathrm{diag}}_{bcj}=\log z_{bcj}",
            r"t^{\mathrm{deficit}}_{bcj}(\tau)=-\max\{0,\tau-z_{bcj}\},\qquad \tau\in\{0.50,0.70\}",
        ], "13"),
        P(
            "The 21 posterior tokens record whether the fit is finite and converged, ELBO drift, cross-seed ROI disagreement, predictive coverage, posterior mass outside plausibility bounds, relative ROI width, margins to safety thresholds, and log ROI centers for paid social, non-brand search, and connected TV together with cross-channel summaries. Raw definitions and the complete importance registry are reported in Appendix C.",
        ),
        H2("8.2", "Validation pillars are token families, not fixed score weights"),
        P(
            "Generalization, structure, causal robustness, and decision coherence remain central scientific dimensions. V5D-SVI does not combine them through a preassigned 20/15/25/40 geometric score. Their component tokens, deficits, model assumptions, temporal context, and posterior diagnostics enter the learned risk model. Importance is therefore conditional and predictive. Correlated tokens can exchange coefficient mass, and no importance estimate establishes a causal effect of a diagnostic on model quality.",
        ),
        H2("8.3", "Truth firewall"),
        P(
            "Candidate IDs, business IDs, simulator-family names, proposal source, search phase, split labels, synthetic parameters, realized decision loss, ROI truth error, contribution truth error, and MAP-to-SVI loss shifts are excluded. Tests verify that posterior token rows contain only information available before truth is revealed.",
        ),
    ])


def add_learning(story: list[Flowable]) -> None:
    story.extend([
        H1("9", "Learning the cross-advertiser selector"),
        H2("9.1", "Model choice: an auditable two-head learner"),
        P(
            "V5D-SVI uses two regularized linear heads over the same token vector. One predicts conditional mean excess loss and the second predicts its conditional 90th percentile. The choice is deliberate: although there are 20,160 candidate rows, candidates within an advertiser are dependent, so the effective number of independent business units is 420. A linear learner permits fold-specific standardization, exact coefficient inspection, deterministic refitting, and a direct audit of which evidence enters each prediction.",
        ),
        image_flow(save_selector_figure(), 152),
        caption("Figure 5", "V5D-SVI selector. Named tokens feed separate mean and P90 loss heads; the declared business risk preference combines their outputs only after prediction."),
        P(
            "The term multi-head therefore refers to these two economic-loss outputs. V5D-SVI does not use auxiliary ROI-error or contribution-error heads. A separate development-only V7 experiment tested a 48/24-unit SiLU neural network with four outputs (mean loss, P90 loss, ROI error, and contribution error), but it failed its predeclared out-of-fold improvement checks and is not the method reported here.",
        ),
        H2("9.2", "Fold-specific preprocessing and predictions"),
        P(
            "Within each training fold, every token is centered and scaled using training rows only, then clipped to eight standard deviations. The held-out advertiser and held-out parameter region never determine the transform.",
        ),
        equation(r"\widetilde x_{bcj}=\operatorname{clip}\!\left(\frac{x_{bcj}-\bar x^{\mathrm{train}}_j}{s^{\mathrm{train}}_j},-8,8\right)", "14"),
        multiline_equation([
            r"\widehat\mu_{bc}=\max\{0,\,\alpha_{\mu}+\widetilde x_{bc}^{\top}\beta_{\mu}\}",
            r"\widehat q_{0.90,bc}=\max\{\widehat\mu_{bc},\,\alpha_q+\widetilde x_{bc}^{\top}\beta_q\}",
            r"\widehat R_{bc}=0.65\,\widehat\mu_{bc}+0.35\,\widehat q_{0.90,bc}",
        ], "15"),
        P(
            "The nonnegativity and quantile-order constraints are applied to outputs, not to the learned coefficients. Consequently, coefficient-based importance is a local linear explanation before output clamping rather than an exact decomposition of every final risk prediction.",
        ),
        H2("9.3", "Economic targets and objective functions"),
        P(
            "Within each advertiser, the eligible candidate with lowest realized economic loss defines the safe oracle. Every candidate receives the uncapped excess-loss target in equation (4). Candidate rows in the empirical upper loss decile receive an additional weight of 0.80; rows from the currently worst-performing simulator family receive an additional weight of 0.70. Family identity affects training emphasis but is prohibited from the deployment token vector.",
        ),
        equation(r"w_{bc}=1+0.80\,\mathbf{1}[\Delta_{bc}\geq Q_{0.90}(\Delta)]+0.70\,\mathbf{1}[f_b=f_{\mathrm{worst}}]", "16"),
        multiline_equation([
            r"\mathcal{L}_{\mu}=\sum_{b,c}w_{bc}\,\mathrm{Huber}_{2.5}(\widehat\mu_{bc}-\Delta_{bc})+\lambda\left\|\beta_{\mu}\right\|_2^2+\gamma\mathcal{L}_{\mathrm{rank}}",
            r"\mathcal{L}_{0.90}=\sum_{b,c}w_{bc}\,\rho_{0.90}(\Delta_{bc}-\widehat q_{0.90,bc})+\lambda\left\|\beta_q\right\|_2^2",
        ], "17"),
        P(
            "Huber loss limits the influence of catastrophic outliers while remaining quadratic near zero. Pinball loss targets the conditional 90th percentile rather than the mean. The frozen constants are lambda=0.012 and gamma=0.55. Both heads are fitted for 28 deterministic gradient epochs with a decaying learning rate.",
        ),
        H2("9.4", "Economically weighted hard-negative ranking"),
        P(
            "For each advertiser and epoch, the learner identifies the challenger that most violates the desired economic ordering relative to the oracle. Let the candidate's excess loss be the required margin. The hardest challenger maximizes the sum of that margin and the amount by which its predicted mean loss is too close to, or below, the oracle's prediction. The update is then weighted by the economic gap, with a small 0.05 floor.",
        ),
        multiline_equation([
            r"c_b^- = \arg\max_{c\neq c_b^\star}\left\{\Delta_{bc}+\widehat\mu_{bc_b^\star}-\widehat\mu_{bc}\right\}",
            r"\mathcal{L}_{\mathrm{rank}}=\sum_b\max\{\Delta_{bc_b^-},0.05\}\,\log\!\left(1+e^{\Delta_{bc_b^-}+\widehat\mu_{bc_b^\star}-\widehat\mu_{bc_b^-}}\right)",
        ], "18"),
        P(
            "Equation (18) states the objective whose sigmoid derivative is implemented in the gradient update. It is related to, but not identical with, the complete all-pairs logistic upper bound in equation (7): V5D-SVI uses one hardest challenger per advertiser per epoch rather than every oracle-versus-candidate pair. This distinction is essential for reproducibility.",
        ),
        H2("9.5", "Two-axis cross-fitting"),
        P(
            "The 420 development advertisers are partitioned into five grouped folds so that no candidate from an assessment advertiser enters its training model. Candidate configurations are also partitioned into four parameter regions. Each assessment row is predicted by a selector that excluded both its advertiser fold and its parameter region. Twenty fitted selectors therefore provide out-of-fold predictions. This design reduces leakage from repeated candidates belonging to the same business and from nearby parameter settings.",
        ),
        H2("9.6", "Search and promotion"),
        P(
            "The learned risk ranks candidates proposed by fixed coverage, uniform random, maximin space-filling, Gaussian-process upper confidence bound, and hybrid global-local search. Algorithms receive equal candidate-evaluation budgets and cannot duplicate evaluations. The Gaussian process proposes where to evaluate next; it is not the MMM posterior and not the V5D-SVI risk learner. Search quality, selector quality, safety, and promotion are evaluated separately: a strong selector cannot choose a candidate the search never proposes, and broader search can expose additional false minima when risk predictions are imperfect.",
        ),
    ])


def add_results(story: list[Flowable]) -> None:
    comparison = DEVELOPMENT["comparison"]
    baseline = comparison["baseline"]
    full = comparison["fullSvi"]
    story.extend([
        H1("10", "Empirical evaluation and interpretation"),
        H2("10.1", "Evaluation questions and study status"),
        P(
            "A methods paper requires empirical results, but an engineering-style version comparison is not the scientific endpoint. We therefore organize the evidence around four questions: whether the selector orders economically consequential candidate pairs, predicts loss magnitude, chooses a low-loss champion, and remains safe when search exposes more candidates. The frozen result uses 420 development advertisers and 20,160 candidate rows. The 80-business adversarial audit was not accessed, so every number below is grouped out-of-fold development evidence rather than a fresh external-validation estimate.",
        ),
        make_table([
            ["Scientific question", "Out-of-fold observation", "Conclusion"],
            ["Does it order costly mistakes?", f"Weighted pair accuracy {pct(full['proxy']['economicallyWeightedPairwiseAccuracy'])} vs. {pct(baseline['proxy']['economicallyWeightedPairwiseAccuracy'])}", "Useful ordering signal in development"],
            ["Does it predict loss magnitude?", f"Correlation {fmt(full['proxy']['candidateLevelCorrelation'], 3)}; MAE {fmt(full['proxy']['meanAbsoluteError'])}", "Only weakly; risk is not calibrated well"],
            ["Does minimum predicted risk win?", f"Champion excess loss {fmt(full['proxy']['riskChampionMeanExcessLoss'])} vs. {fmt(baseline['proxy']['riskChampionMeanExcessLoss'])}", "No; top-one selection worsens"],
            ["Does the safety pool protect search?", f"False champions {pct(full['proxy']['dangerousFalseChampionShare'])}; {DEVELOPMENT['ablation']['safetyAudit']['sviSafe']} safe rows", "No; the current pool is too permissive"],
            ["Is the declared search objective lower?", f"At 16 evaluations: {fmt(full['primary']['objective'])} vs. {fmt(baseline['primary']['objective'])}", "Yes, but this does not rescue promotion"],
        ], widths=[45 * mm, 57 * mm, 60 * mm]),
        caption("Table 4", "Decision-relevant empirical findings. Comparisons are predeclared development ablations, not sealed generalization claims."),
        H2("10.2", "The central falsification result"),
        P(
            "Posterior-aware tokens improve economically weighted ordering, yet the selected champion is worse on mean excess loss and dangerous-false-champion rate. The likely mechanism is winner's curse under correlated prediction errors: a permissive safety set supplies many candidates, and selecting the minimum predicted risk preferentially chooses an underestimated one. Better average ordering does not guarantee a better minimum. This separates three scientific objects that are often conflated: learning a useful risk signal, searching the specification space, and safely promoting one model.",
        ),
        P(
            "The correct conclusion is not that the diagnostic tokens are useless. It is that the current top-one policy is not validated for automatic promotion. V5D-SVI is therefore a falsifiable model-selection benchmark and an interpretable research score, while production deployment still requires repaired joint safety, calibrated selection uncertainty, independent validation, and confirmatory NUTS.",
        ),
        H2("10.3", "Five practitioner pillars"),
        image_flow(save_pillar_figure(), 150),
        caption("Figure 6", "Five-pillar aggregation of frozen V5D-SVI coefficient importance. Pillars explain the learned model; they are not new hand-assigned validation weights."),
        P(
            "A raw token such as log paid-social ROI is not, by itself, a reusable scientific lesson. Its coefficient is conditional on the simulator's channel scale, correlated tokens, and all included interactions. For interpretation, we therefore aggregate the 136 token importances into five predeclared practitioner pillars. The aggregation preserves every coefficient but moves the main text from implementation names to questions an analyst can reproduce.",
        ),
    ])
    pillar_rows = [["Pillar", "Importance", "Practitioner question", "How to reproduce"]]
    for pillar in pillar_importance():
        pillar_rows.append([
            pillar["name"],
            f"{100 * pillar['importance']:.1f}%",
            pillar["question"],
            pillar["reproduce"],
        ])
    story.extend([
        make_table(pillar_rows, widths=[36 * mm, 20 * mm, 53 * mm, 53 * mm], right_columns=[1]),
        caption("Table 5", "Interpretation map from learned token coefficients to reproducible MMM validation practice."),
        H2("10.4", "What importance can and cannot establish"),
        P(
            "For token k and cross-fitted selector m, define the pre-clamp risk coefficient as 0.65 times the mean-head coefficient plus 0.35 times the P90-head coefficient. Token importance is the mean absolute standardized risk coefficient across the 20 selectors, normalized to sum to one. Pillar importance is the sum of its member-token importances.",
        ),
        multiline_equation([
            r"\beta^{R}_{mk}=0.65\,\beta^{\mu}_{mk}+0.35\,\beta^{q}_{mk}",
            r"I_k=\frac{M^{-1}\sum_{m=1}^{M}|\beta^{R}_{mk}|}{\sum_{\ell}M^{-1}\sum_{m=1}^{M}|\beta^{R}_{m\ell}|},\qquad I_p=\sum_{k\in p}I_k",
        ], "19"),
        P(
            "This quantity answers how much a standardized token contributes to the fitted linear risk surface in the development cohort. It does not measure a token's causal effect on model quality. A pillar is a plausible generalizable lesson only if its mass and direction remain stable across business folds, parameter regions, simulator families, an independently authored simulator, and eventually real experiment-linked datasets. The current artifact reports cross-fit sign stability but has not passed the latter external tests. Appendix C gives the exact mapping and raw top-50 audit table.",
        ),
    ])


def add_discussion(story: list[Flowable]) -> None:
    story.extend([
        H1("11", "Discussion"),
        H2("11.1", "What has been learned"),
        P(
            "The study demonstrates that MMM validation can be framed as an empirically learnable mapping from observable model evidence to downstream decision risk. It also demonstrates that this mapping is not sufficient by itself: the candidate pool and promotion rule determine which prediction errors matter. The most important current finding is therefore methodological. A good MMM cannot be reduced to one universally weighted checklist; it must be evaluated through the decisions it supports, while preserving explicit causal, structural, and evidence-source diagnostics.",
        ),
        H2("11.2", "What happens for a new advertiser"),
        P(
            "FluxMMM validates cadence and schema, constructs multiple candidate models, fits posterior approximations, computes truth-blind diagnostics and evidence profiles, and predicts mean and tail excess loss using the frozen cross-advertiser selector. Search proposes candidates under a fixed budget. Safety excludes invalid candidates, the lowest predicted-risk candidate becomes a provisional champion, and NUTS confirms the finalist before production. The real advertiser's data never reveal true economic loss during selection; transfer depends on similarity between the advertiser and the research population.",
        ),
        H2("11.3", "Continuous evidence attribution replaces categorical source labels"),
        P(
            "The scientific output is the continuous four-source attribution vector in equation (9), supplemented by conditional observational information, evidence quality, conflict, and decision dependence. The former 30% and 70% labels for data-led, mixed, and prior-led estimates are not used by the V5D-SVI score and should not appear as a validation verdict in the product. A descriptive source summary may be generated from the continuous shares, but it must not replace the underlying percentages or earn separate validation credit.",
        ),
    ])


def add_limitations(story: list[Flowable]) -> None:
    story.extend([
        H1("12", "Limitations and threats to validity"),
        bullet("Synthetic transport. The selector can learn regularities specific to the implemented generator. Independent simulators and real experiment-linked datasets are required."),
        bullet("Development reuse. The 420-business cohort includes 120 businesses from an earlier opened validation cohort. Grouped cross-fitting limits within-cohort leakage but does not recreate a fresh external test."),
        bullet("Sealed audit unavailable for this release. No claim about the 80 audit businesses is made."),
        bullet("Finite candidate library. The oracle is best among 48 candidates under the active safety policy, not the true model or global optimum."),
        bullet("Approximate inference. FullRankADVI can distort posterior tails, multimodality, and covariance; NUTS auditing is incomplete."),
        bullet("Weak top-one performance. The current posterior safety policy admits many candidates and increases false-champion risk despite improved average ordering."),
        bullet("Linear meta-model. Interactions are engineered rather than fully hierarchical. Correlated tokens exchange coefficient importance."),
        bullet("Evidence-quality proxies. Independence, relevance, and transportability are simplified numeric contracts and cannot substitute for study-level review."),
        bullet("Local evidence attribution. Equations (8)-(12) use a local Gaussian curvature approximation. They describe information sensitivity near the fit, not a global causal decomposition."),
        bullet("Subjective utility. The 70/30 scenario aggregation and 65/35 mean-tail risk mixture represent predeclared business preferences."),
        H2("12.1", "Claims explicitly not supported"),
        P(
            "The study does not establish that V5D-SVI is state of the art, superior to all expert modelers, externally valid across industries, or safe for automatic production deployment. It does not show that posterior evidence profiles identify causal truth. It does not prove generalization from the regret inequality. These remain empirical questions.",
        ),
    ])


def add_future(story: list[Flowable]) -> None:
    story.extend([
        H1("13", "Research agenda"),
        H2("13.1", "Repair top-one selection"),
        bullet("Calibrate predictive risk uncertainty and require a minimum risk margin before promotion."),
        bullet("Train listwise or differentiable top-k objectives that directly target the selected champion."),
        bullet("Model joint safety probability rather than combining permissive marginal gates."),
        bullet("Use hierarchical shrinkage across advertiser families, channels, and response mechanisms."),
        H2("13.2", "Strengthen the economic target"),
        bullet("Evaluate the exact capped all-pairs surrogate from equation (7) alongside uncapped Huber, P90, and hard-negative objectives."),
        bullet("Report mean, P90, P95, CVaR, and scenario-specific loss; keep the business utility declared before audit."),
        bullet("Estimate Pareto frontiers for expected profit, downside regret, and budget stability rather than forcing one scalar preference."),
        H2("13.3", "Independent validation"),
        bullet("Freeze the method and open the 80-business adversarial audit once, with no corrective tuning afterward."),
        bullet("Replicate on an independently authored simulator with different equations and software."),
        bullet("Create a consortium of historical MMM datasets with channel experiments using federated, privacy-safe receipts."),
        bullet("Run prospective precommitted budget tests before making a decision-impact claim."),
        H2("13.4", "Inference robustness"),
        bullet("Run NUTS on stratified candidate-business pairs and all promoted finalists."),
        bullet("Compare posterior ROI centers, covariance, coverage, marginal-return surfaces, allocations, and realized regret."),
        bullet("Train an approximation-error correction only if it is learned without audit leakage and improves decisions."),
        H2("13.5", "Evidence attribution"),
        bullet("Validate continuous source attribution across nonlinear fits, posterior draws, and source-deletion refits without reintroducing categorical score thresholds."),
        bullet("Extend local curvature attribution to posterior-draw perturbation and nonlinear source deletion."),
        bullet("Quantify uncertainty in source shares and validate them under known duplicated-channel and biased-anchor experiments."),
    ])


def add_open_science(story: list[Flowable]) -> None:
    story.extend([
        H1("14", "Reproducibility and open-source contribution"),
        P(
            "FluxMMM is designed as a versioned research platform rather than a one-off simulation. Candidate fits, posterior contracts, token registries, split rules, source hashes, and result receipts are fingerprinted. Tests cover truth leakage, grouped folds, posterior safety, evidence-attribution exhaustiveness, duplicate-channel identification, and the finite-candidate regret inequality. New simulator families, diagnostics, inference engines, or selectors should create new artifacts rather than overwrite frozen releases.",
        ),
        make_table([
            ["Stage", "Truth access", "Required artifact"],
            ["World generation", "Full hidden truth", "Seeded DGP and invariant receipt"],
            ["Candidate fitting", "None", "Model and evidence fingerprint"],
            ["Diagnostic construction", "None", "Token registry and leakage audit"],
            ["Decision labeling", "After action only", "Scenario action, oracle action, and economic loss"],
            ["Selector training", "Loss labels on development only", "Grouped cross-fit receipt"],
            ["Fresh validation", "Opened once", "Frozen paired comparison"],
            ["Deployment", "None", "Risk, safety, evidence profile, and sampling receipt"],
        ], widths=[35 * mm, 39 * mm, 88 * mm]),
        caption("Table 6", "Information timing and reproducibility contract."),
        P(
            "Open sourcing makes the scientific object inspectable: contributors can challenge the data-generating process, add mechanism families, test alternative decision utilities, improve evidence transport, compare inference engines, and submit new selectors under the same truth firewall. Credibility should come from frozen negative results and independent replication, not only from interface polish.",
        ),
    ])


def add_conclusion(story: list[Flowable]) -> None:
    story.extend([
        H1("15", "Conclusion"),
        P(
            "FluxMMM reframes the question 'What is a good marketing mix model?' as an empirical decision problem. Across many advertiser worlds, complete Bayesian MMMs are fitted without causal truth, required to make budget decisions, and then labeled by the economic consequences of those decisions. A meta-model learns which observable validation diagnostics, posterior properties, evidence profiles, and structural assumptions predict low mean and tail loss. The learned rule can then rank models for an advertiser whose truth is unknown.",
        ),
        P(
            "The contribution is not a claim that one approximate posterior beats another. It is the open cross-advertiser protocol linking causal model assessment to economic decisions. The finite-candidate inequality formalizes why costly oracle misrankings matter. The evidence-attribution decomposition formalizes whether channel ROI precision comes from observational data, experiments, benchmarks, or regularization, while quality, conflict, and decision dependence prevent influence from being mistaken for validity.",
        ),
        P(
            "The first frozen development result is deliberately mixed: posterior-aware information improves economically weighted ordering, but the current safety and promotion rules permit damaging false champions. That result narrows the next scientific problem. Independent validation, improved top-one risk control, NUTS confirmation, and real experiment-linked applications are required before deployment claims. By exposing the method, artifacts, tests, and failures, FluxMMM offers a foundation that advertisers and researchers can inspect and improve collectively.",
        ),
    ])


REFERENCES = [
    "[1] Jin, Y., Wang, Y., Sun, Y., Chan, D., and Koehler, J. (2017). Bayesian methods for media mix modeling with carryover and shape effects. Google Research.",
    "[2] Chan, D. and Perry, M. (2017). Challenges and opportunities in media mix modeling. Google Research.",
    "[3] Chen, A., Chan, D., Perry, M., Jin, Y., Sun, Y., Wang, Y., and Koehler, J. (2018). Bias correction for paid search in media mix modeling. arXiv:1807.03292.",
    "[4] Ng, E., Wang, Z., and Dai, A. (2021). Bayesian time varying coefficient model with applications to marketing mix modeling. arXiv:2106.03322.",
    "[5] Zhang, Y., Wurm, M., Li, E., Wakim, A., Kelly, J., Price, B., and Liu, Y. (2024). Media mix model calibration with Bayesian priors. Google Research.",
    "[6] Runge, J. et al. (2024). Robyn: An open-source marketing mix modeling package. arXiv:2403.14674.",
    "[7] Google Meridian Team (2026). Meridian marketing mix modeling, version 1.6.1. GitHub repository.",
    "[8] Google Meridian Team (2026). Prior distribution and ROI calibration documentation. GitHub repository.",
    "[9] Heusch, N. (2026). A synthetic benchmark dataset with endogenous marketing spend for validating marketing mix models. arXiv:2608.21130.",
    "[10] Elmachtoub, A. N. and Grigas, P. (2022). Smart predict, then optimize. Management Science, 68(1), 9-26.",
    "[11] Wang, K. et al. (2024). Decision focused causal learning for direct counterfactual marketing optimization. arXiv:2407.13664.",
    "[12] Doppa, J. R., Fern, A., and Tadepalli, P. (2014). Structured prediction via output space search. Journal of Machine Learning Research, 15, 1317-1350.",
    "[13] Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., and Blei, D. M. (2017). Automatic differentiation variational inference. Journal of Machine Learning Research, 18(14), 1-45.",
    "[14] Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. (2017). Variational inference: A review for statisticians. Journal of the American Statistical Association, 112(518), 859-877.",
    "[15] Hoffman, M. D. and Gelman, A. (2014). The No-U-Turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15, 1593-1623.",
    "[16] Huber, P. J. (1964). Robust estimation of a location parameter. Annals of Mathematical Statistics, 35(1), 73-101.",
    "[17] Cawley, G. C. and Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research, 11, 2079-2107.",
    "[18] Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica, 46(1), 33-50.",
    "[19] Gordon, B. R., Zettelmeyer, F., Bhargava, N., and Chapsky, D. (2019). A comparison of approaches to advertising measurement: Evidence from big field experiments at Facebook. Marketing Science, 38(2), 193-225.",
    "[20] Lewis, R. A. and Rao, J. M. (2015). The unfavorable economics of measuring the returns to advertising. Quarterly Journal of Economics, 130(4), 1941-1973.",
    "[21] Heusch, N. (2026). Structural estimation of marketing mix model parameters from geo-experiments. arXiv:2608.21128.",
    "[22] Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and widely applicable information criterion in singular learning theory. Journal of Machine Learning Research, 11, 3571-3594.",
]


def add_references(story: list[Flowable]) -> None:
    story.extend([PageBreak(), H1("", "References")])
    for ref in REFERENCES:
        story.append(P(ref, "reference"))


def add_appendices(story: list[Flowable]) -> None:
    story.extend([
        PageBreak(),
        H1("A", "Appendix: frozen research contract"),
        H2("A.1", "Fit counts and splits"),
        multiline_equation([
            r"500\ \mathrm{businesses}\times24\ \mathrm{specifications}\times2\ \mathrm{evidence\ arms}=24{,}000\ \mathrm{fits}",
            r"420\ \mathrm{development\ businesses}\times48=20{,}160\ \mathrm{development\ rows}",
            r"80\ \mathrm{audit\ businesses}\times48=3{,}840\ \mathrm{sealed\ rows}",
        ], "A.1"),
        H2("A.2", "SVI settings"),
        make_table([
            ["Setting", "Frozen value"],
            ["Engine", "PyMC 6.2 FullRankADVI"],
            ["Iterations", "5,000 per seed"],
            ["Posterior draws", "256 per accepted fit"],
            ["Primary seeds", "30,071 and 81,119"],
            ["Adjudication seed", "190,081"],
            ["Learning rate", "0.001"],
            ["Gradient norm cap", "10"],
            ["ELBO window", "250 iterations"],
            ["Maximum ELBO drift", "0.05"],
            ["Maximum seed log-ROI difference", "0.25"],
        ], widths=[66 * mm, 96 * mm]),
        caption("Table A.1", "Fingerprint of the approximate posterior contract."),
        H2("A.3", "Selector settings"),
        make_table([
            ["Setting", "Value", "Purpose"],
            ["Epochs", "28", "Gradient passes"],
            ["Learning rate", "0.055 / sqrt(1 + epoch/8)", "Decaying optimization"],
            ["L2 regularization", "0.012", "Coefficient shrinkage"],
            ["Huber delta", "2.5", "Robust mean-loss fit"],
            ["Quantile", "0.90", "Tail-loss target"],
            ["Pairwise weight", "0.55", "Hard-negative ordering pressure"],
            ["Tail row weight", "0.80", "Emphasize high-loss candidates"],
            ["Worst-family weight", "0.70", "Emphasize current hardest family"],
            ["Risk aversion rho", "0.35", "35% P90 protection"],
        ], widths=[48 * mm, 48 * mm, 66 * mm]),
        caption("Table A.2", "Frozen V5D-SVI selector hyperparameters."),
        H1("B", "Appendix: evidence-attribution implementation notes"),
        P(
            "The implementation constructs observational curvature from a weighted Gram matrix and experiment, benchmark, and regularization curvature from positive-precision penalty outer products. It evaluates all sources at the same posterior mode and covariance. For a channel with several time-varying basis coefficients, the ROI gradient spans the channel's coefficient block. Conditional observational information is calculated by projecting that ROI direction on every other design column and measuring weighted residual information.",
        ),
        P(
            "The attribution tests require nonnegative shares, a sum equal to one within floating-point tolerance, and negligible conditional separating information for perfectly duplicated media. The decision layer quality-weights experiment and benchmark shares and gives no identification credit to regularization. Evidence conflict and dependence are computed after a leave-source-out refit.",
        ),
        H1("C", "Appendix: token importance and pillar mapping"),
        H2("C.1", "Exact five-pillar aggregation"),
    ])
    group_counts: dict[str, int] = {}
    for item in IMPORTANCE["risk"]["features"]:
        group_counts[item["group"]] = group_counts.get(item["group"], 0) + 1
    pillar_rows = [["Pillar", "Source groups", "Tokens", "Importance"]]
    for pillar in pillar_importance():
        pillar_rows.append([
            pillar["name"],
            ", ".join(pillar["groups"]),
            sum(group_counts.get(group, 0) for group in pillar["groups"]),
            f"{100 * pillar['importance']:.2f}%",
        ])
    story.extend([
        make_table(pillar_rows, widths=[40 * mm, 82 * mm, 18 * mm, 22 * mm], right_columns=[2, 3]),
        caption("Table C.1", "Deterministic mapping from all 136 raw tokens to the five practitioner pillars. The importance column sums to 100%."),
        P(
            "The pillars are reporting aggregations only. They do not replace the token vector, enter training, or impose new score weights. Reproduction consists of computing token importance with equation (19), assigning each token by its frozen source group, and summing within the mapping above.",
        ),
        H2("C.2", "Top 50 raw predictive tokens"),
    ])
    rows = [["Rank", "Token", "Group", "Importance", "Mean coefficient", "Sign stability"]]
    group_labels = {
        "posterior-roi": "posterior ROI",
        "posterior-reliability": "posterior reliability",
        "posterior-decision-safety": "posterior decision safety",
        "model-specification-and-interactions": "specification / interaction",
        "temporal-evidence": "temporal evidence",
        "decision-coherence": "decision coherence",
        "causal-robustness": "causal robustness",
    }
    for item in IMPORTANCE["top50RiskFeatures"]:
        rows.append([
            item["rank"],
            item["feature"],
            group_labels.get(item["group"], item["group"].replace("-", " ")),
            f"{item['importance']:.4f}",
            f"{item['meanCoefficient']:+.3f}",
            f"{item['signStability']:.2f}",
        ])
    story.extend([
        make_table(rows, widths=[12 * mm, 62 * mm, 40 * mm, 18 * mm, 18 * mm, 18 * mm], right_columns=[0, 3, 4, 5]),
        caption("Table C.2", "Mean absolute standardized risk-coefficient importance across 20 cross-fitted selectors. Raw token names are retained for auditability; coefficients are conditional predictive associations, not general scientific weights."),
        H1("D", "Appendix: algorithmic protocol"),
        H2("D.1", "Training"),
        P("1. Generate advertiser worlds and freeze observed data, hidden truth, experiments, and business decision contracts.", "body_left"),
        P("2. Construct 48 candidates per advertiser without truth and fit all posterior approximations under fingerprinted contracts.", "body_left"),
        P("3. Compute truth-blind validation, posterior, evidence-attribution, and model-specification tokens.", "body_left"),
        P("4. Optimize one posterior-expected action for each candidate and scenario.", "body_left"),
        P("5. Reveal hidden truth, calculate scenario loss and within-business excess loss, then close truth access.", "body_left"),
        P("6. Group advertisers into five folds and candidates into four regions; fit 20 selectors and retain only doubly out-of-fold predictions.", "body_left"),
        P("7. Evaluate risk calibration, mean and tail economic loss, oracle recall, weighted pair ordering, search performance, and safety behavior.", "body_left"),
        H2("D.2", "Deployment"),
        P("1. Validate a new advertiser's schema, cadence, controls, experiments, benchmark choices, and decision contract.", "body_left"),
        P("2. Generate, fit, and diagnose candidate MMMs without causal truth.", "body_left"),
        P("3. Calculate continuous evidence profiles, quality, conflict, and decision dependence.", "body_left"),
        P("4. Predict mean and P90 excess loss, combine them using the declared risk preference, and rank only safe candidates.", "body_left"),
        P("5. Challenge the provisional champion locally, run confirmatory NUTS, and promote only with an auditable receipt.", "body_left"),
        H1("E", "Appendix: audit checklist"),
        make_table([
            ["Claim or component", "Current status", "Requirement before stronger claim"],
            ["Cross-advertiser loss learning", "Implemented on 420 development businesses", "Fresh external cohort"],
            ["Finite-candidate regret inequality", "Proved and code-audited", "Independent proof review; evaluate exact surrogate"],
            ["Continuous evidence attribution", "Implemented and unit-tested", "Nonlinear and posterior-draw validation"],
            ["Posterior-aware representation", "Development signal observed", "Independent simulator and NUTS audit"],
            ["Safe top-one promotion", "Not established", "Repair safety and winner's-curse control"],
            ["Real-world generalization", "Not established", "Historical and prospective experiment-linked applications"],
            ["State-of-the-art performance", "Not claimed", "Predeclared comparison against strong baselines"],
        ], widths=[48 * mm, 48 * mm, 66 * mm]),
        caption("Table E.1", "Claim boundary for peer review."),
    ])


def build() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    story: list[Flowable] = []
    add_title(story)
    add_introduction(story)
    add_related_work(story)
    add_problem(story)
    add_regret_theory(story)
    add_evidence_theory(story)
    add_simulator(story)
    add_models(story)
    add_features(story)
    add_learning(story)
    add_results(story)
    add_discussion(story)
    add_limitations(story)
    add_future(story)
    add_open_science(story)
    add_conclusion(story)
    add_references(story)
    add_appendices(story)
    JournalDoc(str(OUT)).build(story)
    print(OUT)


if __name__ == "__main__":
    build()
