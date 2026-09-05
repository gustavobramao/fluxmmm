#!/usr/bin/env python3
"""Build the FluxMMM V5D-SVI working paper as a publication-style PDF."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    HRFlowable,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "fluxmmm_v5d_svi_working_paper.pdf"
DEVELOPMENT = ROOT / "research" / "svi_score_v5d_svi" / "artifacts" / "svi-score-v5d-svi-development.json"
IMPORTANCE = ROOT / "research" / "svi_score_v5d_svi" / "artifacts" / "svi-score-v5d-svi-feature-importance.json"
SIMULATOR = ROOT / "research" / "score_v3" / "artifacts" / "simulator-audit-v3-summary.json"

PAGE_W, PAGE_H = A4
LEFT = 18 * mm
RIGHT = 18 * mm
TOP = 18 * mm
BOTTOM = 18 * mm

INK = HexColor("#20212B")
MUTED = HexColor("#6F7181")
FAINT = HexColor("#E8E7E2")
PAPER = HexColor("#FCFBF8")
WHITE = colors.white
INDIGO = HexColor("#5B5CE2")
INDIGO_DARK = HexColor("#3738A6")
INDIGO_PALE = HexColor("#EFEEFF")
TEAL = HexColor("#1DA68A")
TEAL_PALE = HexColor("#E8F7F3")
ORANGE = HexColor("#E8894B")
ORANGE_PALE = HexColor("#FFF0E5")
LIME = HexColor("#B8DB25")
RED = HexColor("#C95B62")
RED_PALE = HexColor("#FBEDEE")
NAVY = HexColor("#233C68")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


DEVELOPMENT_DATA = load_json(DEVELOPMENT)
IMPORTANCE_DATA = load_json(IMPORTANCE)
SIMULATOR_DATA = load_json(SIMULATOR)


def register_fonts() -> tuple[str, str, str, str]:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    serif_candidates = [
        Path("/System/Library/Fonts/Supplemental/Georgia.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ]
    body = "Helvetica"
    body_bold = "Helvetica-Bold"
    serif = "Times-Roman"
    serif_bold = "Times-Bold"
    for candidate in candidates:
        if candidate.exists():
            pdfmetrics.registerFont(TTFont("FluxSans", str(candidate)))
            body = "FluxSans"
            break
    for candidate in serif_candidates:
        if candidate.exists():
            pdfmetrics.registerFont(TTFont("FluxSerif", str(candidate)))
            serif = "FluxSerif"
            break
    return body, body_bold, serif, serif_bold


BODY_FONT, BODY_BOLD, SERIF_FONT, SERIF_BOLD = register_fonts()


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=9.15,
            leading=13.3,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6.5,
        ),
        "body_left": ParagraphStyle(
            "BodyLeft",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=9.15,
            leading=13.3,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=6.5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=7.6,
            leading=10.4,
            textColor=MUTED,
            alignment=TA_LEFT,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=7.4,
            leading=10.3,
            textColor=MUTED,
            alignment=TA_LEFT,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName=SERIF_FONT,
            fontSize=21,
            leading=25,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName=SERIF_FONT,
            fontSize=14.4,
            leading=18,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3",
            parent=base["Heading3"],
            fontName=BODY_FONT,
            fontSize=10.2,
            leading=13,
            textColor=INDIGO_DARK,
            spaceBefore=7,
            spaceAfter=3.5,
            keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=8.85,
            leading=12.5,
            leftIndent=13,
            firstLineIndent=-7,
            bulletIndent=0,
            textColor=INK,
            spaceAfter=3.5,
        ),
        "equation": ParagraphStyle(
            "Equation",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8.2,
            leading=11.5,
            leftIndent=8,
            textColor=NAVY,
            alignment=TA_LEFT,
        ),
        "toc_h1": ParagraphStyle(
            "TOCHeading1",
            fontName=BODY_FONT,
            fontSize=9.2,
            leading=13,
            textColor=INK,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=4,
        ),
        "toc_h2": ParagraphStyle(
            "TOCHeading2",
            fontName=BODY_FONT,
            fontSize=8.1,
            leading=11.5,
            textColor=MUTED,
            leftIndent=14,
            firstLineIndent=0,
            spaceBefore=2,
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=7.7,
            leading=10.6,
            leftIndent=14,
            firstLineIndent=-14,
            textColor=INK,
            spaceAfter=4,
        ),
    }


S = make_styles()


class FluxDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title="Learning What Makes a Marketing Mix Model Decision-Useful",
            author="Gustavo Bramao",
            subject="FluxMMM V5D-SVI working paper",
            keywords="marketing mix modeling, Bayesian MMM, decision-focused learning, variational inference, budget optimization",
        )
        frame = Frame(
            LEFT,
            BOTTOM,
            PAGE_W - LEFT - RIGHT,
            PAGE_H - TOP - BOTTOM,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="paper", frames=[frame], onPage=page_decor))

    def afterFlowable(self, flowable: Flowable) -> None:
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            if style_name in ("Heading1", "Heading2"):
                level = 0 if style_name == "Heading1" else 1
                text = flowable.getPlainText()
                key = f"section-{self.seq.nextf('heading')}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=level, closed=False)
                self.notify("TOCEntry", (level, text, self.page, key))


def page_decor(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    if doc.page > 1:
        canvas.setStrokeColor(FAINT)
        canvas.setLineWidth(0.5)
        canvas.line(LEFT, PAGE_H - 11 * mm, PAGE_W - RIGHT, PAGE_H - 11 * mm)
        canvas.setFont(BODY_FONT, 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(LEFT, PAGE_H - 8.5 * mm, "FluxMMM V5D-SVI | Working paper")
        canvas.drawRightString(PAGE_W - RIGHT, PAGE_H - 8.5 * mm, "Gustavo Bramao")
        canvas.line(LEFT, 11 * mm, PAGE_W - RIGHT, 11 * mm)
        canvas.drawString(LEFT, 7.5 * mm, "Public research release | September 2026")
        canvas.drawRightString(PAGE_W - RIGHT, 7.5 * mm, str(doc.page))
    canvas.restoreState()


class ColorBand(Flowable):
    def __init__(self, height=7 * mm):
        super().__init__()
        self.height = height
        self.width = PAGE_W - LEFT - RIGHT

    def draw(self) -> None:
        widths = [0.42, 0.22, 0.22, 0.14]
        palette = [INDIGO, TEAL, ORANGE, LIME]
        x = 0
        for share, color in zip(widths, palette):
            width = self.width * share
            self.canv.setFillColor(color)
            self.canv.rect(x, 0, width, self.height, fill=1, stroke=0)
            x += width


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def H1(text: str) -> Paragraph:
    return Paragraph(text, S["h1"])


def H2(text: str) -> Paragraph:
    return Paragraph(text, S["h2"])


def H3(text: str) -> Paragraph:
    return Paragraph(text, S["h3"])


def bullet(text: str) -> Paragraph:
    return Paragraph(f"<bullet>&bull;</bullet>{text}", S["bullet"])


def equation(*lines: str) -> Table:
    content = "<br/>".join(line.replace(" ", "&nbsp;") for line in lines)
    table = Table([[Paragraph(content, S["equation"])]] , colWidths=[PAGE_W - LEFT - RIGHT])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), INDIGO_PALE),
        ("BOX", (0, 0), (-1, -1), 0.6, HexColor("#D7D5F6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def callout(title: str, body: str, color=INDIGO, background=INDIGO_PALE) -> Table:
    cell = [
        Paragraph(title.upper(), ParagraphStyle(
            "CalloutTitle", fontName=BODY_FONT, fontSize=7.7, leading=10,
            textColor=color, spaceAfter=4,
        )),
        Paragraph(body, ParagraphStyle(
            "CalloutBody", fontName=BODY_FONT, fontSize=8.7, leading=12.3,
            textColor=INK,
        )),
    ]
    table = Table([[cell]], colWidths=[PAGE_W - LEFT - RIGHT])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("LINEBEFORE", (0, 0), (0, -1), 3, color),
        ("BOX", (0, 0), (-1, -1), 0.5, color),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return table


def data_table(
    rows: Sequence[Sequence[object]],
    widths: Sequence[float] | None = None,
    font_size: float = 7.35,
    repeat_rows: int = 1,
    alignments: dict[int, str] | None = None,
) -> Table:
    total_width = PAGE_W - LEFT - RIGHT
    if widths is None:
        widths = [total_width / len(rows[0])] * len(rows[0])
    converted = []
    for row_index, row in enumerate(rows):
        converted.append([
            Paragraph(
                str(value),
                ParagraphStyle(
                    f"Table-{row_index}-{column_index}",
                    fontName=BODY_FONT,
                    fontSize=font_size,
                    leading=font_size + 2.5,
                    textColor=WHITE if row_index == 0 else INK,
                    alignment={"left": TA_LEFT, "right": TA_RIGHT, "center": TA_CENTER}.get(
                        (alignments or {}).get(column_index, "left"), TA_LEFT
                    ),
                ),
            )
            for column_index, value in enumerate(row)
        ])
    table = Table(converted, colWidths=list(widths), repeatRows=repeat_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, FAINT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, HexColor("#F7F6F2")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    table.setStyle(TableStyle(commands))
    return table


def metric_strip(items: Sequence[tuple[str, str, str]], width: float | None = None) -> Table:
    total = width or (PAGE_W - LEFT - RIGHT)
    cells = []
    for label, value, note in items:
        cells.append([
            Paragraph(label.upper(), ParagraphStyle(
                "MetricLabel", fontName=BODY_FONT, fontSize=6.6, leading=8,
                textColor=MUTED,
            )),
            Paragraph(value, ParagraphStyle(
                "MetricValue", fontName=SERIF_FONT, fontSize=19, leading=22,
                textColor=INK, spaceBefore=3, spaceAfter=2,
            )),
            Paragraph(note, S["small"]),
        ])
    table = Table([cells], colWidths=[total / len(cells)] * len(cells))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WHITE),
        ("BOX", (0, 0), (-1, -1), 0.6, FAINT),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, FAINT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def label(d: Drawing, x: float, y: float, text: str, size=7.5, color=INK, anchor="start", font=BODY_FONT):
    d.add(String(x, y, text, fontName=font, fontSize=size, fillColor=color, textAnchor=anchor))


def pipeline_figure() -> Drawing:
    width = PAGE_W - LEFT - RIGHT
    height = 73 * mm
    d = Drawing(width, height)
    stages = [
        ("Advertiser world", "Known only in simulation", TEAL),
        ("Candidate MMMs", "48 truth-blind fits", INDIGO),
        ("Posterior + diagnostics", "Observable model evidence", NAVY),
        ("Budget decisions", "4 declared scenarios", ORANGE),
        ("Revealed economic loss", "Truth used after action", RED),
        ("Learned selector", "Predict mean + tail loss", INDIGO_DARK),
    ]
    box_w = 49 * mm
    box_h = 17 * mm
    positions = [(2, 48), (61, 48), (120, 48), (120, 15), (61, 15), (2, 15)]
    for index, ((title, subtitle, color), (x_mm, y_mm)) in enumerate(zip(stages, positions)):
        x, y = x_mm * mm, y_mm * mm
        d.add(Rect(x, y, box_w, box_h, rx=5, ry=5, fillColor=colors.white, strokeColor=color, strokeWidth=1.2))
        d.add(Rect(x, y, 3.5 * mm, box_h, fillColor=color, strokeColor=None))
        label(d, x + 6 * mm, y + 10.5 * mm, title, 8.2, INK)
        label(d, x + 6 * mm, y + 5.3 * mm, subtitle, 6.4, MUTED)
        if index < len(stages) - 1:
            nx_mm, ny_mm = positions[index + 1]
            if abs(ny_mm - y_mm) < 1:
                x1, y1 = x + box_w, y + box_h / 2
                x2, y2 = nx_mm * mm, ny_mm * mm + box_h / 2
            elif index == 2:
                x1, y1 = x + box_w / 2, y
                x2, y2 = nx_mm * mm + box_w / 2, ny_mm * mm + box_h
            else:
                x1, y1 = x, y + box_h / 2
                x2, y2 = nx_mm * mm + box_w, ny_mm * mm + box_h / 2
            d.add(Line(x1, y1, x2, y2, strokeColor=HexColor("#B9BAC4"), strokeWidth=1.2))
            angle = math.atan2(y2 - y1, x2 - x1)
            for delta in (2.55, -2.55):
                d.add(Line(x2, y2, x2 - 5 * math.cos(angle + delta), y2 - 5 * math.sin(angle + delta), strokeColor=HexColor("#B9BAC4"), strokeWidth=1.2))
    label(d, width / 2, 5 * mm, "Deployment: the learned mapping ranks new truth-blind candidate MMMs; synthetic truth is never an input.", 7.2, MUTED, "middle")
    return d


def dgp_figure() -> Drawing:
    width = PAGE_W - LEFT - RIGHT
    height = 92 * mm
    d = Drawing(width, height)
    nodes = {
        "Latent demand": (3, 70, TEAL),
        "Commercial plan": (62, 70, TEAL),
        "Season and promo": (121, 70, TEAL),
        "Media spend": (18, 47, INDIGO),
        "Baseline outcome": (119, 47, NAVY),
        "Delivery": (18, 24, INDIGO),
        "Adstock": (67, 24, ORANGE),
        "Saturation": (116, 24, ORANGE),
        "Observed outcome": (116, 2, RED),
    }
    sizes = {}
    for text, (x_mm, y_mm, color) in nodes.items():
        width_mm = 35 if len(text) < 13 else 42
        height_mm = 12
        x, y = x_mm * mm, y_mm * mm
        sizes[text] = (x, y, width_mm * mm, height_mm * mm)
        d.add(Rect(x, y, width_mm * mm, height_mm * mm, rx=4, ry=4, fillColor=colors.white, strokeColor=color, strokeWidth=1.1))
        label(d, x + width_mm * mm / 2, y + 4.6 * mm, text, 7.3, INK, "middle")

    edges = [
        ("Latent demand", "Media spend", HexColor("#A8A9B4")),
        ("Latent demand", "Baseline outcome", HexColor("#A8A9B4")),
        ("Commercial plan", "Media spend", HexColor("#A8A9B4")),
        ("Commercial plan", "Baseline outcome", HexColor("#A8A9B4")),
        ("Season and promo", "Media spend", HexColor("#A8A9B4")),
        ("Season and promo", "Baseline outcome", HexColor("#A8A9B4")),
        ("Media spend", "Delivery", INDIGO),
        ("Delivery", "Adstock", ORANGE),
        ("Adstock", "Saturation", ORANGE),
        ("Saturation", "Observed outcome", RED),
        ("Baseline outcome", "Observed outcome", NAVY),
    ]
    for source, target, edge_color in edges:
        sx, sy, sw, sh = sizes[source]
        tx, ty, tw, th = sizes[target]
        if abs(ty - sy) < 1:
            x1, y1 = sx + sw, sy + sh / 2
            x2, y2 = tx, ty + th / 2
        elif ty < sy:
            x1, y1 = sx + sw / 2, sy
            x2, y2 = tx + tw / 2, ty + th
        else:
            y1 = sy + sh
            y2 = ty
            x1, x2 = sx + sw / 2, tx + tw / 2
        d.add(Line(x1, y1, x2, y2, strokeColor=edge_color, strokeWidth=0.9))
        angle = math.atan2(y2 - y1, x2 - x1)
        for delta in (2.55, -2.55):
            d.add(Line(x2, y2, x2 - 4 * math.cos(angle + delta), y2 - 4 * math.sin(angle + delta), strokeColor=edge_color, strokeWidth=0.9))
    label(d, 2 * mm, 87 * mm, "Gray paths create endogeneity; colored paths form the media response mechanism.", 6.6, MUTED)
    return d


def roi_distribution_figure() -> Drawing:
    width = PAGE_W - LEFT - RIGHT
    height = 58 * mm
    d = Drawing(width, height)
    channels = [
        ("Paid social", SIMULATOR_DATA["coverage"]["channel"]["paid_social"]["roi"], TEAL),
        ("Nonbrand search", SIMULATOR_DATA["coverage"]["channel"]["search"]["roi"], INDIGO),
        ("TV / CTV", SIMULATOR_DATA["coverage"]["channel"]["tv"]["roi"], ORANGE),
    ]
    x0, x1 = 41 * mm, width - 8 * mm
    max_value = 10
    for tick in range(0, 11, 2):
        x = x0 + (x1 - x0) * tick / max_value
        d.add(Line(x, 10 * mm, x, 48 * mm, strokeColor=FAINT, strokeWidth=0.6))
        label(d, x, 5 * mm, f"{tick}x", 6.4, MUTED, "middle")
    for row, (name, values, color) in enumerate(channels):
        y = (42 - row * 14) * mm
        label(d, 2 * mm, y - 1.5 * mm, name, 7.5, INK)
        p10 = values["p10"]
        med = values["median"]
        p90 = values["p90"]
        minimum = values["minimum"]
        maximum = values["maximum"]
        scale = lambda value: x0 + (x1 - x0) * min(max(value, 0), max_value) / max_value
        d.add(Line(scale(minimum), y, scale(maximum), y, strokeColor=HexColor("#C7C8D0"), strokeWidth=1.2))
        d.add(Line(scale(p10), y, scale(p90), y, strokeColor=color, strokeWidth=6))
        d.add(Circle(scale(med), y, 3.2, fillColor=WHITE, strokeColor=color, strokeWidth=1.5))
        label(d, min(scale(maximum) + 3, x1 - 15), y + 3.2 * mm, f"median {med:.2f}x", 6.1, color)
    label(d, 2 * mm, 53 * mm, "Injected full-history incremental ROI distributions", 8.3, INK)
    label(d, width - 2 * mm, 53 * mm, "thin: min-max | thick: P10-P90 | dot: median", 6.4, MUTED, "end")
    return d


def ablation_figure() -> Drawing:
    width = PAGE_W - LEFT - RIGHT
    height = 72 * mm
    d = Drawing(width, height)
    labels = ["Base features\nBase safety", "SVI features\nBase safety", "Base features\nSVI safety", "SVI features\nSVI safety"]
    keys = ["baseline", "featureOnly", "safetyOnly", "fullSvi"]
    values = [DEVELOPMENT_DATA["comparison"][key]["primary"]["objective"] for key in keys]
    chart = VerticalBarChart()
    chart.x = 18 * mm
    chart.y = 16 * mm
    chart.height = 44 * mm
    chart.width = width - 30 * mm
    chart.data = [values]
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 14
    chart.valueAxis.valueStep = 2
    chart.valueAxis.labels.fontName = BODY_FONT
    chart.valueAxis.labels.fontSize = 6.5
    chart.valueAxis.strokeColor = FAINT
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontName = BODY_FONT
    chart.categoryAxis.labels.fontSize = 6.2
    chart.categoryAxis.labels.dy = -7
    chart.categoryAxis.strokeColor = FAINT
    chart.bars[0].fillColor = INDIGO
    chart.bars[0].strokeColor = None
    chart.barSpacing = 5
    d.add(chart)
    for index, value in enumerate(values):
        x = chart.x + chart.width * (index + 0.5) / len(values)
        y = chart.y + chart.height * value / 14
        label(d, x, y + 3, f"{value:.2f}", 7, INK, "middle")
    label(d, 2 * mm, 66 * mm, "Primary selection objective (lower is better)", 8.3, INK)
    return d


def result_tradeoff_figure() -> Drawing:
    width = PAGE_W - LEFT - RIGHT
    height = 78 * mm
    d = Drawing(width, height)
    baseline = DEVELOPMENT_DATA["comparison"]["baseline"]["proxy"]
    full = DEVELOPMENT_DATA["comparison"]["fullSvi"]["proxy"]
    metrics = [
        ("Economic pair\naccuracy", baseline["economicallyWeightedPairwiseAccuracy"] * 100, full["economicallyWeightedPairwiseAccuracy"] * 100, 100, "%"),
        ("Champion mean\nexcess loss", baseline["riskChampionMeanExcessLoss"], full["riskChampionMeanExcessLoss"], 2.5, ""),
        ("Champion P90\nexcess loss", baseline["riskChampionP90ExcessLoss"], full["riskChampionP90ExcessLoss"], 6.2, ""),
        ("Dangerous false\nchampions", baseline["dangerousFalseChampionShare"] * 100, full["dangerousFalseChampionShare"] * 100, 60, "%"),
    ]
    x_start = 7 * mm
    group_w = (width - 14 * mm) / len(metrics)
    for index, (name, a, b, max_val, suffix) in enumerate(metrics):
        x = x_start + index * group_w
        y0 = 19 * mm
        usable = 42 * mm
        for offset, value, color in [(0.31, a, HexColor("#BFC0C9")), (0.58, b, INDIGO)]:
            bar_w = 8 * mm
            bar_h = usable * min(value / max_val, 1)
            bx = x + group_w * offset - bar_w / 2
            d.add(Rect(bx, y0, bar_w, bar_h, fillColor=color, strokeColor=None))
            shown = f"{value:.1f}{suffix}"
            label(d, bx + bar_w / 2, y0 + bar_h + 3, shown, 6.5, INK, "middle")
        for tick in range(5):
            y = y0 + usable * tick / 4
            d.add(Line(x + 2 * mm, y, x + group_w - 2 * mm, y, strokeColor=FAINT, strokeWidth=0.4))
        label(d, x + group_w / 2, 6 * mm, name.split("\n")[0], 6.6, INK, "middle")
        label(d, x + group_w / 2, 2 * mm, name.split("\n")[1], 6.6, INK, "middle")
    label(d, 3 * mm, 71 * mm, "A useful representation, an unresolved selection rule", 8.3, INK)
    d.add(Rect(width - 58 * mm, 68 * mm, 4 * mm, 3 * mm, fillColor=HexColor("#BFC0C9"), strokeColor=None))
    label(d, width - 52 * mm, 68 * mm, "V5D baseline", 6.2, MUTED)
    d.add(Rect(width - 28 * mm, 68 * mm, 4 * mm, 3 * mm, fillColor=INDIGO, strokeColor=None))
    label(d, width - 22 * mm, 68 * mm, "V5D-SVI", 6.2, MUTED)
    return d


def safety_pool_figure() -> Drawing:
    width = PAGE_W - LEFT - RIGHT
    height = 51 * mm
    d = Drawing(width, height)
    audit = DEVELOPMENT_DATA["safetyAudit"]
    total = audit["candidateRows"]
    blocks = [
        ("Safe under both", audit["safeUnderBoth"], TEAL),
        ("Base only", audit["mapOnly"], NAVY),
        ("SVI only", audit["sviOnly"], ORANGE),
        ("Neither", total - audit["safeUnderBoth"] - audit["mapOnly"] - audit["sviOnly"], HexColor("#D2D3D8")),
    ]
    x, y, bar_w, bar_h = 4 * mm, 22 * mm, width - 8 * mm, 11 * mm
    cursor = x
    for name, value, color in blocks:
        w = bar_w * value / total
        d.add(Rect(cursor, y, w, bar_h, fillColor=color, strokeColor=PAPER, strokeWidth=0.5))
        if w > 20 * mm:
            label(d, cursor + w / 2, y + 4.1 * mm, f"{value:,}", 7.1, WHITE if color != HexColor("#D2D3D8") else INK, "middle")
        cursor += w
    legend_x = 4 * mm
    for index, (name, value, color) in enumerate(blocks):
        lx = legend_x + (index % 2) * 84 * mm
        ly = (13 - (index // 2) * 8) * mm
        d.add(Rect(lx, ly, 4 * mm, 3 * mm, fillColor=color, strokeColor=None))
        label(d, lx + 6 * mm, ly, f"{name}: {value:,} ({100 * value / total:.1f}%)", 6.5, MUTED)
    label(d, 4 * mm, 42 * mm, "Candidate safety sets across 20,160 development fits", 8.3, INK)
    return d


def group_importance_figure() -> Drawing:
    width = PAGE_W - LEFT - RIGHT
    height = 88 * mm
    d = Drawing(width, height)
    groups = IMPORTANCE_DATA["risk"]["groups"]
    names = [item["group"].replace("-and-", " + ").replace("-", " ") for item in groups]
    values = [100 * item["importance"] for item in groups]
    chart = HorizontalBarChart()
    chart.x = 61 * mm
    chart.y = 9 * mm
    chart.height = 70 * mm
    chart.width = width - 70 * mm
    chart.data = [list(reversed(values))]
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 26
    chart.valueAxis.valueStep = 5
    chart.valueAxis.labels.fontName = BODY_FONT
    chart.valueAxis.labels.fontSize = 6.3
    chart.valueAxis.labelTextFormat = "%d%%"
    chart.valueAxis.strokeColor = FAINT
    chart.categoryAxis.categoryNames = list(reversed(names))
    chart.categoryAxis.labels.fontName = BODY_FONT
    chart.categoryAxis.labels.fontSize = 6.3
    chart.categoryAxis.labels.dx = -4
    chart.categoryAxis.strokeColor = FAINT
    chart.bars[0].fillColor = INDIGO
    chart.bars[0].strokeColor = None
    chart.barSpacing = 2
    d.add(chart)
    label(d, 3 * mm, 82 * mm, "Predictive importance of feature families", 8.3, INK)
    return d


def top_feature_figure() -> Drawing:
    width = PAGE_W - LEFT - RIGHT
    height = 93 * mm
    d = Drawing(width, height)
    features = IMPORTANCE_DATA["top50RiskFeatures"][:15]
    x0 = 78 * mm
    x1 = width - 8 * mm
    maximum = max(100 * item["importance"] for item in features)
    row_h = 5.35 * mm
    for index, item in enumerate(features):
        y = (84 * mm) - index * row_h
        name = item["feature"].replace("svi:", "").replace("diagnostic:", "").replace("interaction:", "")
        if len(name) > 39:
            name = name[:37] + ".."
        label(d, 2 * mm, y, f"{index + 1}. {name}", 5.9, INK)
        value = 100 * item["importance"]
        w = (x1 - x0) * value / maximum
        color = TEAL if item["meanCoefficient"] < 0 else ORANGE
        d.add(Rect(x0, y - 1.1 * mm, w, 2.8 * mm, fillColor=color, strokeColor=None))
        label(d, min(x0 + w + 2, x1 - 9 * mm), y, f"{value:.2f}%", 5.8, MUTED)
    label(d, 2 * mm, 89 * mm, "Top 15 standardized risk coefficients", 8.3, INK)
    label(d, width - 2 * mm, 89 * mm, "teal: predicts lower loss | orange: predicts higher loss", 6.2, MUTED, "end")
    return d


def novelty_matrix() -> Table:
    rows = [
        ["Research stream", "What it contributes", "What remains missing", "FluxMMM bridge"],
        ["Bayesian MMM", "Carryover, saturation, priors, posterior uncertainty", "Model selection often follows fit criteria or analyst judgment", "Treats every Bayesian MMM as a candidate whose decision loss can be learned"],
        ["Automated MMM search", "Large hyperparameter exploration and Pareto fronts", "Objectives are usually proxies such as prediction or decomposition", "Learns a cross-advertiser risk function from revealed synthetic economics"],
        ["Decision-focused learning", "Optimizes prediction systems for downstream actions", "Rarely formalized for selecting causal MMM specifications", "Uses budget regret as the supervised target and MMM diagnostics as features"],
        ["Synthetic MMM benchmarks", "Known causal truth under endogeneity and nonlinear response", "Often compare estimators rather than train a reusable selector", "Turns many synthetic worlds into a meta-learning dataset"],
        ["This study", "Open, reproducible selector plus posterior and causal diagnostics", "No fresh external validation yet", "A testable protocol for learning what makes an MMM decision-useful"],
    ]
    return data_table(rows, widths=[30 * mm, 49 * mm, 48 * mm, 47 * mm], font_size=6.6)


def section_label(text: str) -> Table:
    table = Table([[Paragraph(text.upper(), ParagraphStyle(
        "SectionLabel", fontName=BODY_FONT, fontSize=7.1, leading=9,
        textColor=INDIGO, alignment=TA_LEFT,
    ))]], colWidths=[PAGE_W - LEFT - RIGHT])
    table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.7, INDIGO),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def cover(story: list[Flowable]) -> None:
    story.extend([
        ColorBand(8 * mm),
        Spacer(1, 18 * mm),
        Paragraph("FLUXMMM V5D-SVI", ParagraphStyle(
            "CoverLabel", fontName=BODY_FONT, fontSize=9, leading=11,
            textColor=INDIGO, tracking=1.6,
        )),
        Spacer(1, 5 * mm),
        Paragraph(
            "Learning What Makes a Marketing Mix Model Decision-Useful",
            ParagraphStyle(
                "CoverTitle", fontName=SERIF_FONT, fontSize=32, leading=36,
                textColor=INK, spaceAfter=7 * mm,
            ),
        ),
        Paragraph(
            "A decision-focused, cross-advertiser framework for selecting Bayesian MMM specifications from synthetic causal truth and variational posteriors",
            ParagraphStyle(
                "CoverSubtitle", fontName=BODY_FONT, fontSize=13.2, leading=18,
                textColor=MUTED, spaceAfter=13 * mm,
            ),
        ),
        metric_strip([
            ("Advertiser worlds", "500", "Heterogeneous synthetic businesses"),
            ("Candidate fits", "24,000", "48 truth-blind MMMs per business"),
            ("Development fits", "20,160", "420 businesses in cross-fitting"),
            ("Decision settings", "4", "Reduction, mix, growth, ceiling"),
        ]),
        Spacer(1, 12 * mm),
        Table([
            [Paragraph("AUTHOR", S["small"]), Paragraph("Gustavo Bramao", ParagraphStyle("Author", fontName=SERIF_FONT, fontSize=15, leading=18, textColor=INK))],
            [Paragraph("STATUS", S["small"]), Paragraph("Working paper for methodological review", S["body_left"])],
            [Paragraph("RELEASE", S["small"]), Paragraph("Public research release, frozen as evaluated", S["body_left"])],
            [Paragraph("DATE", S["small"]), Paragraph("September 2026", S["body_left"])],
        ], colWidths=[31 * mm, 140 * mm], style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, FAINT),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])),
        Spacer(1, 10 * mm),
        callout(
            "Central contribution",
            "FluxMMM reframes MMM specification choice as a supervised decision-risk problem. Hidden causal truth is used only to label the economic consequences of candidate models in simulation. A learned selector then maps truth-blind diagnostics, posterior summaries, and model assumptions to expected and tail economic loss for future advertisers.",
            TEAL,
            TEAL_PALE,
        ),
        Spacer(1, 8 * mm),
        P("This manuscript reports a development study, not a production authorization. The sealed 80-business audit remains excluded from V5D-SVI, and no claim of state-of-the-art performance or external generalization is made.", "small"),
        PageBreak(),
    ])


def add_toc(story: list[Flowable]) -> None:
    story.append(H1("Contents"))
    toc = TableOfContents()
    toc.levelStyles = [S["toc_h1"], S["toc_h2"]]
    story.extend([toc, PageBreak()])


def add_abstract(story: list[Flowable]) -> None:
    story.extend([
        section_label("Abstract"),
        Spacer(1, 3 * mm),
        P(
            "Marketing mix models are ultimately used to choose budgets, yet candidate models are commonly selected using predictive fit, decomposition plausibility, calibration error, or expert judgment rather than the economic quality of the decisions they induce. This paper introduces FluxMMM V5D-SVI, an open research framework that learns <i>what makes an MMM decision-useful</i> across heterogeneous advertisers. The procedure generates 500 synthetic direct-to-consumer business worlds with known causal response functions, endogenous media planning, channel-specific delivery, carryover, saturation, seasonality, promotion, and imperfect external evidence. For each world, 24 Bayesian model specifications are crossed with two evidence policies, producing 48 truth-blind candidate MMMs. Full-rank variational inference supplies posterior response and reliability summaries. Each candidate selects budget actions in four predeclared scenarios; synthetic truth is revealed only after the action, producing posterior-expected economic decision loss. A cross-fitted meta-model then predicts each candidate's mean and 90th-percentile excess loss from observable diagnostics, assumptions, interactions, and posterior summaries. Selection minimizes a business-controlled risk functional equal to 65% predicted mean loss plus 35% predicted tail loss within a declared safety pool.",
        ),
        P(
            "The contribution is methodological rather than a new response curve or sampler: MMM selection becomes a learning problem trained on the downstream economic consequences of complete Bayesian models. In the frozen 420-business development cohort (20,160 candidate fits), posterior-aware features increased economically weighted pairwise ordering accuracy from 66.1% to 72.5%, and the full V5D-SVI procedure reduced the primary composite selection objective by 22.9% relative to its predecessor. However, the current posterior safety contract admitted 18,703 candidates versus 6,088 under the earlier safety rule, and champion mean excess loss increased from 1.41 to 2.12. The development evidence therefore supports the value of inference-matched features but rejects the stronger claim that the current selector is production-ready. This failure is scientifically useful: representation quality, safety-set construction, and top-one selection must be evaluated separately. We provide the complete data-generating process, loss definitions, cross-fitting protocol, learned feature importance, failure analysis, and an open roadmap for sealed validation, real-experiment benchmarking, hierarchical meta-learning, and decision-theoretic guarantees.",
        ),
        Spacer(1, 3 * mm),
        callout(
            "Novelty statement",
            "To our knowledge, among the MMM and decision-focused-learning literature reviewed, this is the first openly specified attempt to learn a reusable cross-advertiser MMM selector directly from posterior-expected budget decision loss while keeping synthetic truth out of the deployed feature set. This is a qualified literature claim, not a proof of priority.",
            ORANGE,
            ORANGE_PALE,
        ),
        Spacer(1, 4 * mm),
        P("<b>Keywords:</b> marketing mix modeling; Bayesian causal inference; decision-focused learning; model selection; budget optimization; synthetic data; variational inference; economic regret; open science.", "body_left"),
        PageBreak(),
    ])


def add_main_sections(story: list[Flowable]) -> None:
    story.extend([
        H1("1. Introduction"),
        P(
            "A marketing mix model (MMM) is not merely a forecasting model. Its operational purpose is to estimate incremental channel response, separate that response from baseline demand, and support changes to budget and media mix. The hardest practical question is therefore not only <i>which model fits the observed time series?</i> It is <i>which fitted model will lead to the least damaging decision when causal truth is unknown?</i> Standard fit metrics cannot answer that question on their own. Two candidate MMMs can forecast equally well while assigning radically different effects to correlated channels, implying different marginal returns and different allocations [1-3].",
        ),
        P(
            "This problem is amplified by the geometry of MMM data. Media spends are planned rather than randomized. Search spend often rises when demand rises. Brand campaigns are synchronized with promotions. Carryover makes today's outcome depend on prior exposure. Saturation makes an average return a poor guide to the next dollar. Informative priors and experiments can help, but they also introduce questions about transportability, overlap, conflict, and decision dependence. A plausible posterior is not automatically a useful budget surface.",
        ),
        P(
            "Existing open MMM systems have made major advances. Bayesian MMMs incorporate carryover, saturation, hierarchical structure, prior knowledge, and posterior uncertainty [1,4,5]. Robyn automates hyperparameter search using predictive error, decomposition plausibility, and lift calibration objectives [6]. Meridian supports Bayesian ROI priors, experiment calibration, and budget optimization [7,8]. Synthetic benchmarks increasingly expose endogenous spend and known ground truth [9]. What remains underdeveloped is a reusable, empirically learned rule for selecting among complete MMM candidates based on the <i>economic loss of the decisions those candidates would recommend</i> across many advertiser worlds.",
        ),
        P(
            "FluxMMM V5D-SVI addresses that gap by introducing a second level of learning. The first level fits many candidate MMMs to each advertiser. The second level learns, across simulated advertisers, which observable properties of those fitted candidates predict low economic decision loss. At deployment, a new advertiser has no causal truth. FluxMMM therefore uses only truth-blind features available from the data, model specification, validation diagnostics, evidence contract, and posterior. The hidden truth that created the training labels is never exposed to the selector as a feature.",
        ),
        Spacer(1, 3 * mm),
        pipeline_figure(),
        P("<b>Figure 1.</b> The FluxMMM research loop. Synthetic truth labels the consequences of actions but is excluded from candidate fitting and from the deployed selector.", "caption"),
        H2("1.1 Research objective"),
        P(
            "Let an advertiser be a task, and let each candidate be a full causal modeling contract: baseline, controls, media transformations, prior distribution, calibration route, likelihood, optional time-varying coefficients, and optional latent planning structure. The objective is to learn a function that ranks these candidates by the economic decision loss they are expected to produce for an unseen advertiser. The learned function should be useful even though the true causal response is unavailable in production.",
        ),
        equation(
            "Observable candidate evidence  x_(b,c)  ->  learned risk  R_hat_(b,c)",
            "Selected model  c_hat_b = arg min_c R_hat_(b,c), subject to declared safety",
            "Training label  Delta_(b,c) = L_(b,c) - min_(c' in safe pool) L_(b,c')",
        ),
        H2("1.2 Contributions"),
        bullet("<b>Decision-focused MMM selection.</b> Candidate models are supervised by the economic consequences of the budget actions they induce, not by a hand-weighted aggregate of diagnostics."),
        bullet("<b>Cross-advertiser meta-learning.</b> Heterogeneous synthetic businesses create repeated tasks from which the selector learns relationships that may transfer to a new advertiser."),
        bullet("<b>Truth-blind deployment contract.</b> Causal truth, generator family, business identity, candidate identity, proposal order, and realized decision loss are explicitly forbidden inputs."),
        bullet("<b>Inference-matched evidence.</b> Full-rank variational posteriors supply ROI location, cross-channel dispersion, uncertainty, predictive coverage, convergence, and cross-seed agreement features."),
        bullet("<b>Mean-tail economic risk.</b> The selector estimates both conditional mean excess loss and conditional P90 excess loss; business risk appetite determines their final combination."),
        bullet("<b>Two-axis cross-fitting.</b> Evaluation excludes both the held-out advertiser and the held-out candidate-parameter region, limiting memorization of businesses and model families."),
        bullet("<b>Open failure analysis.</b> The development study reports both improved weighted ranking and degraded champion loss, isolating a permissive safety set as a major unresolved mechanism."),
        bullet("<b>Reproducible open research.</b> The simulator, candidates, inference contracts, feature construction, learner, frozen artifacts, and generated importance receipts are available in source form."),
        H2("1.3 Scope and non-claims"),
        P(
            "V5D-SVI is a research release. It does not establish that its synthetic population represents every advertiser, that variational inference exactly reproduces the Bayesian posterior, that the learned selector generalizes to real firms, or that the current safety gates are optimal. The sealed audit businesses were not used in this release, and there is no fresh external validation cohort. Accordingly, the paper makes a contribution of formulation, infrastructure, transparent evidence, and falsifiable hypotheses rather than a final state-of-the-art claim.",
        ),
        PageBreak(),

        H1("2. Positioning in the literature"),
        H2("2.1 Bayesian marketing mix modeling"),
        P(
            "Bayesian MMMs provide a coherent language for carryover, diminishing returns, prior knowledge, and uncertainty. Jin et al. formalized carryover and shape effects and showed that priors can materially affect posterior estimates when time series are short [1]. Later work introduced flexible time-varying coefficients [4] and direct ROI reparameterization for experiment-calibrated priors [5]. These contributions concern how to specify and estimate an MMM. FluxMMM instead asks how to choose among many plausible complete specifications when their observational fit and business implications disagree.",
        ),
        H2("2.2 Automated model selection in MMM"),
        P(
            "Robyn demonstrates the practical value of automated search over adstock, saturation, regularization, and validation windows. Its multi-objective procedure considers normalized prediction error, decomposition distance, and experiment calibration error [6]. This reduces manual tuning, but the objective remains a designed proxy for model quality. FluxMMM uses those types of diagnostics as inputs and learns their conditional relevance from simulated decision loss. In other words, a diagnostic is not assigned a universal weight in advance; its contribution is estimated jointly with model assumptions, temporal evidence, posterior summaries, and interactions.",
        ),
        H2("2.3 Synthetic truth for causal validation"),
        P(
            "Ground-truth causal ROI is rarely available for observational MMM data. Synthetic simulation makes it possible to evaluate recovery under controlled confounding, nonlinear response, and evidence error. The recent Heusch benchmark explicitly models endogenous marketing spend and documents the causal decomposition [9]. FluxMMM follows the same scientific motivation but uses a broader population of heterogeneous business worlds and converts each fitted model into a downstream decision label. It also simulates window-aligned incrementality evidence rather than equating a short experiment with full-history ROI.",
        ),
        H2("2.4 Decision-focused learning and learning to rank"),
        P(
            "Decision-focused learning evaluates a predictive system by the quality of the optimization decision it supports rather than by prediction error alone [10,11]. Learning-to-rank methods similarly learn orderings when top candidates matter more than calibrated point prediction [12]. FluxMMM imports these ideas into causal model selection. The unit being ranked is not an advertisement or customer; it is an entire fitted MMM specification. The label is not a click or a sale; it is excess economic loss after a model has recommended an allocation.",
        ),
        H2("2.5 Relation to causal learning"),
        P(
            "The selector does not make an observational MMM causal by prediction. Causal credibility still depends on design, controls, external experiments, assumptions, and sensitivity analysis. The meta-model instead learns which combinations of those observable signals have historically accompanied low synthetic decision loss. This is closer to empirical model-risk estimation than to causal identification itself. The distinction is essential: a learned score can prioritize candidates, but it cannot manufacture randomization or prove the absence of unobserved confounding.",
        ),
        Spacer(1, 3 * mm),
        novelty_matrix(),
        P("<b>Table 1.</b> FluxMMM connects adjacent research streams but does not replace their identification assumptions.", "caption"),
        PageBreak(),

        H1("3. Formal problem statement"),
        H2("3.1 Advertisers, candidates, and observations"),
        P(
            "Index advertisers by <i>b</i>, candidate specifications by <i>c</i>, media channels by <i>j</i>, and time by <i>t</i>. An advertiser supplies an outcome y, media spends X, and observed controls Z. A candidate c defines a transformation and inference contract. A generic candidate model is:",
        ),
        equation(
            "y_(b,t) = baseline_(b,t) + sum_j beta_(b,j,t) * s_(c,j)(a_(c,j)(X_(b,j,1:t)))",
            "            + gamma_c' Z_(b,t) + epsilon_(b,t)",
            "a(.) = carryover operator;  s(.) = saturation operator",
        ),
        P(
            "The baseline contains intercept, trend, Fourier seasonality, cycle, promotions, and available demand controls. Depending on the contract, beta can be constant or time-varying, and an optional latent planning factor can absorb common commercial intensity. The likelihood can be Gaussian, Student-t, or log-normal. External evidence can enter through informative coefficient or ROI priors, or as a noisy measurement likelihood.",
        ),
        H2("3.2 Candidate posterior and decision"),
        P(
            "Fitting candidate c yields an approximate posterior q_(b,c)(theta). For a declared scenario k and feasible allocation set A_(b,k), the candidate chooses the action that maximizes posterior expected incremental profit:",
        ),
        equation(
            "a_hat_(b,c,k) = arg max_(a in A_(b,k)) E_q[ Profit_(b,c,k)(a, theta) ]",
            "Profit_(b,c,k)(a, theta) = margin_b * DeltaOutcome_(b,c)(a, theta) - NetSpend_b(a)",
        ),
        P(
            "Crucially, a single action is chosen from the posterior expected decision surface. The procedure does not allow each posterior draw to choose its own optimal action and then average those optimistic draw-specific outcomes. That separation prevents an artificial information advantage.",
        ),
        H2("3.3 Revealed economic loss"),
        P(
            "Only after the action is selected does the simulator reveal true profit. The scenario-specific economic loss compares the oracle profit with the realized profit of the candidate action and normalizes by a business-relevant scale:",
        ),
        equation(
            "gap_(b,c,k) = max(0, Profit*_(b,k) - Profit_true_(b,k)(a_hat_(b,c,k)))",
            "ell_(b,c,k) = gap_(b,c,k) / max(|Profit*_(b,k)|, 0.02 * historicalSpend_b, 1)",
            "L_(b,c) = 0.70 * mean_k ell_(b,c,k) + 0.30 * max_k ell_(b,c,k)",
        ),
        P(
            "The 70/30 aggregation values routine performance while retaining a penalty for the worst declared scenario. It is a business-protection preference, not a law of nature. The per-scenario losses are preserved so later work can estimate a full multi-task loss vector or let a user specify a different risk utility without retraining the MMMs.",
        ),
        H2("3.4 Excess loss and the safe oracle"),
        P(
            "Absolute decision difficulty varies by advertiser. V5D-SVI therefore learns within-business excess loss relative to the best candidate allowed by the same safety policy:",
        ),
        equation(
            "c*_(b) = arg min_(c in S_b) L_(b,c)",
            "Delta_(b,c) = max(0, L_(b,c) - L_(b,c*)) / s_econ",
            "s_econ = 1 in the frozen release; Delta is uncapped",
        ),
        P(
            "This target asks a deployable question: among the candidates FluxMMM would permit itself to consider, how much economic opportunity does a candidate lose relative to the best available alternative? The oracle is local to the candidate library and safety policy. It is not the globally optimal causal model, and changing the safety set changes the target.",
        ),
        H2("3.5 Learned risk"),
        P(
            "The meta-model estimates a conditional mean and a conditional 90th percentile for Delta. The final risk is:",
        ),
        equation(
            "R_hat_(b,c) = mu_hat_(b,c) + rho * (q90_hat_(b,c) - mu_hat_(b,c))",
            "               = (1-rho) * mu_hat_(b,c) + rho * q90_hat_(b,c)",
            "rho = 0.35 in V5D-SVI, so Risk = 0.65 * mean + 0.35 * P90",
        ),
        P(
            "Lower risk is better. The parameter rho represents business aversion to costly tail mistakes. It is intentionally separated from learned feature coefficients. The machine learns how features predict loss; the business chooses how much to emphasize expected loss versus a bad-but-plausible outcome.",
        ),
        H2("3.6 Finite-candidate regret bound"),
        P(
            "The ranking term has a direct decision-theoretic motivation. For any finite safe candidate set, let c* be the economic oracle, let c_hat be the candidate with minimum predicted risk, and let Delta_c be non-negative excess economic loss. Then selected-model excess loss is bounded by the sum of economic gaps for every candidate that the learned risk misranks ahead of the oracle:",
        ),
        equation(
            "Delta_(c_hat) <= sum_(c != c*) Delta_c * I[ R_hat_c <= R_hat_(c*) ]",
            "Proof: if c_hat=c*, the left side is zero. Otherwise c_hat is one term",
            "on the right because selection guarantees R_hat_(c_hat) <= R_hat_(c*).",
        ),
        P(
            "The bound is exact but can be loose. It explains why economically weighted oracle-versus-candidate ranking is relevant: reducing costly oracle misrankings reduces an upper bound on selected excess loss. It does not guarantee transfer to a new advertiser, and the development result shows that better average weighted ordering can still coexist with poor top-one behavior when correlated errors or a permissive safety set place one harmful candidate at the minimum.",
        ),
        PageBreak(),

        H1("4. Synthetic advertiser population"),
        H2("4.1 Why a population rather than one benchmark"),
        P(
            "A learned selector needs variation across tasks. One synthetic dataset can test whether a model recovers one known truth, but it cannot teach how diagnostic meaning changes across advertisers. FluxMMM generates 500 weekly direct-to-consumer business worlds spanning different histories, noise, confounding, delivery mechanisms, response curves, margins, evidence quality, and decision opportunities. Each world is reproducible from a fixed seed and retains both an observed table and hidden truth arrays.",
        ),
        metric_strip([
            ("Train families", "300", "Balanced DTC, search harvesting, social pressure"),
            ("Development families", "120", "Delayed TV and correlated planning"),
            ("Sealed audit", "80", "Wrong evidence and swapped mechanics"),
            ("History length", "104-260", "Weekly observations"),
        ]),
        Spacer(1, 4 * mm),
        H2("4.2 Causal data-generating graph"),
        dgp_figure(),
        P("<b>Figure 2.</b> Simplified data-generating process. Media is endogenous because latent demand, commercial planning, seasonality, and promotion affect both spend and the non-media baseline.", "caption"),
        H2("4.3 Latent demand, planning, promotion, and baseline"),
        P(
            "Latent demand D, commercial intensity C, and planning intensity P follow persistent stochastic processes. Planning is partly driven by commercial intensity, so it is neither an independent control nor merely random noise. Promotions are more likely during high commercial and demand periods. The hidden baseline is:",
        ),
        equation(
            "D_t = rho_D D_(t-1) + u_t;   C_t = rho_C C_(t-1) + v_t",
            "P_t = rho_P P_(t-1) + 0.45 C_t + 0.8 w_t",
            "logit Pr(promo_t=1) = -2 + 0.55 C_t + 0.20 D_t",
            "B_t = max(30000, 205000 + 105t + 34000D_t + 28000C_t",
            "      + 17000 sin(2*pi*t/52.18) + 46000 promo_t + 24000 event_t)",
        ),
        P(
            "An observed demand proxy is available frequently in balanced businesses and only occasionally in harder families. The unresolved part remains in the error term from the fitted model's perspective. This creates the central endogeneity challenge: a channel can appear effective because it follows demand rather than causes it.",
        ),
        H2("4.4 Media planning and delivery"),
        P(
            "For each channel, latent budget intent is log-linear in demand, planning, annual seasonality, and channel-specific spend volatility. Delivery then converts intent into a channel-appropriate exposure equivalent. Auction media is constrained by demand-linked inventory and stochastic price. Reach media loses delivery efficiency as frequency inflates. TV is flighted through start and continuation probabilities. The simulator therefore distinguishes money spent from causal delivery.",
        ),
        equation(
            "Intent_(j,t) = AvgSpend_j * exp(d_j D_t + p_j P_t + season_t + sigma_j z_(j,t))",
            "Auction: clicks = min(Intent/price, 1.35 * availableClicks)",
            "Reach:   equivalent reach = impressions / (1 + frequency inflation)",
            "TV:      spend is delivered only while a stochastic flight is active",
        ),
        H2("4.5 Carryover and saturation"),
        P(
            "Each channel receives either geometric or Weibull carryover and a Hill saturation curve. TV is much more likely to use Weibull persistence; search is usually short-memory geometric; paid social varies between geometric and Weibull. This channel heterogeneity prevents the estimator from relying on a single universal memory assumption.",
        ),
        equation(
            "Geometric: A_(j,t) = sum_(l=0..39) theta_j^l * delivery_(j,t-l)",
            "Weibull:   A_(j,t) = sum_(l=0..51) w_(j,l;k_j,lambda_j) * delivery_(j,t-l)",
            "Hill:      S_(j,t) = A_(j,t)^h_j / (A_(j,t)^h_j + K_j^h_j)",
            "Contribution_(j,t) = beta_j * S_(j,t)",
        ),
        P(
            "The coefficient beta_j is scaled so that the realized full-history average incremental ROI equals the sampled latent target. This makes causal truth exact while allowing marginal ROI to differ from average ROI because the observed operating point may already be saturated.",
        ),
        Spacer(1, 2 * mm),
        roi_distribution_figure(),
        P("<b>Figure 3.</b> Heterogeneous full-history ROI truth across all 500 generated businesses. Values are deliberately broad and should not be interpreted as empirical industry benchmarks.", "caption"),
        H2("4.6 Outcome noise"),
        P(
            "The deterministic outcome adds baseline and channel contributions. Revenue is then perturbed by mean-corrected multiplicative log-normal noise whose scale increases during event shocks. Across the population, noise share ranges from 1.8% to 13.8%, with a median of 6.2%.",
        ),
        equation(
            "Y_t^det = B_t + sum_j Contribution_(j,t)",
            "Y_t = max(1, Y_t^det * exp(sigma_t e_t - 0.5 sigma_t^2))",
        ),
        PageBreak(),

        H1("5. External evidence and causal anchors"),
        H2("5.1 Industry benchmark arm"),
        P(
            "Each business has two evidence policies. The experiments-only arm uses available experiment evidence and leaves other channels without an industry prior. The benchmark-gap-fill arm keeps experiments where available and supplies a benchmark prior only for channels without an experiment. The benchmark registry is distinct from the latent ROI population; benchmark values are perturbed within ordinary businesses and deliberately corrupted in the sealed wrong-evidence family.",
        ),
        P(
            "This separation matters. If the benchmark were generated directly from true ROI with negligible error, a selector could win by learning benchmark agreement rather than causal modeling. The observed benchmark log-error distribution is wide: median 0.63, P90 1.15, and maximum 1.75 across businesses. The sealed wrong-evidence family increases this error by multiplying benchmarks downward or upward. Those audit businesses remain unused in this release.",
        ),
        H2("5.2 Window-aligned simulated experiments"),
        P(
            "An experiment is generated for a declared 8-to-13-week spend window when the channel's evidence contract makes a study available. The outcome window extends through the lag required to capture 95% of the channel's carryover kernel. True experiment ROI compares observed delivery with a counterfactual that removes only delivery during the tested spend window, while retaining the rest of the business trajectory.",
        ),
        equation(
            "ROI_true_(j,W) = [sum_(t in W+) beta_j {S(A_j,t) - S(A_j,t without W delivery)}]",
            "                   / sum_(t in W) Spend_(j,t)",
            "ROI_exp_(j,W) = max(0.05, ROI_true_(j,W)*(1+b_j) + eta_j)",
            "eta_j ~ Normal(0, SE_j^2);  SE share in [0.08, 0.38]",
        ),
        P(
            "The study represents a noisy platform holdout or geo experiment and is a source of causal evidence for a specific campaign and time window. It is not assumed to equal full-history average ROI. The experiment record retains start date, spend end date, outcome end date, incremental outcome, incremental spend, standard error, confidence, and scope.",
        ),
        H2("5.3 Prior calibration versus likelihood calibration"),
        P(
            "Candidate specifications decide how evidence enters. Prior calibration encodes the experiment or benchmark as information about an ROI-linked coefficient before observing the time series. Likelihood calibration adds a noisy measurement equation for the experiment. Both can be appropriate, but the latter requires explicit accounting for data overlap. If the experiment outcome or controls reuse the MMM sample, treating the experiment as independent can double-count information. FluxMMM records the route as a candidate feature because its appropriateness depends on evidence quality and overlap.",
        ),
        callout(
            "Evidence is not truth by label",
            "An experiment can be internally valid but weakly transported to the national, full-history response surface. An industry benchmark can prevent an absurd decomposition yet be badly matched to the advertiser. V5D-SVI therefore exposes evidence compatibility, quality, conflict, and decision dependence as separate signals rather than assuming that more prior influence is always better.",
            ORANGE,
            ORANGE_PALE,
        ),
        PageBreak(),

        H1("6. Candidate Bayesian MMM library"),
        H2("6.1 Why candidate diversity matters"),
        P(
            "The meta-learning problem is only meaningful if candidate MMMs disagree for scientifically plausible reasons. FluxMMM defines 24 base specifications, 12 Bayesian and 12 advanced, and crosses each with the two evidence policies. This yields 48 candidates per advertiser and 24,000 posterior fits over 500 businesses. All candidates are fitted without access to synthetic truth.",
        ),
        H2("6.2 Baseline and controls"),
        P(
            "Candidate baselines include an intercept, linear trend, observed promotions and demand controls where available, Fourier seasonality, and a declared cycle. The library varies one to three Fourier pairs and 13-, 26-, or 52-week cycles. Ridge regularization controls weak or collinear components. Because baseline flexibility can absorb media variation, the specification and its interaction with temporal diagnostics are visible to the selector.",
        ),
        H2("6.3 Media response assumptions"),
        P(
            "Candidates vary geometric versus Weibull adstock, memory parameters, Hill shape, half-saturation quantiles, and kernel normalization. Advanced candidates permit channel-specific response contracts, so TV, search, and paid social need not share carryover or saturation. This is necessary because the simulator itself assigns channel-specific delivery and memory mechanisms.",
        ),
        H2("6.4 Time-varying coefficients"),
        P(
            "Advanced candidates may represent beta_j(t) with a smooth radial-basis expansion. Kernel knots and bandwidth control how quickly effectiveness can change. The intent follows the time-varying coefficient literature [4]: allow gradual response drift without estimating one unrestricted coefficient per week.",
        ),
        equation(
            "beta_j(t) = beta_j,0 * exp( sum_r delta_(j,r) phi_r(t) )",
            "phi_r(t) = exp( -0.5 * ((t-knot_r)/bandwidth)^2 )",
        ),
        H2("6.5 Latent planning intensity"),
        P(
            "Some advanced candidates include a smooth latent planning factor shared across media and outcome structure. The factor is designed to capture commercial intensity that simultaneously drives campaign timing and baseline demand. It is not an instrument and cannot guarantee identification. Its usefulness must be judged through sensitivity, temporal placebo behavior, experiment coherence, and downstream economic loss.",
        ),
        H2("6.6 Prior and likelihood families"),
        P(
            "Media coefficients use positive half-normal or log-normal prior contracts, with evidence-informed ROI potentials when appropriate. Outcomes can use Gaussian, Student-t, or log-normal likelihoods. Student-t likelihoods protect against episodic shocks; log-normal forms may suit multiplicative revenue noise. The candidate library treats these as assumptions to be evaluated rather than universal defaults.",
        ),
        data_table([
            ["Dimension", "Candidate alternatives", "Scientific role"],
            ["Estimator family", "Bayesian; Advanced", "Constant versus optional flexible latent structure"],
            ["Adstock", "Geometric; Weibull; channel-specific", "Immediate decay versus delayed or peaked carryover"],
            ["Saturation", "Hill shape and half-saturation ranges", "Diminishing marginal return"],
            ["Baseline", "Fourier order 1-3; cycles 13/26/52", "Recurring non-media demand"],
            ["Coefficients", "Constant; smooth time-varying", "Stable versus drifting effectiveness"],
            ["Planning factor", "Off; on", "Shared latent commercial intensity"],
            ["Calibration", "Prior; noisy measurement likelihood", "How external evidence enters"],
            ["Coefficient prior", "Half-normal; log-normal", "Positive media effects with different tail shapes"],
            ["Outcome likelihood", "Gaussian; Student-t; log-normal", "Residual and shock behavior"],
        ], widths=[34 * mm, 65 * mm, 75 * mm], font_size=6.9),
        P("<b>Table 2.</b> Principal candidate dimensions. Appendix B documents the frozen search contract in greater detail.", "caption"),
        PageBreak(),

        H1("7. Posterior inference with FullRank ADVI"),
        H2("7.1 Role of SVI in this study"),
        P(
            "Stochastic variational inference (SVI) is used to approximate each candidate posterior at a scale large enough to label the full candidate library. It is not itself the novelty claim. Its role is to provide posterior-expected decisions and uncertainty-aware observable features for all 24,000 fits. Unlike a point estimate, the full-rank variational family represents covariance among unconstrained parameters, which matters when media channels compete to explain the same outcome.",
        ),
        H2("7.2 Variational objective"),
        P(
            "Let phi denote the unconstrained parameter vector. FullRank ADVI chooses a multivariate Gaussian q_lambda(phi) whose covariance is not restricted to be diagonal. It maximizes the evidence lower bound (ELBO), equivalently minimizing the reverse Kullback-Leibler divergence from q to the exact posterior [13,14].",
        ),
        equation(
            "q_lambda(phi) = Normal(m, L L')",
            "ELBO(lambda) = E_q[ log p(y, phi) - log q_lambda(phi) ]",
            "lambda_hat = arg max_lambda ELBO(lambda)",
        ),
        H2("7.3 Frozen inference contract"),
        metric_strip([
            ("Iterations", "5,000", "Full-rank optimization per seed"),
            ("Posterior draws", "256", "Retained per accepted fit"),
            ("Primary seeds", "2", "30,071 and 81,119"),
            ("Adjudication", "1", "Seed 190,081 when required"),
        ]),
        Spacer(1, 4 * mm),
        P(
            "The learning rate is 0.001, initial scale 0.01, and gradient norm cap 10. ELBO drift is evaluated over a 250-iteration window and must not exceed 0.05. Independent seeds must agree within 0.25 on the maximum channel log-ROI difference. A third seed adjudicates unstable ELBO behavior or seed disagreement. This contract is fingerprinted so cached posterior artifacts cannot silently mix inference versions.",
        ),
        H2("7.4 Why posterior features matter"),
        P(
            "A model can have acceptable residuals while its economically relevant posterior is broad, multimodally approximated, prior-dominated, or highly sensitive to initialization. V5D-SVI therefore makes posterior summaries visible to the selector: channel ROI centers, cross-channel dispersion, interval width, implausible mass, predictive coverage, ELBO stability, and seed agreement. These are candidate observables in a real application; no synthetic truth is required to compute them.",
        ),
        H2("7.5 Approximation limitations"),
        P(
            "Reverse-KL variational inference can underrepresent tail mass, prefer one mode, and produce biased transformed moments. ELBO stability is not proof of posterior accuracy. The study uses two seeds and a small NUTS audit contract to reduce this risk, but confirmatory MCMC is still appropriate for a promoted production candidate [15]. The selector must also avoid learning artifacts specific to FullRank ADVI. Future work should test whether features and rankings remain stable under NUTS, sequential Monte Carlo, and alternative variational families.",
        ),
        PageBreak(),

        H1("8. Observable feature system"),
        H2("8.1 Design principle"),
        P(
            "Every deployed feature must be computable for a new advertiser without causal truth. The feature registry explicitly forbids business ID, generator family, split label, candidate ID, proposal method, search phase, synthetic truth, realized decision loss, ROI truth error, contribution truth error, and MAP-to-SVI loss shift. This contract prevents the most obvious leakage paths.",
        ),
        H2("8.2 Predictive generalization"),
        P(
            "Generalization features summarize chronological out-of-sample performance, posterior predictive coverage, and behavior in low- and high-spend regimes. They test whether a candidate can reproduce observed outcomes under temporal or distributional shifts. Predictive adequacy is necessary for a useful response surface, but it is not sufficient for causal attribution.",
        ),
        H2("8.3 Structural adequacy"),
        P(
            "Structural features summarize residual pattern, residual independence, heteroskedasticity, likelihood-shape agreement, multicollinearity, numerical condition, effective rank, clipping, and functional-form adequacy. Residual independence means that fitted errors lack systematic time or regressor patterns; it does not prove the unobservable condition E[epsilon | X]=0. Collinearity diagnostics are conditional because duplicated media columns identify sums of effects rather than individual channel coefficients.",
        ),
        H2("8.4 Causal robustness"),
        P(
            "Causal features include future-media placebo behavior, confounder sensitivity, evidence anchor recovery, and stability to added or removed data. A future-media placebo asks whether spend that occurs after the outcome spuriously explains the present. Confounder sensitivity perturbs plausible omitted demand structure. Experiment recovery is evaluated over the experiment's declared spend and carryover outcome window.",
        ),
        H2("8.5 Decision coherence"),
        P(
            "Decision features test ROI plausibility, marginal-return consistency, benchmark compatibility, evidence conflict, and dependence of allocation on an evidence source. They are not a command to match benchmarks. A channel can conflict with an industry prior for legitimate reasons; the conflict becomes an observable risk signal whose learned effect depends on evidence quality and the rest of the feature vector.",
        ),
        H2("8.6 Temporal evidence"),
        P(
            "Temporal features measure whether the dataset contains enough spend variation, flights, off-periods, and post-flight observations to distinguish competing carryover kernels. They also interact with candidate memory, likelihood, planning, and family. These interactions let the learned selector distinguish a theoretically flexible model from one whose flexibility is actually supported by the observed time series.",
        ),
        H2("8.7 Posterior reliability and ROI geometry"),
        P(
            "V5D-SVI adds 21 posterior-observable features: label and finite status; ELBO stability; seed agreement; adjudication; log ELBO drift; maximum cross-seed log-ROI difference; predictive coverage; maximum implausible ROI probability; log relative interval width; margins to the three posterior gates; channel log-ROI centers; and their mean, standard deviation, minimum, maximum, and range. The log transform makes multiplicative ROI differences more regular.",
        ),
        H2("8.8 Model assumptions and interactions"),
        P(
            "Specification indicators and carefully declared interactions capture the conditional meaning of a diagnostic. For example, a carryover-support score should matter differently for a long-memory Weibull TV model than for a short-memory search model. The feature system therefore encodes adstock family, likelihood, calibration route, time variation, planning factor, evidence arm, and interactions with temporal and decision diagnostics.",
        ),
        callout(
            "Features are not scores",
            "Generalization, structure, causal robustness, decision coherence, temporal evidence, posterior diagnostics, and model assumptions enter as separate machine-learning features. V5D-SVI does not assign fixed 20/15/25/40 weights to four headline dimensions. The learned coefficients are conditional predictive associations, while 65/35 is only the post-prediction business risk preference between mean and P90 loss.",
            TEAL,
            TEAL_PALE,
        ),
        PageBreak(),

        H1("9. Learning the economic-risk selector"),
        H2("9.1 Training examples"),
        P(
            "Each candidate fit is a training row, but rows from the same advertiser are statistically dependent. V5D-SVI groups all 48 candidates of a business together when defining folds. Within each training fold, continuous features are standardized and clamped to eight standard deviations. The resulting development matrix contains 20,160 rows from 420 businesses.",
        ),
        H2("9.2 Mean excess-loss model"),
        P(
            "The conditional mean is a linear model trained with Huber loss for robustness. Candidate rows above the empirical 90th-percentile loss threshold receive additional weight, and rows from the current worst-performing generator family are also upweighted. L2 regularization limits coefficient magnitude. This creates an interpretable meta-model while reducing the influence of a small number of extreme loss labels [16].",
        ),
        equation(
            "L_mean = sum_i w_i * Huber(mu_hat_i - Delta_i; delta=2.5) + 0.012 ||w||_2^2",
            "w_i = 1 + 0.8 I[tail row] + 0.7 I[current worst family]",
        ),
        H2("9.3 Hard-negative ranking term"),
        P(
            "Point prediction is not enough: the system must order the safest low-loss candidate above economically costly challengers. For each business and epoch, the learner identifies the candidate that most strongly violates the oracle ordering. A logistic soft-margin term is weighted by the candidate's economic gap, with a minimum weight of 0.05. The frozen pairwise weight is 0.55.",
        ),
        equation(
            "v_(b,c) = Delta_(b,c) + mu_hat_(b,c*) - mu_hat_(b,c)",
            "L_rank = sum_b gapWeight_(b,c_hard) * log(1 + exp(v_(b,c_hard)))",
            "Lower predicted loss is preferred; costly misorderings receive larger updates.",
        ),
        H2("9.4 Tail model"),
        P(
            "A second linear model predicts the conditional 90th percentile with pinball loss. It shares the feature representation and standardization but has separate coefficients. The predicted P90 is constrained at output to be at least the predicted mean.",
        ),
        equation(
            "L_tail = sum_i w_i * pinball_0.90(q90_hat_i - Delta_i) + 0.012 ||v||_2^2",
            "q90_output = max(mu_output, q90_linear)",
        ),
        H2("9.5 Two-axis cross-fitting"),
        P(
            "The procedure fits 20 selector models: five outer advertiser folds crossed with four candidate-parameter regions. For an assessment row, the predictor excludes both that advertiser's fold and that candidate's region. Each fitted selector uses 336 businesses and 12,096 training rows. This is stricter than a random row split, which would leak advertiser identity through sibling candidates, and stricter than business-only splitting, which could memorize candidate neighborhoods.",
        ),
        H2("9.6 Truth-blind deployment"),
        P(
            "For a new advertiser, FluxMMM validates the data, constructs the candidate library, fits each candidate, calculates the observable feature vector, applies the cross-fitted or frozen production selector, filters by safety, and ranks remaining candidates by predicted risk. Synthetic truth and realized decision loss are absent. The promoted candidate should then undergo confirmatory NUTS sampling and human review before budget deployment.",
        ),
        PageBreak(),

        H1("10. Candidate safety and governance"),
        H2("10.1 Posterior safety contract"),
        P(
            "A V5D-SVI candidate is posterior-safe only when the fit is labelled and finite, ELBO drift is stable, independent seeds agree, posterior predictive coverage is at least 80%, maximum posterior probability beyond an evidence-informed ROI bound is at most 20%, and maximum relative ROI interval width is at most 3. If no candidate passes, FluxMMM may select among converged candidates for review but must not describe that result as decision-grade.",
        ),
        data_table([
            ["Gate", "Frozen rule", "Purpose"],
            ["Finite posterior", "All stored outputs finite", "Reject numerical failure"],
            ["ELBO stability", "Maximum drift <= 0.05", "Reject unfinished variational optimization"],
            ["Seed agreement", "Max log-ROI difference <= 0.25", "Reject initialization-sensitive fits"],
            ["Predictive coverage", ">= 0.80", "Require minimum posterior predictive calibration"],
            ["ROI plausibility", "Max implausible probability <= 0.20", "Limit mass beyond evidence-informed bounds"],
            ["ROI precision", "Max relative interval width <= 3", "Reject excessively diffuse decisions"],
        ], widths=[39 * mm, 53 * mm, 82 * mm], font_size=6.9),
        P("<b>Table 3.</b> Frozen posterior safety contract. Thresholds are predeclared governance choices, not learned causal truths.", "caption"),
        H2("10.2 Non-compensatory safety"),
        P(
            "Safety is non-compensatory when at least one safe candidate exists: an excellent predicted risk cannot rescue a failed hard gate. This separates model-risk governance from the learned economic objective. The separation is valuable, but it creates an interaction: the safe oracle and every excess-loss target depend on which candidates enter the pool.",
        ),
        H2("10.3 Governance state"),
        P(
            "The frozen release explicitly records: fresh validation not accessed; sealed audit not accessed; public research release permitted; fresh generalization not established; state-of-the-art claim not permitted; production activation not permitted. These flags prevent a promising development result from silently becoming a product claim.",
        ),
        PageBreak(),

        H1("11. Evaluation design"),
        H2("11.1 Development cohort"),
        P(
            "The V5D-SVI development cohort combines the original 300 training businesses with 120 previously opened validation businesses. Those 420 businesses are now treated entirely as development. The remaining 80 adversarial businesses stay sealed. Consequently, results in this paper are cross-fitted development estimates rather than fresh validation estimates.",
        ),
        H2("11.2 Controlled 2x2 ablation"),
        P(
            "To distinguish the value of posterior features from the effect of changing the safety pool, the study holds the economic-loss target, learner, folds, candidate regions, search algorithms, and budgets fixed while crossing two feature systems with two safety policies. The four cells are: base features plus base safety; SVI features plus base safety; base features plus SVI safety; and SVI features plus SVI safety.",
        ),
        H2("11.3 Selection and search metrics"),
        P(
            "The primary search budget is 16 candidate evaluations per advertiser, with checkpoints at 4, 8, 12, 16, 24, 32, and 48. Search algorithms receive equal candidate-evaluation budgets. The composite primary objective summarizes mean loss, P90 loss, CVaR90, worst-family mean loss, oracle recall, and review-only fallback behavior. Additional proxy metrics assess candidate-level prediction, pairwise ordering, and top-one champion quality.",
        ),
        data_table([
            ["Metric", "Interpretation", "Preferred direction"],
            ["Mean loss", "Average realized economic decision loss of selected candidates", "Lower"],
            ["P90 loss", "Loss below which 90% of selections fall", "Lower"],
            ["CVaR90", "Average loss in the worst 10% tail", "Lower"],
            ["Worst-family mean", "Average loss in the hardest generator family", "Lower"],
            ["Weighted pair accuracy", "Correct order weighted by economic gap", "Higher"],
            ["Champion mean excess", "Mean gap of the predicted best full-pool candidate", "Lower"],
            ["Dangerous false champion", "Share of champions with excess loss >= 1", "Lower"],
            ["P90 coverage", "Share of labels below predicted P90", "Near 90%"],
        ], widths=[43 * mm, 98 * mm, 33 * mm], font_size=6.9),
        P("<b>Table 4.</b> Evaluation endpoints. Strong average ranking does not guarantee a strong top-one champion.", "caption"),
        PageBreak(),

        H1("12. Development results"),
        H2("12.1 Primary result"),
        P(
            "The full V5D-SVI cell achieved a primary objective of 9.24 compared with 11.98 for the frozen V5D baseline, a relative reduction of 22.9%. At the full 48-candidate pool, the objective decreased from 13.34 to 10.33, a 22.6% reduction. These results indicate that the joint posterior feature and safety representation changes which candidates are selected in economically meaningful ways within the development cohort.",
        ),
        ablation_figure(),
        P("<b>Figure 4.</b> Controlled 2x2 ablation of posterior features and posterior safety. The composite objective is lower when posterior information is used, with the largest development improvement in the full cell.", "caption"),
        H2("12.2 Ranking improves, champion selection worsens"),
        P(
            "Economically weighted pairwise ordering accuracy increased from 66.1% to 72.5%. This means the model became more accurate on pairwise comparisons that involve larger economic gaps. Candidate-level correlation remained low, rising only from 0.154 to 0.190, and mean absolute loss-prediction error worsened from 2.67 to 4.05. More importantly, champion mean excess loss increased from 1.41 to 2.12, champion P90 excess loss increased from 3.68 to 5.59, and dangerous false champions increased from 27.9% to 48.1%.",
        ),
        result_tradeoff_figure(),
        P("<b>Figure 5.</b> V5D-SVI better orders economically important pairs but is worse at selecting the single full-pool champion. The objectives are related but not equivalent.", "caption"),
        callout(
            "Scientific interpretation",
            "Posterior-aware observables contain useful decision-risk signal, while the current safety and top-one selection system does not yet convert that signal reliably into the best promoted candidate. A meta-selector must be evaluated at the actual deployment decision, not only by candidate-level accuracy.",
            ORANGE,
            ORANGE_PALE,
        ),

        H1("13. Why the result is mixed"),
        H2("13.1 Safety-set expansion"),
        P(
            "The base safety rule accepted 6,088 of 20,160 development candidates. The SVI rule accepted 18,703. Only 5,667 were safe under both; 13,036 were unique to SVI safety, whereas only 421 were unique to base safety. Every development business had at least 12 SVI-safe candidates, the median had all 48, and the 90th percentile and maximum were also 48. The SVI gates therefore behave more like broad convergence checks than a selective causal-decision screen.",
        ),
        safety_pool_figure(),
        P("<b>Figure 6.</b> Posterior safety expands the candidate pool substantially. Because the oracle and excess-loss target are defined inside the active pool, this is not a neutral implementation change.", "caption"),
        H2("13.2 A changed pool changes the learning target"),
        P(
            "When safety admits more candidates, it can introduce both genuinely useful models and harmful but numerically well-behaved models. It also changes the within-business oracle c*. Thus the full 2x2 cell differs not only in available features but in the population of allowed decisions and the target baseline. The safety-only cell illustrates this interaction: primary objective improves relative to baseline, yet full-pool champion behavior deteriorates.",
        ),
        H2("13.3 Pairwise success is not top-one success"),
        P(
            "Weighted pair accuracy averages over many comparisons. A selector can correctly order most costly pairs yet place one unsafe or poorly calibrated candidate just above the true champion. Top-one selection amplifies local errors near the minimum predicted risk. With 48 alternatives, even modest correlated prediction error creates a winner's-curse effect: the smallest predicted loss is likely to be an optimistic error.",
        ),
        H2("13.4 Linear representation and interactions"),
        P(
            "The current learner is interpretable but linear in transformed features. It can represent declared interactions but may miss nonlinear thresholds, business-conditional effects, and complex feature substitution. For example, a posterior interval width may be benign under a flat profit surface and dangerous near a marginal-return boundary. A future hierarchical or tree-based model could represent such context while preserving grouped validation and monotonic safety constraints.",
        ),
        H2("13.5 Development reuse"),
        P(
            "The 120 businesses formerly described as validation were opened during earlier method development and are now correctly reclassified as development. Cross-fitting protects row-level estimates inside this pool but does not create a new untouched cohort. Repeated design choices informed by the same families can still overfit the research program [17]. A fresh population and the sealed adversarial audit are required before external claims.",
        ),
        PageBreak(),

        H1("14. What the model learned"),
        H2("14.1 Feature-family importance"),
        P(
            "Importance is the mean absolute standardized risk coefficient across the 20 cross-fitted models, normalized to sum to one. Model specification and declared interactions account for 23.9%, posterior ROI geometry for 21.9%, and temporal evidence for 21.4%. Together they contribute 67.2% of absolute coefficient mass. Structure contributes 7.9%, decision coherence 6.7%, posterior decision safety 5.1%, generalization 4.7%, causal robustness 4.1%, posterior reliability 3.6%, and posterior predictive features 0.6%.",
        ),
        group_importance_figure(),
        P("<b>Figure 7.</b> Learned risk importance by feature family. Values describe this frozen development model and are not universal causal weights.", "caption"),
        H2("14.2 Leading features"),
        P(
            "The largest individual coefficient is paid-social posterior log-ROI center (7.38%). Cross-channel mean, minimum, and maximum posterior log-ROI centers add 9.77%. The saturation-by-spend-regime interaction contributes 2.99%; cross-seed log-ROI disagreement contributes 2.73%; functional-form adequacy contributes 2.31%; and spend-regime generalization contributes 2.14%. These results suggest that the absolute geometry of the inferred response surface, its temporal support, and its consistency across posterior fits are central to decision risk.",
        ),
        top_feature_figure(),
        P("<b>Figure 8.</b> Top 15 standardized risk features. Sign is conditional on all other features. Appendix C reports the complete top 50.", "caption"),
        H2("14.3 Why signs must not be read causally"),
        P(
            "A negative coefficient means that, conditional on all other transformed features in this development sample, larger feature values predict lower excess loss. It does not mean an advertiser should intervene to increase that feature. Posterior ROI centers, their aggregates, specification indicators, and interactions are strongly correlated. They can exchange coefficient mass and produce counterintuitive conditional signs. Importance is useful for understanding the learned representation, but permutation stability, grouped bootstrap uncertainty, and nonlinear partial dependence are needed before substantive interpretation.",
        ),
        PageBreak(),

        H1("15. Scientific novelty"),
        H2("15.1 The estimator is not the central novelty"),
        P(
            "FluxMMM uses familiar MMM components: Fourier baselines, geometric and Weibull adstock, Hill saturation, positive media priors, Student-t robustness, optional time variation, latent planning structure, and Bayesian posterior approximation. The contribution is the orchestration and learning target. Entire fitted causal models become objects in a supervised decision-risk problem.",
        ),
        H2("15.2 From hand-crafted validation score to learned economic risk"),
        P(
            "A hand-crafted validation score assumes in advance how much prediction, residual behavior, causal sensitivity, and ROI plausibility should matter. FluxMMM instead records those diagnostics separately and asks which combinations predict downstream loss across many known-truth worlds. This does not eliminate judgment: simulation design, feature choice, safety thresholds, risk appetite, and candidate library remain human commitments. It moves one crucial judgment - the conditional weighting of observable model evidence - from intuition to an auditable empirical learning problem.",
        ),
        H2("15.3 A cross-advertiser scientific object"),
        P(
            "The learned selector is a meta-model over advertisers. Its scientific object is the mapping from observable model evidence to decision risk under a population of business mechanisms. This invites questions that ordinary single-dataset MMM papers cannot answer: Which diagnostics transfer? Which interactions are stable? How much does evidence quality matter? Does a posterior diagnostic predict budget regret beyond predictive fit? How does the safe set change the oracle? These questions are testable through held-out mechanism families and real experiment-linked datasets.",
        ),
        H2("15.4 Open-source contribution"),
        P(
            "An open implementation allows researchers to contribute new generators, evidence contracts, candidate models, posterior engines, diagnostics, decision scenarios, and search algorithms while retaining a common evaluation contract. Reproducible artifacts also make negative results visible. The field can compare selectors on the same hidden-truth tasks rather than compare screenshots or proprietary case studies.",
        ),
        callout(
            "Qualified priority claim",
            "The reviewed literature contains automated MMM hyperparameter search, Bayesian calibration, synthetic causal benchmarks, and decision-focused prediction. We did not identify an open MMM study that combines these into cross-advertiser supervised selection of complete Bayesian candidates using posterior-expected budget excess loss as the target. A formal systematic review is still required before claiming absolute priority.",
            TEAL,
            TEAL_PALE,
        ),
        PageBreak(),

        H1("16. Limitations and threats to validity"),
        H2("16.1 Synthetic-to-real transport"),
        P(
            "The strongest limitation is domain transport. Every label is generated from a simulator. If the simulator omits a real mechanism - competitive response, creative wear-out, channel auctions under policy change, retailer distribution, stockouts, cross-channel synergies, measurement error in outcomes, regional heterogeneity, or advertiser learning - the selector may reward features that do not transfer. More simulated businesses do not solve misspecified worlds.",
        ),
        H2("16.2 Population and channel scope"),
        P(
            "The current population focuses on weekly direct-to-consumer businesses with three channel archetypes. It is not representative of lead-generation businesses, subscriptions with long conversion delay, marketplaces with two-sided demand, offline retail distribution, app installs, heavily localized businesses, or advertisers with ten or more correlated media channels. ROI ranges are plausible design ranges, not estimates of an industry population.",
        ),
        H2("16.3 Candidate-library oracle"),
        P(
            "The safe oracle is the best of 48 candidates, not the true structural model. If every candidate omits the same confounder or response mechanism, the oracle can still be poor. The learned selector can at best imitate the economically strongest option in the supplied library. Candidate coverage must therefore expand and be audited separately from selector quality.",
        ),
        H2("16.4 Variational approximation"),
        P(
            "FullRank ADVI provides scalable posteriors but may understate tail uncertainty or settle on one region of a non-Gaussian posterior. Because both labels and observable posterior features come from the same approximation family, the meta-model may learn approximation-specific regularities. A stratified NUTS audit is necessary to quantify label error and feature stability.",
        ),
        H2("16.5 Selection-induced optimism"),
        P(
            "Ranking 48 noisy risk predictions creates winner's-curse bias. The chosen minimum is more optimistic than a typical prediction. The current learner does not explicitly estimate uncertainty in predicted risk, correct for multiple comparisons, or require a margin over the runner-up. This likely contributes to poor champion behavior.",
        ),
        H2("16.6 Safety thresholds"),
        P(
            "Posterior coverage, implausible-mass, and interval-width thresholds are governance choices. Their development behavior shows they are too permissive for top-one selection. They also encode evidence-informed ROI bounds, whose quality varies. Future safety should combine posterior reliability with evidence quality, conflict, identification, and decision dependence rather than use independent marginal thresholds alone.",
        ),
        H2("16.7 Feature importance uncertainty"),
        P(
            "The published feature importances are descriptive coefficients from 20 correlated cross-fitted models. They lack confidence intervals and are sensitive to feature scaling, redundancy, and regularization. They should not be marketed as universal laws of good MMM. Grouped bootstrap, stability selection, ablation, and cross-generator replication are required.",
        ),
        H2("16.8 No fresh validation in this release"),
        P(
            "The current comparison is fully development-only. The sealed 80-business audit is intact but was not opened for V5D-SVI. No independent simulator, historical advertiser dataset, or prospective budget experiment has evaluated the selector. This is the decisive boundary between an interesting method and a publishable external performance claim.",
        ),
        PageBreak(),

        H1("17. Future development program"),
        H2("17.1 Freeze the present method"),
        P(
            "The first step is governance: freeze code, feature registry, candidate universe, inference fingerprint, economic-loss definition, risk appetite, and safety thresholds for this release. Preserve the mixed result. Any subsequent changes should receive a new version and a new predeclared evaluation plan.",
        ),
        H2("17.2 Repair top-one selection"),
        bullet("Learn calibrated uncertainty for predicted economic risk and penalize candidates whose risk estimate is unstable across selector folds."),
        bullet("Require a minimum predicted-risk margin before declaring a champion; otherwise retain a shortlist for confirmatory sampling."),
        bullet("Use listwise or differentiable top-k losses that target the promoted candidate rather than average pair accuracy."),
        bullet("Apply shrinkage or hierarchical partial pooling across advertiser families, channels, and candidate mechanisms."),
        bullet("Re-estimate safety as a joint risk model rather than a permissive conjunction of marginal posterior gates."),
        H2("17.3 Expand and stress the simulator"),
        bullet("Add more channels, regional panels, price, distribution, stockouts, competitive spend, creative quality, measurement error, cross-channel synergies, and structural breaks."),
        bullet("Vary time cadence and aggregation, missingness, sparse flights, zero-variance channels, high dimensionality, and extreme scale."),
        bullet("Add mechanism-held-out generators from independent teams and compare against the Heusch endogenous-spend benchmark [9]."),
        bullet("Model experiment noncompliance, contamination, overlap, platform attribution bias, and transport from campaign windows to national response."),
        H2("17.4 Independent validation hierarchy"),
        data_table([
            ["Stage", "Evidence", "Claim permitted"],
            ["Development", "Grouped cross-fitting within 420 businesses", "Method behavior and hypothesis generation"],
            ["Sealed synthetic audit", "Wrong evidence and swapped mechanics", "Robustness to predeclared adversarial shifts"],
            ["Independent simulator", "Externally authored causal worlds", "Cross-simulator transport"],
            ["Historical experiment-linked MMMs", "Real aggregated time series plus channel lift studies", "Retrospective external validity"],
            ["Prospective deployment", "Precommitted budget actions and outcomes", "Decision impact in practice"],
        ], widths=[35 * mm, 83 * mm, 56 * mm], font_size=6.9),
        P("<b>Table 5.</b> Claims should expand only as evidence advances through the validation hierarchy.", "caption"),
        H2("17.5 MCMC confirmation and inference robustness"),
        P(
            "Run NUTS on a stratified sample of candidate-business pairs and on every promoted production finalist. Compare SVI and NUTS on ROI centers, covariance, tail probability, predictive coverage, selected allocation, and realized synthetic regret. If approximation error changes decisions, train an explicit correction or reserve SVI for screening.",
        ),
        H2("17.6 Real experiment partnerships"),
        P(
            "The highest-value empirical extension is a consortium of historical MMM datasets with channel experiments. The data need not be public at row level. A federated or bring-code-to-data protocol could compute standardized diagnostics and decision comparisons inside each partner environment, export only privacy-safe receipts, and test whether the learned ordering transfers. Prospective partnerships should pre-register the evaluated candidate set and business action before observing the experimental outcome.",
        ),
        H2("17.7 Theory"),
        P(
            "A useful theoretical program would connect selector ranking error with bounded excess decision regret, characterize top-one winner's-curse under correlated risk errors, and formalize evidence attribution. In a conjugate Gaussian case, local source sensitivity can be decomposed through posterior precision. Under exact collinearity, only sums of channel coefficients are likelihood-identified, clarifying why external evidence or assumptions determine the split. These results would support interpretation but cannot substitute for external validation.",
        ),
        PageBreak(),

        H1("18. Reproducibility and open science"),
        H2("18.1 Frozen artifacts"),
        P(
            "The paper is generated directly from the frozen development, feature-importance, and simulator-audit JSON artifacts. The development receipt records SHA-256 provenance for upstream V5A, V5D, and SVI record stores. The feature-importance receipt records the 20 cross-fitted model receipts and forbids access to fresh validation or audit data.",
        ),
        H2("18.2 Reproduction sequence"),
        bullet("Generate the 500 seeded advertiser population and verify that injected full-history ROI, contribution vectors, allocation constraints, and noisy experiments satisfy internal invariants."),
        bullet("Construct the 24 base specifications and two evidence arms without using hidden truth."),
        bullet("Fit all 24,000 FullRank ADVI candidate posteriors under the fingerprinted contract and retain posterior decision draws and diagnostics."),
        bullet("Compute the four scenario actions from each posterior expected decision surface, reveal hidden truth after action selection, and store economic loss labels."),
        bullet("Build truth-blind diagnostic and posterior feature rows; verify forbidden fields are absent."),
        bullet("Fit 20 two-axis cross-fitted selectors, generate order-independent search traces, and calculate primary, full-pool, proxy, and safety receipts."),
        bullet("Generate feature importance as mean absolute standardized coefficients; publish signs, stability, and descriptions."),
        H2("18.3 Open contribution model"),
        P(
            "Contributors should add new simulator families, model specifications, inference engines, or features behind explicit versioned contracts. New work should not overwrite old artifacts. Pull requests should include unit tests, leakage tests, deterministic seeds, computational receipts, and a declared claim boundary. This keeps the project useful as a communal benchmark rather than an ever-changing demo.",
        ),
        H2("18.4 Recommended artifact package for peer review"),
        bullet("Source code and environment lock files."),
        bullet("Machine-readable simulator card and feature dictionary."),
        bullet("Immutable candidate, inference, safety, and economic-loss contracts."),
        bullet("Cross-fitting assignments and hashes without hidden truth leakage."),
        bullet("Development and sealed-audit analysis scripts separated by access controls."),
        bullet("A computational appendix reporting hardware, wall time, failed fits, reruns, and storage."),
        PageBreak(),

        H1("19. Discussion"),
        P(
            "FluxMMM asks a different question from conventional MMM model comparison. Instead of declaring that predictive accuracy, residual normality, experiment agreement, or ROI plausibility is the correct objective, it treats each as evidence whose usefulness can vary with business context and model assumptions. Synthetic causal truth supplies the only information unavailable in production: what would actually have happened under alternative budget actions. That truth is used to train the selection rule, not to fit or repair the candidate MMM.",
        ),
        P(
            "The frozen V5D-SVI study shows why this framing is valuable even before it succeeds fully. Posterior features materially improve ordering where economic differences are large. Yet a permissive safety rule and noisy top-one minimization produce worse champions. If evaluation had stopped at pairwise accuracy or a composite objective, the method could have been overstated. By following the full chain from posterior to action to revealed profit, the failure becomes visible.",
        ),
        P(
            "This result also clarifies the distinction between a good causal model and a useful selector. A candidate's causal credibility depends on identification and evidence. A selector's job is to predict which candidate will make a better decision given imperfect signals of that credibility. The selector can generalize only if its training worlds cover the mechanisms and evidence failures of future advertisers. Therefore, the simulator is part of the scientific model, not merely a test harness.",
        ),
        P(
            "For practitioners, the method suggests a disciplined operating model. Preserve individual diagnostics rather than collapse them into arbitrary points. Fit diverse causal candidates. Require explicit evidence contracts. Use posterior uncertainty in both the action and the meta-model. Separate safety from ranking. Confirm the finalist with full sampling. Most importantly, evaluate the economic consequence of the model's recommendation rather than equating a high validation score with a good budget decision.",
        ),
        H2("19.1 What would make this top-journal evidence"),
        P(
            "A top-journal submission would require a preregistered sealed evaluation, independent simulator replication, a substantial NUTS audit, uncertainty around feature importance and selection metrics, comparison with strong baselines such as WAIC/LOO, Robyn-style multi-objective selection, experiment-calibration error, and expert modeler choice, and at least one real experiment-linked application. The current paper provides the method and development evidence needed to design that study honestly.",
        ),
        PageBreak(),

        H1("20. Conclusion"),
        P(
            "FluxMMM V5D-SVI reframes marketing mix model selection as cross-advertiser decision-risk learning. Many Bayesian MMM candidates are fitted without truth; each candidate makes budget decisions; the simulator reveals economic loss only afterward; and a truth-blind meta-model learns which diagnostics, posterior properties, and assumptions predict that loss. This formulation aligns model selection with the purpose of MMM: making economically sound decisions under causal and posterior uncertainty.",
        ),
        P(
            "The initial development study is informative but not conclusive. Posterior-aware features improve economically weighted ranking and the composite search objective. The current posterior safety pool, however, is too broad, and champion-level loss worsens. The appropriate conclusion is neither that SVI wins nor that learned selection has failed. It is that a usable signal has been identified, while safety and top-one selection remain the critical research frontier.",
        ),
        P(
            "The broader contribution is an open scientific protocol. It makes the definition of a good MMM testable: a good candidate should use defensible assumptions and evidence, quantify uncertainty, survive structural and causal challenges, and most importantly produce low regret when its recommended action is evaluated against known or experimentally anchored consequences. By opening the simulator, labels, feature registry, selector, and failures, FluxMMM invites the field to improve that protocol collectively.",
        ),
        Spacer(1, 5 * mm),
        callout(
            "Bottom line",
            "V5D-SVI is the first frozen open research version of a larger idea: learn model quality from the decisions models cause, not from a universal hand-crafted score. The present evidence justifies further research and open collaboration, not automatic production deployment.",
            INDIGO,
            INDIGO_PALE,
        ),
        PageBreak(),
    ])


def add_appendices(story: list[Flowable]) -> None:
    story.extend([
        H1("References"),
    ])
    references = [
        "[1] Jin, Y., Wang, Y., Sun, Y., Chan, D., and Koehler, J. (2017). Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects. Google Research. https://research.google/pubs/bayesian-methods-for-media-mix-modeling-with-carryover-and-shape-effects/",
        "[2] Chan, D. and Perry, M. (2017). Challenges and Opportunities in Media Mix Modeling. Google Research. https://research.google/pubs/challenges-and-opportunities-in-media-mix-modeling/",
        "[3] Chen, A., Chan, D., Perry, M., Jin, Y., Sun, Y., Wang, Y., and Koehler, J. (2018). Bias Correction for Paid Search in Media Mix Modeling. arXiv:1807.03292. https://arxiv.org/abs/1807.03292",
        "[4] Ng, E., Wang, Z., and Dai, A. (2021). Bayesian Time Varying Coefficient Model with Applications to Marketing Mix Modeling. arXiv:2106.03322. https://arxiv.org/abs/2106.03322",
        "[5] Zhang, Y., Wurm, M., Li, E., Wakim, A., Kelly, J., Price, B., and Liu, Y. (2024). Media Mix Model Calibration With Bayesian Priors. Google Research. https://research.google/pubs/media-mix-model-calibration-with-bayesian-priors/",
        "[6] Runge, J., et al. (2024). Robyn: An Open-source Marketing Mix Modeling Package. arXiv:2403.14674. https://arxiv.org/abs/2403.14674",
        "[7] Google Meridian Marketing Mix Modeling Team. (2026). Meridian: Marketing Mix Modeling, version 1.6.1. https://github.com/google/meridian",
        "[8] Google Meridian Marketing Mix Modeling Team. Prior distribution and ROI calibration documentation. https://github.com/google/meridian/blob/main/meridian/model/prior_distribution.py",
        "[9] Heusch, N. (2026). A Synthetic Benchmark Dataset with Endogenous Marketing Spend for Validating Marketing Mix Models. arXiv:2608.21130. https://arxiv.org/abs/2608.21130",
        "[10] Elmachtoub, A. N. and Grigas, P. (2022). Smart Predict, then Optimize. Management Science 68(1), 9-26. https://doi.org/10.1287/mnsc.2020.3922",
        "[11] Wang, K., et al. (2024). Decision Focused Causal Learning for Direct Counterfactual Marketing Optimization. arXiv:2407.13664. https://arxiv.org/abs/2407.13664",
        "[12] Doppa, J. R., Fern, A., and Tadepalli, P. (2014). Structured Prediction via Output Space Search. Journal of Machine Learning Research 15, 1317-1350. https://www.jmlr.org/papers/volume15/doppa14a/doppa14a.pdf",
        "[13] Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., and Blei, D. M. (2017). Automatic Differentiation Variational Inference. Journal of Machine Learning Research 18(14), 1-45. https://www.jmlr.org/papers/v18/16-107.html",
        "[14] Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. (2017). Variational Inference: A Review for Statisticians. Journal of the American Statistical Association 112(518), 859-877. https://doi.org/10.1080/01621459.2017.1285773",
        "[15] Hoffman, M. D. and Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research 15, 1593-1623. https://jmlr.org/papers/v15/hoffman14a.html",
        "[16] Huber, P. J. (1964). Robust Estimation of a Location Parameter. Annals of Mathematical Statistics 35(1), 73-101. https://doi.org/10.1214/aoms/1177703732",
        "[17] Cawley, G. C. and Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. Journal of Machine Learning Research 11, 2079-2107. https://www.jmlr.org/papers/v11/cawley10a.html",
        "[18] Koenker, R. and Bassett, G. (1978). Regression Quantiles. Econometrica 46(1), 33-50. https://doi.org/10.2307/1913643",
        "[19] Gordon, B. R., Zettelmeyer, F., Bhargava, N., and Chapsky, D. (2019). A Comparison of Approaches to Advertising Measurement: Evidence from Big Field Experiments at Facebook. Marketing Science 38(2), 193-225. https://doi.org/10.1287/mksc.2018.1135",
        "[20] Lewis, R. A. and Rao, J. M. (2015). The Unfavorable Economics of Measuring the Returns to Advertising. Quarterly Journal of Economics 130(4), 1941-1973. https://doi.org/10.1093/qje/qjv023",
        "[21] Heusch, N. (2026). Structural Estimation of Marketing Mix Model Parameters from Geo-Experiments. arXiv:2608.21128. https://arxiv.org/abs/2608.21128",
        "[22] Watanabe, S. (2010). Asymptotic Equivalence of Bayes Cross Validation and Widely Applicable Information Criterion in Singular Learning Theory. Journal of Machine Learning Research 11, 3571-3594. https://www.jmlr.org/papers/v11/watanabe10a.html",
    ]
    for reference in references:
        story.append(Paragraph(reference, S["reference"]))
    story.append(PageBreak())

    story.extend([
        H1("Appendix A. Data-generating process card"),
        H2("A.1 Population families"),
    ])
    family_rows = [["Family", "Split", "Businesses", "Mechanism"]]
    for family in SIMULATOR_DATA["families"]:
        family_rows.append([
            family["label"],
            family["split"],
            str(family["businessCount"]),
            family["description"],
        ])
    story.extend([
        data_table(family_rows, widths=[37 * mm, 23 * mm, 23 * mm, 91 * mm], font_size=6.6),
        P("<b>Table A1.</b> Simulator families. Audit families were not accessed by V5D-SVI.", "caption"),
        H2("A.2 Population coverage"),
        data_table([
            ["Quantity", "Minimum", "P10", "Median", "P90", "Maximum"],
            ["History weeks", "104", "104", "156", "260", "260"],
            ["Noise share", "1.81%", "2.71%", "6.18%", "9.47%", "13.83%"],
            ["Effective revenue margin", "26.3%", "42.1%", "75.7%", "100%", "100%"],
            ["Benchmark log error", "0.04", "0.28", "0.63", "1.15", "1.75"],
        ], widths=[49 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm], font_size=6.8, alignments={1:"right",2:"right",3:"right",4:"right",5:"right"}),
        P("<b>Table A2.</b> Coverage over all 500 businesses.", "caption"),
        H2("A.3 Channel truth coverage"),
        data_table([
            ["Channel", "ROI P10", "ROI median", "ROI P90", "Marginal ROI median", "Weibull share", "Memory median"],
            ["Paid social", "1.09x", "2.46x", "5.47x", "1.17x", "23.0%", "0.39"],
            ["Nonbrand search", "0.50x", "1.22x", "3.23x", "0.75x", "11.4%", "0.31"],
            ["TV / CTV", "0.74x", "2.13x", "5.66x", "1.10x", "74.6%", "6.53"],
        ], widths=[34 * mm, 23 * mm, 27 * mm, 23 * mm, 36 * mm, 28 * mm, 27 * mm], font_size=6.5, alignments={1:"right",2:"right",3:"right",4:"right",5:"right",6:"right"}),
        P("<b>Table A3.</b> ROI and memory distributions. Memory is theta for geometric response and scale in weeks for Weibull response, so the final column is descriptive rather than directly comparable across families.", "caption"),
        H2("A.4 Persistence and confounding ranges"),
        P(
            "Latent demand persistence is sampled from 0.35 to 0.94, planning persistence from 0.30 to 0.94, and commercial persistence from 0.35 to 0.93. Ordinary channel demand coupling ranges approximately from 0.02 to 0.62, while search-demand-harvesting businesses raise search coupling to 0.75-1.15. Planning coupling typically ranges from 0.04 to 0.72 and rises to 0.72-1.08 in correlated-planning businesses. Event-shock probability ranges from 1.5% to 20% per week.",
        ),
        H2("A.5 Response ranges"),
        P(
            "Geometric decay is sampled from 0.01 up to 0.55 for ordinary digital media and up to 0.78 for TV. Weibull shape ranges from 1.15 to 4.8. Weibull scale ranges from 1.5 to 8 weeks for ordinary channels and 3.5 to 13 weeks for TV. Hill shape ranges from 0.90 to 3.10 for paid social and 0.72 to 2.35 for search and TV. Half-saturation quantile ranges from 0.30 to 0.82.",
        ),
        PageBreak(),

        H1("Appendix B. Candidate and inference contract"),
        H2("B.1 Candidate count"),
        equation(
            "500 businesses * 24 base specifications * 2 evidence arms = 24,000 SVI fits",
            "420 development businesses * 48 candidates = 20,160 development rows",
            "80 audit businesses * 48 candidates = 3,840 sealed rows",
        ),
        P(
            "The 24 base specifications contain 12 Bayesian and 12 Advanced models. Evidence arms are experiments-only and benchmark-gap-fill. A candidate ID, proposal source, and generator family are retained for audit but forbidden from the learned feature vector.",
        ),
        H2("B.2 Inference settings"),
        data_table([
            ["Setting", "Frozen value"],
            ["Engine", "PyMC 6.2 FullRankADVI"],
            ["Iterations", "5,000 per seed"],
            ["Posterior draws", "256 per accepted fit"],
            ["Primary seeds", "30,071 and 81,119"],
            ["Adjudication seed", "190,081"],
            ["Learning rate", "0.001"],
            ["Initial scale", "0.01"],
            ["Gradient norm cap", "10"],
            ["ELBO window", "250 iterations"],
            ["Maximum ELBO drift", "0.05"],
            ["Maximum seed log-ROI difference", "0.25"],
        ], widths=[78 * mm, 96 * mm], font_size=7.1),
        P("<b>Table B1.</b> Fingerprinted inference contract.", "caption"),
        H2("B.3 Learner settings"),
        data_table([
            ["Setting", "Value", "Role"],
            ["Epochs", "28", "Gradient passes"],
            ["Learning rate", "0.055 / sqrt(1 + epoch/8)", "Decaying optimization step"],
            ["L2 regularization", "0.012", "Coefficient shrinkage"],
            ["Huber delta", "2.5", "Robust mean loss"],
            ["Quantile", "0.90", "Tail-loss target"],
            ["Pairwise weight", "0.55", "Hard-negative ordering pressure"],
            ["Tail row weight", "0.80", "Extra emphasis on high-loss rows"],
            ["Worst-family row weight", "0.70", "Extra emphasis on current hardest family"],
            ["Risk aversion", "0.35", "Weight on P90-minus-mean width"],
            ["Economic scale", "1", "Uncapped target normalization"],
        ], widths=[47 * mm, 54 * mm, 73 * mm], font_size=6.8),
        P("<b>Table B2.</b> Frozen V5D-SVI learner configuration.", "caption"),
        PageBreak(),

        H1("Appendix C. Full top-50 feature importance"),
        P(
            "Importance is normalized mean absolute standardized risk coefficient across 20 cross-fitted models. Coefficient sign is conditional. Stability is the share of models agreeing with the majority sign.",
        ),
    ])
    features = IMPORTANCE_DATA["top50RiskFeatures"]
    feature_rows: list[list[str]] = [["Rank", "Feature", "Family", "Imp.", "Coef.", "Stable", "Description"]]
    for index, item in enumerate(features, start=1):
        feature_rows.append([
            str(index),
            item["feature"],
            item["group"].replace("-and-", " + ").replace("-", " "),
            f"{100 * item['importance']:.2f}%",
            f"{item['meanCoefficient']:+.3f}",
            f"{100 * item['signStability']:.0f}%",
            item["description"],
        ])
    story.extend([
        data_table(feature_rows, widths=[9 * mm, 43 * mm, 31 * mm, 15 * mm, 17 * mm, 16 * mm, 43 * mm], font_size=5.2),
        P("<b>Table C1.</b> Complete frozen top-50 importance table.", "caption"),
        PageBreak(),

        H1("Appendix D. Algorithm"),
        H2("D.1 Development protocol"),
        equation(
            "INPUT: seeded advertiser population B; candidate contracts C; evidence arms E",
            "FOR each advertiser b:",
            "  generate observed data O_b and retain hidden truth T_b",
            "  FOR each candidate c in C x E:",
            "    fit q_(b,c)(theta) using only O_b and declared evidence",
            "    compute truth-blind diagnostics x_(b,c)",
            "    choose one posterior-expected action in each scenario k",
            "    reveal T_b and calculate L_(b,c)",
            "build grouped business folds and candidate parameter regions",
            "fit cross-fitted mean, P90, and ranking models",
            "evaluate safety, search, ordering, and promoted-champion outcomes",
        ),
        H2("D.2 Deployment protocol"),
        equation(
            "INPUT: new advertiser observations O_new; no hidden truth",
            "fit candidate posterior library under immutable contracts",
            "compute observable feature vector x_(new,c)",
            "predict mean loss, P90 loss, and risk for each candidate",
            "filter candidates through the declared safety policy",
            "select lowest predicted risk, or mark review-only if none are safe",
            "run confirmatory NUTS and compare posterior decision surfaces",
            "promote only after diagnostics, evidence, and human review",
        ),
        H2("D.3 Leakage audit"),
        data_table([
            ["Forbidden field", "Reason"],
            ["Business or generator identity", "Would memorize training worlds rather than transport evidence"],
            ["Split label", "Would reveal research partition"],
            ["Candidate ID or proposal order", "Would memorize hand-designed candidate ordering"],
            ["Synthetic truth", "Unavailable for a real advertiser"],
            ["Realized decision loss", "The supervised label, unavailable at deployment"],
            ["ROI or contribution truth error", "Direct leakage of the causal target"],
            ["MAP-to-SVI loss shift", "Uses SVI label information not available as a standalone deployment feature"],
        ], widths=[57 * mm, 117 * mm], font_size=6.9),
        P("<b>Table D1.</b> Explicit forbidden-feature contract.", "caption"),
        PageBreak(),

        H1("Appendix E. Development results"),
        H2("E.1 Primary 16-evaluation search"),
        data_table([
            ["Cell", "Objective", "Mean", "P90", "CVaR90", "Worst family", "Oracle recall", "Fallback"],
            ["Base feat. + base safety", "11.98", "2.93", "6.88", "14.20", "3.89", "37.6%", "12.1%"],
            ["SVI feat. + base safety", "11.10", "2.81", "6.74", "13.21", "3.37", "36.4%", "12.1%"],
            ["Base feat. + SVI safety", "9.84", "2.50", "6.38", "11.42", "3.27", "6.9%", "0.0%"],
            ["SVI feat. + SVI safety", "9.24", "2.39", "6.18", "10.34", "3.36", "6.7%", "0.0%"],
        ], widths=[39 * mm, 19 * mm, 18 * mm, 18 * mm, 20 * mm, 24 * mm, 21 * mm, 18 * mm], font_size=5.9, alignments={1:"right",2:"right",3:"right",4:"right",5:"right",6:"right",7:"right"}),
        P("<b>Table E1.</b> Frozen primary search metrics.", "caption"),
        H2("E.2 Full 48-candidate pool"),
        data_table([
            ["Cell", "Objective", "Mean", "P90", "CVaR90", "Worst family", "Oracle recall", "Fallback"],
            ["Base feat. + base safety", "13.34", "3.24", "6.99", "16.05", "4.15", "37.6%", "8.3%"],
            ["SVI feat. + base safety", "12.08", "3.01", "6.83", "14.59", "3.54", "39.5%", "8.3%"],
            ["Base feat. + SVI safety", "10.78", "2.79", "7.59", "12.33", "3.66", "4.3%", "0.0%"],
            ["SVI feat. + SVI safety", "10.33", "2.74", "6.99", "11.53", "3.64", "4.0%", "0.0%"],
        ], widths=[39 * mm, 19 * mm, 18 * mm, 18 * mm, 20 * mm, 24 * mm, 21 * mm, 18 * mm], font_size=5.9, alignments={1:"right",2:"right",3:"right",4:"right",5:"right",6:"right",7:"right"}),
        P("<b>Table E2.</b> Frozen full-pool metrics.", "caption"),
        H2("E.3 Candidate prediction and champion metrics"),
        data_table([
            ["Metric", "Base V5D", "V5D-SVI", "Interpretation"],
            ["Prediction MAE", "2.67", "4.05", "Worse point calibration"],
            ["Candidate correlation", "0.154", "0.190", "Slightly stronger, still weak"],
            ["Pairwise accuracy", "55.8%", "58.4%", "Modest ordering gain"],
            ["Economically weighted pair accuracy", "66.1%", "72.5%", "Stronger ordering on costly gaps"],
            ["P90 coverage", "90.0%", "91.0%", "Close to target"],
            ["Champion oracle recall", "37.6%", "4.0%", "Materially worse top-one recovery"],
            ["Champion mean excess loss", "1.41", "2.12", "51.0% worse"],
            ["Champion P90 excess loss", "3.68", "5.59", "51.7% worse"],
            ["Dangerous false champions", "27.9%", "48.1%", "Materially worse"],
        ], widths=[57 * mm, 28 * mm, 28 * mm, 61 * mm], font_size=6.5, alignments={1:"right",2:"right"}),
        P("<b>Table E3.</b> The exact tradeoff that motivates the next research phase.", "caption"),
        PageBreak(),

        H1("Appendix F. Claim and review checklist"),
        data_table([
            ["Claim", "Current evidence", "Status"],
            ["Posterior observables contain decision-risk signal", "Weighted pair accuracy rises from 66.1% to 72.5%", "Supported in development"],
            ["Full V5D-SVI improves composite selection objective", "Primary objective improves 22.9%", "Supported in development"],
            ["Current V5D-SVI chooses better champions", "Champion loss and false champion rate worsen", "Rejected"],
            ["Selector generalizes to fresh synthetic businesses", "No fresh V5D-SVI cohort opened", "Not established"],
            ["Selector generalizes to real advertisers", "No real experiment-linked evaluation", "Not established"],
            ["V5D-SVI is state of the art", "No sealed or external benchmark victory", "Not permitted"],
            ["V5D-SVI is production ready", "Safety and top-one behavior unresolved", "Not permitted"],
            ["Method is an open cross-advertiser decision-risk formulation", "Code and frozen contracts implement the full pipeline", "Documented contribution"],
        ], widths=[68 * mm, 76 * mm, 30 * mm], font_size=6.7),
        P("<b>Table F1.</b> Review boundary. A journal revision should update this table before changing the abstract or conclusion.", "caption"),
        H2("F.1 Questions for methodological review"),
        bullet("Is within-business excess loss relative to a safety-defined oracle the right supervised target, or should absolute scenario loss also be modeled directly?"),
        bullet("Should safety be fixed ex ante, learned as a joint probability of unacceptable loss, or expressed as constraints on evidence and identification?"),
        bullet("Which top-one or listwise loss best converts weighted pair signal into reliable champion selection?"),
        bullet("How should uncertainty in meta-model risk predictions affect shortlisting and confirmatory sampling?"),
        bullet("Which real experiment-linked datasets can test transport without exposing advertiser-confidential data?"),
        bullet("What systematic-review protocol is sufficient to substantiate the qualified novelty claim?"),
        Spacer(1, 8 * mm),
        HRFlowable(width="100%", thickness=0.7, color=INDIGO),
        Spacer(1, 4 * mm),
        P("<b>End of working paper.</b> Generated from frozen V5D-SVI research artifacts for author review.", "small"),
    ])


def build_story() -> list[Flowable]:
    story: list[Flowable] = []
    cover(story)
    add_toc(story)
    add_abstract(story)
    add_main_sections(story)
    add_appendices(story)
    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = FluxDocTemplate(str(OUTPUT))
    doc.multiBuild(build_story())
    print(OUTPUT)


if __name__ == "__main__":
    main()
