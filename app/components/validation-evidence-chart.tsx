"use client";

import type { EChartsCoreOption, EChartsType } from "echarts/core";
import { useEffect, useMemo, useRef } from "react";
import type { ValidationEvidence } from "../../lib/mmm/validation";

export type ValidationChartMode =
  | "forecast"
  | "regimes"
  | "residuals"
  | "residual-fitted"
  | "acf"
  | "distribution"
  | "vifs"
  | "anchors"
  | "coherence"
  | "stability"
  | "confounders"
  | "placebo";

interface ValidationEvidenceChartProps {
  evidence: ValidationEvidence;
  mode: ValidationChartMode;
  label: string;
  height?: number;
  compact?: boolean;
}

const COLORS = {
  ink: "#363445",
  violet: "#5b5bd6",
  violetLight: "#b9b8e7",
  orange: "#ef9560",
  mint: "#27a589",
  danger: "#d75f6c",
  grid: "#ecebf0",
  muted: "#8e8d99",
};

const CHANNEL_COLORS = [
  COLORS.violet,
  COLORS.orange,
  COLORS.mint,
  "#d3648f",
];

const VALUE_FORMATTER = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 2,
});

const DATE_FORMATTER = new Intl.DateTimeFormat("en-US", {
  day: "2-digit",
  month: "short",
  year: "numeric",
});

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function tooltipRows(params: unknown): string {
  const points = (Array.isArray(params) ? params : [params]) as {
    axisValueLabel?: string;
    dataIndex?: number;
    marker?: string;
    seriesName?: string;
    value?: number | string | [number, number];
  }[];
  const rawHeading = points[0]?.axisValueLabel ?? "";
  const heading = escapeHtml(
    /^\d{4}-\d{2}/.test(rawHeading) ? dateLabel(rawHeading) : rawHeading,
  );
  const rows = points
    .filter((point) => !point.seriesName?.includes("base"))
    .map((point) => {
      const raw = Array.isArray(point.value)
        ? point.value.at(-1)
        : point.value;
      const value =
        typeof raw === "number"
          ? VALUE_FORMATTER.format(raw)
          : escapeHtml(String(raw ?? ""));
      return `<div class="flux-tooltip-row"><span>${point.marker ?? ""}${escapeHtml(point.seriesName ?? "")}</span><b>${value}</b></div>`;
    })
    .join("");
  return `<div class="flux-tooltip-date">${heading}</div>${rows}`;
}

function forecastTooltip(params: unknown): string {
  const points = (Array.isArray(params) ? params : [params]) as {
    axisValueLabel?: string;
    marker?: string;
    seriesName?: string;
    value?: number;
  }[];
  const rawDate = points[0]?.axisValueLabel ?? "";
  const lower = Number(
    points.find((point) => point.seriesName === "95% interval base")?.value,
  );
  const width = Number(
    points.find((point) => point.seriesName === "95% interval")?.value,
  );
  const rows = points
    .filter(
      (point) =>
        point.seriesName === "Observed" ||
        point.seriesName === "Forecast",
    )
    .map(
      (point) =>
        `<div class="flux-tooltip-row"><span>${point.marker ?? ""}${escapeHtml(point.seriesName ?? "")}</span><b>${VALUE_FORMATTER.format(Number(point.value) || 0)}</b></div>`,
    )
    .join("");
  const interval =
    Number.isFinite(lower) && Number.isFinite(width)
      ? `<div class="flux-tooltip-row"><span><i class="flux-tooltip-band"></i>95% interval</span><b>${VALUE_FORMATTER.format(lower)} – ${VALUE_FORMATTER.format(lower + width)}</b></div>`
      : "";
  return `<div class="flux-tooltip-date">${escapeHtml(dateLabel(rawDate))}</div>${rows}${interval}`;
}

function anchorTooltip(
  params: unknown,
  evidence: Extract<ValidationEvidence, { kind: "causal" }>,
): string {
  const points = (Array.isArray(params) ? params : [params]) as {
    dataIndex?: number;
  }[];
  const anchor = evidence.anchors[points[0]?.dataIndex ?? -1];
  if (!anchor) return "";
  const basis = anchor.comparisonLabel
    ? `<div class="flux-tooltip-row"><span>Comparison</span><b>${escapeHtml(anchor.comparisonLabel)}</b></div>`
    : "";
  return `<div class="flux-tooltip-date">${escapeHtml(anchor.channel)}</div>
    <div class="flux-tooltip-row"><span><i class="flux-tooltip-dot violet"></i>MMM</span><b>${VALUE_FORMATTER.format(anchor.modelRoi)} <small>${VALUE_FORMATTER.format(anchor.modelLow)}–${VALUE_FORMATTER.format(anchor.modelHigh)}</small></b></div>
    <div class="flux-tooltip-row"><span><i class="flux-tooltip-dot orange"></i>Experiment</span><b>${VALUE_FORMATTER.format(anchor.experimentRoi)} <small>${VALUE_FORMATTER.format(anchor.experimentLow)}–${VALUE_FORMATTER.format(anchor.experimentHigh)}</small></b></div>${basis}`;
}

function decisionTooltip(
  params: unknown,
  evidence: Extract<ValidationEvidence, { kind: "decision" }>,
): string {
  const points = (Array.isArray(params) ? params : [params]) as {
    dataIndex?: number;
  }[];
  const material = evidence.channels.filter((channel) => channel.material);
  const channels = material.length ? material : evidence.channels;
  const channel = channels[points[0]?.dataIndex ?? -1];
  if (!channel) return "";
  const evidenceRange =
    channel.evidenceLow === undefined || channel.evidenceHigh === undefined
      ? "Unanchored"
      : `${channel.evidenceLow.toFixed(2)}–${channel.evidenceHigh.toFixed(2)}×`;
  const posteriorMass =
    channel.posteriorMass === undefined
      ? "Not testable"
      : `${Math.round(channel.posteriorMass * 100)}%`;
  return `<div class="flux-tooltip-date">${escapeHtml(channel.channel)}</div>
    <div class="flux-tooltip-row"><span><i class="flux-tooltip-dot violet"></i>MMM ROI</span><b>${channel.roi.toFixed(2)}× <small>${channel.roiLow.toFixed(2)}–${channel.roiHigh.toFixed(2)}×</small></b></div>
    <div class="flux-tooltip-row"><span><i class="flux-tooltip-dot orange"></i>Evidence range</span><b>${evidenceRange}</b></div>
    <div class="flux-tooltip-row"><span>Posterior mass in range</span><b>${posteriorMass}</b></div>
    <div class="flux-tooltip-row"><span>Decision score</span><b>${Math.round(channel.score)}/100</b></div>`;
}

function dateLabel(value: string): string {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed)
    ? DATE_FORMATTER.format(new Date(parsed))
    : value;
}

function commonOption(
  compact: boolean,
  showLegend = true,
): EChartsCoreOption {
  return {
    animationDuration: compact ? 350 : 520,
    animationEasing: "cubicOut",
    color: CHANNEL_COLORS,
    grid: {
      left: compact ? 5 : 10,
      right: compact ? 7 : 16,
      top: showLegend ? (compact ? 27 : 38) : compact ? 9 : 18,
      bottom: compact ? 7 : 18,
      containLabel: true,
    },
    legend: showLegend
      ? {
          top: 0,
          right: 4,
          itemWidth: 13,
          itemHeight: 3,
          textStyle: {
            color: COLORS.muted,
            fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
            fontSize: compact ? 8 : 9,
          },
        }
      : undefined,
    tooltip: tooltipOption(),
  };
}

function tooltipOption() {
  return {
    trigger: "axis" as const,
    confine: true,
    className: "flux-chart-tooltip",
    backgroundColor: "rgba(255,255,255,.98)",
    borderColor: "#dedde5",
    borderWidth: 1,
    padding: [10, 12] as [number, number],
    extraCssText:
      "border-radius:10px;box-shadow:0 12px 32px rgba(35,33,54,.14);",
    axisPointer: {
      type: "cross" as const,
      lineStyle: { color: "#9291a2", type: "dashed" as const, width: 1 },
      label: { show: false },
    },
    formatter: tooltipRows,
  };
}

function valueAxis(compact: boolean) {
  return {
    type: "value" as const,
    scale: true,
    splitNumber: compact ? 3 : 5,
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: {
      color: "#9a99a3",
      fontSize: compact ? 8 : 9,
      formatter: (value: number) => VALUE_FORMATTER.format(value),
    },
    splitLine: {
      lineStyle: { color: COLORS.grid, type: "dashed" },
    },
  };
}

function categoryAxis(
  data: string[],
  compact: boolean,
) {
  return {
    type: "category" as const,
    data,
    boundaryGap: false,
    axisLine: { lineStyle: { color: "#dedde4" } },
    axisTick: { show: false },
    axisLabel: {
      color: "#9a99a3",
      fontSize: compact ? 8 : 9,
      hideOverlap: true,
      formatter: (value: string) => dateLabel(value),
    },
  };
}

function forecastOption(
  evidence: Extract<ValidationEvidence, { kind: "generalization" }>,
  compact: boolean,
): EChartsCoreOption {
  const dates: string[] = [];
  const actual: (number | null)[] = [];
  const predicted: (number | null)[] = [];
  const lower: (number | null)[] = [];
  const interval: (number | null)[] = [];
  evidence.folds.forEach((fold, foldIndex) => {
    if (foldIndex > 0) {
      dates.push("");
      actual.push(null);
      predicted.push(null);
      lower.push(null);
      interval.push(null);
    }
    fold.dates.forEach((date, index) => {
      dates.push(date);
      actual.push(fold.actual[index]);
      predicted.push(fold.predicted[index]);
      lower.push(fold.lower[index]);
      interval.push(fold.upper[index] - fold.lower[index]);
    });
  });
  return {
    ...commonOption(compact),
    tooltip: { ...tooltipOption(), formatter: forecastTooltip },
    xAxis: categoryAxis(dates, compact),
    yAxis: valueAxis(compact),
    series: [
      {
        name: "95% interval base",
        type: "line",
        data: lower,
        stack: "forecast-band",
        symbol: "none",
        lineStyle: { opacity: 0 },
        areaStyle: { opacity: 0 },
        emphasis: { disabled: true },
      },
      {
        name: "95% interval",
        type: "line",
        data: interval,
        stack: "forecast-band",
        symbol: "none",
        lineStyle: { opacity: 0 },
        areaStyle: { color: "rgba(91,91,214,.18)" },
        emphasis: { disabled: true },
      },
      {
        name: "Observed",
        type: "line",
        data: actual,
        showSymbol: !compact,
        symbolSize: 5,
        connectNulls: false,
        lineStyle: { color: COLORS.ink, width: compact ? 2 : 2.5 },
        itemStyle: { color: COLORS.ink },
        emphasis: { focus: "series" },
      },
      {
        name: "Forecast",
        type: "line",
        data: predicted,
        showSymbol: !compact,
        symbolSize: 5,
        connectNulls: false,
        lineStyle: {
          color: COLORS.violet,
          width: compact ? 2 : 2.5,
          type: "dashed",
        },
        itemStyle: { color: COLORS.violet },
        emphasis: { focus: "series" },
      },
    ],
  };
}

function regimeOption(
  evidence: Extract<ValidationEvidence, { kind: "generalization" }>,
  compact: boolean,
): EChartsCoreOption {
  const available = evidence.regimes.filter((regime) => regime.available);
  return {
    ...commonOption(compact, false),
    grid: {
      left: 8,
      right: 12,
      top: 18,
      bottom: 8,
      containLabel: true,
    },
    xAxis: {
      type: "category",
      data: available.map((regime) => regime.name),
      axisLine: { lineStyle: { color: "#dedde4" } },
      axisTick: { show: false },
      axisLabel: { color: COLORS.muted, fontSize: compact ? 8 : 9 },
    },
    yAxis: {
      ...valueAxis(compact),
      min: 0,
      axisLabel: {
        color: COLORS.muted,
        fontSize: compact ? 8 : 9,
        formatter: (value: number) => `${Math.round(value)}%`,
      },
    },
    tooltip: {
      ...tooltipOption(),
      valueFormatter: (value: number) => `${value.toFixed(1)}% WAPE`,
    },
    series: [
      {
        name: "Holdout error",
        type: "bar",
        data: available.map((regime) => regime.wape * 100),
        barWidth: compact ? 28 : 42,
        itemStyle: {
          color: COLORS.orange,
          borderRadius: [6, 6, 0, 0],
        },
      },
    ],
  };
}

function structuralOption(
  evidence: Extract<ValidationEvidence, { kind: "structure" }>,
  mode: ValidationChartMode,
  compact: boolean,
): EChartsCoreOption {
  if (mode === "residual-fitted") {
    const minimum = Math.min(...evidence.fitted);
    const maximum = Math.max(...evidence.fitted);
    return {
      ...commonOption(compact, false),
      xAxis: {
        type: "value",
        scale: true,
        name: compact ? undefined : "Fitted outcome",
        nameTextStyle: { color: COLORS.muted, fontSize: 9 },
        axisLabel: { color: COLORS.muted, fontSize: compact ? 8 : 9 },
        splitLine: { lineStyle: { color: COLORS.grid, type: "dashed" } },
      },
      yAxis: {
        ...valueAxis(compact),
        name: compact ? undefined : "Residual",
        nameTextStyle: { color: COLORS.muted, fontSize: 9 },
      },
      tooltip: { ...tooltipOption(), trigger: "item" },
      series: [
        {
          name: "Residual",
          type: "scatter",
          data: evidence.fitted.map((fitted, index) => [
            fitted,
            evidence.residuals[index],
          ]),
          symbolSize: compact ? 4 : 6,
          itemStyle: { color: COLORS.violet, opacity: 0.62 },
        },
        {
          name: "Zero",
          type: "line",
          data: [
            [minimum, 0],
            [maximum, 0],
          ],
          symbol: "none",
          lineStyle: { color: COLORS.muted, type: "dashed", width: 1 },
        },
      ],
    };
  }
  if (mode === "acf") {
    return {
      ...commonOption(compact, false),
      xAxis: {
        type: "category",
        data: evidence.autocorrelation.map((point) => `${point.lag}w`),
        axisLine: { lineStyle: { color: "#dedde4" } },
        axisTick: { show: false },
        axisLabel: { color: COLORS.muted, fontSize: compact ? 8 : 9 },
      },
      yAxis: { ...valueAxis(compact), min: -1, max: 1 },
      series: [
        {
          name: "Residual ACF",
          type: "bar",
          data: evidence.autocorrelation.map((point) => ({
            value: point.value,
            itemStyle: {
              color:
                Math.abs(point.value) > 0.25
                  ? COLORS.danger
                  : COLORS.violet,
              borderRadius:
                point.value >= 0 ? [4, 4, 0, 0] : [0, 0, 4, 4],
            },
          })),
          barMaxWidth: compact ? 16 : 28,
        },
      ],
    };
  }
  if (mode === "vifs") {
    const sorted = [...evidence.vifs].sort((a, b) => a.value - b.value);
    return {
      ...commonOption(compact, false),
      grid: {
        left: 8,
        right: 22,
        top: 15,
        bottom: 8,
        containLabel: true,
      },
      xAxis: {
        ...valueAxis(compact),
        name: compact ? undefined : "Variance inflation factor",
        nameTextStyle: { color: COLORS.muted, fontSize: 9 },
      },
      yAxis: {
        type: "category",
        data: sorted.map((point) => point.label),
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          color: COLORS.muted,
          fontSize: compact ? 8 : 9,
          width: compact ? 70 : 130,
          overflow: "truncate",
        },
      },
      series: [
        {
          name: "VIF",
          type: "bar",
          data: sorted.map((point) => ({
            value: point.value,
            itemStyle: {
              color: point.value > 5 ? COLORS.danger : COLORS.mint,
              borderRadius: [0, 5, 5, 0],
            },
          })),
          barMaxWidth: compact ? 10 : 18,
        },
      ],
    };
  }
  if (mode === "distribution") {
    const extent = Math.max(
      ...evidence.quantiles.flatMap((point) => [
        Math.abs(point.expected),
        Math.abs(point.observed),
      ]),
      1,
    );
    return {
      ...commonOption(compact, false),
      xAxis: {
        type: "value",
        min: -extent,
        max: extent,
        name: compact ? undefined : "Expected quantile",
        nameTextStyle: { color: COLORS.muted, fontSize: 9 },
        axisLabel: { color: COLORS.muted, fontSize: compact ? 8 : 9 },
        splitLine: { lineStyle: { color: COLORS.grid, type: "dashed" } },
      },
      yAxis: {
        ...valueAxis(compact),
        min: -extent,
        max: extent,
        name: compact ? undefined : "Observed quantile",
        nameTextStyle: { color: COLORS.muted, fontSize: 9 },
      },
      tooltip: { ...tooltipOption(), trigger: "item" },
      series: [
        {
          name: "Residual quantiles",
          type: "scatter",
          data: evidence.quantiles.map((point) => [
            point.expected,
            point.observed,
          ]),
          symbolSize: compact ? 4 : 6,
          itemStyle: { color: COLORS.violet, opacity: 0.7 },
        },
        {
          name: "Reference",
          type: "line",
          data: [
            [-extent, -extent],
            [extent, extent],
          ],
          symbol: "none",
          lineStyle: { color: COLORS.orange, type: "dashed", width: 1.5 },
        },
      ],
    };
  }
  const band = evidence.residualScale * 2;
  return {
    ...commonOption(compact, !compact),
    xAxis: categoryAxis(evidence.dates, compact),
    yAxis: valueAxis(compact),
    series: [
      {
        name: "Residual",
        type: "line",
        data: evidence.residuals,
        showSymbol: false,
        lineStyle: { color: COLORS.violet, width: compact ? 1.8 : 2.2 },
        areaStyle: { color: "rgba(91,91,214,.07)" },
      },
      {
        name: "+2σ",
        type: "line",
        data: evidence.dates.map(() => band),
        symbol: "none",
        lineStyle: { color: COLORS.orange, type: "dashed", width: 1 },
      },
      {
        name: "−2σ",
        type: "line",
        data: evidence.dates.map(() => -band),
        symbol: "none",
        lineStyle: { color: COLORS.orange, type: "dashed", width: 1 },
      },
      {
        name: "Zero",
        type: "line",
        data: evidence.dates.map(() => 0),
        symbol: "none",
        lineStyle: { color: COLORS.muted, type: "dotted", width: 1 },
      },
    ],
  };
}

function causalOption(
  evidence: Extract<ValidationEvidence, { kind: "causal" }>,
  mode: ValidationChartMode,
  compact: boolean,
): EChartsCoreOption {
  if (mode === "stability") {
    return {
      ...commonOption(compact),
      xAxis: {
        type: "value",
        min: 0.68,
        max: 1.02,
        axisLabel: {
          color: COLORS.muted,
          fontSize: compact ? 8 : 9,
          formatter: (value: number) => `${Math.round(value * 100)}%`,
        },
        splitLine: { lineStyle: { color: COLORS.grid, type: "dashed" } },
      },
      yAxis: { ...valueAxis(compact), name: compact ? undefined : "ROI" },
      series: evidence.stability.map((channel, index) => ({
        name: channel.channel,
        type: "line",
        data: channel.points.map((point) => [
          point.historyShare,
          point.roi,
        ]),
        symbol: "circle",
        symbolSize: compact ? 5 : 7,
        lineStyle: { color: CHANNEL_COLORS[index], width: 2 },
        itemStyle: { color: CHANNEL_COLORS[index] },
      })),
    };
  }
  if (mode === "confounders") {
    return {
      ...commonOption(compact),
      xAxis: {
        type: "value",
        min: 0,
        max: 0.62,
        name: compact ? undefined : "Confounder strength",
        nameTextStyle: { color: COLORS.muted, fontSize: 9 },
        axisLabel: { color: COLORS.muted, fontSize: compact ? 8 : 9 },
        splitLine: { lineStyle: { color: COLORS.grid, type: "dashed" } },
      },
      yAxis: {
        ...valueAxis(compact),
        name: compact ? undefined : "ROI relative to original",
        axisLabel: {
          color: COLORS.muted,
          fontSize: compact ? 8 : 9,
          formatter: (value: number) => `${Math.round(value * 100)}%`,
        },
      },
      series: evidence.confounders.map((channel, index) => ({
        name: channel.channel,
        type: "line",
        data: channel.points.map((point) => [
          point.strength,
          point.normalizedRoi,
        ]),
        symbol: "circle",
        symbolSize: compact ? 5 : 7,
        lineStyle: { color: CHANNEL_COLORS[index], width: 2 },
        itemStyle: { color: CHANNEL_COLORS[index] },
      })),
    };
  }
  if (mode === "placebo") {
    return {
      ...commonOption(compact, false),
      xAxis: {
        type: "category",
        data: evidence.placebo.map((point) => `+${point.lead}w`),
        axisLine: { lineStyle: { color: "#dedde4" } },
        axisTick: { show: false },
        axisLabel: { color: COLORS.muted, fontSize: compact ? 8 : 9 },
      },
      yAxis: { ...valueAxis(compact), min: -1, max: 1 },
      series: [
        {
          name: "Future spend correlation",
          type: "bar",
          data: evidence.placebo.map((point) => ({
            value: point.correlation,
            itemStyle: {
              color:
                Math.abs(point.correlation) > 0.25
                  ? COLORS.danger
                  : COLORS.mint,
              borderRadius:
                point.correlation >= 0 ? [5, 5, 0, 0] : [0, 0, 5, 5],
            },
          })),
          barMaxWidth: compact ? 22 : 38,
        },
      ],
    };
  }
  const channels = evidence.anchors.map((anchor) => anchor.channel);
  return {
    ...commonOption(compact, !compact),
    tooltip: {
      ...tooltipOption(),
      formatter: (params: unknown) => anchorTooltip(params, evidence),
    },
    grid: {
      left: 8,
      right: 18,
      top: compact ? 10 : 42,
      bottom: 15,
      containLabel: true,
    },
    xAxis: {
      type: "value",
      scale: true,
      name: compact ? undefined : "Incremental ROI",
      nameTextStyle: { color: COLORS.muted, fontSize: 9 },
      axisLabel: { color: COLORS.muted, fontSize: compact ? 8 : 9 },
      splitLine: { lineStyle: { color: COLORS.grid, type: "dashed" } },
    },
    yAxis: {
      type: "category",
      data: channels,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: COLORS.muted,
        fontSize: compact ? 8 : 9,
        width: compact ? 65 : 130,
        overflow: "truncate",
      },
    },
    series: [
      {
        name: "MMM interval base",
        type: "bar",
        stack: "mmm",
        data: evidence.anchors.map((anchor) => anchor.modelLow),
        itemStyle: { color: "transparent" },
        emphasis: { disabled: true },
        barWidth: compact ? 3 : 5,
      },
      {
        name: "MMM interval",
        type: "bar",
        stack: "mmm",
        data: evidence.anchors.map(
          (anchor) => anchor.modelHigh - anchor.modelLow,
        ),
        itemStyle: {
          color: COLORS.violetLight,
          borderRadius: 4,
        },
        barWidth: compact ? 3 : 5,
      },
      {
        name: "Experiment interval base",
        type: "bar",
        stack: "experiment",
        data: evidence.anchors.map((anchor) => anchor.experimentLow),
        itemStyle: { color: "transparent" },
        emphasis: { disabled: true },
        barWidth: compact ? 3 : 5,
      },
      {
        name: "Experiment interval",
        type: "bar",
        stack: "experiment",
        data: evidence.anchors.map(
          (anchor) => anchor.experimentHigh - anchor.experimentLow,
        ),
        itemStyle: { color: "#f3c4a9", borderRadius: 4 },
        barWidth: compact ? 3 : 5,
      },
      {
        name: "MMM estimate",
        type: "scatter",
        data: evidence.anchors.map((anchor, index) => [
          anchor.modelRoi,
          index,
        ]),
        symbol: "diamond",
        symbolSize: compact ? 8 : 11,
        itemStyle: { color: COLORS.violet },
      },
      {
        name: "Experiment",
        type: "scatter",
        data: evidence.anchors.map((anchor, index) => [
          anchor.experimentRoi,
          index,
        ]),
        symbol: "circle",
        symbolSize: compact ? 7 : 10,
        itemStyle: { color: COLORS.orange },
      },
    ],
  };
}

function decisionOption(
  evidence: Extract<ValidationEvidence, { kind: "decision" }>,
  compact: boolean,
): EChartsCoreOption {
  const material = evidence.channels.filter((channel) => channel.material);
  const channels = (material.length ? material : evidence.channels)
    .slice()
    .sort((left, right) => left.spendShare - right.spendShare);
  return {
    ...commonOption(compact, !compact),
    tooltip: {
      ...tooltipOption(),
      formatter: (params: unknown) => decisionTooltip(params, {
        ...evidence,
        channels,
      }),
    },
    grid: {
      left: 8,
      right: 18,
      top: compact ? 10 : 42,
      bottom: 15,
      containLabel: true,
    },
    xAxis: {
      type: "value",
      min: 0,
      scale: true,
      name: compact ? undefined : "Incremental ROI",
      nameTextStyle: { color: COLORS.muted, fontSize: 9 },
      axisLabel: {
        color: COLORS.muted,
        fontSize: compact ? 8 : 9,
        formatter: (value: number) => `${value.toFixed(1)}×`,
      },
      splitLine: { lineStyle: { color: COLORS.grid, type: "dashed" } },
    },
    yAxis: {
      type: "category",
      data: channels.map((channel) => channel.channel),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: COLORS.muted,
        fontSize: compact ? 8 : 9,
        width: compact ? 65 : 130,
        overflow: "truncate",
      },
    },
    series: [
      {
        name: "Evidence range base",
        type: "bar",
        stack: "evidence",
        data: channels.map((channel) => channel.evidenceLow ?? null),
        itemStyle: { color: "transparent" },
        emphasis: { disabled: true },
        barWidth: compact ? 5 : 7,
      },
      {
        name: "Evidence range",
        type: "bar",
        stack: "evidence",
        data: channels.map((channel) =>
          channel.evidenceLow === undefined || channel.evidenceHigh === undefined
            ? null
            : channel.evidenceHigh - channel.evidenceLow,
        ),
        itemStyle: { color: "#f3c4a9", borderRadius: 4 },
        barWidth: compact ? 5 : 7,
      },
      {
        name: "MMM interval base",
        type: "bar",
        stack: "posterior",
        data: channels.map((channel) => channel.roiLow),
        itemStyle: { color: "transparent" },
        emphasis: { disabled: true },
        barWidth: compact ? 4 : 6,
      },
      {
        name: "MMM interval",
        type: "bar",
        stack: "posterior",
        data: channels.map((channel) => channel.roiHigh - channel.roiLow),
        itemStyle: { color: COLORS.violetLight, borderRadius: 4 },
        barWidth: compact ? 4 : 6,
      },
      {
        name: "MMM estimate",
        type: "scatter",
        data: channels.map((channel, index) => [channel.roi, index]),
        symbol: "diamond",
        symbolSize: compact ? 8 : 11,
        itemStyle: {
          color: (params: { dataIndex: number }) =>
            channels[params.dataIndex]?.blocking
              ? COLORS.danger
              : COLORS.violet,
        },
      },
    ],
  };
}

function chartOption(
  evidence: ValidationEvidence,
  mode: ValidationChartMode,
  compact: boolean,
): EChartsCoreOption {
  if (evidence.kind === "generalization") {
    return mode === "regimes"
      ? regimeOption(evidence, compact)
      : forecastOption(evidence, compact);
  }
  if (evidence.kind === "structure") {
    return structuralOption(evidence, mode, compact);
  }
  if (evidence.kind === "decision") {
    return decisionOption(evidence, compact);
  }
  return causalOption(evidence, mode, compact);
}

export function ValidationEvidenceChart({
  evidence,
  mode,
  label,
  height = 300,
  compact = false,
}: ValidationEvidenceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const option = useMemo(
    () => chartOption(evidence, mode, compact),
    [compact, evidence, mode],
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let cancelled = false;
    let chart: EChartsType | undefined;
    let observer: ResizeObserver | undefined;

    void (async () => {
      const [echarts, charts, components, renderers] = await Promise.all([
        import("echarts/core"),
        import("echarts/charts"),
        import("echarts/components"),
        import("echarts/renderers"),
      ]);
      echarts.use([
        charts.BarChart,
        charts.LineChart,
        charts.ScatterChart,
        components.AriaComponent,
        components.GridComponent,
        components.LegendComponent,
        components.TooltipComponent,
        renderers.CanvasRenderer,
      ]);
      if (cancelled) return;
      chart = echarts.init(container, undefined, {
        renderer: "canvas",
        useDirtyRect: false,
      });
      chart.setOption(option, { notMerge: true });
      observer = new ResizeObserver(() => chart?.resize());
      observer.observe(container);
    })();

    return () => {
      cancelled = true;
      observer?.disconnect();
      chart?.dispose();
    };
  }, [option]);

  return (
    <div
      ref={containerRef}
      className={`validation-evidence-chart${compact ? " compact" : ""}`}
      style={{ height }}
      role="img"
      aria-label={label}
    />
  );
}
