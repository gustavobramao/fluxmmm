"use client";

import type { EChartsCoreOption, EChartsType } from "echarts/core";
import { useEffect, useMemo, useRef } from "react";
import type { SamplingResult } from "../../lib/mmm/sampling";

export type SamplingChartMode =
  | "roi"
  | "predictive"
  | "trace"
  | "rank"
  | "density";

interface SamplingChartProps {
  result: SamplingResult;
  mode: SamplingChartMode;
  label: string;
  parameterIndex?: number;
  channelIndex?: number;
  height?: number;
}

const COLORS = {
  ink: "#363445",
  violet: "#5b5bd6",
  orange: "#ef9560",
  mint: "#27a589",
  grid: "#ecebf0",
  muted: "#8e8d99",
  violetFill: "rgba(91,91,214,.18)",
  orangeFill: "rgba(239,149,96,.12)",
};

const VALUE_FORMATTER = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 2,
});

const DATE_FORMATTER = new Intl.DateTimeFormat("en-US", {
  day: "2-digit",
  month: "short",
  year: "numeric",
});

function cleanChannel(value: string): string {
  return value.replace(/[_-](S|spend)$/i, "").replaceAll("_", " ");
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function roiOption(result: SamplingResult): EChartsCoreOption {
  const channels = [...result.channels].reverse();
  return {
    animationDuration: 520,
    aria: { enabled: true },
    grid: { left: 10, right: 24, top: 34, bottom: 18, containLabel: true },
    legend: {
      top: 0,
      right: 4,
      itemWidth: 13,
      itemHeight: 4,
      textStyle: { color: COLORS.muted, fontSize: 9 },
    },
    tooltip: {
      trigger: "axis",
      confine: true,
      backgroundColor: "rgba(255,255,255,.98)",
      borderColor: "#dedde5",
      borderWidth: 1,
      padding: [10, 12],
      extraCssText:
        "border-radius:10px;box-shadow:0 12px 32px rgba(35,33,54,.14);",
      formatter: (params: unknown) => {
        const points = (Array.isArray(params) ? params : [params]) as {
          axisValue?: string;
          marker?: string;
          seriesName?: string;
          value?: number[];
        }[];
        const heading = escapeHtml(String(points[0]?.axisValue ?? ""));
        const rows = points
          .filter((point) => Array.isArray(point.value))
          .map((point) => {
            const values = point.value ?? [];
            return `<div class="flux-tooltip-row"><span>${point.marker ?? ""}${escapeHtml(point.seriesName ?? "")}</span><b>${VALUE_FORMATTER.format(values[2] ?? 0)}× <small>${VALUE_FORMATTER.format(values[0] ?? 0)}–${VALUE_FORMATTER.format(values[4] ?? 0)}</small></b></div>`;
          })
          .join("");
        return `<div class="flux-tooltip-date">${heading}</div>${rows}`;
      },
    },
    xAxis: {
      type: "value",
      name: "Incremental ROI",
      nameLocation: "middle",
      nameGap: 27,
      nameTextStyle: { color: COLORS.muted, fontSize: 9 },
      axisLine: { lineStyle: { color: "#dedde4" } },
      axisTick: { show: false },
      axisLabel: { color: "#9a99a3", fontSize: 9, formatter: "{value}×" },
      splitLine: { lineStyle: { color: COLORS.grid, type: "dashed" } },
    },
    yAxis: {
      type: "category",
      data: channels.map((channel) => cleanChannel(channel.channel)),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: COLORS.ink, fontSize: 9 },
    },
    series: [
      {
        name: "Analytic screening",
        type: "boxplot",
        boxWidth: [4, 8],
        data: channels.map((channel) => [
          channel.mapLow,
          channel.mapRoi,
          channel.mapRoi,
          channel.mapRoi,
          channel.mapHigh,
        ]),
        itemStyle: {
          color: COLORS.orangeFill,
          borderColor: COLORS.orange,
          borderWidth: 1.3,
        },
      },
      {
        name: "MCMC posterior",
        type: "boxplot",
        boxWidth: [9, 15],
        data: channels.map((channel) => [
          channel.posteriorLow,
          channel.posteriorMedian,
          channel.posteriorMedian,
          channel.posteriorMedian,
          channel.posteriorHigh,
        ]),
        itemStyle: {
          color: COLORS.violetFill,
          borderColor: COLORS.violet,
          borderWidth: 1.6,
        },
      },
    ],
  };
}

function predictiveOption(result: SamplingResult): EChartsCoreOption {
  const predictive = result.predictive;
  const width = predictive.high.map(
    (value, index) => value - predictive.low[index],
  );
  return {
    animationDuration: 520,
    aria: { enabled: true },
    grid: { left: 9, right: 14, top: 36, bottom: 28, containLabel: true },
    legend: {
      top: 0,
      right: 4,
      itemWidth: 13,
      itemHeight: 3,
      textStyle: { color: COLORS.muted, fontSize: 9 },
    },
    tooltip: {
      trigger: "axis",
      confine: true,
      backgroundColor: "rgba(255,255,255,.98)",
      borderColor: "#dedde5",
      borderWidth: 1,
      padding: [10, 12],
      extraCssText:
        "border-radius:10px;box-shadow:0 12px 32px rgba(35,33,54,.14);",
      formatter: (params: unknown) => {
        const points = (Array.isArray(params) ? params : [params]) as {
          dataIndex?: number;
          marker?: string;
          seriesName?: string;
          value?: number;
        }[];
        const index = points[0]?.dataIndex ?? 0;
        const rawDate = predictive.dates[index] ?? "";
        const parsed = Date.parse(rawDate);
        const date = Number.isFinite(parsed)
          ? DATE_FORMATTER.format(new Date(parsed))
          : rawDate;
        const visibleRows = points
          .filter((point) =>
            ["Observed", "Posterior mean response", "Analytic screening"].includes(
              point.seriesName ?? "",
            ),
          )
          .map(
            (point) =>
              `<div class="flux-tooltip-row"><span>${point.marker ?? ""}${escapeHtml(point.seriesName ?? "")}</span><b>${VALUE_FORMATTER.format(Number(point.value) || 0)}</b></div>`,
          )
          .join("");
        return `<div class="flux-tooltip-date">${escapeHtml(date)}</div>${visibleRows}<div class="flux-tooltip-row"><span><i class="flux-tooltip-band"></i>95% predictive</span><b>${VALUE_FORMATTER.format(predictive.low[index])}–${VALUE_FORMATTER.format(predictive.high[index])}</b></div><div class="flux-tooltip-row"><span>Mean-response HDI</span><b>${VALUE_FORMATTER.format(predictive.meanLow[index])}–${VALUE_FORMATTER.format(predictive.meanHigh[index])}</b></div>`;
      },
    },
    xAxis: {
      type: "category",
      data: predictive.dates,
      boundaryGap: false,
      axisLine: { lineStyle: { color: "#dedde4" } },
      axisTick: { show: false },
      axisLabel: {
        color: "#9a99a3",
        fontSize: 9,
        hideOverlap: true,
        formatter: (value: string) => {
          const parsed = Date.parse(value);
          return Number.isFinite(parsed)
            ? new Intl.DateTimeFormat("en-US", {
                month: "short",
                year: "2-digit",
              }).format(new Date(parsed))
            : value;
        },
      },
    },
    yAxis: {
      type: "value",
      scale: true,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: "#9a99a3", fontSize: 9 },
      splitLine: { lineStyle: { color: COLORS.grid, type: "dashed" } },
    },
    dataZoom: [{ type: "inside", filterMode: "none" }],
    series: [
      {
        name: "95% predictive base",
        type: "line",
        data: predictive.low,
        stack: "posterior-band",
        symbol: "none",
        lineStyle: { opacity: 0 },
        areaStyle: { opacity: 0 },
        emphasis: { disabled: true },
      },
      {
        name: "95% predictive",
        type: "line",
        data: width,
        stack: "posterior-band",
        symbol: "none",
        lineStyle: { opacity: 0 },
        areaStyle: { color: COLORS.violetFill },
        emphasis: { disabled: true },
      },
      {
        name: "Observed",
        type: "line",
        data: predictive.actual,
        showSymbol: false,
        lineStyle: { color: COLORS.ink, width: 2.3 },
      },
      {
        name: "Posterior mean response",
        type: "line",
        data: predictive.meanMedian,
        showSymbol: false,
        lineStyle: { color: COLORS.violet, width: 2.4 },
      },
      {
        name: "Analytic screening",
        type: "line",
        data: predictive.map,
        showSymbol: false,
        lineStyle: { color: COLORS.orange, width: 1.8, type: "dashed" },
      },
    ],
  };
}

function chainOption(
  result: SamplingResult,
  parameterIndex: number,
  mode: "trace" | "rank",
): EChartsCoreOption {
  const parameter =
    result.parameters[parameterIndex] ?? result.parameters[0];
  const chains = mode === "trace" ? parameter?.traces : parameter?.ranks;
  return {
    animationDuration: 360,
    aria: { enabled: true },
    grid: { left: 8, right: 12, top: 34, bottom: 20, containLabel: true },
    legend: {
      top: 0,
      right: 4,
      itemWidth: 13,
      itemHeight: 3,
      textStyle: { color: COLORS.muted, fontSize: 9 },
    },
    tooltip: {
      trigger: "axis",
      confine: true,
      backgroundColor: "rgba(255,255,255,.98)",
      borderColor: "#dedde5",
      borderWidth: 1,
    },
    xAxis: {
      type: "category",
      boundaryGap: false,
      axisLine: { lineStyle: { color: "#dedde4" } },
      axisTick: { show: false },
      axisLabel: { color: "#9a99a3", fontSize: 8, interval: 39 },
    },
    yAxis: {
      type: "value",
      scale: mode === "trace",
      min: mode === "rank" ? 0 : undefined,
      max: mode === "rank" ? 1 : undefined,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: "#9a99a3", fontSize: 8 },
      splitLine: { lineStyle: { color: COLORS.grid, type: "dashed" } },
    },
    series: (chains ?? []).map((values, index) => ({
      name: `Chain ${index + 1}`,
      type: "line",
      data: values,
      showSymbol: false,
      smooth: mode === "rank" ? 0.08 : 0,
      lineStyle: {
        color: [COLORS.violet, COLORS.orange, COLORS.mint, "#d3648f", "#6b8fcf", "#8d76c7"][index % 6],
        width: mode === "rank" ? 1.7 : 1.1,
        opacity: mode === "trace" ? 0.82 : 0.95,
      },
    })),
  };
}

function densityOption(
  result: SamplingResult,
  channelIndex: number,
): EChartsCoreOption {
  const channel = result.channels[channelIndex] ?? result.channels[0];
  const density = channel?.explanation?.density;
  if (!channel || !density) return { series: [] };
  const roiValue = (coordinate: number) =>
    density.scale === "log1p" ? Math.expm1(coordinate) : coordinate;
  const coordinate = (roi: number) =>
    density.scale === "log1p" ? Math.log1p(Math.max(roi, 0)) : roi;
  return {
    animationDuration: 520,
    aria: { enabled: true },
    grid: { left: 8, right: 18, top: 38, bottom: 31, containLabel: true },
    legend: {
      top: 0,
      right: 4,
      itemWidth: 14,
      itemHeight: 4,
      textStyle: { color: COLORS.muted, fontSize: 9 },
    },
    tooltip: {
      trigger: "axis",
      confine: true,
      backgroundColor: "rgba(255,255,255,.98)",
      borderColor: "#dedde5",
      borderWidth: 1,
      padding: [10, 12],
      extraCssText:
        "border-radius:10px;box-shadow:0 12px 32px rgba(35,33,54,.14);",
      formatter: (params: unknown) => {
        const points = (Array.isArray(params) ? params : [params]) as {
          axisValue?: number;
          marker?: string;
          seriesName?: string;
          value?: number[];
        }[];
        const roi = roiValue(Number(points[0]?.axisValue) || 0);
        const rows = points
          .map(
            (point) =>
              `<div class="flux-tooltip-row"><span>${point.marker ?? ""}${escapeHtml(point.seriesName ?? "")}</span><b>${Math.round((point.value?.[1] ?? 0) * 100)}% relative density</b></div>`,
          )
          .join("");
        return `<div class="flux-tooltip-date">${VALUE_FORMATTER.format(roi)}× ROI</div>${rows}`;
      },
    },
    xAxis: {
      type: "value",
      name:
        density.scale === "log1p"
          ? "Incremental ROI · log scale reveals the tail"
          : "Incremental ROI",
      nameLocation: "middle",
      nameGap: 25,
      nameTextStyle: { color: COLORS.muted, fontSize: 9 },
      axisLine: { lineStyle: { color: "#dedde4" } },
      axisTick: { show: false },
      axisLabel: {
        color: "#9a99a3",
        fontSize: 9,
        formatter: (value: number) => `${VALUE_FORMATTER.format(roiValue(value))}×`,
      },
      splitLine: { lineStyle: { color: COLORS.grid, type: "dashed" } },
    },
    yAxis: {
      type: "value",
      min: 0,
      max: 1.08,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { show: false },
      splitLine: { show: false },
    },
    series: [
      {
        name: "Screening local approximation",
        type: "line",
        data: density.grid.map((value, index) => [value, density.laplace[index]]),
        showSymbol: false,
        smooth: 0.32,
        lineStyle: { color: COLORS.orange, width: 2, type: "dashed" },
        areaStyle: { color: COLORS.orangeFill },
      },
      {
        name: "MCMC posterior",
        type: "line",
        data: density.grid.map((value, index) => [value, density.posterior[index]]),
        showSymbol: false,
        smooth: 0.32,
        lineStyle: { color: COLORS.violet, width: 2.5 },
        areaStyle: { color: COLORS.violetFill },
        markLine: {
          symbol: "none",
          silent: true,
          label: { color: COLORS.ink, fontSize: 9 },
          data: [
            {
              name: `Screen ${channel.mapRoi.toFixed(2)}×`,
              xAxis: coordinate(channel.mapRoi),
              lineStyle: { color: COLORS.orange, width: 1.4, type: "dashed" },
              label: { formatter: `MAP ${channel.mapRoi.toFixed(2)}×`, position: "insideEndTop" },
            },
            {
              name: `MCMC ${channel.posteriorMedian.toFixed(2)}×`,
              xAxis: coordinate(channel.posteriorMedian),
              lineStyle: { color: COLORS.violet, width: 1.5 },
              label: { formatter: `MCMC ${channel.posteriorMedian.toFixed(2)}×`, position: "insideEndBottom" },
            },
          ],
        },
      },
    ],
  };
}

export function SamplingChart({
  result,
  mode,
  label,
  parameterIndex = 0,
  channelIndex = 0,
  height = 280,
}: SamplingChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const option = useMemo(() => {
    if (mode === "roi") return roiOption(result);
    if (mode === "predictive") return predictiveOption(result);
    if (mode === "density") return densityOption(result, channelIndex);
    return chainOption(result, parameterIndex, mode);
  }, [channelIndex, mode, parameterIndex, result]);

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
        charts.BoxplotChart,
        charts.LineChart,
        components.AriaComponent,
        components.DataZoomComponent,
        components.GridComponent,
        components.LegendComponent,
        components.MarkLineComponent,
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
      className="sampling-chart"
      style={{ height }}
      role="img"
      aria-label={label}
    />
  );
}
