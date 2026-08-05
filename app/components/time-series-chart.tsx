"use client";

import type { EChartsCoreOption, EChartsType } from "echarts/core";
import { useEffect, useMemo, useRef } from "react";

export interface TimeSeries {
  name: string;
  values: number[];
  color: string;
  dashed?: boolean;
  area?: boolean;
}

interface TooltipPoint {
  axisValue?: number | string;
  color?: string;
  marker?: string;
  seriesName?: string;
  value?: [number, number] | number;
}

interface TimeSeriesChartProps {
  dates: string[];
  series: TimeSeries[];
  label: string;
  height?: number;
  showLegend?: boolean;
  showZoom?: boolean;
}

const DATE_FORMATTER = new Intl.DateTimeFormat("en-US", {
  day: "2-digit",
  month: "short",
  year: "numeric",
});

const VALUE_FORMATTER = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 2,
});

function timestamp(date: string, index: number): number {
  const parsed = Date.parse(date);
  return Number.isFinite(parsed)
    ? parsed
    : Date.UTC(1970, 0, 1 + index * 7);
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function tooltip(params: unknown): string {
  const points = (Array.isArray(params) ? params : [params]) as TooltipPoint[];
  const first = points[0];
  const rawDate = Array.isArray(first?.value)
    ? first.value[0]
    : Number(first?.axisValue);
  const date = Number.isFinite(rawDate)
    ? DATE_FORMATTER.format(new Date(rawDate))
    : String(first?.axisValue ?? "");
  const rows = points
    .map((point) => {
      const rawValue = Array.isArray(point.value)
        ? point.value[1]
        : Number(point.value);
      return `<div class="flux-tooltip-row">
        <span>${point.marker ?? ""}${escapeHtml(point.seriesName ?? "")}</span>
        <b>${VALUE_FORMATTER.format(Number(rawValue) || 0)}</b>
      </div>`;
    })
    .join("");

  return `<div class="flux-tooltip-date">${escapeHtml(date)}</div>${rows}`;
}

export function TimeSeriesChart({
  dates,
  series,
  label,
  height = 244,
  showLegend = false,
  showZoom = true,
}: TimeSeriesChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const points = useMemo(
    () =>
      series.map((item) => ({
        ...item,
        data: item.values.map((value, index) => [
          timestamp(dates[index] ?? "", index),
          value,
        ]),
      })),
    [dates, series],
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
        charts.LineChart,
        components.AriaComponent,
        components.DataZoomComponent,
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
      const option: EChartsCoreOption = {
      animationDuration: 520,
      animationEasing: "cubicOut",
      aria: {
        enabled: true,
        description: label,
      },
      color: points.map((item) => item.color),
      grid: {
        left: 8,
        right: 10,
        top: showLegend ? 34 : 12,
        bottom: showZoom ? 42 : 25,
        containLabel: true,
      },
      legend: showLegend
        ? {
            top: 0,
            right: 4,
            itemWidth: 14,
            itemHeight: 3,
            textStyle: {
              color: "#73727d",
              fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
              fontSize: 9,
            },
          }
        : undefined,
      tooltip: {
        trigger: "axis",
        confine: true,
        appendToBody: false,
        className: "flux-chart-tooltip",
        backgroundColor: "rgba(255, 255, 255, 0.98)",
        borderColor: "#dedde5",
        borderWidth: 1,
        padding: [10, 12],
        extraCssText:
          "border-radius:10px;box-shadow:0 12px 32px rgba(35,33,54,.14);",
        axisPointer: {
          type: "cross",
          snap: true,
          lineStyle: { color: "#9291a2", type: "dashed", width: 1 },
          label: {
            show: false,
          },
        },
        formatter: tooltip,
      },
      xAxis: {
        type: "time",
        boundaryGap: false,
        axisLine: { lineStyle: { color: "#dedde4" } },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: {
          color: "#9a99a3",
          fontSize: 9,
          hideOverlap: true,
          formatter: {
            year: "{yyyy}",
            month: "{MMM} {yyyy}",
            day: "{dd} {MMM}",
          },
        },
      },
      yAxis: {
        type: "value",
        scale: true,
        splitNumber: 4,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          color: "#9a99a3",
          fontSize: 9,
          formatter: (value: number) => VALUE_FORMATTER.format(value),
        },
        splitLine: {
          lineStyle: {
            color: "#ecebf0",
            type: "dashed",
          },
        },
      },
      dataZoom: showZoom
        ? [
            {
              type: "inside",
              filterMode: "none",
              zoomOnMouseWheel: "shift",
              moveOnMouseWheel: true,
              moveOnMouseMove: true,
            },
            {
              type: "slider",
              height: 14,
              bottom: 2,
              borderColor: "transparent",
              backgroundColor: "#f1f0ed",
              fillerColor: "rgba(91, 91, 214, 0.14)",
              dataBackground: {
                lineStyle: { color: "#b6b5ca", opacity: 0.7 },
                areaStyle: { color: "#e5e4f0", opacity: 0.7 },
              },
              selectedDataBackground: {
                lineStyle: { color: "#5b5bd6" },
                areaStyle: { color: "#c9c8ed" },
              },
              handleStyle: {
                color: "#ffffff",
                borderColor: "#7776d8",
                borderWidth: 1,
              },
              moveHandleStyle: { color: "#8b8ade", opacity: 0.75 },
              showDetail: false,
              brushSelect: false,
            },
          ]
        : [{ type: "inside", filterMode: "none", disabled: false }],
        series: points.map((item) => ({
          name: item.name,
          type: "line",
          data: item.data,
          showSymbol: false,
          symbol: "circle",
          symbolSize: 7,
          smooth: 0.14,
          sampling: "lttb",
          connectNulls: false,
          lineStyle: {
            color: item.color,
            width: item.dashed ? 2 : 2.5,
            type: item.dashed ? "dashed" : "solid",
          },
          itemStyle: {
            color: "#ffffff",
            borderColor: item.color,
            borderWidth: 2,
          },
          areaStyle: item.area
            ? {
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                  { offset: 0, color: `${item.color}35` },
                  { offset: 1, color: `${item.color}00` },
                ]),
              }
            : undefined,
          emphasis: {
            focus: "series",
            lineStyle: { width: 3 },
          },
        })),
      };

      chart.setOption(option, { notMerge: true });
      observer = new ResizeObserver(() => chart?.resize());
      observer.observe(container);
    })();

    return () => {
      cancelled = true;
      observer?.disconnect();
      chart?.dispose();
    };
  }, [label, points, showLegend, showZoom]);

  return (
    <div
      ref={containerRef}
      className="time-series-chart"
      style={{ height }}
      role="img"
      aria-label={label}
    />
  );
}
