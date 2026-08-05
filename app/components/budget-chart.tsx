"use client";

import type { EChartsCoreOption, EChartsType } from "echarts/core";
import { useEffect, useMemo, useRef } from "react";
import type {
  BudgetOptimizationContract,
  BudgetOptimizationResult,
  BudgetReference,
} from "../../lib/mmm/budget";

export type BudgetChartMode = "frontier" | "allocation" | "guide";

interface BudgetChartProps {
  result?: BudgetOptimizationResult;
  reference: BudgetReference;
  contract: BudgetOptimizationContract;
  mode: BudgetChartMode;
  label: string;
  height?: number;
  currency?: boolean;
}

const COLORS = {
  ink: "#363445",
  violet: "#5b5bd6",
  violetFill: "rgba(91,91,214,.16)",
  orange: "#ef9560",
  mint: "#27a589",
  grid: "#ecebf0",
  muted: "#8e8d99",
};

const INTERVAL_BASE_SERIES = "Posterior interval base";
const INTERVAL_SERIES = "80% posterior range";

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatter(currency: boolean) {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
    ...(currency ? { style: "currency", currency: "USD" } : {}),
  });
}

function illustration(
  reference: BudgetReference,
  contract: BudgetOptimizationContract,
) {
  const currentBudget = Math.max(reference.currentBudget, 1);
  const baseline = Math.max(reference.baselineOutcome, 1);
  const currentIncremental = Math.max(reference.currentIncrementalOutcome, baseline * 0.08);
  return Array.from({ length: 21 }, (_, index) => {
    const multiplier = 0.5 + (2 * index) / 20;
    const budget = currentBudget * multiplier;
    const incremental =
      currentIncremental *
      (1 - Math.exp(-1.35 * multiplier)) /
      (1 - Math.exp(-1.35));
    const total = baseline + incremental;
    const incrementalLow = incremental * 0.88;
    const incrementalHigh = incremental * 1.12;
    return {
      budget,
      incrementalOutcome: incremental,
      incrementalLow,
      incrementalHigh,
      totalOutcome: total,
      low: baseline + incrementalLow,
      high: baseline + incrementalHigh,
      profit: incremental * contract.grossMargin - budget,
      targetProbability: 0,
      feasible: true,
    };
  });
}

function frontierOption(
  result: BudgetOptimizationResult | undefined,
  reference: BudgetReference,
  contract: BudgetOptimizationContract,
  currency: boolean,
  guide: boolean,
): EChartsCoreOption {
  const points = result?.frontier ?? illustration(reference, contract);
  const economic = contract.scenario === "economic";
  const yValue = (point: (typeof points)[number]) =>
    economic ? point.profit : point.totalOutcome;
  const low = points.map((point) =>
    economic
      ? point.incrementalLow * contract.grossMargin - point.budget
      : point.low,
  );
  const high = points.map((point) =>
    economic
      ? point.incrementalHigh * contract.grossMargin - point.budget
      : point.high,
  );
  const width = high.map((value, index) => Math.max(value - low[index], 0));
  const valueFormat = formatter(currency);
  const target =
    contract.targetOutcome > 0
      ? contract.targetOutcome
      : reference.defaultTargetOutcome;
  const targetSeries =
    contract.scenario === "target"
      ? [
          {
            name: "Outcome target",
            type: "line" as const,
            data: points.map((point) => [point.budget, target]),
            showSymbol: false,
            lineStyle: { color: COLORS.orange, width: 1.5, type: "dashed" },
          },
        ]
      : [];
  const currentY = economic
    ? reference.currentIncrementalOutcome * contract.grossMargin -
      reference.currentBudget
    : reference.currentTotalOutcome;
  const selectedBudget =
    result?.recommendedBudget ??
    (contract.scenario === "fixed"
      ? reference.currentBudget * (1 + contract.budgetChange)
      : points[Math.floor(points.length * 0.55)]?.budget ?? reference.currentBudget);
  const selectedPoint = points.reduce((closest, point) =>
    Math.abs(point.budget - selectedBudget) <
    Math.abs(closest.budget - selectedBudget)
      ? point
      : closest,
  points[0]);
  return {
    animationDuration: 520,
    aria: { enabled: true },
    grid: { left: 8, right: 18, top: 36, bottom: 31, containLabel: true },
    legend: {
      top: 0,
      right: 3,
      selectedMode: false,
      data: [
        INTERVAL_SERIES,
        economic ? "Expected profit" : "Efficient frontier",
        ...(contract.scenario === "target" ? ["Outcome target"] : []),
        "Current plan",
        result ? "Recommendation" : "Selected point",
      ],
      itemWidth: 13,
      itemHeight: 4,
      textStyle: { color: COLORS.muted, fontSize: 8 },
    },
    tooltip: {
      trigger: "axis",
      hideDelay: 0,
      confine: true,
      backgroundColor: "rgba(255,255,255,.98)",
      borderColor: "#dedde5",
      borderWidth: 1,
      padding: [10, 12],
      extraCssText:
        "border-radius:10px;box-shadow:0 12px 32px rgba(35,33,54,.14);",
      formatter: (params: unknown) => {
        const rows = (Array.isArray(params) ? params : [params]) as {
          marker?: string;
          seriesName?: string;
          value?: [number, number];
        }[];
        const visible = rows.filter(
          (row) =>
            row.seriesName !== INTERVAL_BASE_SERIES &&
            row.seriesName !== INTERVAL_SERIES,
        );
        const budget = Number(rows[0]?.value?.[0]) || 0;
        return `<div class="flux-tooltip-date">${valueFormat.format(budget)} budget</div>${visible
          .map(
            (row) =>
              `<div class="flux-tooltip-row"><span>${row.marker ?? ""}${escapeHtml(row.seriesName ?? "")}</span><b>${valueFormat.format(Number(row.value?.[1]) || 0)}</b></div>`,
          )
          .join("")}`;
      },
    },
    xAxis: {
      type: "value",
      name: "Total media budget",
      nameLocation: "middle",
      nameGap: 25,
      nameTextStyle: { color: COLORS.muted, fontSize: 8 },
      axisLine: { lineStyle: { color: "#dedde4" } },
      axisTick: { show: false },
      axisLabel: {
        color: "#9a99a3",
        fontSize: 8,
        formatter: (value: number) => valueFormat.format(value),
      },
      splitLine: { lineStyle: { color: COLORS.grid, type: "dashed" } },
    },
    yAxis: {
      type: "value",
      scale: true,
      name: economic ? "Incremental profit" : "Expected outcome",
      nameTextStyle: { color: COLORS.muted, fontSize: 8 },
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: "#9a99a3",
        fontSize: 8,
        formatter: (value: number) => valueFormat.format(value),
      },
      splitLine: { lineStyle: { color: COLORS.grid, type: "dashed" } },
    },
    series: [
      {
        name: INTERVAL_BASE_SERIES,
        type: "line",
        data: points.map((point, index) => [point.budget, low[index]]),
        stack: "budget-band",
        symbol: "none",
        lineStyle: { opacity: 0 },
        itemStyle: { color: "transparent" },
        areaStyle: { opacity: 0 },
        emphasis: { disabled: true },
      },
      {
        name: INTERVAL_SERIES,
        type: "line",
        data: points.map((point, index) => [point.budget, width[index]]),
        stack: "budget-band",
        symbol: "none",
        lineStyle: { opacity: 0 },
        itemStyle: {
          color: COLORS.violetFill,
          borderColor: COLORS.violet,
          borderWidth: 1,
        },
        areaStyle: { color: COLORS.violetFill },
        emphasis: { disabled: true },
      },
      {
        name: economic ? "Expected profit" : "Efficient frontier",
        type: "line",
        data: points.map((point) => [point.budget, yValue(point)]),
        showSymbol: false,
        smooth: 0.28,
        lineStyle: { color: COLORS.violet, width: 2.5 },
      },
      ...targetSeries,
      {
        name: "Current plan",
        type: "scatter",
        data: [[reference.currentBudget, currentY]],
        symbolSize: guide ? 8 : 10,
        itemStyle: { color: COLORS.ink },
      },
      {
        name: result ? "Recommendation" : "Selected point",
        type: "scatter",
        data: [[selectedPoint.budget, yValue(selectedPoint)]],
        symbolSize: guide ? 10 : 12,
        itemStyle: {
          color: economic ? COLORS.mint : COLORS.orange,
          borderColor: "white",
          borderWidth: 2,
        },
      },
    ],
  };
}

function allocationOption(
  result: BudgetOptimizationResult,
  currency: boolean,
): EChartsCoreOption {
  const channels = [...result.channels].reverse();
  const labels = channels.map((channel) =>
    channel.channel.replace(/_S$/i, "").replaceAll("_", " "),
  );
  const valueFormat = formatter(currency);
  return {
    animationDuration: 480,
    aria: { enabled: true },
    color: ["#d9d8df", COLORS.violet],
    grid: { left: 8, right: 15, top: 34, bottom: 20, containLabel: true },
    legend: {
      top: 0,
      right: 3,
      selectedMode: false,
      itemWidth: 13,
      itemHeight: 4,
      textStyle: { color: COLORS.muted, fontSize: 8 },
    },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      hideDelay: 0,
      confine: true,
      backgroundColor: "rgba(255,255,255,.98)",
      borderColor: "#dedde5",
      formatter: (params: unknown) => {
        const rows = (Array.isArray(params) ? params : [params]) as {
          marker?: string;
          seriesName?: string;
          axisValue?: string;
          value?: [number, string];
        }[];
        const channel = rows[0]?.value?.[1] ?? rows[0]?.axisValue ?? "";
        return `<div class="flux-tooltip-date">${escapeHtml(channel)}</div>${rows
          .map(
            (row) =>
              `<div class="flux-tooltip-row"><span>${row.marker ?? ""}${escapeHtml(row.seriesName ?? "")}</span><b>${valueFormat.format(Number(row.value?.[0]) || 0)}</b></div>`,
          )
          .join("")}`;
      },
    },
    xAxis: {
      type: "value",
      axisLine: { lineStyle: { color: "#dedde4" } },
      axisTick: { show: false },
      axisLabel: {
        color: "#9a99a3",
        fontSize: 8,
        formatter: (value: number) => valueFormat.format(value),
      },
      splitLine: { lineStyle: { color: COLORS.grid, type: "dashed" } },
    },
    yAxis: {
      type: "category",
      data: labels,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: COLORS.ink, fontSize: 8 },
    },
    series: [
      {
        name: "Current",
        type: "bar",
        data: channels.map((channel, index) => [
          channel.currentBudget,
          labels[index],
        ]),
        encode: { x: 0, y: 1 },
        barWidth: 7,
        itemStyle: { borderRadius: [0, 4, 4, 0] },
      },
      {
        name: "Recommended",
        type: "bar",
        data: channels.map((channel, index) => [
          channel.recommendedBudget,
          labels[index],
        ]),
        encode: { x: 0, y: 1 },
        barWidth: 11,
        itemStyle: { borderRadius: [0, 5, 5, 0] },
      },
    ],
  };
}

export function BudgetChart({
  result,
  reference,
  contract,
  mode,
  label,
  height = 280,
  currency = true,
}: BudgetChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const option = useMemo(() => {
    if (mode === "allocation" && result) {
      return allocationOption(result, currency);
    }
    return frontierOption(
      result,
      reference,
      contract,
      currency,
      mode === "guide",
    );
  }, [contract, currency, mode, reference, result]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let cancelled = false;
    let chart: EChartsType | undefined;
    let observer: ResizeObserver | undefined;
    let hideOwnTooltip: (() => void) | undefined;
    let handleDocumentMouseMove: ((event: MouseEvent) => void) | undefined;

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
        // Axis tooltips repeatedly clear narrow regions. Full redraws keep the
        // stacked posterior band and reference lines continuous while hovering.
        useDirtyRect: false,
      });
      chart.setOption(option, { notMerge: true });
      hideOwnTooltip = () => {
        const tooltip = container.querySelector<HTMLElement>(
          ".flux-tooltip-date",
        )?.parentElement;
        if (!tooltip || tooltip.style.display === "none") return;
        chart?.dispatchAction({ type: "hideTip" });
        chart?.dispatchAction({
          type: "updateAxisPointer",
          currTrigger: "leave",
        });
        tooltip.style.display = "none";
      };
      container.addEventListener("mouseleave", hideOwnTooltip);
      handleDocumentMouseMove = (event) => {
        if (!container.contains(event.target as Node)) hideOwnTooltip?.();
      };
      document.addEventListener("mousemove", handleDocumentMouseMove, true);
      observer = new ResizeObserver(() => chart?.resize());
      observer.observe(container);
    })();

    return () => {
      cancelled = true;
      observer?.disconnect();
      if (hideOwnTooltip) {
        container.removeEventListener("mouseleave", hideOwnTooltip);
      }
      if (handleDocumentMouseMove) {
        document.removeEventListener(
          "mousemove",
          handleDocumentMouseMove,
          true,
        );
      }
      chart?.dispose();
    };
  }, [option]);

  return (
    <div
      ref={containerRef}
      className="budget-chart"
      style={{ height }}
      role="img"
      aria-label={label}
    />
  );
}
