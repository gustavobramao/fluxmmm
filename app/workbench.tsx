"use client";

import {
  type ReactNode,
  useCallback,
  useEffect,
  useEffectEvent,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { createPortal } from "react-dom";
import { TimeSeriesChart } from "./components/time-series-chart";
import {
  ValidationEvidenceChart,
  type ValidationChartMode,
} from "./components/validation-evidence-chart";
import {
  SamplingChart,
  type SamplingChartMode,
} from "./components/sampling-chart";
import { BudgetChart } from "./components/budget-chart";
import { ScoreLabView } from "./components/score-lab-view";
import {
  getCachedBudget,
  getCachedSampling,
  getCachedValidation,
  getCachedModel,
  persistBudget,
  persistDataset,
  persistModel,
  persistSampling,
  persistValidation,
} from "../lib/mmm/api";
import {
  budgetFingerprint,
  createBudgetReference,
  DEFAULT_BUDGET_CONTRACT,
  defaultBudgetConstraints,
  runBudgetOptimization,
} from "../lib/mmm/budget";
import type {
  BudgetChannelConstraint,
  BudgetOptimizationContract,
  BudgetOptimizationProgress,
  BudgetOptimizationResult,
  BudgetScenarioType,
} from "../lib/mmm/budget";
import {
  advancedModelFingerprint,
  DEFAULT_ADVANCED_CONFIG,
  runAdvancedModel,
} from "../lib/mmm/advanced";
import {
  agenticSearchStage,
  createAgenticForcePromotionAudit,
  DEFAULT_AGENTIC_SEARCH_CONTRACT,
  findAgenticBenchmarkRescueRecommendations,
  findAgenticRoiGuardrailViolations,
  passesAgenticEligibility,
  rankAgenticCandidates,
  selectAgenticWinner,
} from "../lib/mmm/agentic";
import type {
  AgenticCandidateRun,
  AgenticForcePromotionAudit,
  AgenticSearchContract,
} from "../lib/mmm/agentic";
import {
  agenticAdaptiveMinimumBudget,
  agenticAdaptiveMaximumBudget,
  agenticAdvancedChallengeBudget,
  agenticAdvancedCoverage,
  agenticBoundaryParameters,
  agenticChannelResponseBudget,
  agenticChannelResponseCoverage,
  agenticSeedBudget,
  agenticLocalChallengeBudget,
  agenticSearchConfidence,
  agenticStoppingDecision,
  generateAgenticAdvancedChallenge,
  generateAgenticChannelResponseChallenge,
  generateAgenticLocalChallenge,
  generateAgenticSeeds,
  proposeAgenticCandidate,
} from "../lib/mmm/agentic-search";
import {
  activeIndustryPrior,
  channelExperiments,
  INDUSTRY_BENCHMARKS,
  inferIndustryPrior,
  industryPriorPercentile,
  isHighlyImprobableIndustryRoi,
} from "../lib/mmm/benchmarks";
import { formatCompact, formatFull, parseCsv, toNumber } from "../lib/mmm/csv";
import { runEda } from "../lib/mmm/eda";
import { modelExperimentWindowRoi } from "../lib/mmm/experiment-window";
import { mean } from "../lib/mmm/math";
import {
  ACTIVE_SCORE_CONTRACT,
  activeScoreFormula,
  scoreWeightPercent,
} from "../lib/mmm/score-contract";
import { responseForChannel } from "../lib/mmm/response";
import {
  defaultExperimentsForDataset,
  DEFAULT_CONFIG,
  modelFingerprint,
  runModel,
} from "../lib/mmm/models";
import type { DatasetExperimentOrigin } from "../lib/mmm/models";
import {
  createDataset,
  updateColumnRole,
  validateDataset,
} from "../lib/mmm/schema";
import {
  compileSamplingModel,
  DEFAULT_SAMPLING_CONTRACT,
  sameSamplingContract,
  SAMPLING_PRESETS,
  samplingContractLabel,
  samplingFingerprint,
} from "../lib/mmm/sampling";
import type {
  SamplingChannelPosterior,
  SamplingContract,
  SamplingJobProgress,
  SamplingResult,
} from "../lib/mmm/sampling";
import {
  getSamplingJob,
  samplingServiceHealth,
  startSamplingJob,
} from "../lib/mmm/sampling-api";
import {
  assessExternalAnchorEligibility,
  runModelValidation,
  validationFingerprint,
} from "../lib/mmm/validation";
import {
  createWorkspaceCheckpoint,
  persistWorkspaceCheckpoint,
  readLatestWorkspaceCheckpoint,
  type WorkspaceCheckpointEnvelope,
} from "../lib/mmm/workspace-checkpoint";
import type {
  AdvancedModelConfig,
  ColumnRole,
  Dataset,
  EdaResult,
  Experiment,
  ModelConfig,
  ModelResult,
  ValidationResult,
} from "../lib/mmm/types";
import type {
  ValidationLayerId,
  ValidationLayerResult,
  ValidationModelKind,
  ValidationProgress,
  ValidationResult as ModelValidationResult,
  ValidationStatus,
} from "../lib/mmm/validation";

type View =
  | "overview"
  | "data"
  | "eda"
  | "models"
  | "advanced"
  | "calibration"
  | "validation"
  | "scorelab"
  | "agentic"
  | "sampling"
  | "budget";
type JobStatus = "idle" | "queued" | "running" | "complete" | "cached";
type BenchmarkGuardrailMode = "suggest" | "auto" | "off";
type WorkspaceSaveStatus = "restoring" | "saving" | "saved" | "unavailable";

const EMPTY_AGENTIC_CHECKPOINT_RUNS: AgenticCandidateRun[] = [];

interface StoredBenchmarkPolicy {
  mode: BenchmarkGuardrailMode;
  channels: string[];
}

const BENCHMARK_POLICY_STORAGE_KEY = "fluxmmm-benchmark-policy-v1";

function eligibleStoredBenchmarkChannels(
  dataset: Dataset,
  experiments: Experiment[],
  channels: readonly string[],
): string[] {
  const mediaByName = new Map(
    dataset.mediaColumns.map((channel) => [channel.toLowerCase(), channel]),
  );
  return Array.from(
    new Set(
      channels.flatMap((channel) => {
        const canonical = mediaByName.get(channel.toLowerCase());
        return canonical && !channelExperiments(experiments, canonical).length
          ? [canonical]
          : [];
      }),
    ),
  );
}

function readStoredBenchmarkPolicy(
  dataset: Dataset,
  experiments: Experiment[],
): StoredBenchmarkPolicy | undefined {
  if (typeof window === "undefined") return undefined;
  try {
    const stored = JSON.parse(
      window.localStorage.getItem(BENCHMARK_POLICY_STORAGE_KEY) ?? "{}",
    ) as Record<string, Partial<StoredBenchmarkPolicy>>;
    const policy = stored[dataset.hash];
    if (!policy) return undefined;
    const mode =
      policy.mode === "auto" || policy.mode === "off"
        ? policy.mode
        : "suggest";
    return {
      mode,
      channels:
        mode === "off"
          ? []
          : eligibleStoredBenchmarkChannels(
              dataset,
              experiments,
              Array.isArray(policy.channels) ? policy.channels : [],
            ),
    };
  } catch {
    return undefined;
  }
}

function writeStoredBenchmarkPolicy(
  dataset: Dataset,
  policy: StoredBenchmarkPolicy,
): void {
  if (typeof window === "undefined") return;
  try {
    const stored = JSON.parse(
      window.localStorage.getItem(BENCHMARK_POLICY_STORAGE_KEY) ?? "{}",
    ) as Record<string, StoredBenchmarkPolicy>;
    stored[dataset.hash] = policy;
    window.localStorage.setItem(
      BENCHMARK_POLICY_STORAGE_KEY,
      JSON.stringify(stored),
    );
  } catch {
    // Storage can be disabled; the in-memory evidence contract still works.
  }
}

const NAV_ITEMS: { id: View; label: string; glyph: string }[] = [
  { id: "overview", label: "Overview", glyph: "⌂" },
  { id: "data", label: "Data & schema", glyph: "▦" },
  { id: "eda", label: "Explore", glyph: "⌁" },
  { id: "models", label: "Models", glyph: "∿" },
  { id: "calibration", label: "Calibration", glyph: "◎" },
  { id: "advanced", label: "Advanced", glyph: "◇" },
  { id: "validation", label: "Validation", glyph: "✓" },
  { id: "scorelab", label: "Score research", glyph: "◈" },
  { id: "agentic", label: "Agentic", glyph: "✦" },
  { id: "sampling", label: "Production", glyph: "◉" },
  { id: "budget", label: "Budget", glyph: "$" },
];

const ROLE_LABELS: Record<ColumnRole, string> = {
  date: "Date",
  outcome: "Outcome",
  media_spend: "Media spend",
  media_exposure: "Media exposure",
  control: "Control",
  categorical: "Categorical",
  ignore: "Ignore",
};

const CHANNEL_COLORS = [
  "#5b5bd6",
  "#f29d62",
  "#27a589",
  "#d3648f",
  "#8d76c7",
  "#6b8fcf",
];

function cyclePeriodsForDataset(dataset: Dataset): readonly number[] {
  return dataset.modelCadence === "monthly" ? [3, 6, 12] : [13, 26, 52];
}

function periodLabel(dataset: Dataset, count: number, short = false): string {
  if (short) return dataset.periodUnit === "month" ? "mo" : "wk";
  return `${dataset.periodUnit}${count === 1 ? "" : "s"}`;
}

function defaultConfigForDataset(dataset: Dataset): ModelConfig {
  return {
    ...DEFAULT_CONFIG,
    cyclePeriod: dataset.modelCadence === "monthly" ? 6 : DEFAULT_CONFIG.cyclePeriod,
  };
}

function responseProfileForDataset(dataset: Dataset, config: ModelConfig) {
  return dataset.mediaColumns.map((channel) => ({
    channel,
    response: responseForChannel(config, channel),
    explicit: Boolean(
      Object.keys(config.channelResponses ?? {}).find(
        (key) => key.toLowerCase() === channel.toLowerCase(),
      ),
    ),
  }));
}

function delay(milliseconds: number) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function subscribeToClientRuntime() {
  return () => undefined;
}

function getClientRuntimeSnapshot() {
  return true;
}

function getServerRuntimeSnapshot() {
  return false;
}

function DrawerShell({
  children,
  className = "",
  labelledBy,
  onClose,
}: {
  children: ReactNode;
  className?: string;
  labelledBy: string;
  onClose: () => void;
}) {
  const isClient = useSyncExternalStore(
    subscribeToClientRuntime,
    getClientRuntimeSnapshot,
    getServerRuntimeSnapshot,
  );
  const panelRef = useRef<HTMLElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const closeDrawer = useEffectEvent(onClose);

  useEffect(() => {
    const previousFocus =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const previousBodyOverflow = document.body.style.overflow;
    const previousBodyPaddingRight = document.body.style.paddingRight;
    const previousHtmlOverflow = document.documentElement.style.overflow;
    const appShell = document.querySelector<HTMLElement>(".app-shell");
    const appWasInert = appShell?.hasAttribute("inert") ?? false;
    const scrollbarWidth =
      window.innerWidth - document.documentElement.clientWidth;

    document.body.style.overflow = "hidden";
    document.documentElement.style.overflow = "hidden";
    if (scrollbarWidth > 0) {
      document.body.style.paddingRight = `${scrollbarWidth}px`;
    }
    appShell?.setAttribute("inert", "");

    const focusDrawer = window.requestAnimationFrame(() => {
      scrollRef.current?.scrollTo({ top: 0 });
      panelRef.current
        ?.querySelector<HTMLElement>(".drawer-close")
        ?.focus();
    });

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeDrawer();
        return;
      }
      if (event.key !== "Tab" || !panelRef.current) return;
      const focusable = Array.from(
        panelRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => !element.hasAttribute("hidden"));
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      } else if (!panelRef.current.contains(document.activeElement)) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.cancelAnimationFrame(focusDrawer);
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousBodyOverflow;
      document.body.style.paddingRight = previousBodyPaddingRight;
      document.documentElement.style.overflow = previousHtmlOverflow;
      if (!appWasInert) appShell?.removeAttribute("inert");
      previousFocus?.focus();
    };
  }, []);

  if (!isClient) return null;

  return createPortal(
    <div
      className="assumptions-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <aside
        ref={panelRef}
        className={`assumptions-drawer ${className}`.trim()}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
      >
        <div ref={scrollRef} className="drawer-scroll-region">
          {children}
        </div>
      </aside>
    </div>,
    document.body,
  );
}

function cleanChannel(channel: string): string {
  return channel
    .replace(/_S$/i, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatCorrelation(value: number): string {
  const rounded = Math.abs(value) < 0.005 ? 0 : value;
  if (rounded === 1) return "1.00";
  if (rounded === -1) return "−1.00";
  return `${rounded > 0 ? "+" : ""}${rounded.toFixed(2)}`.replace("-", "−");
}

function correlationColor(value: number): string {
  const strength = 0.07 + Math.min(Math.abs(value), 1) * 0.82;
  return value < 0
    ? `rgba(218, 111, 72, ${strength})`
    : `rgba(91, 91, 214, ${strength})`;
}

function SkeletonCard({ height = 170 }: { height?: number }) {
  return (
    <div className="card skeleton-card" style={{ minHeight: height }}>
      <span className="skeleton-line wide" />
      <span className="skeleton-line medium" />
      <span className="skeleton-block" />
    </div>
  );
}

function FluxGlyph({ large = false }: { large?: boolean }) {
  return (
    <span className={large ? "empty-mark" : "brand-mark"} aria-hidden="true">
      <span className="flux-glyph">
        <i />
        <i />
        <i />
      </span>
    </span>
  );
}

function EmptyState() {
  return (
    <main className="empty-main">
      <FluxGlyph large />
      <h1>Preparing your measurement workspace</h1>
      <p>Loading the public weekly demo dataset and validating its schema.</p>
      <div className="empty-progress">
        <span />
      </div>
    </main>
  );
}

function MetricCard({
  eyebrow,
  value,
  detail,
  tone,
}: {
  eyebrow: string;
  value: string;
  detail: string;
  tone?: "violet" | "mint" | "orange";
}) {
  return (
    <article className={`metric-card ${tone ?? ""}`}>
      <span>{eyebrow}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

function JobPill({ status }: { status: JobStatus }) {
  return (
    <span className={`job-pill ${status}`}>
      <i />
      {status === "complete"
        ? "Ready"
        : status === "cached"
          ? "Cached"
          : status === "running"
            ? "Running"
            : status === "queued"
              ? "Queued"
              : "Not run"}
    </span>
  );
}

function Header({
  dataset,
  onUpload,
  demoMode,
  saveStatus,
  savedAt,
}: {
  dataset: Dataset;
  onUpload: () => void;
  demoMode: boolean;
  saveStatus: WorkspaceSaveStatus;
  savedAt?: string;
}) {
  const savedLabel =
    saveStatus === "restoring"
      ? "Restoring"
      : saveStatus === "saving"
        ? "Saving…"
        : saveStatus === "unavailable"
          ? "Save unavailable"
          : "Saved locally";
  return (
    <header className="topbar">
      <div className="breadcrumb">
        <span>Projects</span>
        <b>›</b>
        <strong>{dataset.name.replace(/\.csv$/i, "")}</strong>
      </div>
      <div className="header-actions">
        <span
          className={`saved-state ${saveStatus}`}
          title={
            savedAt
              ? `Local workspace checkpoint saved ${new Date(savedAt).toLocaleString()}`
              : "Flux automatically checkpoints this workspace on this browser."
          }
        >
          <i /> {savedLabel}
        </span>
        {demoMode ? (
          <span className="demo-mode-pill">◇ Public fixture · read only</span>
        ) : (
          <button className="button secondary" onClick={onUpload}>
            ↑ Upload data
          </button>
        )}
      </div>
    </header>
  );
}

function Sidebar({
  view,
  setView,
  validation,
}: {
  view: View;
  setView: (view: View) => void;
  validation: ValidationResult;
}) {
  return (
    <aside className="sidebar">
      <button className="brand" onClick={() => setView("overview")}>
        <FluxGlyph />
        <span>
          <b>Flux</b>
          <small>MMM</small>
        </span>
      </button>
      <nav aria-label="Workspace navigation">
        <p>Workspace</p>
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className={view === item.id ? "active" : ""}
            onClick={() => setView(item.id)}
          >
            <span>{item.glyph}</span>
            {item.label}
            {item.id === "data" && (
              <i className={`nav-status ${validation.status}`} />
            )}
          </button>
        ))}
      </nav>
      <div className="side-bottom">
        <div className="open-badge">
          <i>◎</i>
          <span>
            <b>RegretSet-MMM</b>
            <small>Adaptive V11 · externally audited</small>
          </span>
        </div>
        <button className="user-chip">
          <span>GB</span>
          <p>
            <b>Demo workspace</b>
            <small>Advertiser analyst</small>
          </p>
          <i>···</i>
        </button>
      </div>
    </aside>
  );
}

function OverviewView({
  dataset,
  validation,
  eda,
  edaStatus,
  models,
  onNavigate,
}: {
  dataset: Dataset;
  validation: ValidationResult;
  eda: EdaResult | null;
  edaStatus: JobStatus;
  models: Partial<Record<"frequentist" | "bayesian", ModelResult>>;
  onNavigate: (view: View) => void;
}) {
  const bestModel = models.bayesian ?? models.frequentist;
  const dates = dataset.rows.map((row) => String(row[dataset.dateColumn]));
  const cadenceSummary =
    dataset.sourceCadence === "daily"
      ? `We safely converted ${dataset.sourceRowCount} daily observations into ${dataset.rows.length} consecutive weekly modeling periods`
      : `We validated ${dataset.rows.length} ${dataset.modelCadence} observations`;
  return (
    <div className="view">
      <section className="page-heading">
        <div>
          <span className="kicker">Measurement workspace</span>
          <h1>Good morning. Your data is ready to model.</h1>
          <p>
            {cadenceSummary} and started the diagnostics that matter before
            estimating channel ROI.
          </p>
        </div>
        <div className="health-orbit" aria-label={`${validation.score} data health`}>
          <strong>{validation.score}</strong>
          <span>health</span>
        </div>
      </section>

      <section className="metric-grid">
        <MetricCard
          eyebrow="Date coverage"
          value={`${validation.frequency} · ${dataset.rows.length}`}
          detail={`${validation.startDate} — ${validation.endDate}`}
          tone="violet"
        />
        <MetricCard
          eyebrow="Total outcome"
          value={eda ? formatCompact(eda.totalOutcome, true) : "Calculating"}
          detail={dataset.outcomeColumn}
          tone="mint"
        />
        <MetricCard
          eyebrow="Media investment"
          value={eda ? formatCompact(eda.totalSpend, true) : "Calculating"}
          detail={`${dataset.mediaColumns.length} paid channels`}
          tone="orange"
        />
        <MetricCard
          eyebrow="Model status"
          value={bestModel ? `${(bestModel.r2 * 100).toFixed(0)}% fit` : "Ready to run"}
          detail={bestModel ? `${bestModel.kind} · ${bestModel.cached ? "cache hit" : "fresh run"}` : "Baseline + calibrated Bayesian"}
        />
      </section>

      <section className="overview-grid">
        <article className="card performance-card">
          <div className="card-heading">
            <div>
              <span className="eyebrow">Outcome over time</span>
              <h2>{dataset.outcomeColumn}</h2>
            </div>
            <div className="chart-legend">
              <span><i className="violet-dot" /> Actual</span>
              {bestModel && <span><i className="orange-dot" /> Model fit</span>}
            </div>
          </div>
          {eda ? (
            <>
              <TimeSeriesChart
                dates={dates}
                series={[
                  {
                    name: "Actual",
                    values: eda.outcomeTrend,
                    color: "#5b5bd6",
                    area: !bestModel,
                  },
                  ...(bestModel
                    ? [{
                        name: "Model fit",
                        values: bestModel.predicted,
                        color: "#ee9c61",
                        dashed: true,
                      }]
                    : []),
                ]}
                label="Actual and modeled outcome over time"
                showZoom
              />
            </>
          ) : (
            <div className="chart-loading">Computing the first time-series view…</div>
          )}
        </article>

        <article className="card pipeline-card">
          <div className="card-heading">
            <div>
              <span className="eyebrow">Analysis pipeline</span>
              <h2>Progressive results</h2>
            </div>
            <span className="live-label">Live</span>
          </div>
          <div className="pipeline-list">
            <button onClick={() => onNavigate("data")}>
              <span className="pipeline-icon done">✓</span>
              <p><b>Schema validation</b><small>{validation.issues.length} checks completed</small></p>
              <JobPill status="complete" />
            </button>
            <button onClick={() => onNavigate("eda")}>
              <span className={`pipeline-icon ${edaStatus === "complete" ? "done" : "working"}`}>
                {edaStatus === "complete" ? "✓" : "↻"}
              </span>
              <p><b>MMM diagnostics</b><small>Variation, sparsity & collinearity</small></p>
              <JobPill status={edaStatus} />
            </button>
            <button onClick={() => onNavigate("models")}>
              <span className={`pipeline-icon ${models.frequentist ? "done" : ""}`}>
                {models.frequentist ? "✓" : "3"}
              </span>
              <p><b>Frequentist baseline</b><small>Adstock, Hill & Fourier cycle</small></p>
              <JobPill status={models.frequentist?.cached ? "cached" : models.frequentist ? "complete" : "idle"} />
            </button>
            <button onClick={() => onNavigate("models")}>
              <span className={`pipeline-icon ${models.bayesian ? "done" : ""}`}>
                {models.bayesian ? "✓" : "4"}
              </span>
              <p><b>Calibrated Bayesian</b><small>Experiment-first priors + optional benchmarks</small></p>
              <JobPill status={models.bayesian?.cached ? "cached" : models.bayesian ? "complete" : "idle"} />
            </button>
          </div>
        </article>
      </section>

      <section className="lower-grid">
        <article className="card">
          <div className="card-heading">
            <div>
              <span className="eyebrow">Spend profile</span>
              <h2>Channel mix</h2>
            </div>
            <button className="text-button" onClick={() => onNavigate("eda")}>Full EDA →</button>
          </div>
          {eda ? (
            <div className="channel-list compact">
              {eda.channels.slice(0, 5).map((channel, index) => (
                <div key={channel.channel}>
                  <span className="channel-name">
                    <i style={{ background: CHANNEL_COLORS[index] }} />
                    {cleanChannel(channel.channel)}
                  </span>
                  <div className="bar-track">
                    <i
                      style={{
                        width: `${Math.max(channel.share * 100, 2)}%`,
                        background: CHANNEL_COLORS[index],
                      }}
                    />
                  </div>
                  <b>{(channel.share * 100).toFixed(0)}%</b>
                </div>
              ))}
            </div>
          ) : <span className="subtle">Queued behind schema checks.</span>}
        </article>

        <article className="card insight-card">
          <div className="insight-mark">✦</div>
          <div>
            <span className="eyebrow">Modeling note</span>
            <h2>{eda?.readiness[0] ?? "Reading the signal structure…"}</h2>
            <p>
              Flux surfaces identification risks before model output, so ROI is
              interpreted with the data-generating context in view.
            </p>
          </div>
        </article>
      </section>
    </div>
  );
}

function DataView({
  dataset,
  validation,
  onDatasetChange,
  onUpload,
  demoMode,
}: {
  dataset: Dataset;
  validation: ValidationResult;
  onDatasetChange: (dataset: Dataset) => void;
  onUpload: () => void;
  demoMode: boolean;
}) {
  return (
    <div className="view">
      <section className="page-heading compact-heading">
        <div>
          <span className="kicker">Data contract</span>
          <h1>Schema & validation</h1>
          <p>Review the inferred semantic roles before modeling.</p>
        </div>
        {demoMode ? (
          <span className="demo-mode-pill">Public demo dataset</span>
        ) : (
          <button className="button primary" onClick={onUpload}>↑ Replace dataset</button>
        )}
      </section>
      <section className="metric-grid three">
        <MetricCard
          eyebrow="Validation score"
          value={`${validation.score}/100`}
          detail={
            validation.status === "ready"
              ? "Model ready"
              : validation.status === "review"
                ? "Review recommendations"
                : "Action required"
          }
          tone="mint"
        />
        <MetricCard
          eyebrow="Dataset shape"
          value={dataset.sourceCadence === "daily" ? `${dataset.sourceRowCount} → ${dataset.rows.length}` : `${dataset.rows.length} × ${dataset.columns.length}`}
          detail={dataset.sourceCadence === "daily" ? "Daily rows → weekly model periods" : `${dataset.modelCadence} rows × columns`}
          tone="violet"
        />
        <MetricCard eyebrow="Fingerprint" value={dataset.hash.slice(0, 10)} detail="Content-addressed version" tone="orange" />
      </section>
      <section className="data-layout">
        <article className="card schema-card">
          <div className="card-heading">
            <div><span className="eyebrow">Canonical mapping</span><h2>Column roles</h2></div>
            <span className="subtle">{dataset.name}</span>
          </div>
          <div className="schema-table">
            <div className="schema-row schema-head">
              <span>Column</span><span>Detected type</span><span>Semantic role</span><span>Example</span>
            </div>
            {dataset.specs.map((column) => (
              <div className="schema-row" key={column.name}>
                <strong>{column.name}</strong>
                <span><i className={`type-dot ${column.numeric ? "numeric" : ""}`} />{column.numeric ? "Numeric" : "Text"}</span>
                <select
                  aria-label={`Role for ${column.name}`}
                  value={column.role}
                  onChange={(event) => {
                    void updateColumnRole(
                        dataset,
                        column.name,
                        event.target.value as ColumnRole,
                      ).then(onDatasetChange);
                  }}
                >
                  {Object.entries(ROLE_LABELS).map(([value, label]) => (
                    <option value={value} key={value}>{label}</option>
                  ))}
                </select>
                <code>{String(dataset.rows[0]?.[column.name] ?? "—").slice(0, 18)}</code>
              </div>
            ))}
          </div>
        </article>
        <aside className="card validation-card">
          <div className="validation-score">
            <div className={`score-ring ${validation.status}`}>
              <strong>{validation.score}</strong><span>/100</span>
            </div>
            <div><span className="eyebrow">Readiness</span><h2>{validation.status === "ready" ? "Model ready" : validation.status === "review" ? "Ready with notes" : "Action required"}</h2></div>
          </div>
          <div className="issue-list">
            {validation.issues.map((issue) => (
              <div className={`issue ${issue.level}`} key={issue.title}>
                <span>{issue.level === "error" ? "!" : issue.level === "warning" ? "△" : "✓"}</span>
                <p><b>{issue.title}</b><small>{issue.detail}</small></p>
              </div>
            ))}
          </div>
          <div className="contract-note">
            <b>Cadence-aware data contract</b>
            <p>
              Flux accepts consecutive daily, weekly, or monthly rows. Daily
              data is safely aggregated into complete seven-day modeling
              periods; weekly and monthly data model at their native cadence.
              Every contract requires one numeric outcome, non-negative media
              spend, and no empty modeled cells.
            </p>
            <p>
              <strong>{validation.frequency}</strong> · {validation.startDate} — {validation.endDate}
              {dataset.controlColumns.length > 0
                ? ` · ${Math.min(dataset.controlColumns.length, 4)} active control${Math.min(dataset.controlColumns.length, 4) === 1 ? "" : "s"}`
                : " · No active controls"}
            </p>
          </div>
        </aside>
      </section>
    </div>
  );
}

function EdaView({
  dataset,
  eda,
  status,
}: {
  dataset: Dataset;
  eda: EdaResult | null;
  status: JobStatus;
}) {
  if (!eda) {
    return (
      <div className="view">
        <section className="page-heading compact-heading">
          <div><span className="kicker">Progressive EDA</span><h1>Model readiness</h1><p>Fast checks render independently as each analysis completes.</p></div>
          <JobPill status={status} />
        </section>
        <div className="eda-skeleton-grid"><SkeletonCard height={290} /><SkeletonCard height={290} /><SkeletonCard height={230} /></div>
      </div>
    );
  }

  return (
    <div className="view">
      <section className="page-heading compact-heading">
        <div><span className="kicker">Progressive EDA</span><h1>Model readiness</h1><p>Signal quality, channel variation, and identification risks.</p></div>
        <JobPill status={status} />
      </section>
      <section className="eda-grid">
        <article className="card span-two">
          <div className="card-heading">
            <div><span className="eyebrow">Media investment</span><h2>Spend over time</h2></div>
            <b>{formatCompact(eda.totalSpend, true)} total</b>
          </div>
          <TimeSeriesChart
            dates={dataset.rows.map((row) => String(row[dataset.dateColumn]))}
            series={[{
              name: "Total spend",
              values: eda.spendTrend,
              color: "#5b5bd6",
              area: true,
            }]}
            label="Total media spend over time"
            height={230}
            showZoom
          />
        </article>
        <article className="card">
          <div className="card-heading"><div><span className="eyebrow">Coverage</span><h2>Channel variation</h2></div></div>
          <div className="variation-list">
            {eda.channels.map((channel, index) => (
              <div key={channel.channel}>
                <span><i style={{ background: CHANNEL_COLORS[index % CHANNEL_COLORS.length] }} />{cleanChannel(channel.channel)}</span>
                <b>{channel.activePeriods}<small> active</small></b>
                <em>{Math.round(channel.zeroShare * 100)}% zero</em>
              </div>
            ))}
          </div>
        </article>
        <article className="card span-two correlation-card">
          <div className="card-heading">
            <div><span className="eyebrow">Identification</span><h2>Media correlation</h2></div>
            <span className="matrix-method">Pearson r</span>
          </div>
          <div className="heatmap" role="region" aria-label="Pairwise media correlation matrix">
            <table className="correlation-matrix">
              <thead>
                <tr>
                  <th className="matrix-corner" aria-label="Channel" />
                  {dataset.mediaColumns.map((column) => (
                    <th key={column} scope="col" title={cleanChannel(column)}>
                      {cleanChannel(column)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {eda.correlations.map((row, rowIndex) => (
                  <tr key={dataset.mediaColumns[rowIndex]}>
                    <th scope="row">{cleanChannel(dataset.mediaColumns[rowIndex])}</th>
                    {row.map((value, columnIndex) => {
                      const isDiagonal = rowIndex === columnIndex;
                      const channelPair = `${cleanChannel(dataset.mediaColumns[rowIndex])} × ${cleanChannel(dataset.mediaColumns[columnIndex])}`;
                      return (
                        <td
                          key={`${rowIndex}-${columnIndex}`}
                          className={isDiagonal ? "diagonal" : value < 0 ? "negative" : "positive"}
                          title={`${channelPair}: ${formatCorrelation(value)}`}
                          style={{
                            backgroundColor: correlationColor(value),
                            color: Math.abs(value) > 0.58 ? "white" : "#35344a",
                          }}
                        >
                          {formatCorrelation(value)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="correlation-legend" aria-label="Correlation color scale">
              <span><i className="negative" />−1 negative</span>
              <span><i className="neutral" />0 none</span>
              <span><i className="positive" />+1 positive</span>
            </div>
            <p className="heatmap-note">Values near zero mean the channels move independently; the 1.00 diagonal is each channel compared with itself.</p>
          </div>
        </article>
        <article className="card recommendation-card">
          <div className="card-heading"><div><span className="eyebrow">Automatic review</span><h2>What the modeler should know</h2></div><span className="ai-tag">✦ Generated</span></div>
          <div className="recommendation-list">
            {eda.readiness.map((note, index) => (
              <div key={note}><span>{index + 1}</span><p><b>{index === 0 ? "Interpretability note" : "Readiness signal"}</b><small>{note}</small></p></div>
            ))}
            <div><span>✓</span><p><b>Baseline specification</b><small>Use an annual Fourier basis plus a {dataset.modelCadence === "monthly" ? "6-month" : "26-week"} cycle, keeping media effects non-negative.</small></p></div>
          </div>
        </article>
      </section>
    </div>
  );
}

function ModelCard({
  kind,
  dates,
  result,
  status,
  onRun,
}: {
  kind: "frequentist" | "bayesian";
  dates: string[];
  result?: ModelResult;
  status: JobStatus;
  onRun: () => void;
}) {
  const title = kind === "frequentist" ? "Frequentist baseline" : "Bayesian calibrated";
  return (
    <article className={`card model-card ${kind}`}>
      <div className="model-title">
        <div className="model-symbol">{kind === "frequentist" ? "F" : "B"}</div>
        <div><span className="eyebrow">{kind === "frequentist" ? "Naive benchmark" : "Evidence-informed"}</span><h2>{title}</h2></div>
        <JobPill status={status} />
      </div>
      <p className="model-description">
        {kind === "frequentist"
          ? "Regularized regression with geometric adstock, Hill saturation, trend, Fourier seasonality, and cycle."
          : "Analytic Bayesian regression calibrated through channel ROI priors—not experimental observations in the likelihood."}
      </p>
      {status === "running" || status === "queued" ? (
        <div className="model-running">
          <span className="run-orbit" />
          <b>{kind === "bayesian" ? "Mapping calibration evidence to priors…" : "Optimizing constrained coefficients…"}</b>
          <small>The workspace remains interactive while this job runs.</small>
        </div>
      ) : result ? (
        <>
          <div className="model-metrics">
            <div><span>R²</span><b>{result.r2.toFixed(3)}</b></div>
            <div><span>MAPE</span><b>{result.mape.toFixed(1)}%</b></div>
            <div><span>RMSE</span><b>{formatCompact(result.rmse)}</b></div>
          </div>
          <TimeSeriesChart
            dates={dates}
            series={[
              { name: "Actual", values: result.actual, color: "#5b5bd6" },
              { name: "Model fit", values: result.predicted, color: "#ee9c61", dashed: true },
            ]}
            label={`${title} actual versus fitted outcome`}
            height={180}
            showLegend
            showZoom={false}
          />
          <div className="diagnostic-grid">
            {result.diagnostics.map((diagnostic) => (
              <div key={diagnostic.label}><span className={diagnostic.state} /> <small>{diagnostic.label}</small><b>{diagnostic.value}</b></div>
            ))}
          </div>
          <NumericalReliability result={result} />
        </>
      ) : (
        <div className="model-empty">
          <div className="formula">y<sub>t</sub> = baseline + Σ media<sub>c,t</sub> + ε<sub>t</sub></div>
          <p>Configuration is valid and ready for the first run.</p>
        </div>
      )}
      <button className={`button ${kind === "bayesian" ? "primary" : "secondary"} full`} onClick={onRun} disabled={status === "running" || status === "queued"}>
        {result ? "Run again" : `Run ${kind === "frequentist" ? "baseline" : "calibrated model"}`} <span>→</span>
      </button>
    </article>
  );
}

function NumericalReliability({ result }: { result: ModelResult }) {
  const numerical = result.numerical;
  if (!numerical) return null;
  const condition = numerical.dataConditionNumber;
  const conditionLabel =
    condition === null
      ? "Not finite"
      : condition < 1_000
        ? condition.toFixed(1)
        : condition.toExponential(1);
  const statusLabel =
    numerical.status === "stable"
      ? "Stable"
      : numerical.status === "rank-deficient"
        ? "Rank review"
        : "Review";

  return (
    <details className="numerical-reliability">
      <summary>
        <span>
          Numerical reliability
          <small>Solver, identification, and constraint checks</small>
        </span>
        <b className={numerical.status === "stable" ? "good" : "warn"}>
          {statusLabel}
        </b>
      </summary>
      <div className="numerical-facts">
        <span>
          Solver
          <b>{numerical.solver === "pivoted-qr" ? "Pivoted QR" : "SVD fallback"}</b>
        </span>
        <span>
          Solver rank
          <b>{numerical.rank}/{numerical.parameterCount}</b>
        </span>
        <span>
          Data rank
          <b>{numerical.dataRank}/{numerical.parameterCount}</b>
        </span>
        <span>
          Data condition
          <b>{conditionLabel}</b>
        </span>
      </div>
      {numerical.evidenceAttribution?.length ? (
        <div className="identification-section evidence-attribution-section">
          <div className="identification-heading">
            <span>Channel evidence attribution</span>
            <small>Share of local ROI precision</small>
          </div>
          <div className="evidence-attribution-list">
            {numerical.evidenceAttribution.map((channel) => {
              const shares = channel.uncertaintyShare;
              const sources = [
                { id: "observational", label: "Data", value: shares.observational },
                { id: "experiment", label: "Experiment", value: shares.experiment },
                { id: "benchmark", label: "Benchmark", value: shares.benchmark },
                { id: "regularization", label: "Regularization", value: shares.regularization },
              ] as const;
              return (
                <div key={channel.channel}>
                  <span>
                    <b>{cleanChannel(channel.channel)}</b>
                    <small>
                      Conditional data separation {Math.round(channel.conditionalDataShare * 100)}%
                    </small>
                  </span>
                  <div className="evidence-attribution-detail">
                    <div
                      className="evidence-attribution-bar"
                      aria-label={`${cleanChannel(channel.channel)} evidence attribution`}
                    >
                      {sources.map((source) => (
                        <i
                          key={source.id}
                          className={source.id}
                          style={{ width: `${Math.max(0, source.value) * 100}%` }}
                          title={`${source.label}: ${Math.round(source.value * 100)}%`}
                        />
                      ))}
                    </div>
                    <small className="evidence-attribution-values">
                      {sources.map((source) => `${source.label} ${Math.round(source.value * 100)}%`).join(" · ")}
                    </small>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="evidence-attribution-legend" aria-label="Evidence sources">
            <span className="observational">Data</span>
            <span className="experiment">Experiment</span>
            <span className="benchmark">Benchmark</span>
            <span className="regularization">Regularization</span>
          </div>
          <small className="identification-caption">
            Shares describe where local ROI precision comes from. Observational credit is interpreted with the conditional separation score; evidence quality, conflict, and decision dependence remain separate checks.
          </small>
        </div>
      ) : null}
      {numerical.clipping.applied ? (
        <div className={`clipping-disclosure ${numerical.clipping.material ? "warn" : "good"}`}>
          <div className="identification-heading">
            <span>Non-negative boundary</span>
            <small>{numerical.clipping.material ? "Structural review" : "Informational"}</small>
          </div>
          {numerical.clipping.channels.map((channel) => (
            <div className="clipping-row" key={channel.channel}>
              <span>
                <b>{cleanChannel(channel.channel)}</b>
                <small>β {channel.unconstrainedCoefficient.toPrecision(3)} → {channel.constrainedCoefficient.toPrecision(3)}</small>
              </span>
              <span>
                <b>{formatCompact(channel.contributionChange, true)}</b>
                <small>contribution change</small>
              </span>
              <span>
                <b>{channel.unconstrainedRoi.toFixed(2)}× → {channel.constrainedRoi.toFixed(2)}×</b>
                <small>{channel.roiIntervalMaterial ? "Interval materially affected" : "Interval impact small"}</small>
              </span>
            </div>
          ))}
          <small className="identification-caption">
            Fitted predictions moved by {(numerical.clipping.predictionShiftShare * 100).toFixed(1)}% of outcome variation. The coefficient is unchanged beyond the existing zero boundary; reported analytic intervals remain a local unconstrained Gaussian approximation.
          </small>
        </div>
      ) : null}
      <small className="numerical-method-note">
        QR solves the same penalized MAP objective without forming a matrix inverse. SVD is used only when the design is practically singular or extremely ill-conditioned.
      </small>
    </details>
  );
}

type GuideTopic = "adstock" | "saturation" | "regularization" | "baseline";

function AdstockMiniCurve({
  type,
  config,
}: {
  type: ModelConfig["adstockType"];
  config: ModelConfig;
}) {
  const values =
    type === "geometric"
      ? Array.from({ length: 12 }, (_, lag) => config.adstock ** lag)
      : Array.from({ length: 12 }, (_, lag) => {
          const time = lag + 0.5;
          const shape = config.weibullShape;
          const scale = config.weibullScale;
          const ratio = time / scale;
          return (
            (shape / scale) *
            ratio ** (shape - 1) *
            Math.exp(-(ratio ** shape))
          );
        });
  const maximum = Math.max(...values);

  return (
    <div
      className={`adstock-mini-curve ${type}`}
      role="img"
      aria-label={`${type === "geometric" ? "Geometric immediate decay" : "Weibull delayed peak"} example`}
    >
      {values.map((value, index) => (
        <i
          key={index}
          style={{ height: `${Math.max((value / maximum) * 100, 3)}%` }}
        />
      ))}
    </div>
  );
}

function ParameterBars({
  values,
  label,
  tone = "violet",
  reference,
  domain,
}: {
  values: number[];
  label: string;
  tone?: "violet" | "orange" | "mint";
  reference?: number;
  domain?: [number, number];
}) {
  const minimum = domain?.[0] ?? Math.min(...values);
  const maximum = domain?.[1] ?? Math.max(...values);
  const span = maximum - minimum || 1;
  return (
    <div
      className={`parameter-bars ${tone}`}
      role="img"
      aria-label={label}
      style={reference === undefined ? undefined : { backgroundPositionY: `${100 - reference}%` }}
    >
      {values.map((value, index) => (
        <i
          key={index}
          style={{ height: `${Math.max(((value - minimum) / span) * 88 + 8, 3)}%` }}
        />
      ))}
    </div>
  );
}

function GuideSlider({
  label,
  display,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  display: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="guide-slider">
      <span>{label}<b>{display}</b></span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function AdstockTopic({
  dataset,
  config,
  setConfig,
}: {
  dataset: Dataset;
  config: ModelConfig;
  setConfig: (config: ModelConfig) => void;
}) {
  const halfLife =
    config.adstock > 0 && config.adstock < 1
      ? Math.log(0.5) / Math.log(config.adstock)
      : 0;
  const peakPeriod =
    config.weibullShape > 1
      ? config.weibullScale *
        ((config.weibullShape - 1) / config.weibullShape) **
          (1 / config.weibullShape)
      : 0;

  return (
    <>
      <div className="assumption-options">
        <article className={config.adstockType === "geometric" ? "selected" : ""}>
          <div className="assumption-title">
            <div><span>Recommended baseline</span><h3>Geometric</h3></div>
            <b>1 parameter</b>
          </div>
          <AdstockMiniCurve type="geometric" config={config} />
          <p>Effect peaks immediately, then a constant share carries into every following period.</p>
          <dl>
            <div><dt>Use when</dt><dd>Response is prompt, history is limited, or interpretability matters most.</dd></div>
            <div><dt>Watch for</dt><dd>It cannot represent a delayed peak or changing decay rate.</dd></div>
          </dl>
          <button
            className={config.adstockType === "geometric" ? "button secondary full" : "button primary full"}
            onClick={() => setConfig({ ...config, adstockType: "geometric" })}
          >
            {config.adstockType === "geometric" ? "✓ Currently selected" : "Use geometric"}
          </button>
        </article>

        <article className={config.adstockType === "weibull" ? "selected" : ""}>
          <div className="assumption-title">
            <div><span>Flexible response</span><h3>Weibull PDF</h3></div>
            <b>2 parameters</b>
          </div>
          <AdstockMiniCurve type="weibull" config={config} />
          <p>Shape and scale allow the effect to build, peak later, and then decay at a changing rate.</p>
          <dl>
            <div><dt>Use when</dt><dd>Offline or considered purchases plausibly respond after several periods.</dd></div>
            <div><dt>Watch for</dt><dd>Extra flexibility needs more history and can weaken identification.</dd></div>
          </dl>
          <button
            className={config.adstockType === "weibull" ? "button secondary full" : "button primary full"}
            onClick={() => setConfig({ ...config, adstockType: "weibull" })}
          >
            {config.adstockType === "weibull" ? "✓ Currently selected" : "Use Weibull"}
          </button>
        </article>
      </div>

      <section className="parameter-live-card">
        <div>
          <span className="eyebrow">Current parameter</span>
          <h3>
            {config.adstockType === "geometric"
              ? `${Math.round(config.adstock * 100)}% carries into the next ${dataset.periodUnit}`
              : `Response peaks around ${dataset.periodUnit} ${peakPeriod.toFixed(1)}`}
          </h3>
          <p>
            {config.adstockType === "geometric"
              ? `A decay of ${config.adstock.toFixed(2)} means 1.00 of effect becomes ${config.adstock.toFixed(2)} next ${dataset.periodUnit} and ${(config.adstock ** 2).toFixed(2)} two ${periodLabel(dataset, 2)} later. The implied half-life is about ${halfLife.toFixed(1)} ${periodLabel(dataset, halfLife)}.`
              : `Shape ${config.weibullShape.toFixed(1)} controls whether response peaks immediately or later. Scale ${config.weibullScale.toFixed(1)} sets the timing in ${periodLabel(dataset, 2)}; larger values push the peak and tail outward.`}
          </p>
        </div>
        <div className="parameter-live-controls">
          {config.adstockType === "geometric" ? (
            <GuideSlider label="Decay θ" display={config.adstock.toFixed(2)} value={config.adstock} min={0.05} max={0.9} step={0.05} onChange={(adstock) => setConfig({ ...config, adstock })} />
          ) : (
            <>
              <GuideSlider label="Shape k" display={config.weibullShape.toFixed(1)} value={config.weibullShape} min={0.6} max={6} step={0.1} onChange={(weibullShape) => setConfig({ ...config, weibullShape })} />
              <GuideSlider label="Scale λ" display={`${config.weibullScale.toFixed(1)} ${periodLabel(dataset, 2, true)}`} value={config.weibullScale} min={1} max={12} step={0.5} onChange={(weibullScale) => setConfig({ ...config, weibullScale })} />
            </>
          )}
        </div>
      </section>

      <DecisionGuide />
    </>
  );
}

function SaturationTopic({
  config,
  setConfig,
}: {
  config: ModelConfig;
  setConfig: (config: ModelConfig) => void;
}) {
  const values = Array.from({ length: 24 }, (_, index) => {
    const spend = (index + 1) / 12;
    return spend ** config.saturation /
      (spend ** config.saturation + 1);
  });
  const description =
    config.saturation < 1
      ? "Strong early gains with a long, gradual tail."
      : config.saturation < 1.5
        ? "A mild S-curve: gradual takeoff, then diminishing returns."
        : "A stronger threshold effect followed by faster saturation.";

  return (
    <>
      <section className="parameter-explainer-card">
        <div className="parameter-copy">
          <span className="eyebrow">Diminishing returns</span>
          <h3>Hill shape α = {config.saturation.toFixed(1)}</h3>
          <p>{description} At the median transformed spend, response is always 50%; shape controls how sharply the curve bends around that point.</p>
          <div className="parameter-facts">
            <span><b>Below 1</b> Fast early response</span>
            <span><b>Around 1</b> Smooth saturation</span>
            <span><b>Above 1</b> Increasing threshold</span>
          </div>
        </div>
        <ParameterBars values={values} label={`Hill response curve with shape ${config.saturation.toFixed(1)}`} tone="mint" reference={50} />
        <GuideSlider label="Hill shape α" display={config.saturation.toFixed(1)} value={config.saturation} min={0.5} max={3} step={0.1} onChange={(saturation) => setConfig({ ...config, saturation })} />
      </section>
      <section className="parameter-caution">
        <b>What to validate</b>
        <p>Do not interpret a steep curve as proof of a spending threshold. Check that observed spend covers both sides of the bend and that ROI remains stable.</p>
      </section>
    </>
  );
}

function RegularizationTopic({
  config,
  setConfig,
}: {
  config: ModelConfig;
  setConfig: (config: ModelConfig) => void;
}) {
  const retention = 1 / (1 + config.ridge);
  const values = [1, 0.84, 0.68, 0.52, 0.37].map(
    (coefficient) => coefficient * retention,
  );
  return (
    <>
      <section className="parameter-explainer-card">
        <div className="parameter-copy">
          <span className="eyebrow">Stability versus flexibility</span>
          <h3>Ridge penalty λ = {config.ridge.toFixed(2)}</h3>
          <p>Ridge discourages large, unstable coefficients when channels move together. Higher values shrink attribution more strongly toward zero; lower values let the data move coefficients more freely.</p>
          <div className="parameter-facts">
            <span><b>Low λ</b> Less bias, more variance</span>
            <span><b>High λ</b> More stability, more bias</span>
          </div>
        </div>
        <ParameterBars values={values} label={`Illustrative coefficient shrinkage at ridge ${config.ridge.toFixed(2)}`} tone="orange" domain={[0, 1]} />
        <GuideSlider label="Ridge penalty λ" display={config.ridge.toFixed(2)} value={config.ridge} min={0.01} max={2} step={0.01} onChange={(ridge) => setConfig({ ...config, ridge })} />
      </section>
      <section className="parameter-caution">
        <b>Illustrative preview</b>
        <p>The bars show the direction of shrinkage, not exact fitted coefficients. Actual shrinkage depends on feature scale, signal strength, and media correlation.</p>
      </section>
    </>
  );
}

function BaselineTopic({
  dataset,
  config,
  setConfig,
}: {
  dataset: Dataset;
  config: ModelConfig;
  setConfig: (config: ModelConfig) => void;
}) {
  const seasonal = Array.from({ length: 36 }, (_, index) =>
    Array.from({ length: config.fourierOrder }, (_, order) =>
      Math.sin((2 * Math.PI * (order + 1) * index) / 36) / (order + 1),
    ).reduce((sum, value) => sum + value, 0),
  );
  const cycle = Array.from({ length: 36 }, (_, index) =>
    Math.sin((2 * Math.PI * index) / config.cyclePeriod),
  );
  const cyclePeriods = cyclePeriodsForDataset(dataset);

  return (
    <>
      <div className="baseline-guide-grid">
        <section className="parameter-explainer-card">
          <div className="parameter-copy">
            <span className="eyebrow">Annual seasonality</span>
            <h3>{config.fourierOrder} Fourier pair{config.fourierOrder === 1 ? "" : "s"}</h3>
            <p>Each pair adds one sine and one cosine at a finer annual frequency. More pairs capture richer seasonal shapes, but can also absorb media-driven variation.</p>
          </div>
          <ParameterBars values={seasonal} label={`Seasonality basis with Fourier order ${config.fourierOrder}`} />
          <div className="guide-choice-row">
            {[1, 2, 3].map((order) => (
              <button key={order} className={config.fourierOrder === order ? "active" : ""} onClick={() => setConfig({ ...config, fourierOrder: order })}>{order} pair{order === 1 ? "" : "s"}</button>
            ))}
          </div>
        </section>
        <section className="parameter-explainer-card">
          <div className="parameter-copy">
            <span className="eyebrow">Low-frequency cycle</span>
            <h3>{config.cyclePeriod}-{dataset.periodUnit} period</h3>
            <p>The cycle repeats every {config.cyclePeriod} {periodLabel(dataset, config.cyclePeriod)}. Shorter periods allow faster recurring movement; longer periods reserve the baseline for slower business rhythms.</p>
          </div>
          <ParameterBars values={cycle} label={`Repeating baseline cycle every ${config.cyclePeriod} ${periodLabel(dataset, config.cyclePeriod)}`} tone="orange" />
          <div className="guide-choice-row">
            {cyclePeriods.map((period) => (
              <button key={period} className={config.cyclePeriod === period ? "active" : ""} onClick={() => setConfig({ ...config, cyclePeriod: period })}>{period} {periodLabel(dataset, period)}</button>
            ))}
          </div>
        </section>
      </div>
      <section className="parameter-caution">
        <b>Guardrail</b>
        <p>Increase baseline flexibility only when residual seasonality remains. An overly flexible baseline can explain away real media contribution.</p>
      </section>
    </>
  );
}

function DecisionGuide() {
  return (
    <section className="decision-guide">
      <span className="eyebrow">Decision rule</span>
      <h3>Prefer the simpler model unless flexibility earns its place.</h3>
      <div>
        <p><b>1</b><span><strong>Validate out of sample</strong><small>Compare predictive error, not in-sample fit alone.</small></span></p>
        <p><b>2</b><span><strong>Check plausibility</strong><small>The response shape should match how the channel works.</small></span></p>
        <p><b>3</b><span><strong>Check stability</strong><small>ROI should remain credible across reasonable settings.</small></span></p>
      </div>
    </section>
  );
}

function ModelParameterGuide({
  dataset,
  topic,
  setTopic,
  config,
  setConfig,
  onClose,
}: {
  dataset: Dataset;
  topic: GuideTopic;
  setTopic: (topic: GuideTopic) => void;
  config: ModelConfig;
  setConfig: (config: ModelConfig) => void;
  onClose: () => void;
}) {
  const headings: Record<GuideTopic, { title: string; description: string }> = {
    adstock: {
      title: "How should memory decay?",
      description: "Carryover controls how past media remains active in the current period.",
    },
    saturation: {
      title: "How quickly do returns flatten?",
      description: "The Hill curve converts media pressure into a bounded, diminishing response.",
    },
    regularization: {
      title: "How much should estimates shrink?",
      description: "Regularization trades a little bias for more stable attribution.",
    },
    baseline: {
      title: "How flexible should baseline be?",
      description: "Fourier terms and cycles capture recurring demand that media should not claim.",
    },
  };

  return (
    <DrawerShell
      labelledBy="parameter-guide-title"
      onClose={onClose}
    >
        <div className="assumptions-header">
          <div>
            <span className="kicker">Modeling assumptions</span>
            <h2 id="parameter-guide-title">{headings[topic].title}</h2>
            <p>{headings[topic].description}</p>
          </div>
          <button className="drawer-close" onClick={onClose} aria-label="Close parameter guide">×</button>
        </div>
        <nav className="guide-tabs" aria-label="Parameter guide topics">
          {([
            ["adstock", "Carryover"],
            ["saturation", "Saturation"],
            ["regularization", "Ridge"],
            ["baseline", "Baseline"],
          ] as [GuideTopic, string][]).map(([id, label]) => (
            <button key={id} className={topic === id ? "active" : ""} onClick={() => setTopic(id)}>{label}</button>
          ))}
        </nav>

        {topic === "adstock" && <AdstockTopic dataset={dataset} config={config} setConfig={setConfig} />}
        {topic === "saturation" && <SaturationTopic config={config} setConfig={setConfig} />}
        {topic === "regularization" && <RegularizationTopic config={config} setConfig={setConfig} />}
        {topic === "baseline" && <BaselineTopic dataset={dataset} config={config} setConfig={setConfig} />}
    </DrawerShell>
  );
}

function ParameterHelpButton({
  label,
  onClick,
}: {
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="parameter-help"
      onClick={(event) => {
        event.preventDefault();
        event.stopPropagation();
        onClick();
      }}
      aria-label={`Explain ${label}`}
    >
      ?
    </button>
  );
}

interface PromotedAgenticSpecification {
  run: AgenticCandidateRun;
  datasetHash: string;
  experiments: Experiment[];
  industryPriorChannels: string[];
  override?: AgenticForcePromotionAudit;
}

interface WorkspaceCheckpointPayload {
  demoMode: boolean;
  view: View;
  dataset: Dataset;
  validation: ValidationResult;
  eda: EdaResult | null;
  experiments: Experiment[];
  guardrailMode: BenchmarkGuardrailMode;
  industryPriorChannels: string[];
  benchmarkScreeningRois: Record<string, number>;
  config: ModelConfig;
  advancedConfig: AdvancedModelConfig;
  models: Partial<Record<"frequentist" | "bayesian", ModelResult>>;
  advancedResult?: ModelResult;
  validationResults: Partial<
    Record<ValidationModelKind, ModelValidationResult>
  >;
  anchorIndependenceConfirmed: boolean;
  agenticContract: AgenticSearchContract;
  agenticRuns: AgenticCandidateRun[];
  agenticStatus: AgenticWorkspaceStatus;
  agenticStopReason?: string;
  promotedAgenticSpecification?: PromotedAgenticSpecification;
  samplingContract: SamplingContract;
  samplingResult?: SamplingResult;
  samplingHistory: SamplingResult[];
  samplingStatus: JobStatus;
  budgetContract: BudgetOptimizationContract;
  budgetResult?: BudgetOptimizationResult;
  budgetStatus: JobStatus;
}

interface SpecificationParameter {
  label: string;
  value: string;
}

interface SpecificationGroup {
  title: string;
  description: string;
  parameters: SpecificationParameter[];
  status?: string;
  countsAsParameters?: boolean;
}

function agenticFamilyLabel(kind: ValidationModelKind): string {
  return kind === "frequentist"
    ? "Frequentist"
    : kind === "bayesian"
      ? "Bayesian"
      : "Advanced";
}

function experimentSignature(experiments: Experiment[]): string {
  return JSON.stringify(
    [...experiments].sort((left, right) =>
      [
        left.channel,
        left.startDate,
        left.endDate,
        left.source,
      ].join("|").localeCompare(
        [
          right.channel,
          right.startDate,
          right.endDate,
          right.source,
        ].join("|"),
      ),
    ),
  );
}

function channelSignature(channels: string[]): string {
  return [...channels]
    .map((channel) => channel.toLowerCase())
    .sort()
    .join("|");
}

function configurationsMatch<T>(left: T, right: T): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function agenticSpecificationGroups(
  run: AgenticCandidateRun,
): SpecificationGroup[] {
  const { config, advancedConfig, family } = run.spec;
  const cycleUnit = config.cyclePeriod <= 12 ? "months" : "weeks";
  const groups: SpecificationGroup[] = [
    {
      title: "Media response",
      description: "How spend carries over and reaches diminishing returns.",
      parameters: [
        {
          label: "Estimator family",
          value: agenticFamilyLabel(family),
        },
        {
          label: "Adstock family",
          value:
            config.adstockType === "geometric"
              ? "Geometric"
              : "Weibull PDF",
        },
        ...(config.adstockType === "geometric"
          ? [
              {
                label: "Geometric decay θ",
                value: config.adstock.toFixed(2),
              },
            ]
          : [
              {
                label: "Weibull shape k",
                value: config.weibullShape.toFixed(2),
              },
              {
                label: "Weibull scale λ",
                value: `${config.weibullScale.toFixed(2)} ${cycleUnit}`,
              },
            ]),
        {
          label: "Hill shape",
          value: config.saturation.toFixed(2),
        },
      ],
    },
    {
      title: "Baseline & regularization",
      description: "How the model separates recurring demand from media.",
      parameters: [
        {
          label: "Ridge penalty",
          value: config.ridge.toFixed(2),
        },
        {
          label: "Fourier order",
          value: `${config.fourierOrder} pair${config.fourierOrder === 1 ? "" : "s"}`,
        },
        {
          label: "Cycle period",
          value: `${config.cyclePeriod} ${cycleUnit}`,
        },
      ],
    },
  ];

  const channelResponses = Object.entries(config.channelResponses ?? {});
  if (channelResponses.length) {
    groups.push({
      title: "Channel-specific response",
      description:
        "The V5 search lets each media channel carry over and saturate independently.",
      parameters: channelResponses.flatMap(([channel, response]) => {
        const adstockType = response.adstockType ?? config.adstockType;
        return [
          {
            label: `${cleanChannel(channel)} response`,
            value:
              adstockType === "geometric"
                ? `Geometric θ ${(response.adstock ?? config.adstock).toFixed(2)}`
                : `Weibull k ${(response.weibullShape ?? config.weibullShape).toFixed(1)} · λ ${(response.weibullScale ?? config.weibullScale).toFixed(1)}`,
          },
          {
            label: `${cleanChannel(channel)} saturation`,
            value: `Hill ${(response.saturation ?? config.saturation).toFixed(1)} · Q${Math.round((response.halfSaturationQuantile ?? 0.5) * 100)} · ${response.kernelNormalization ?? "peak"} normalized`,
          },
        ];
      }),
    });
  }

  if (family === "advanced") {
    groups.push(
      {
        title: "Advanced structure",
        description: "Optional flexibility used by this candidate.",
        parameters: [
          {
            label: "Media coefficients",
            value: advancedConfig.timeVarying
              ? "Time-varying · KTR-style"
              : "Static",
          },
          ...(advancedConfig.timeVarying
            ? [
                {
                  label: "Kernel knots",
                  value: String(advancedConfig.kernelKnots),
                },
                {
                  label: "Kernel bandwidth",
                  value: advancedConfig.kernelBandwidth.toFixed(2),
                },
              ]
            : []),
          {
            label: "Planning intensity factor",
            value: advancedConfig.planningIntensity ? "Included" : "Off",
          },
        ],
      },
      {
        title: "Calibration & distributions",
        description: "Where external evidence enters and how uncertainty is shaped.",
        parameters: [
          {
            label: "Experiment calibration",
            value:
              advancedConfig.calibrationMode === "prior"
                ? "Informative coefficient prior"
                : "Noisy likelihood measurement",
          },
          {
            label: "Coefficient prior",
            value:
              advancedConfig.priorDistribution === "log-normal"
                ? "Log-normal"
                : "Half-normal",
          },
          {
            label: "Outcome likelihood",
            value:
              advancedConfig.likelihoodDistribution === "student-t"
                ? "Student-t"
                : advancedConfig.likelihoodDistribution === "log-normal"
                  ? "Log-normal"
                  : "Gaussian",
          },
          ...(advancedConfig.likelihoodDistribution === "student-t"
            ? [
                {
                  label: "Student-t degrees of freedom",
                  value: advancedConfig.studentTDegreesFreedom.toFixed(1),
                },
              ]
            : []),
          {
            label: "Inference engine",
            value: "Analytic MAP / Laplace",
          },
        ],
      },
    );
  } else {
    groups.push(
      {
        title: "Estimation contract",
        description:
          family === "bayesian"
            ? "The fast Bayesian estimator used for this search."
            : "The reference estimator used for this search.",
        parameters: [
          {
            label: "Experiment calibration",
            value:
              family === "bayesian"
                ? "Informative coefficient prior"
                : "Not applied",
          },
          {
            label: "Uncertainty",
            value:
              family === "bayesian"
                ? "Analytic Gaussian posterior"
                : "Analytic sampling interval",
          },
          {
            label: "Inference engine",
            value:
              family === "bayesian"
                ? "MAP / Laplace"
                : "Penalized least squares",
          },
        ],
      },
      {
        title: "Advanced search dimensions",
        status: "Not applied",
        countsAsParameters: false,
        description: `These dimensions were searched in competing Advanced candidates. This ${agenticFamilyLabel(family)} candidate has no winning values for them because they did not enter its fit.`,
        parameters: [
          {
            label: "Time-varying coefficients",
            value: "Advanced candidates · on / off",
          },
          {
            label: "Kernel smoothness",
            value: "3–10 knots · 0.08–0.35 bandwidth",
          },
          {
            label: "Planning intensity factor",
            value: "Advanced candidates · on / off",
          },
          {
            label: "Prior distribution",
            value: "Half-normal / log-normal",
          },
          {
            label: "Outcome likelihood",
            value: "Gaussian / Student-t / log-normal",
          },
          {
            label: "Student-t degrees of freedom",
            value: "3–30 · when Student-t is active",
          },
        ],
      },
    );
  }

  return groups;
}

function specificationParameterCount(run: AgenticCandidateRun): number {
  return agenticSpecificationGroups(run).reduce(
    (total, group) =>
      total +
      (group.countsAsParameters === false ? 0 : group.parameters.length),
    0,
  );
}

function SpecificationInspector({
  run,
  experiments,
  industryPriorChannels,
  onClose,
}: {
  run: AgenticCandidateRun;
  experiments: Experiment[];
  industryPriorChannels: string[];
  onClose: () => void;
}) {
  const groups = agenticSpecificationGroups(run);
  const boundaryParameters = agenticBoundaryParameters(run.spec);
  const eligible = passesAgenticEligibility(run);

  return (
    <DrawerShell
      className="specification-drawer"
      labelledBy="specification-inspector-title"
      onClose={onClose}
    >
        <div className="assumptions-header specification-header">
          <div>
            <span className="kicker">
              {run.spec.id} · {run.spec.searchPhase === "adaptive"
                ? "Family-aware adaptive proposal"
                : run.spec.searchPhase === "advanced-challenge"
                  ? "Paired Advanced challenge"
                  : run.spec.searchPhase === "local-challenge"
                    ? `Local challenge of ${run.spec.challengeOf ?? "champion"}`
                  : run.spec.searchPhase === "rescue"
                    ? `Evidence rescue of ${run.spec.rescueOf ?? "candidate"}`
                    : run.spec.searchPhase === "response-coverage"
                      ? "Channel-response covering array"
                      : "Space-filling seed"}
            </span>
            <h2 id="specification-inspector-title">{run.spec.label}</h2>
            <p>{run.spec.hypothesis}</p>
          </div>
          <button className="drawer-close" onClick={onClose} aria-label="Close specification inspector">×</button>
        </div>

        <section className="specification-summary-strip">
          <div>
            <span>Flux score</span>
            <strong>{run.validation?.finalScore?.toFixed(1) ?? "—"}</strong>
          </div>
          <div>
            <span>Eligibility</span>
            <strong className={eligible ? "pass" : "review"}>
              {run.validation ? (eligible ? "Pass" : "Review") : "Pending"}
            </strong>
          </div>
          <div>
            <span>Active parameters</span>
            <strong>{specificationParameterCount(run)}</strong>
          </div>
          <div>
            <span>Search source</span>
            <strong>
              {run.spec.proposal.method === "tpe"
                ? `Start ${run.spec.restart}`
                : run.spec.proposal.method === "response-covering-array"
                  ? `Response grid · Start ${run.spec.restart}`
                : run.spec.proposal.method === "covering-array"
                  ? "Paired"
                  : run.spec.proposal.method === "local-challenge"
                    ? `Local · Start ${run.spec.restart}`
                  : run.spec.proposal.method === "evidence-rescue"
                    ? "Rescue"
                    : "Seed"}
            </strong>
          </div>
        </section>

        <div className="specification-groups">
          {groups.map((group) => (
            <section
              key={group.title}
              className={`specification-group ${group.status ? "inactive" : ""}`}
            >
              <div>
                <div className="specification-group-heading">
                  <span className="eyebrow">{group.title}</span>
                  {group.status && (
                    <span className="specification-group-status">
                      {group.status}
                    </span>
                  )}
                </div>
                <p>{group.description}</p>
              </div>
              <dl>
                {group.parameters.map((parameter) => (
                  <div key={parameter.label}>
                    <dt>{parameter.label}</dt>
                    <dd>{parameter.value}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>

        <section className="specification-evidence">
          <div className="card-heading">
            <div>
              <span className="eyebrow">Locked evidence</span>
              <h3>What the search could not change</h3>
            </div>
            <span className="agentic-lock">◆ Immutable</span>
          </div>
          <div className="specification-evidence-grid">
            <div>
              <span>Experiments</span>
              <b>{experiments.length || "None"}</b>
              <small>
                {experiments.length
                  ? experiments
                      .map((experiment) => cleanChannel(experiment.channel))
                      .join(", ")
                  : "No experimental anchors"}
              </small>
            </div>
            <div>
              <span>Industry guardrails</span>
              <b>{industryPriorChannels.length || "None"}</b>
              <small>
                {industryPriorChannels.length
                  ? industryPriorChannels.map(cleanChannel).join(", ")
                  : "No benchmark priors active"}
              </small>
            </div>
            {run.validation?.evidenceCoherence && (
              <div>
                <span>Evidence coherence</span>
                <b>
                  {run.validation.evidenceCoherence.status === "pass"
                    ? "Aligned"
                    : run.validation.evidenceCoherence.status === "fail"
                      ? "Blocking"
                      : "Review"}
                </b>
                <small>{run.validation.evidenceCoherence.summary}</small>
              </div>
            )}
            <div>
              <span>Proposal evidence</span>
              <b>
                {run.spec.proposal.expectedScore === undefined
                  ? "Coverage seed"
                  : `${run.spec.proposal.expectedScore.toFixed(1)} expected`}
              </b>
              <small>{run.spec.proposal.reason}</small>
            </div>
            <div>
              <span>Declared bounds</span>
              <b>
                {boundaryParameters.length
                  ? `${boundaryParameters.length} active`
                  : "Interior"}
              </b>
              <small>
                {boundaryParameters.length
                  ? boundaryParameters.join(", ")
                  : "No parameter reached a search limit"}
              </small>
            </div>
          </div>
        </section>
        {run.validation?.evidenceCoherence && (
          <details className="specification-channel-evidence">
            <summary>
              <span>
                <b>Channel evidence receipt</b>
                <small>
                  Material channels · benchmark use never adds score
                </small>
              </span>
              <i className={run.validation.evidenceCoherence.status}>
                {run.validation.evidenceCoherence.blockingChannels.length
                  ? `${run.validation.evidenceCoherence.blockingChannels.length} blocking`
                  : "Aligned"}
              </i>
            </summary>
            <div className="specification-channel-evidence-table">
              <div className="head">
                <span>Channel</span>
                <span>Model ROI</span>
                <span>Evidence</span>
                <span>Assessment</span>
              </div>
              {run.validation.evidenceCoherence.channels
                .filter((channel) => channel.material)
                .sort((left, right) => right.spendShare - left.spendShare)
                .map((channel) => (
                  <div key={channel.channel}>
                    <strong>
                      {cleanChannel(channel.channel)}
                      <small>{Math.round(channel.spendShare * 100)}% spend</small>
                    </strong>
                    <span>
                      {channel.roi.toFixed(2)}×
                      <small>
                        {channel.roiLow.toFixed(2)}–{channel.roiHigh.toFixed(2)}× · {channel.comparisonLabel}
                      </small>
                    </span>
                    <span>
                      {channel.evidenceCenter === undefined
                        ? "Unanchored"
                        : `${channel.evidenceCenter.toFixed(2)}×`}
                      <small>{channel.evidenceLabel}</small>
                    </span>
                    <span
                      className={`evidence-status ${channel.blocking ? "fail" : channel.status === "aligned" ? "pass" : "review"}`}
                      title={channel.detail}
                    >
                      {channel.status.replaceAll("-", " ")}
                    </span>
                  </div>
                ))}
            </div>
          </details>
        )}
    </DrawerShell>
  );
}

function PromotedSpecificationBanner({
  destination,
  promotion,
  datasetHash,
  config,
  advancedConfig,
  result,
  experiments,
  industryPriorChannels,
  onInspect,
}: {
  destination: "models" | "advanced";
  promotion: PromotedAgenticSpecification;
  datasetHash: string;
  config: ModelConfig;
  advancedConfig: AdvancedModelConfig;
  result?: ModelResult;
  experiments: Experiment[];
  industryPriorChannels: string[];
  onInspect: () => void;
}) {
  const promotedFamily = promotion.run.spec.family;
  const applicableIndustryChannels =
    promotedFamily === "frequentist" ? [] : industryPriorChannels;
  const contextMatches =
    promotion.datasetHash === datasetHash &&
    experimentSignature(promotion.experiments) ===
      experimentSignature(experiments) &&
    channelSignature(promotion.industryPriorChannels) ===
      channelSignature(applicableIndustryChannels);
  const baseMatches = configurationsMatch(
    promotion.run.spec.config,
    config,
  );
  const advancedMatches =
    promotedFamily !== "advanced" ||
    configurationsMatch(
      promotion.run.spec.advancedConfig,
      advancedConfig,
    );
  const artifactMatches =
    result?.fingerprint === promotion.run.model?.fingerprint;
  const inherited =
    destination === "advanced" &&
    promotedFamily !== "advanced" &&
    baseMatches &&
    contextMatches;
  const exact =
    !inherited &&
    baseMatches &&
    advancedMatches &&
    contextMatches &&
    artifactMatches;
  const state = exact ? "exact" : inherited ? "inherited" : "modified";
  const parameterCount = specificationParameterCount(promotion.run);
  const forced = Boolean(promotion.override);
  const failedEvidenceCount =
    (promotion.override?.failedGates.length ?? 0) +
    (promotion.override?.roiGuardrailViolations.length ?? 0);
  const title =
    forced && state === "exact"
      ? "Forced candidate configuration loaded"
      : forced && state === "inherited"
        ? "Forced base settings inherited"
        : state === "exact"
      ? "Exact winning configuration loaded"
      : state === "inherited"
        ? "Winning base settings inherited"
        : contextMatches
          ? "Modified since promotion"
          : "Evidence changed since promotion";
  const detail =
    forced && (state === "exact" || state === "inherited")
      ? `${promotion.run.spec.id} was manually promoted with ${failedEvidenceCount} unresolved evidence item${failedEvidenceCount === 1 ? "" : "s"} preserved. Its score and eligibility were not changed.`
      : state === "exact"
      ? `All ${parameterCount} applicable settings and the fitted artifact match ${promotion.run.spec.id}.`
      : state === "inherited"
        ? `${promotion.run.spec.id} was a ${agenticFamilyLabel(promotedFamily)} winner. Its base parameters are loaded; Advanced assumptions were not part of that fit.`
        : "The promoted receipt is preserved for comparison, but the active controls, fitted artifact, or locked evidence no longer match it.";

  return (
    <section className={`promotion-receipt ${state}${forced ? " forced" : ""}`}>
      <span className="promotion-receipt-mark">
        {forced && state !== "modified"
          ? "!"
          : state === "exact"
            ? "✓"
            : state === "inherited"
              ? "↗"
              : "△"}
      </span>
      <div className="promotion-receipt-copy">
        <span className="eyebrow">
          {promotion.run.spec.id} {forced ? "force promoted" : "promoted"} from Agentic
        </span>
        <h2>{title}</h2>
        <p>{detail}</p>
      </div>
      <div className="promotion-receipt-meta">
        <span>
          {promotion.run.validation?.finalScore?.toFixed(1) ?? "—"} score
        </span>
        <span>{agenticFamilyLabel(promotedFamily)}</span>
        {forced && <span>Forced override</span>}
      </div>
      <button className="text-button" onClick={onInspect}>
        View all {parameterCount} parameters →
      </button>
    </section>
  );
}

function ModelsView({
  dataset,
  config,
  setConfig,
  results,
  statuses,
  promotion,
  experiments,
  industryPriorChannels,
  onRun,
}: {
  dataset: Dataset;
  config: ModelConfig;
  setConfig: (config: ModelConfig) => void;
  results: Partial<Record<"frequentist" | "bayesian", ModelResult>>;
  statuses: Record<"frequentist" | "bayesian", JobStatus>;
  promotion?: PromotedAgenticSpecification;
  experiments: Experiment[];
  industryPriorChannels: string[];
  onRun: (kind: "frequentist" | "bayesian") => void;
}) {
  const [guideTopic, setGuideTopic] = useState<GuideTopic | null>(null);
  const [showPromotedSpecification, setShowPromotedSpecification] =
    useState(false);
  const available = results.bayesian ?? results.frequentist;
  const dates = dataset.rows.map((row) => String(row[dataset.dateColumn]));
  const cyclePeriods = cyclePeriodsForDataset(dataset);

  return (
    <div className="view">
      <section className="page-heading compact-heading">
        <div><span className="kicker">Comparable estimators</span><h1>Model studio</h1><p>One feature pipeline, two estimators, reproducible fingerprints.</p></div>
        {available && (
          <button className="button secondary" onClick={() => {
            const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = "flux-mmm-model-results.json";
            link.click();
            URL.revokeObjectURL(link.href);
          }}>↓ Export results</button>
        )}
      </section>
      {promotion && promotion.run.spec.family !== "advanced" && (
        <PromotedSpecificationBanner
          destination="models"
          promotion={promotion}
          datasetHash={dataset.hash}
          config={config}
          advancedConfig={promotion.run.spec.advancedConfig}
          result={results[promotion.run.spec.family]}
          experiments={experiments}
          industryPriorChannels={industryPriorChannels}
          onInspect={() => setShowPromotedSpecification(true)}
        />
      )}
      <section className="model-layout">
        <div className="model-comparison">
          <ModelCard kind="frequentist" dates={dates} result={results.frequentist} status={statuses.frequentist} onRun={() => onRun("frequentist")} />
          <ModelCard kind="bayesian" dates={dates} result={results.bayesian} status={statuses.bayesian} onRun={() => onRun("bayesian")} />
        </div>
        <aside className="card config-panel">
          <div className="card-heading"><div><span className="eyebrow">Shared pipeline</span><h2>Specification</h2></div><span className="version-tag">V2</span></div>
          <div className="adstock-control">
            <div className="control-label">
              <span>Adstock family</span>
              <button onClick={() => setGuideTopic("adstock")}>Assumptions guide ↗</button>
            </div>
            <div className="segmented-control" aria-label="Adstock family">
              <button
                className={config.adstockType === "geometric" ? "active" : ""}
                onClick={() => setConfig({ ...config, adstockType: "geometric" })}
              >
                Geometric
              </button>
              <button
                className={config.adstockType === "weibull" ? "active" : ""}
                onClick={() => setConfig({ ...config, adstockType: "weibull" })}
              >
                Weibull
              </button>
            </div>
            <p>
              {config.adstockType === "geometric"
                ? "Immediate peak · fixed decay · easiest to explain"
                : "Flexible peak · changing decay · more data needed"}
            </p>
          </div>
          {config.adstockType === "geometric" ? (
            <label>
              <span><span className="parameter-name">Decay θ <ParameterHelpButton label="geometric decay" onClick={() => setGuideTopic("adstock")} /></span><b>{config.adstock.toFixed(2)}</b></span>
              <input type="range" min="0.05" max="0.9" step="0.05" value={config.adstock} onChange={(event) => setConfig({ ...config, adstock: Number(event.target.value) })} />
            </label>
          ) : (
            <div className="weibull-controls">
              <label>
                <span><span className="parameter-name">Shape k <ParameterHelpButton label="Weibull shape" onClick={() => setGuideTopic("adstock")} /></span><b>{config.weibullShape.toFixed(1)}</b></span>
                <input type="range" min="0.6" max="6" step="0.1" value={config.weibullShape} onChange={(event) => setConfig({ ...config, weibullShape: Number(event.target.value) })} />
              </label>
              <label>
                <span><span className="parameter-name">Scale λ <ParameterHelpButton label="Weibull scale" onClick={() => setGuideTopic("adstock")} /></span><b>{config.weibullScale.toFixed(1)} {periodLabel(dataset, 2, true)}</b></span>
                <input type="range" min="1" max="12" step="0.5" value={config.weibullScale} onChange={(event) => setConfig({ ...config, weibullScale: Number(event.target.value) })} />
              </label>
            </div>
          )}
          <label>
            <span><span className="parameter-name">Hill shape <ParameterHelpButton label="Hill shape" onClick={() => setGuideTopic("saturation")} /></span><b>{config.saturation.toFixed(1)}</b></span>
            <input type="range" min="0.5" max="3" step="0.1" value={config.saturation} onChange={(event) => setConfig({ ...config, saturation: Number(event.target.value) })} />
          </label>
          <label>
            <span><span className="parameter-name">Ridge penalty <ParameterHelpButton label="ridge penalty" onClick={() => setGuideTopic("regularization")} /></span><b>{config.ridge.toFixed(2)}</b></span>
            <input type="range" min="0.01" max="2" step="0.01" value={config.ridge} onChange={(event) => setConfig({ ...config, ridge: Number(event.target.value) })} />
          </label>
          <div className="select-group">
            <label><span><span className="parameter-name">Fourier order <ParameterHelpButton label="Fourier order" onClick={() => setGuideTopic("baseline")} /></span></span><select value={config.fourierOrder} onChange={(event) => setConfig({ ...config, fourierOrder: Number(event.target.value) })}><option value="1">1 pair</option><option value="2">2 pairs</option><option value="3">3 pairs</option></select></label>
            <label><span><span className="parameter-name">Cycle period <ParameterHelpButton label="cycle period" onClick={() => setGuideTopic("baseline")} /></span></span><select value={config.cyclePeriod} onChange={(event) => setConfig({ ...config, cyclePeriod: Number(event.target.value) })}>{cyclePeriods.map((period) => <option key={period} value={period}>{period} {periodLabel(dataset, period)}</option>)}</select></label>
          </div>
          <div className="spec-list">
            <div><span>✓</span><p><b>Annual seasonality</b><small>{dataset.periodsPerYear.toFixed(dataset.modelCadence === "monthly" ? 0 : 2)}-{dataset.periodUnit} Fourier basis</small></p></div>
            <div><span>✓</span><p><b>Low-frequency cycle</b><small>Deterministic sine and cosine</small></p></div>
            <div><span>✓</span><p><b>Media transforms</b><small>{config.adstockType === "geometric" ? "Geometric" : "Weibull PDF"} adstock + Hill</small></p></div>
            <div><span>⌁</span><p><b>Cache key</b><small>Data + experiments + config + code</small></p></div>
          </div>
        </aside>
      </section>
      {available && (
        <section className="card roi-card">
          <div className="card-heading"><div><span className="eyebrow">Marginal return</span><h2>Channel ROI estimates</h2></div><span className="subtle">{available.kind === "bayesian" ? "95% posterior interval" : "95% analytic interval"}</span></div>
          <div className="roi-table">
            <div className="roi-row roi-head"><span>Channel</span><span>Contribution</span><span>Share</span><span>ROI interval</span><span>Point ROI</span></div>
            {available.channels.map((channel, index) => (
              <div className="roi-row" key={channel.channel}>
                <strong><i style={{ background: CHANNEL_COLORS[index % CHANNEL_COLORS.length] }} />{cleanChannel(channel.channel)}</strong>
                <span>{formatCompact(channel.contribution, true)}</span>
                <span>{(channel.contributionShare * 100).toFixed(1)}%</span>
                <span className="interval"><i style={{ left: `${Math.min(channel.roiLow / Math.max(channel.roiHigh, 1) * 70, 70)}%`, width: `${Math.max((channel.roiHigh - channel.roiLow) / Math.max(channel.roiHigh, 1) * 65, 8)}%` }} /><em>{channel.roiLow.toFixed(2)}–{channel.roiHigh.toFixed(2)}</em></span>
                <b>{channel.roi.toFixed(2)}×</b>
              </div>
            ))}
          </div>
        </section>
      )}
      {guideTopic && (
        <ModelParameterGuide
          dataset={dataset}
          topic={guideTopic}
          setTopic={setGuideTopic}
          config={config}
          setConfig={setConfig}
          onClose={() => setGuideTopic(null)}
        />
      )}
      {showPromotedSpecification && promotion && (
        <SpecificationInspector
          run={promotion.run}
          experiments={promotion.experiments}
          industryPriorChannels={promotion.industryPriorChannels}
          onClose={() => setShowPromotedSpecification(false)}
        />
      )}
    </div>
  );
}

type AdvancedGuideTopic =
  | "dynamic"
  | "planning"
  | "calibration"
  | "distributions";

type DistributionPreset = "balanced" | "robust" | "multiplicative";

function distributionPreset(
  config: AdvancedModelConfig,
): DistributionPreset | "custom" {
  if (
    config.priorDistribution === "log-normal" &&
    config.likelihoodDistribution === "gaussian"
  ) {
    return "balanced";
  }
  if (
    config.priorDistribution === "half-normal" &&
    config.likelihoodDistribution === "student-t"
  ) {
    return "robust";
  }
  if (
    config.priorDistribution === "log-normal" &&
    config.likelihoodDistribution === "log-normal"
  ) {
    return "multiplicative";
  }
  return "custom";
}

function applyDistributionPreset(
  config: AdvancedModelConfig,
  preset: DistributionPreset,
): AdvancedModelConfig {
  if (preset === "robust") {
    return {
      ...config,
      priorDistribution: "half-normal",
      likelihoodDistribution: "student-t",
      studentTDegreesFreedom: 4,
    };
  }
  if (preset === "multiplicative") {
    return {
      ...config,
      priorDistribution: "log-normal",
      likelihoodDistribution: "log-normal",
    };
  }
  return {
    ...config,
    priorDistribution: "log-normal",
    likelihoodDistribution: "gaussian",
  };
}

function AdvancedToggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      className={`advanced-toggle ${checked ? "on" : ""}`}
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
    >
      <i />
    </button>
  );
}

function AdvancedAssumptionsGuide({
  dataset,
  topic,
  setTopic,
  config,
  setConfig,
  onClose,
}: {
  dataset: Dataset;
  topic: AdvancedGuideTopic;
  setTopic: (topic: AdvancedGuideTopic) => void;
  config: AdvancedModelConfig;
  setConfig: (config: AdvancedModelConfig) => void;
  onClose: () => void;
}) {
  const headings: Record<
    AdvancedGuideTopic,
    { kicker: string; title: string; summary: string }
  > = {
    dynamic: {
      kicker: "Dynamic effectiveness",
      title: "Let media effectiveness evolve smoothly",
      summary:
        "A compact Gaussian-kernel basis turns one channel coefficient into a smooth path through time.",
    },
    planning: {
      kicker: "Shared planning pressure",
      title: "Represent coordinated media intensity",
      summary:
        "A latent factor summarizes the common movement across channel plans without adding another uploaded variable.",
    },
    calibration: {
      kicker: "Experimental evidence",
      title: "Choose where lift evidence enters",
      summary:
        "Prior and likelihood calibration answer the same business need through different statistical contracts.",
    },
    distributions: {
      kicker: "Probability model",
      title: "Match distributions to the data-generating story",
      summary:
        "Choose a positive media prior and an outcome likelihood based on residual shape—not whichever specification reports the highest fit.",
    },
  };

  return (
    <DrawerShell
      className="advanced-guide"
      labelledBy="advanced-guide-title"
      onClose={onClose}
    >
        <div className="assumptions-header">
          <div>
            <span className="kicker">{headings[topic].kicker}</span>
            <h2 id="advanced-guide-title">{headings[topic].title}</h2>
            <p>{headings[topic].summary}</p>
          </div>
          <button
            className="drawer-close"
            onClick={onClose}
            aria-label="Close advanced assumptions guide"
          >
            ×
          </button>
        </div>
        <div className="guide-tabs advanced-tabs" aria-label="Advanced assumption topics">
          <button
            className={topic === "dynamic" ? "active" : ""}
            onClick={() => setTopic("dynamic")}
          >
            Dynamic effects
          </button>
          <button
            className={topic === "planning" ? "active" : ""}
            onClick={() => setTopic("planning")}
          >
            Planning factor
          </button>
          <button
            className={topic === "calibration" ? "active" : ""}
            onClick={() => setTopic("calibration")}
          >
            Calibration
          </button>
          <button
            className={topic === "distributions" ? "active" : ""}
            onClick={() => setTopic("distributions")}
          >
            Distributions
          </button>
        </div>

        {topic === "dynamic" && (
          <div className="advanced-guide-body">
            <section className="advanced-formula-card">
              <div>
                <span className="eyebrow">KTR-style representation</span>
                <h3>β<sub>c,t</sub> = Σ K(t, knot<sub>j</sub>) · b<sub>c,j</sub></h3>
                <p>
                  Each {dataset.periodUnit}ly coefficient is a weighted blend of a small set of
                  knot coefficients. Nearby knots receive more weight, so the
                  path changes smoothly instead of jumping {dataset.periodUnit} to {dataset.periodUnit}.
                </p>
              </div>
              <div className="kernel-sketch" aria-label="Overlapping smooth Gaussian kernels">
                {Array.from({ length: config.kernelKnots }, (_, index) => (
                  <i
                    key={index}
                    style={{
                      left: `${(index / Math.max(config.kernelKnots - 1, 1)) * 86}%`,
                      opacity: 0.42 + (index % 3) * 0.18,
                    }}
                  />
                ))}
              </div>
            </section>
            <section className="advanced-detail-grid">
              <article>
                <span className="detail-icon mint">✓</span>
                <h3>Use it when</h3>
                <p>
                  Creative quality, targeting, inventory, or market response
                  plausibly changed during a long history.
                </p>
              </article>
              <article>
                <span className="detail-icon orange">△</span>
                <h3>Watch for</h3>
                <p>
                  Short histories and correlated channels can make changing
                  coefficients look more certain than the data supports.
                </p>
              </article>
            </section>
            <section className="advanced-control-card">
              <div>
                <span className="eyebrow">Complexity controls</span>
                <h3>{config.kernelKnots} knots · {config.kernelBandwidth.toFixed(2)} bandwidth</h3>
                <p>
                  More knots permit more local change. Larger bandwidth blends
                  information across a longer portion of the timeline.
                </p>
              </div>
              <label>
                <span>Kernel knots <b>{config.kernelKnots}</b></span>
                <input
                  type="range"
                  min="3"
                  max="10"
                  step="1"
                  value={config.kernelKnots}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      kernelKnots: Number(event.target.value),
                    })
                  }
                />
              </label>
              <label>
                <span>Smoothing bandwidth <b>{config.kernelBandwidth.toFixed(2)}</b></span>
                <input
                  type="range"
                  min="0.08"
                  max="0.35"
                  step="0.01"
                  value={config.kernelBandwidth}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      kernelBandwidth: Number(event.target.value),
                    })
                  }
                />
              </label>
            </section>
            <p className="advanced-method-note">
              Flux uses an analytic Gaussian-kernel approximation inspired by
              Orbit KTR. It is intentionally lighter than a full posterior
              sampler and is labeled accordingly in exported results.
            </p>
          </div>
        )}

        {topic === "planning" && (
          <div className="advanced-guide-body">
            <section className="advanced-formula-card planning">
              <div>
                <span className="eyebrow">Latent context variable</span>
                <h3>p<sub>t</sub> = smooth(PC₁(log(1 + spend<sub>·,t</sub>)))</h3>
                <p>
                  Flux extracts the dominant shared movement across standardized
                  channel plans, smooths it, and includes it in the baseline as
                  planning context.
                </p>
              </div>
              <div className="planning-sketch" aria-hidden="true">
                <span /><span /><span /><b />
              </div>
            </section>
            <section className="advanced-detail-grid">
              <article>
                <span className="detail-icon mint">↗</span>
                <h3>What it can capture</h3>
                <p>
                  Coordinated bursts, campaign waves, and common budget pressure
                  that move several channels together.
                </p>
              </article>
              <article>
                <span className="detail-icon orange">!</span>
                <h3>What it cannot claim</h3>
                <p>
                  It is a proxy—not observed demand and not a causal media
                  effect. It may absorb signal that would otherwise be assigned
                  to channels.
                </p>
              </article>
            </section>
            <section className="advanced-decision-card">
              <div>
                <span className="eyebrow">Decision rule</span>
                <h3>Use as a sensitivity specification</h3>
                <p>
                  Compare ROI stability with the factor on and off. Keep it when
                  it improves residual structure without implausibly erasing
                  media contribution.
                </p>
              </div>
              <AdvancedToggle
                checked={config.planningIntensity}
                onChange={(planningIntensity) =>
                  setConfig({ ...config, planningIntensity })
                }
                label="Include planning intensity factor"
              />
            </section>
          </div>
        )}

        {topic === "calibration" && (
          <div className="advanced-guide-body">
            <section className="calibration-choice-grid">
              <button
                className={config.calibrationMode === "prior" ? "selected" : ""}
                onClick={() =>
                  setConfig({ ...config, calibrationMode: "prior" })
                }
              >
                <span>Recommended default</span>
                <h3>Prior calibration</h3>
                <p>
                  Lift ROI shapes plausible coefficient values before the MMM
                  outcome is fitted. Uncertainty controls prior strength.
                </p>
                <b>Best for regularizing toward ground truth</b>
              </button>
              <button
                className={
                  config.calibrationMode === "likelihood" ? "selected" : ""
                }
                onClick={() =>
                  setConfig({ ...config, calibrationMode: "likelihood" })
                }
              >
                <span>Advanced alternative</span>
                <h3>Likelihood calibration</h3>
                <p>
                  Each experiment becomes a separate noisy ROI measurement
                  linked to the coefficient path over its study window.
                </p>
                <b>Best for a joint evidence model</b>
              </button>
            </section>
            <section className="calibration-contract">
              <div>
                <span className="eyebrow">Prior route</span>
                <h3>Experiment → coefficient prior → outcome fit</h3>
                <p>
                  Easier to reason about and less likely to over-count the
                  experimental result as if it were another {dataset.periodUnit}ly outcome.
                </p>
              </div>
              <div>
                <span className="eyebrow">Likelihood route</span>
                <h3>{dataset.modelCadence === "monthly" ? "Monthly" : "Weekly"} outcome + noisy experiment measurement</h3>
                <p>
                  More direct, but the standard error and experiment window must
                  be trustworthy because they set the measurement weight.
                </p>
              </div>
            </section>
            <div className="parameter-caution">
              <b>Important</b>
              <p>
                If the experiment and MMM data share outcomes or controls,
                likelihood calibration can double-count information. Record
                overlap explicitly before using it for decision-grade ROI.
              </p>
            </div>
          </div>
        )}

        {topic === "distributions" && (
          <div className="advanced-guide-body">
            <section className="distribution-preset-grid">
              <button
                className={
                  distributionPreset(config) === "balanced" ? "selected" : ""
                }
                onClick={() =>
                  setConfig(applyDistributionPreset(config, "balanced"))
                }
              >
                <span>Recommended start</span>
                <i className="distribution-shape log-normal" />
                <h3>Balanced continuous</h3>
                <p>Log-normal media prior + Gaussian outcome likelihood.</p>
                <b>Stable revenue or conversion series with roughly symmetric residuals.</b>
              </button>
              <button
                className={
                  distributionPreset(config) === "robust" ? "selected" : ""
                }
                onClick={() =>
                  setConfig(applyDistributionPreset(config, "robust"))
                }
              >
                <span>Robust alternative</span>
                <i className="distribution-shape student-t" />
                <h3>Noisy campaigns</h3>
                <p>Half-normal media prior + Student-t outcome likelihood.</p>
                <b>Promotions, tracking disruptions, or isolated demand spikes.</b>
              </button>
              <button
                className={
                  distributionPreset(config) === "multiplicative"
                    ? "selected"
                    : ""
                }
                onClick={() =>
                  setConfig(
                    applyDistributionPreset(config, "multiplicative"),
                  )
                }
              >
                <span>Positive & skewed</span>
                <i className="distribution-shape log-normal" />
                <h3>Multiplicative growth</h3>
                <p>Log-normal media prior + log-normal outcome likelihood.</p>
                <b>Positive KPIs whose variability grows with their level.</b>
              </button>
            </section>

            <section className="distribution-custom-card">
              <div>
                <span className="eyebrow">Custom specification</span>
                <h3>Choose each family directly</h3>
                <p>
                  Presets are starting points. Confirm the outcome choice with
                  residual diagnostics and compare ROI stability.
                </p>
              </div>
              <label>
                <span>Media coefficient prior</span>
                <select
                  value={config.priorDistribution}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      priorDistribution: event.target
                        .value as AdvancedModelConfig["priorDistribution"],
                    })
                  }
                >
                  <option value="half-normal">Half-normal</option>
                  <option value="log-normal">Log-normal</option>
                </select>
              </label>
              <label>
                <span>Outcome likelihood</span>
                <select
                  value={config.likelihoodDistribution}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      likelihoodDistribution: event.target
                        .value as AdvancedModelConfig["likelihoodDistribution"],
                    })
                  }
                >
                  <option value="gaussian">Gaussian</option>
                  <option value="student-t">Student-t</option>
                  <option value="log-normal">Log-normal</option>
                </select>
              </label>
            </section>

            {config.likelihoodDistribution === "student-t" && (
              <section className="student-t-control">
                <div>
                  <span className="eyebrow">Tail weight</span>
                  <h3>Degrees of freedom ν = {config.studentTDegreesFreedom}</h3>
                  <p>
                    Lower values discount extreme residuals more strongly.
                    Values around 4–7 are a practical robust starting range.
                  </p>
                </div>
                <label>
                  <span>More robust</span>
                  <input
                    type="range"
                    min="3"
                    max="30"
                    step="1"
                    value={config.studentTDegreesFreedom}
                    onChange={(event) =>
                      setConfig({
                        ...config,
                        studentTDegreesFreedom: Number(event.target.value),
                      })
                    }
                  />
                  <span>More Gaussian</span>
                </label>
              </section>
            )}

            <section className="distribution-explanation-grid">
              <article>
                <span className="eyebrow">Half-normal prior</span>
                <h3>Positive with shrinkage toward zero</h3>
                <p>
                  A conservative default when media effects should not be
                  negative and weak channels should remain close to zero.
                </p>
              </article>
              <article>
                <span className="eyebrow">Log-normal prior</span>
                <h3>Positive with a longer upper tail</h3>
                <p>
                  Useful when uncertainty is multiplicative and unusually large
                  effects are possible, but it can be sensitive to scale.
                </p>
              </article>
              <article>
                <span className="eyebrow">Gaussian likelihood</span>
                <h3>Symmetric additive noise</h3>
                <p>
                  Use when residual spread is fairly constant and large misses
                  are genuinely rare.
                </p>
              </article>
              <article>
                <span className="eyebrow">Student-t likelihood</span>
                <h3>Heavy-tailed additive noise</h3>
                <p>
                  Reduces the influence of isolated spikes without deleting
                  those observations from the training data.
                </p>
              </article>
              <article>
                <span className="eyebrow">Log-normal likelihood</span>
                <h3>Positive multiplicative noise</h3>
                <p>
                  Fits the outcome on the log scale when percentage-sized
                  errors are more plausible than fixed-sized errors.
                </p>
              </article>
            </section>
            <p className="advanced-method-note">
              Flux estimates these families with an analytic MAP/Laplace
              approximation. Student-t uses iteratively reweighted regression;
              log-normal likelihood fits the positive outcome on its log scale.
              This outcome-likelihood choice is separate from likelihood
              calibration, where an experiment enters as an additional noisy
              ROI measurement using its reported standard error.
            </p>
          </div>
        )}
    </DrawerShell>
  );
}

function AdvancedModelerView({
  dataset,
  config,
  setConfig,
  advancedConfig,
  setAdvancedConfig,
  result,
  status,
  experiments,
  industryPriorChannels,
  promotion,
  onRun,
}: {
  dataset: Dataset;
  config: ModelConfig;
  setConfig: (config: ModelConfig) => void;
  advancedConfig: AdvancedModelConfig;
  setAdvancedConfig: (config: AdvancedModelConfig) => void;
  result?: ModelResult;
  status: JobStatus;
  experiments: Experiment[];
  industryPriorChannels: string[];
  promotion?: PromotedAgenticSpecification;
  onRun: () => void;
}) {
  const [guideTopic, setGuideTopic] = useState<AdvancedGuideTopic | null>(null);
  const [showPromotedSpecification, setShowPromotedSpecification] =
    useState(false);
  const dates = dataset.rows.map((row) => String(row[dataset.dateColumn]));
  const advanced = result?.advanced;
  const artifactIndustryChannels = Array.from(
    new Set(
      advanced?.calibrationReceipts
        ?.filter((receipt) => receipt.source === "industry")
        .map((receipt) => receipt.channel) ?? [],
    ),
  );
  const experimentChannels = Array.from(
    new Set(experiments.map((experiment) => experiment.channel.toLowerCase())),
  );
  const activeIndustryChannels = industryPriorChannels.filter(
    (channel) =>
      !experimentChannels.some(
        (experimentChannel) =>
          experimentChannel === channel.toLowerCase(),
      ),
  );
  const displayedIndustryChannels = result
    ? artifactIndustryChannels
    : activeIndustryChannels;
  const experimentContract =
    advancedConfig.calibrationMode === "prior" ? "prior" : "likelihood";
  const evidenceLabels = [
    ...experimentChannels.map((channel) => ({
      channel,
      label: `Experiment ${experimentContract}`,
      source: "experiment" as const,
    })),
    ...activeIndustryChannels.map((channel) => ({
      channel,
      label: "Industry prior",
      source: "industry" as const,
    })),
  ];
  const coefficientSeries =
    advanced?.coefficientPaths.map((path, index) => {
      const pathMean = mean(path.values);
      return {
        name: cleanChannel(path.channel),
        values: path.values.map(
          (value) => (value / Math.max(pathMean, 1e-9)) * 100,
        ),
        color: CHANNEL_COLORS[index % CHANNEL_COLORS.length],
      };
    }) ?? [];
  const activeDistributionPreset = distributionPreset(advancedConfig);
  const responseProfiles = responseProfileForDataset(dataset, config);
  const channelResponseEnabled = responseProfiles.some(
    (profile) => profile.explicit,
  );
  const toggleChannelResponses = (enabled: boolean) => {
    setConfig({
      ...config,
      channelResponses: enabled
        ? Object.fromEntries(
            dataset.mediaColumns.map((channel) => [
              channel,
              {
                adstockType: config.adstockType,
                adstock: config.adstock,
                weibullShape: config.weibullShape,
                weibullScale: config.weibullScale,
                saturation: config.saturation,
                halfSaturationQuantile: 0.5,
                kernelNormalization: "sum" as const,
              },
            ]),
          )
        : undefined,
    });
  };

  return (
    <div className="view advanced-modeler-view">
      <section className="page-heading compact-heading">
        <div>
          <span className="kicker">Opt-in complexity</span>
          <h1>Advanced modeler</h1>
          <p>
            Add one assumption at a time, understand what it changes, and keep
            every specification reproducible.
          </p>
        </div>
        <span className="advanced-stage-pill">
          {promotion ? "Agentic receipt active" : "Experimental · V2"}
        </span>
      </section>

      {promotion && (
        <PromotedSpecificationBanner
          destination="advanced"
          promotion={promotion}
          datasetHash={dataset.hash}
          config={config}
          advancedConfig={advancedConfig}
          result={result}
          experiments={experiments}
          industryPriorChannels={industryPriorChannels}
          onInspect={() => setShowPromotedSpecification(true)}
        />
      )}

      <section className="advanced-intro">
        <span>Start with a question, not a feature.</span>
        <p>
          The standard Model Studio remains your benchmark. This workspace is
          for testing changing effectiveness, coordinated planning, and
          alternative experiment calibration and probability assumptions.
        </p>
      </section>

      <details className="card channel-response-editor" open={channelResponseEnabled}>
        <summary>
          <span><span className="eyebrow">Media response contract</span><b>Fit carryover and saturation by channel</b><small>Recommended when channel mechanics differ materially</small></span>
          <AdvancedToggle
            checked={channelResponseEnabled}
            onChange={toggleChannelResponses}
            label="Use channel-specific response curves"
          />
        </summary>
        {channelResponseEnabled ? (
          <div className="channel-response-table">
            <div className="head"><span>Channel</span><span>Carryover</span><span>Memory</span><span>Hill shape</span><span>Half-saturation</span></div>
            {responseProfiles.map(({ channel, response }) => (
              <div key={channel}>
                <strong>{cleanChannel(channel)}</strong>
                <select value={response.adstockType} onChange={(event) => setConfig({ ...config, channelResponses: { ...config.channelResponses, [channel]: { ...config.channelResponses?.[channel], adstockType: event.target.value as ModelConfig["adstockType"] } } })}><option value="geometric">Geometric</option><option value="weibull">Weibull</option></select>
                {response.adstockType === "geometric" ? (
                  <label><b>θ {response.adstock.toFixed(2)}</b><input type="range" min="0.05" max="0.9" step="0.05" value={response.adstock} onChange={(event) => setConfig({ ...config, channelResponses: { ...config.channelResponses, [channel]: { ...config.channelResponses?.[channel], adstock: Number(event.target.value) } } })} /></label>
                ) : (
                  <label><b>k {response.weibullShape.toFixed(1)} · λ {response.weibullScale.toFixed(1)}</b><input type="range" min="1" max="12" step="0.5" value={response.weibullScale} onChange={(event) => setConfig({ ...config, channelResponses: { ...config.channelResponses, [channel]: { ...config.channelResponses?.[channel], weibullScale: Number(event.target.value) } } })} /></label>
                )}
                <label><b>{response.saturation.toFixed(1)}</b><input type="range" min="0.5" max="3" step="0.1" value={response.saturation} onChange={(event) => setConfig({ ...config, channelResponses: { ...config.channelResponses, [channel]: { ...config.channelResponses?.[channel], saturation: Number(event.target.value) } } })} /></label>
                <label><b>Q{Math.round(response.halfSaturationQuantile * 100)}</b><input type="range" min="0.2" max="0.8" step="0.05" value={response.halfSaturationQuantile} onChange={(event) => setConfig({ ...config, channelResponses: { ...config.channelResponses, [channel]: { ...config.channelResponses?.[channel], halfSaturationQuantile: Number(event.target.value) } } })} /></label>
              </div>
            ))}
            <p>Each candidate and every downstream validation refit, MCMC compile, and budget curve uses these same channel-level settings. Sum-normalized kernels keep total carryover scale comparable.</p>
          </div>
        ) : (
          <p className="channel-response-empty">All channels currently inherit the shared Model Studio curve. Enable this only when there is enough variation to identify the additional response parameters.</p>
        )}
      </details>

      <section className="advanced-module-grid">
        <article className={`advanced-module ${advancedConfig.timeVarying ? "enabled" : ""}`}>
          <div className="advanced-module-top">
            <span className="module-number">01</span>
            <AdvancedToggle
              checked={advancedConfig.timeVarying}
              onChange={(timeVarying) =>
                setAdvancedConfig({ ...advancedConfig, timeVarying })
              }
              label="Use time-varying media coefficients"
            />
          </div>
          <span className="eyebrow">Dynamic effectiveness</span>
          <h2>Time-varying coefficients</h2>
          <p>
            Smooth KTR-style coefficient paths let channel effectiveness change
            across the measurement history.
          </p>
          {advancedConfig.timeVarying && (
            <div className="advanced-inline-settings">
              <span><b>{advancedConfig.kernelKnots}</b> knots</span>
              <span><b>{advancedConfig.kernelBandwidth.toFixed(2)}</b> smoothing</span>
            </div>
          )}
          <button className="advanced-learn" onClick={() => setGuideTopic("dynamic")}>
            Understand the assumption <span>→</span>
          </button>
        </article>

        <article className={`advanced-module ${advancedConfig.planningIntensity ? "enabled" : ""}`}>
          <div className="advanced-module-top">
            <span className="module-number">02</span>
            <AdvancedToggle
              checked={advancedConfig.planningIntensity}
              onChange={(planningIntensity) =>
                setAdvancedConfig({ ...advancedConfig, planningIntensity })
              }
              label="Include latent planning intensity"
            />
          </div>
          <span className="eyebrow">Shared context</span>
          <h2>Planning intensity factor</h2>
          <p>
            A smoothed latent factor captures common planning pressure across
            channels as baseline context.
          </p>
          <div className="advanced-inline-settings">
            <span><b>{advancedConfig.planningIntensity ? "On" : "Off"}</b> sensitivity factor</span>
          </div>
          <button className="advanced-learn" onClick={() => setGuideTopic("planning")}>
            Understand the assumption <span>→</span>
          </button>
        </article>

        <article className="advanced-module enabled">
          <div className="advanced-module-top">
            <span className="module-number">03</span>
            <span className="module-evidence">
              {experiments.length + activeIndustryChannels.length} evidence{" "}
              source{experiments.length + activeIndustryChannels.length === 1
                ? ""
                : "s"}
            </span>
          </div>
          <span className="eyebrow">Experiment calibration</span>
          <h2>Where evidence enters</h2>
          <p>
            Choose coefficient priors for regularization or a separate noisy
            measurement in the likelihood.
          </p>
          <div className="segmented-control advanced-calibration-toggle">
            <button
              className={advancedConfig.calibrationMode === "prior" ? "active" : ""}
              onClick={() =>
                setAdvancedConfig({
                  ...advancedConfig,
                  calibrationMode: "prior",
                })
              }
            >
              Prior
            </button>
            <button
              className={advancedConfig.calibrationMode === "likelihood" ? "active" : ""}
              onClick={() =>
                setAdvancedConfig({
                  ...advancedConfig,
                  calibrationMode: "likelihood",
                })
              }
            >
              Likelihood
            </button>
          </div>
          <div className="advanced-evidence-preview">
            <div>
              <span>
                <b>{experiments.length}</b> experiment
                {experiments.length === 1 ? "" : "s"} as {experimentContract}
              </span>
              <span>
                <b>{activeIndustryChannels.length}</b> benchmark guardrail
                {activeIndustryChannels.length === 1 ? "" : "s"} as prior
              </span>
            </div>
            <p>
              {evidenceLabels.length
                ? evidenceLabels
                    .slice(0, 3)
                    .map(
                      (evidence) =>
                        `${cleanChannel(evidence.channel)} · ${evidence.label}`,
                    )
                    .join("  ·  ")
                : "No calibration evidence is currently connected."}
              {evidenceLabels.length > 3
                ? `  ·  +${evidenceLabels.length - 3} more`
                : ""}
            </p>
          </div>
          <button className="advanced-learn" onClick={() => setGuideTopic("calibration")}>
            Compare the contracts <span>→</span>
          </button>
        </article>

        <article className="advanced-module enabled distribution-module">
          <div className="advanced-module-top">
            <span className="module-number">04</span>
            <span className="module-evidence">
              {activeDistributionPreset === "balanced"
                ? "Recommended"
                : activeDistributionPreset === "robust"
                  ? "Robust"
                  : activeDistributionPreset === "multiplicative"
                    ? "Multiplicative"
                    : "Custom"}
            </span>
          </div>
          <span className="eyebrow">Probability model</span>
          <h2>Distribution set</h2>
          <p>
            Match positive media effects and outcome noise to the way the KPI
            actually behaves.
          </p>
          <div className="distribution-quick-presets" aria-label="Distribution presets">
            {([
              ["balanced", "Balanced"],
              ["robust", "Noisy"],
              ["multiplicative", "Growth"],
            ] as const).map(([preset, label]) => (
              <button
                key={preset}
                className={activeDistributionPreset === preset ? "active" : ""}
                onClick={() =>
                  setAdvancedConfig(
                    applyDistributionPreset(advancedConfig, preset),
                  )
                }
              >
                {label}
              </button>
            ))}
          </div>
          <div className="advanced-inline-settings">
            <span><b>{advancedConfig.priorDistribution}</b> prior</span>
            <span><b>{advancedConfig.likelihoodDistribution}</b> likelihood</span>
          </div>
          <button className="advanced-learn" onClick={() => setGuideTopic("distributions")}>
            Choose with guidance <span>→</span>
          </button>
        </article>
      </section>

      <section className="advanced-run-card">
        <div>
          <span className="eyebrow">Current specification</span>
          <h2>
            {advancedConfig.timeVarying ? "Dynamic" : "Static"} effects ·{" "}
            {advancedConfig.planningIntensity ? "planning factor on" : "planning factor off"} ·{" "}
            {advancedConfig.calibrationMode} calibration · {config.adstockType} adstock
          </h2>
          <p>
            {advancedConfig.priorDistribution} prior +{" "}
            {advancedConfig.likelihoodDistribution} likelihood. The cache key
            includes every advanced option.
          </p>
          <div className="advanced-run-evidence">
            <span>
              <b>{experiments.length}</b> experiment
              {experiments.length === 1 ? "" : "s"} as {experimentContract}
            </span>
            <span>
              <b>{displayedIndustryChannels.length}</b> benchmark guardrail
              {displayedIndustryChannels.length === 1 ? "" : "s"} as prior
            </span>
            <em>{result ? "Artifact evidence receipt" : "Will be applied on the next run"}</em>
          </div>
          {result && (
            <p className="advanced-artifact-evidence">
              Displayed artifact: {advanced?.calibratedExperiments ?? 0} experiment calibration
              {(advanced?.calibratedExperiments ?? 0) === 1 ? "" : "s"}
              {displayedIndustryChannels.length
                ? ` · industry prior on ${displayedIndustryChannels.map(cleanChannel).join(", ")}`
                : " · no industry priors"}
            </p>
          )}
        </div>
        <div className="advanced-run-action">
          <JobPill status={status} />
          <button
            className="button primary"
            onClick={onRun}
            disabled={status === "running" || status === "queued"}
          >
            {status === "running" || status === "queued"
              ? "Estimating…"
              : result
                ? "Run this specification again"
                : "Run advanced model"}{" "}
            <span>→</span>
          </button>
        </div>
      </section>

      {(status === "running" || status === "queued") && (
        <section className="advanced-loading-grid">
          <SkeletonCard height={280} />
          <SkeletonCard height={280} />
        </section>
      )}

      {result && status !== "running" && status !== "queued" && (
        <>
          <section className="advanced-result-heading">
            <div>
              <span className="kicker">Specification results</span>
              <h2>What changed over time</h2>
              <div className="advanced-result-evidence">
                <span>
                  ◎ {advanced?.calibratedExperiments ?? 0} experiment
                  {(advanced?.calibratedExperiments ?? 0) === 1 ? "" : "s"} via{" "}
                  {advanced?.calibrationMode ?? experimentContract}
                </span>
                <span>
                  ◇ {advanced?.calibratedIndustryPriors ?? 0} benchmark
                  {(advanced?.calibratedIndustryPriors ?? 0) === 1 ? "" : "s"}{" "}
                  via prior
                </span>
              </div>
            </div>
            <div className="advanced-result-metrics">
              <span>R² <b>{result.r2.toFixed(3)}</b></span>
              <span>MAPE <b>{result.mape.toFixed(1)}%</b></span>
              <span>RMSE <b>{formatCompact(result.rmse)}</b></span>
            </div>
          </section>
          <section className="advanced-result-grid">
            <article className="card">
              <div className="card-heading">
                <div><span className="eyebrow">Model fit</span><h2>Actual versus fitted</h2></div>
                <span className="subtle">{result.cached ? "Restored from cache" : "Fresh estimate"}</span>
              </div>
              <TimeSeriesChart
                dates={dates}
                series={[
                  { name: "Actual", values: result.actual, color: "#5b5bd6" },
                  { name: "Model fit", values: result.predicted, color: "#ee9c61", dashed: true },
                ]}
                label="Advanced model actual versus fitted outcome"
                height={260}
                showLegend
                showZoom
              />
            </article>
            <article className="card">
              <div className="card-heading">
                <div><span className="eyebrow">Dynamic effectiveness</span><h2>Relative coefficient index</h2></div>
                <span className="subtle">Channel mean = 100</span>
              </div>
              <TimeSeriesChart
                dates={dates}
                series={coefficientSeries}
                label="Relative media coefficient paths over time"
                height={260}
                showLegend
                showZoom
              />
              <p className="advanced-chart-note">
                This index shows within-channel change. It does not compare ROI
                levels across channels.
              </p>
            </article>
            {advanced?.planningIntensity.length ? (
              <article className="card advanced-planning-chart">
                <div className="card-heading">
                  <div><span className="eyebrow">Latent context</span><h2>Media planning intensity</h2></div>
                  <span className="subtle">Relative index · 0–100</span>
                </div>
                <TimeSeriesChart
                  dates={dates}
                  series={[{
                    name: "Planning intensity",
                    values: advanced.planningIntensity,
                    color: "#27a589",
                    area: true,
                  }]}
                  label="Latent media planning intensity over time"
                  height={210}
                  showZoom
                />
              </article>
            ) : null}
          </section>
          <section className="card roi-card advanced-roi-card">
            <div className="card-heading">
              <div><span className="eyebrow">Decision output</span><h2>Channel ROI estimates</h2></div>
              <span className="subtle">95% analytic interval</span>
            </div>
            <div className="roi-table">
              <div className="roi-row roi-head"><span>Channel & evidence</span><span>Contribution</span><span>Share</span><span>ROI interval</span><span>Full-history MAP ROI</span></div>
              {result.channels.map((channel, index) => {
                const receipt = advanced?.calibrationReceipts?.find(
                  (candidate) =>
                    candidate.channel.toLowerCase() ===
                    channel.channel.toLowerCase(),
                );
                const experiment = experiments.find(
                  (candidate) =>
                    candidate.channel.toLowerCase() ===
                    channel.channel.toLowerCase(),
                );
                const windowComparison = experiment
                  ? modelExperimentWindowRoi(
                      dataset,
                      result,
                      config,
                      experiment,
                    )
                  : undefined;
                return (
                <div className="roi-row" key={channel.channel}>
                  <strong className="advanced-channel-evidence">
                    <i style={{ background: CHANNEL_COLORS[index % CHANNEL_COLORS.length] }} />
                    <span>
                      {cleanChannel(channel.channel)}
                      <small
                        className={
                          channel.priorSource
                            ? `calibrated ${channel.priorSource}`
                            : "unanchored"
                        }
                        title={channel.priorLabel}
                      >
                        {channel.priorSource === "experiment"
                          ? `Experiment ${advanced?.calibrationMode ?? experimentContract}`
                          : channel.priorSource === "industry"
                            ? "Industry prior"
                            : "Unanchored"}
                      </small>
                    </span>
                  </strong>
                  <span>{formatCompact(channel.contribution, true)}</span>
                  <span>{(channel.contributionShare * 100).toFixed(1)}%</span>
                  <span className="interval"><i style={{ left: `${Math.min(channel.roiLow / Math.max(channel.roiHigh, 1) * 70, 70)}%`, width: `${Math.max((channel.roiHigh - channel.roiLow) / Math.max(channel.roiHigh, 1) * 65, 8)}%` }} /><em>{channel.roiLow.toFixed(2)}–{channel.roiHigh.toFixed(2)}</em></span>
                  <span className="advanced-roi-value">
                    <b>{channel.roi.toFixed(2)}×</b>
                    <small>
                      {receipt?.source === "experiment"
                        ? `Window ${(windowComparison?.roi ?? receipt.modelRoi).toFixed(2)}× · evidence ${receipt.targetRoi.toFixed(2)}×`
                        : receipt?.source === "industry"
                          ? `Prior median ${receipt.targetRoi.toFixed(2)}× · 80% ${receipt.targetLow.toFixed(2)}–${receipt.targetHigh.toFixed(2)}×`
                      : channel.priorRoi !== undefined
                        ? `Evidence ${channel.priorRoi.toFixed(2)}×`
                        : "No external target"}
                    </small>
                  </span>
                </div>
                );
              })}
            </div>
            <NumericalReliability result={result} />
          </section>
        </>
      )}

      {guideTopic && (
        <AdvancedAssumptionsGuide
          dataset={dataset}
          topic={guideTopic}
          setTopic={setGuideTopic}
          config={advancedConfig}
          setConfig={setAdvancedConfig}
          onClose={() => setGuideTopic(null)}
        />
      )}
      {showPromotedSpecification && promotion && (
        <SpecificationInspector
          run={promotion.run}
          experiments={promotion.experiments}
          industryPriorChannels={promotion.industryPriorChannels}
          onClose={() => setShowPromotedSpecification(false)}
        />
      )}
    </div>
  );
}

type ValidationGuideTopic =
  | "generalization"
  | "structure"
  | "causal"
  | "decision"
  | "scoring";

const VALIDATION_LAYER_META: Record<
  ValidationLayerId,
  { number: string; title: string; eyebrow: string }
> = {
  generalization: {
    number: "01",
    title: "Generalization",
    eyebrow: "Unseen data",
  },
  structure: {
    number: "02",
    title: "Structural adequacy",
    eyebrow: "Assumption-aware",
  },
  causal: {
    number: "03",
    title: "Causal credibility",
    eyebrow: "Decision evidence",
  },
  decision: {
    number: "04",
    title: "ROI decision coherence",
    eyebrow: "Business plausibility",
  },
};

const VALIDATION_MODEL_LABELS: Record<ValidationModelKind, string> = {
  frequentist: "Frequentist",
  bayesian: "Bayesian",
  advanced: "Advanced",
};

function validationStatusLabel(status: ValidationStatus): string {
  return status === "pass"
    ? "Pass"
    : status === "review"
      ? "Review"
      : status === "fail"
        ? "Fail"
        : "Incomplete";
}

function ValidationGuide({
  dataset,
  topic,
  setTopic,
  onClose,
}: {
  dataset: Dataset;
  topic: ValidationGuideTopic;
  setTopic: (topic: ValidationGuideTopic) => void;
  onClose: () => void;
}) {
  const headings: Record<
    ValidationGuideTopic,
    { kicker: string; title: string; summary: string }
  > = {
    generalization: {
      kicker: `Layer 01 · ${scoreWeightPercent("generalization")}% learned weight`,
      title: "Can the model handle unseen conditions?",
      summary:
        "Temporal folds and observed spend-regime holdouts test prediction without allowing future outcomes into training.",
    },
    structure: {
      kicker: `Layer 02 · ${scoreWeightPercent("structure")}% learned weight`,
      title: "Does the fitted structure match its assumptions?",
      summary:
        "Diagnostics adapt to the selected likelihood and coefficient structure instead of applying one universal regression checklist.",
    },
    causal: {
      kicker: `Layer 03 · ${scoreWeightPercent("causal")}% learned weight`,
      title: "How credible is the causal interpretation?",
      summary:
        "Qualified external prediction, refit stability, confounder stress, and temporal placebos test whether ROI survives reasonable challenges.",
    },
    decision: {
      kicker: `Layer 04 · ${scoreWeightPercent("decision")}% learned weight`,
      title: "Are the channel ROI estimates usable for business decisions?",
      summary:
        "Posterior plausibility, stability, resolution, identification, and economic consistency test every material channel under its declared evidence contract.",
    },
    scoring: {
      kicker: "Winner methodology",
      title: "Decision regret ranks candidates only after gates",
      summary:
        "Offline simulation learned twenty diagnostic weights from known economic decision loss. Evidence, identification, and two-sided ROI plausibility gates remain immutable and cannot be traded for a higher score.",
    },
  };
  return (
    <DrawerShell
      className="validation-guide"
      labelledBy="validation-guide-title"
      onClose={onClose}
    >
        <div className="assumptions-header">
          <div>
            <span className="kicker">{headings[topic].kicker}</span>
            <h2 id="validation-guide-title">{headings[topic].title}</h2>
            <p>{headings[topic].summary}</p>
          </div>
          <button
            className="drawer-close"
            onClick={onClose}
            aria-label="Close validation explanation"
          >
            ×
          </button>
        </div>
        <div className="guide-tabs validation-guide-tabs">
          {([
            ["generalization", "Generalization"],
            ["structure", "Structure"],
            ["causal", "Causal"],
            ["decision", "ROI coherence"],
            ["scoring", "Winner score"],
          ] as const).map(([id, label]) => (
            <button
              key={id}
              className={topic === id ? "active" : ""}
              onClick={() => setTopic(id)}
            >
              {label}
            </button>
          ))}
        </div>

        {topic === "generalization" && (
          <div className="validation-guide-body">
            <section className="validation-timeline">
              {[58, 72, 86].map((trainShare, index) => (
                <div key={trainShare}>
                  <span>Fold {index + 1}</span>
                  <i style={{ width: `${trainShare}%` }} />
                  <b style={{ left: `${trainShare}%` }} />
                </div>
              ))}
              <p><i /> Training history <b /> Next {dataset.modelCadence === "monthly" ? "3–6 months" : "6–13 weeks"}</p>
            </section>
            <section className="validation-explanation-grid">
              <article>
                <span className="detail-icon mint">→</span>
                <h3>Rolling origin</h3>
                <p>
                  Every fold trains only on earlier observations. Adstock moves
                  forward into the holdout, while outcome data never moves
                  backward.
                </p>
              </article>
              <article>
                <span className="detail-icon orange">↕</span>
                <h3>Spend regimes</h3>
                <p>
                  Low- and high-spend periods are withheld with a purge buffer.
                  If the history lacks support, the test is incomplete—not a
                  pass.
                </p>
              </article>
            </section>
            <section className="validation-weight-card">
              <div><span>45%</span><p><b>OOS accuracy</b><small>WAPE, R² and seasonal-naive skill</small></p></div>
              <div><span>20%</span><p><b>Interval coverage</b><small>Unseen outcomes inside predictive bands</small></p></div>
              <div><span>25%</span><p><b>Regime robustness</b><small>Held-out high and low spend</small></p></div>
              <div><span>10%</span><p><b>Fold stability</b><small>Error consistency over time</small></p></div>
            </section>
          </div>
        )}

        {topic === "structure" && (
          <div className="validation-guide-body">
            <section className="validation-principle">
              <span>Adaptive diagnostics</span>
              <h3>The chosen likelihood defines what “well behaved” means.</h3>
              <p>
                Gaussian models are checked for symmetric constant-variance
                residuals. Student-t models permit heavy tails. Log-normal
                models are evaluated on the log scale.
              </p>
            </section>
            <section className="validation-explanation-grid three">
              <article><span className="detail-icon">≈</span><h3>Identification</h3><p>VIF on transformed media and controls tests whether channel effects can be separated.</p></article>
              <article><span className="detail-icon">⌁</span><h3>Residual memory</h3><p>Short and seasonal autocorrelation reveal baseline signal still left unexplained.</p></article>
              <article><span className="detail-icon">σ</span><h3>Variance</h3><p>Residual spread is compared across fitted outcome levels under the selected scale.</p></article>
              <article><span className="detail-icon">ƒ</span><h3>Functional form</h3><p>Residual relationships with time and fitted values identify missing trend or nonlinearity.</p></article>
              <article><span className="detail-icon">!</span><h3>Influence</h3><p>Extreme standardized residuals flag periods capable of steering the estimate.</p></article>
              <article><span className="detail-icon">β</span><h3>Dynamic paths</h3><p>Time-varying coefficients are checked for unnecessary curvature and instability.</p></article>
            </section>
            <div className="parameter-caution">
              <b>Causal priority</b>
              <p>
                Identification and residual dependence receive more weight than
                mild non-normality. Normal residuals alone do not establish an
                unconfounded effect.
              </p>
            </div>
          </div>
        )}

        {topic === "causal" && (
          <div className="validation-guide-body">
            <section className="causal-validation-flow">
              <div><span>Anchor</span><b>Remove</b></div>
              <i>→</i>
              <div><span>MMM</span><b>Refit</b></div>
              <i>→</i>
              <div><span>ROI</span><b>Recover</b></div>
              <i>→</i>
              <div><span>Ground truth</span><b>Compare</b></div>
            </section>
            <section className="validation-explanation-grid">
              <article><span className="detail-icon mint">◎</span><h3>External prediction · up to 40%</h3><p>Runs only when a non-overlapping, same-channel and same-scope anchor remains in calibration after the holdout.</p></article>
              <article><span className="detail-icon">+</span><h3>New-data stability · 20 parts</h3><p>ROI is compared using 70%, 85%, and the complete measurement history.</p></article>
              <article><span className="detail-icon orange">U</span><h3>Confounder stress · 25 parts</h3><p>A latent control is made progressively more related to media and unexplained outcome.</p></article>
              <article><span className="detail-icon">⇥</span><h3>Future placebo · 15 parts</h3><p>Future media should not explain current residuals; if it does, timing bias may remain.</p></article>
            </section>
            <div className="parameter-caution">
              <b>Conditional scoring</b>
              <p>
                When external prediction is not genuinely testable, it receives
                no score or penalty. The remaining 20/25/15 parts are
                automatically reweighted, while the evidence grade is capped
                and decision-grade eligibility remains unavailable.
              </p>
            </div>
          </div>
        )}

        {topic === "decision" && (
          <div className="validation-guide-body">
            <section className="validation-principle">
              <span>Material-channel contract</span>
              <h3>Plausibility is necessary, but matching a prior is not independent proof.</h3>
              <p>
                Flux evaluates the full ROI interval against experiments or
                channel-specific industry ranges, then separately checks
                stability, resolution, identification, and economic integrity.
              </p>
            </section>
            <section className="validation-explanation-grid three">
              <article><span className="detail-icon mint">%</span><h3>Posterior plausibility</h3><p>Measures how much posterior probability lies inside the registered ROI range, not only whether the median looks reasonable.</p></article>
              <article><span className="detail-icon">↔</span><h3>Decision stability</h3><p>Tracks material-channel ROI as the available measurement history expands.</p></article>
              <article><span className="detail-icon orange">±</span><h3>Resolution</h3><p>Flags intervals too wide to support a practical channel decision.</p></article>
              <article><span className="detail-icon">β</span><h3>Evidence attribution</h3><p>Attributes ROI precision continuously to observational data, experiments, benchmarks, and regularization, with observational credit conditioned on the other regressors.</p></article>
              <article><span className="detail-icon mint">$</span><h3>Economic consistency</h3><p>Checks finite non-negative ROI, contribution accounting, and material coefficient clipping.</p></article>
              <article><span className="detail-icon orange">≠</span><h3>No circular credit</h3><p>Evidence used to calibrate a channel is a contract check and cannot earn independent validation points for matching itself.</p></article>
            </section>
            <div className="parameter-caution">
              <b>Portfolio protection</b>
              <p>
                Spend weighting is capped and the weakest material channel
                receives explicit tail weight, so one implausible ROI cannot be
                hidden behind stronger channels.
              </p>
            </div>
          </div>
        )}

        {topic === "scoring" && (
          <div className="validation-guide-body">
            <section className="winner-formula">
              <span>{ACTIVE_SCORE_CONTRACT.kind === "learned" ? "Learned winner score" : "Fallback winner score"}</span>
              <h3>{activeScoreFormula()}</h3>
              <p>
                V6 combines twenty diagnostic scores with non-negative learned
                weights. Missing diagnostics receive a neutral research value;
                immutable evidence and ROI-coherence gates determine eligibility
                separately and cannot be overridden by the numeric rank.
              </p>
            </section>
            <section className="score-weight-visual">
              {(["generalization", "structure", "causal", "decision"] as const).map((id) => (
                <div key={id} style={{ width: `${scoreWeightPercent(id)}%` }}>
                  <b>{scoreWeightPercent(id)}%</b>
                  <span>{VALIDATION_LAYER_META[id].title}</span>
                </div>
              ))}
            </section>
            <section className="validation-explanation-grid">
              <article><span className="detail-icon mint">✓</span><h3>Learned rank</h3><p>Compares candidates within the highest available immutable eligibility tier using twenty diagnostic signals.</p></article>
              <article><span className="detail-icon orange">◆</span><h3>Gates</h3><p>Gates determine eligibility, not score points. A higher V6 score can never repair a failed gate.</p></article>
              <article><span className="detail-icon">A</span><h3>Evidence grade</h3><p>Reports how complete the validation evidence is independently of the numerical score.</p></article>
              <article><span className="detail-icon">≠</span><h3>No false winner</h3><p>An untestable anchor does not lower the numeric score, but it caps the evidence grade and prevents a decision-grade label.</p></article>
            </section>
          </div>
        )}
    </DrawerShell>
  );
}

const VALIDATION_DIAGNOSTICS: Record<
  ValidationLayerId,
  { mode: ValidationChartMode; label: string; description: string }[]
> = {
  generalization: [
    {
      mode: "forecast",
      label: "Observed vs forecast",
      description:
        "Only genuinely held-out periods are shown. The shaded range is the predictive interval.",
    },
    {
      mode: "regimes",
      label: "Spend regimes",
      description:
        "Compares holdout error when unusually low- and high-spend periods are removed from training.",
    },
  ],
  structure: [
    {
      mode: "residuals",
      label: "Residual timeline",
      description:
        "Runs, changing spread, and isolated shocks reveal structure the model has not captured.",
    },
    {
      mode: "residual-fitted",
      label: "Residuals vs fitted",
      description:
        "A stable cloud around zero supports the selected variance and functional-form assumptions.",
    },
    {
      mode: "acf",
      label: "Residual memory",
      description:
        "Correlation at short and seasonal lags reveals unexplained temporal structure.",
    },
    {
      mode: "distribution",
      label: "Likelihood shape",
      description:
        "Observed residual quantiles are compared with the selected model’s reference scale.",
    },
    {
      mode: "vifs",
      label: "Identification",
      description:
        "Variance inflation shows where transformed media or controls are difficult to separate.",
    },
  ],
  causal: [
    {
      mode: "anchors",
      label: "External prediction",
      description:
        "Qualified MMM estimates and intervals are compared with experiments that were genuinely withheld from calibration.",
    },
    {
      mode: "stability",
      label: "New-data stability",
      description:
        "Leading channel ROI is tracked as the available measurement history grows.",
    },
    {
      mode: "confounders",
      label: "Confounder stress",
      description:
        "ROI movement is measured as a plausible unobserved driver becomes progressively stronger.",
    },
    {
      mode: "placebo",
      label: "Future placebo",
      description:
        "Future media should not explain current residuals if demand anticipation is adequately controlled.",
    },
  ],
  decision: [
    {
      mode: "coherence",
      label: "Channel ROI coherence",
      description:
        "Compares each material channel posterior with a like-for-like experiment window or full-history industry plausibility range.",
    },
  ],
};

function diagnosticForTest(
  layer: ValidationLayerId,
  testId: string,
): ValidationChartMode {
  const mappings: Record<string, ValidationChartMode> = {
    "rolling-oos": "forecast",
    "predictive-coverage": "forecast",
    "fold-stability": "forecast",
    "spend-regimes": "regimes",
    "residual-independence": "acf",
    "variance-structure": "residual-fitted",
    "likelihood-shape": "distribution",
    multicollinearity: "vifs",
    "functional-form": "residuals",
    "influence-stability": "residuals",
    "anchor-recovery": "anchors",
    "new-data-stability": "stability",
    "confounder-sensitivity": "confounders",
    "future-media-placebo": "placebo",
    "roi-posterior-plausibility": "coherence",
    "roi-decision-stability": "coherence",
    "roi-resolution": "coherence",
    "roi-identification": "coherence",
    "roi-economic-consistency": "coherence",
  };
  return mappings[testId] ?? VALIDATION_DIAGNOSTICS[layer][0].mode;
}

function primaryDiagnostic(layer: ValidationLayerId): ValidationChartMode {
  return VALIDATION_DIAGNOSTICS[layer][0].mode;
}

function ValidationScoreSummary({
  result,
}: {
  result: ModelValidationResult;
}) {
  const applicableGates = result.gates.filter((gate) => gate.applicable);
  const unavailableGates = result.gates.filter((gate) => !gate.applicable);
  const passedGates = applicableGates.filter((gate) => gate.passed);
  const eligibilityLabel = result.eligible
    ? "Eligible"
    : unavailableGates.length
      ? "Evidence-limited"
      : "Gated";
  return (
    <section className="validation-score-summary">
      <div className="validation-score-outcome">
        <span className="eyebrow">Winner score</span>
        <div>
          <strong>
            {result.finalScore === null ? "—" : Math.round(result.finalScore)}
          </strong>
          <small>{result.finalScore === null ? "Incomplete" : "/100"}</small>
        </div>
        <span
          className={`validation-eligibility ${
            result.eligible ? "pass" : unavailableGates.length ? "limited" : "review"
          }`}
        >
          {eligibilityLabel}
        </span>
      </div>
      <div className="validation-score-story">
        <span className="eyebrow">Validation fingerprint</span>
        <h2>{result.recommendation}</h2>
        <div className="validation-profile">
          {(["generalization", "structure", "causal", "decision"] as const).map((id) => {
            const layer = result.layers[id];
            return (
              <div key={id}>
                <span>
                  {VALIDATION_LAYER_META[id].title}
                  <small>{scoreWeightPercent(id)}% learned weight</small>
                </span>
                <i>
                  <b
                    className={layer.status}
                    style={{ width: `${Math.max(layer.score, 2)}%` }}
                  />
                </i>
                <strong>{Math.round(layer.score)}</strong>
              </div>
            );
          })}
        </div>
      </div>
      <div className="validation-score-evidence">
        <span className="eyebrow">
          {result.scoreContract.kind === "learned" ? "Learned score · evidence" : "Fallback score · evidence"}
        </span>
        <strong>{result.evidenceGrade}</strong>
        <p>
          {passedGates.length}/{applicableGates.length} applicable gates passed
          {unavailableGates.length
            ? ` · ${unavailableGates.length} not testable`
            : ""}
        </p>
        {result.heuristicScore !== null && result.finalScore !== null && (
          <small className="validation-score-receipt">
            Fixed heuristic {result.heuristicScore.toFixed(1)} · artifact {result.scoreContract.version.replace("flux-score-learner-", "")}
          </small>
        )}
        <div className="validation-gates compact">
          {result.gates.map((gate) => (
            <span
              key={gate.id}
              className={
                !gate.applicable
                  ? "not-applicable"
                  : gate.passed
                    ? "pass"
                    : "fail"
              }
              title={gate.detail}
            >
              {!gate.applicable ? "—" : gate.passed ? "✓" : "×"}{" "}
              {gate.label}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}

function ValidationLayerCard({
  id,
  layer,
  isRunning,
  onExplain,
  onExplore,
}: {
  id: ValidationLayerId;
  layer?: ModelValidationResult["layers"][ValidationLayerId];
  isRunning: boolean;
  onExplain: () => void;
  onExplore: (mode: ValidationChartMode) => void;
}) {
  const meta = VALIDATION_LAYER_META[id];
  const importantTests = layer
    ? [...layer.tests]
        .sort((a, b) => {
          const importance = { critical: 0, high: 1, supporting: 2 };
          return (
            importance[a.importance] - importance[b.importance] ||
            a.score - b.score
          );
        })
        .slice(0, 2)
    : [];
  const weakestTest = layer
    ? [...layer.tests].sort((a, b) => a.score - b.score)[0]
    : undefined;
  const previewMode =
    id === "causal" &&
    layer?.evidence.kind === "causal" &&
    layer.evidence.anchorAssessment.status !== "qualified"
      ? "confounders"
      : primaryDiagnostic(id);
  return (
    <article className={`validation-layer-card ${layer?.status ?? "pending"}`}>
      <div className="validation-layer-top">
        <span className="module-number">{meta.number}</span>
        {layer ? (
          <span className={`validation-status ${layer.status}`}>
            {validationStatusLabel(layer.status)}
          </span>
        ) : (
          <span className={`validation-status ${isRunning ? "running" : "pending"}`}>
            {isRunning ? "Queued" : "Not run"}
          </span>
        )}
      </div>
      <span className="eyebrow">{meta.eyebrow}</span>
      <div className="validation-layer-title">
        <h2>{meta.title}</h2>
        <div className="validation-layer-score">
          <strong>{layer ? Math.round(layer.score) : "—"}</strong>
          <small>/100</small>
        </div>
      </div>
      <p className="validation-layer-summary">
        {layer?.summary ??
          (isRunning
            ? "Waiting for the preceding validation layer."
            : "Run validation to evaluate this layer.")}
      </p>
      {layer ? (
        <>
          <button
            className="validation-preview"
            onClick={() => onExplore(previewMode)}
            aria-label={`Explore ${meta.title} evidence`}
          >
            <ValidationEvidenceChart
              evidence={layer.evidence}
              mode={previewMode}
              height={150}
              compact
              label={`${meta.title} evidence preview`}
            />
          </button>
          <div className="validation-key-metrics">
            {importantTests.map((item) => (
              <button
                key={item.id}
                onClick={() => onExplore(diagnosticForTest(id, item.id))}
              >
                <span><i className={item.status} />{item.name}</span>
                <strong>{item.metric}</strong>
              </button>
            ))}
          </div>
          {weakestTest && (
            <p className="validation-key-finding">
              <span>Key signal</span>
              {weakestTest.name}: {weakestTest.detail}
            </p>
          )}
        </>
      ) : (
        <div className="validation-layer-placeholder">
          {isRunning ? <span className="run-orbit" /> : <b>{meta.number}</b>}
        </div>
      )}
      <div className="validation-layer-actions">
        <button className="advanced-learn" onClick={onExplain}>
          How it works
        </button>
        {layer && (
          <button
            className="validation-explore-button"
            onClick={() => onExplore(previewMode)}
          >
            Explore evidence <span>→</span>
          </button>
        )}
      </div>
    </article>
  );
}

function ValidationEvidenceWorkspace({
  layer,
  evidenceCoherence,
  initialMode,
  onBack,
  onExplain,
}: {
  layer: ValidationLayerResult;
  evidenceCoherence?: ModelValidationResult["evidenceCoherence"];
  initialMode: ValidationChartMode;
  onBack: () => void;
  onExplain: () => void;
}) {
  const [mode, setMode] = useState(initialMode);
  const meta = VALIDATION_LAYER_META[layer.id];
  const diagnostics = VALIDATION_DIAGNOSTICS[layer.id];
  const active =
    diagnostics.find((diagnostic) => diagnostic.mode === mode) ??
    diagnostics[0];
  const issues = [...layer.tests]
    .filter((test) => test.status !== "pass")
    .sort((a, b) => a.score - b.score);
  const anchorAssessment =
    layer.evidence.kind === "causal"
      ? layer.evidence.anchorAssessment
      : undefined;
  const showAnchorAssessment =
    active.mode === "anchors" &&
    anchorAssessment?.status !== "qualified";
  const showEvidenceCoherence =
    active.mode === "coherence" && Boolean(evidenceCoherence);

  return (
    <section className={`validation-evidence-workspace ${layer.status}`}>
      <button className="validation-back" onClick={onBack}>
        ← Validation overview
      </button>
      <div className="validation-workspace-heading">
        <div>
          <span className="kicker">
            Layer {meta.number} · {meta.eyebrow}
          </span>
          <h1>{meta.title} evidence</h1>
          <p>{layer.summary}</p>
        </div>
        <div className="validation-workspace-score">
          <span className={`validation-status ${layer.status}`}>
            {validationStatusLabel(layer.status)}
          </span>
          <strong>{Math.round(layer.score)}</strong>
          <small>/100</small>
        </div>
      </div>
      <div className="validation-diagnostic-tabs" role="tablist">
        {diagnostics.map((diagnostic) => (
          <button
            key={diagnostic.mode}
            className={active.mode === diagnostic.mode ? "active" : ""}
            onClick={() => setMode(diagnostic.mode)}
            role="tab"
            aria-selected={active.mode === diagnostic.mode}
          >
            {diagnostic.label}
            {diagnostic.mode === "anchors" && anchorAssessment && (
              <small>
                {anchorAssessment.status === "qualified"
                  ? `${anchorAssessment.qualifiedCount} qualified`
                  : anchorAssessment.status === "needs-confirmation"
                    ? "Confirm independence"
                    : "Not testable"}
              </small>
            )}
          </button>
        ))}
      </div>
      <div className="validation-evidence-layout">
        <article className="validation-chart-stage">
          <div className="validation-chart-heading">
            <div>
              <span className="eyebrow">Diagnostic evidence</span>
              <h2>{active.label}</h2>
            </div>
            <span>Hover to inspect</span>
          </div>
          {showEvidenceCoherence && evidenceCoherence ? (
            <div className="validation-coherence-receipt">
              <div className="validation-coherence-score">
                <span>
                  Decision score
                  <b>{Math.round(evidenceCoherence.decisionScore ?? layer.score)}</b>
                </span>
                <i>→</i>
                <span>
                  Aligned channels
                  <b>
                    {evidenceCoherence.channels.filter(
                      (channel) =>
                        channel.material && channel.status === "aligned",
                    ).length}
                    /
                    {evidenceCoherence.channels.filter(
                      (channel) => channel.material,
                    ).length}
                  </b>
                </span>
                <p>{evidenceCoherence.summary}</p>
              </div>
              <div className="validation-coherence-table">
                <div className="head">
                  <span>Channel</span>
                  <span>Comparable model ROI</span>
                  <span>Evidence</span>
                  <span>Assessment</span>
                </div>
                {evidenceCoherence.channels
                  .filter((channel) => channel.material)
                  .sort((left, right) => right.spendShare - left.spendShare)
                  .map((channel) => (
                    <div key={channel.channel}>
                      <strong>
                        {cleanChannel(channel.channel)}
                        <small>{Math.round(channel.spendShare * 100)}% spend</small>
                      </strong>
                      <span>
                        <b>{channel.roi.toFixed(2)}×</b>
                        <small>
                          {channel.roiLow.toFixed(2)}–{channel.roiHigh.toFixed(2)}× · {channel.comparisonLabel}
                        </small>
                      </span>
                      <span>
                        <b>
                          {channel.evidenceCenter === undefined
                            ? "Unanchored"
                            : `${channel.evidenceCenter.toFixed(2)}×`}
                        </b>
                        <small>
                          {channel.evidenceLow === undefined || channel.evidenceHigh === undefined
                            ? channel.evidenceLabel
                            : `${channel.evidenceLow.toFixed(2)}–${channel.evidenceHigh.toFixed(2)}× · ${channel.evidenceLabel}`}
                        </small>
                      </span>
                      <span className={`evidence-status ${channel.blocking ? "fail" : channel.status === "aligned" ? "pass" : "review"}`}>
                        <b>{channel.status.replaceAll("-", " ")}</b>
                        <small>{channel.detail}</small>
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          ) : showAnchorAssessment && anchorAssessment ? (
            <div className="anchor-qualification">
              <div className="anchor-qualification-status">
                <span>
                  {anchorAssessment.status === "needs-confirmation"
                    ? "Needs confirmation"
                    : "Not testable"}
                </span>
                <strong>0% score weight</strong>
              </div>
              <h3>
                {anchorAssessment.status === "needs-confirmation"
                  ? "Confirm that the holdout was invisible to the modeling process."
                  : "Holding out this evidence would remove its identifying prior."}
              </h3>
              <p>{anchorAssessment.summary}</p>
              <div className="anchor-criteria">
                {anchorAssessment.criteria.map((criterion) => (
                  <div key={criterion.label}>
                    <i className={criterion.state}>
                      {criterion.state === "pass"
                        ? "✓"
                        : criterion.state === "confirm"
                          ? "?"
                          : "×"}
                    </i>
                    <span>
                      <b>{criterion.label}</b>
                      <small>{criterion.detail}</small>
                    </span>
                  </div>
                ))}
              </div>
              <div className="anchor-reweight-note">
                <span>Score behavior</span>
                <p>
                  External prediction is excluded. Stability, confounder
                  sensitivity, and placebo evidence are reweighted from
                  20/25/15 parts, while the evidence grade is capped at C.
                </p>
              </div>
            </div>
          ) : (
            <ValidationEvidenceChart
              evidence={layer.evidence}
              mode={active.mode}
              height={350}
              label={`${meta.title}: ${active.label}`}
            />
          )}
          <p className="validation-chart-caption">
            <span>How to read it</span>
            {showEvidenceCoherence
              ? "Experiment evidence is evaluated over its registered date window; calibrated industry priors are contract checks and never earn independent validation credit."
              : showAnchorAssessment
              ? "This is an applicability result, not a model failure. The diagnostic activates only after the evidence satisfies the qualification checks."
              : active.description}
          </p>
        </article>
        <aside className="validation-attention-panel">
          <div>
            <span className="eyebrow">Attention queue</span>
            <h2>{issues.length ? `${issues.length} checks to review` : "No material flags"}</h2>
            <p>
              Selecting a check opens the diagnostic that explains its score.
            </p>
          </div>
          <div className="validation-attention-list">
            {(issues.length ? issues : layer.tests.slice(0, 2)).map((test) => (
              <button
                key={test.id}
                className={
                  diagnosticForTest(layer.id, test.id) === active.mode
                    ? "active"
                    : ""
                }
                onClick={() =>
                  setMode(diagnosticForTest(layer.id, test.id))
                }
              >
                <i className={test.status} />
                <span><b>{test.name}</b><small>{test.metric}</small></span>
                <strong>{Math.round(test.score)}</strong>
              </button>
            ))}
          </div>
          <button className="advanced-learn" onClick={onExplain}>
            Read the methodology <span>↗</span>
          </button>
        </aside>
      </div>
      <details className="validation-all-checks">
        <summary>
          <span>All validation checks</span>
          <small>{layer.tests.length} tests · details on demand</small>
        </summary>
        <div className="validation-test-list">
          {layer.tests.map((item) => (
            <button
              key={item.id}
              onClick={() => setMode(diagnosticForTest(layer.id, item.id))}
            >
              <i className={item.status} />
              <p><b>{item.name}</b><small>{item.detail}</small></p>
              <span>{item.metric}</span>
            </button>
          ))}
        </div>
      </details>
    </section>
  );
}

function ValidationView({
  dataset,
  models,
  results,
  statuses,
  progress,
  anchorQualificationAvailable,
  anchorIndependenceConfirmed,
  onAnchorIndependenceChange,
  onRun,
  onRunAll,
  onNavigate,
}: {
  dataset: Dataset;
  models: Partial<Record<ValidationModelKind, ModelResult>>;
  results: Partial<Record<ValidationModelKind, ModelValidationResult>>;
  statuses: Record<ValidationModelKind, JobStatus>;
  progress: Partial<Record<ValidationModelKind, ValidationProgress>>;
  anchorQualificationAvailable: boolean;
  anchorIndependenceConfirmed: boolean;
  onAnchorIndependenceChange: (confirmed: boolean) => void;
  onRun: (kind: ValidationModelKind) => void;
  onRunAll: () => void;
  onNavigate: (view: View) => void;
}) {
  const availableKinds = ([
    "frequentist",
    "bayesian",
    "advanced",
  ] as const).filter((kind) => Boolean(models[kind]));
  const [selectedKind, setSelectedKind] = useState<ValidationModelKind>(
    availableKinds.includes("advanced")
      ? "advanced"
      : availableKinds.includes("bayesian")
        ? "bayesian"
        : "frequentist",
  );
  const [guideTopic, setGuideTopic] =
    useState<ValidationGuideTopic | null>(null);
  const [focusedLayer, setFocusedLayer] =
    useState<ValidationLayerId | null>(null);
  const [focusedMode, setFocusedMode] =
    useState<ValidationChartMode>("forecast");
  const activeKind = availableKinds.includes(selectedKind)
    ? selectedKind
    : (availableKinds[0] ?? "frequentist");
  if (!availableKinds.length) {
    return (
      <div className="view">
        <section className="page-heading compact-heading">
          <div><span className="kicker">Model validation</span><h1>Validation lab</h1><p>Challenge a fitted model across prediction, structure, and causal evidence.</p></div>
        </section>
        <section className="validation-empty card">
          <span>✓</span>
          <h2>Run a model before validating it</h2>
          <p>
            Validation refits the exact selected specification across time,
            spend regimes, experimental anchors, and sensitivity scenarios.
          </p>
          <button className="button primary" onClick={() => onNavigate("models")}>
            Go to Model Studio →
          </button>
        </section>
      </div>
    );
  }

  const selectedResult = results[activeKind];
  const selectedProgress = progress[activeKind];
  const selectedStatus = statuses[activeKind];
  const partialLayers = selectedProgress?.layers ?? {};
  const leaderBoard = availableKinds
    .map((kind) => results[kind])
    .filter((result): result is ModelValidationResult => Boolean(result))
    .sort(
      (a, b) =>
        (b.finalScore ?? -1) - (a.finalScore ?? -1),
    );
  const winner =
    leaderBoard.find((result) => result.eligible) ??
    leaderBoard.find((result) => result.finalScore !== null);
  const focusedResultLayer = focusedLayer
    ? selectedResult?.layers[focusedLayer] ?? partialLayers[focusedLayer]
    : undefined;
  const openEvidence = (
    layer: ValidationLayerId,
    mode: ValidationChartMode,
  ) => {
    setFocusedMode(mode);
    setFocusedLayer(layer);
  };

  return (
    <div className="view validation-view">
      <section className="page-heading compact-heading">
        <div>
          <span className="kicker">Decision-grade review</span>
          <h1>Validation lab</h1>
          <p>
            One model, four complementary challenges: generalization,
            structural adequacy, causal credibility, and ROI decision coherence.
          </p>
        </div>
        <button className="button secondary" onClick={() => setGuideTopic("scoring")}>
          How winner scoring works ↗
        </button>
      </section>

      <section className="validation-control-bar">
        <div>
          <span className="eyebrow">Model under review</span>
          <div className="validation-model-tabs">
            {availableKinds.map((kind) => (
              <button
                key={kind}
                className={activeKind === kind ? "active" : ""}
                onClick={() => {
                  setFocusedLayer(null);
                  setSelectedKind(kind);
                }}
              >
                <span>{kind === "frequentist" ? "F" : kind === "bayesian" ? "B" : "A"}</span>
                {VALIDATION_MODEL_LABELS[kind]}
                {results[kind] && <i className={results[kind]?.eligible ? "pass" : "review"} />}
              </button>
            ))}
          </div>
        </div>
        <div className="validation-run-controls">
          {anchorQualificationAvailable && (
            <label className="validation-independence-toggle">
              <input
                type="checkbox"
                checked={anchorIndependenceConfirmed}
                onChange={(event) =>
                  onAnchorIndependenceChange(event.target.checked)
                }
                disabled={
                  selectedStatus === "running" ||
                  selectedStatus === "queued"
                }
              />
              <span>
                <b>Independent holdout</b>
                <small>No prior or model-selection leakage</small>
              </span>
            </label>
          )}
          <JobPill status={selectedStatus} />
          <button
            className="button secondary"
            onClick={onRunAll}
            disabled={availableKinds.some(
              (kind) =>
                statuses[kind] === "running" || statuses[kind] === "queued",
            )}
          >
            Validate all
          </button>
          <button
            className="button primary"
            onClick={() => onRun(activeKind)}
            disabled={selectedStatus === "running" || selectedStatus === "queued"}
          >
            {selectedResult ? "Run again" : "Run validation"} →
          </button>
        </div>
      </section>

      {(selectedStatus === "running" || selectedStatus === "queued") && (
        <section className="validation-progress-banner">
          <span className="run-orbit" />
          <div>
            <span className="eyebrow">Progressive validation</span>
            <h2>{selectedProgress?.detail ?? "Preparing leakage-safe refits…"}</h2>
            <p>Completed layers render immediately while heavier causal stress tests continue.</p>
          </div>
        </section>
      )}

      {focusedLayer && focusedResultLayer ? (
        <ValidationEvidenceWorkspace
          key={`${activeKind}-${focusedLayer}`}
          layer={focusedResultLayer}
          evidenceCoherence={selectedResult?.evidenceCoherence}
          initialMode={focusedMode}
          onBack={() => setFocusedLayer(null)}
          onExplain={() => setGuideTopic(focusedLayer)}
        />
      ) : (
        <>
          {selectedResult && (
            <ValidationScoreSummary result={selectedResult} />
          )}
          <section className="validation-layer-grid">
            {(["generalization", "structure", "causal", "decision"] as const).map((id) => (
              <ValidationLayerCard
                key={id}
                id={id}
                layer={selectedResult?.layers[id] ?? partialLayers[id]}
                isRunning={
                  selectedStatus === "running" ||
                  selectedStatus === "queued"
                }
                onExplain={() => setGuideTopic(id)}
                onExplore={(mode) => openEvidence(id, mode)}
              />
            ))}
          </section>

          {selectedResult && (
            <section className="validation-ranking-wrap">
              <article className="card validation-ranking-card">
                <div className="card-heading">
                  <div><span className="eyebrow">Comparable evidence</span><h2>Model leaderboard</h2></div>
                  {winner && (
                    <span className="winner-tag">
                      {winner.eligible ? "Eligible winner" : "Top score · not eligible"} ·{" "}
                      {VALIDATION_MODEL_LABELS[winner.modelKind]}
                    </span>
                  )}
                </div>
                <div className="validation-ranking">
                  {leaderBoard.map((result, index) => (
                    <button
                      key={result.modelKind}
                      onClick={() => {
                        setFocusedLayer(null);
                        setSelectedKind(result.modelKind);
                      }}
                    >
                      <span>{index + 1}</span>
                      <p><b>{VALIDATION_MODEL_LABELS[result.modelKind]}</b><small>{result.recommendation}</small></p>
                      <div>
                        {(["generalization", "structure", "causal", "decision"] as const).map((id) => (
                          <i key={id} title={`${VALIDATION_LAYER_META[id].title}: ${Math.round(result.layers[id].score)}`} style={{ height: `${Math.max(result.layers[id].score, 5)}%` }} />
                        ))}
                      </div>
                      <strong>{result.finalScore === null ? "—" : Math.round(result.finalScore)}</strong>
                    </button>
                  ))}
                  {leaderBoard.length < availableKinds.length && (
                    <p className="validation-ranking-note">
                      Validate the remaining fitted models to complete the winner comparison.
                    </p>
                  )}
                </div>
              </article>
            </section>
          )}
        </>
      )}

      {guideTopic && (
        <ValidationGuide
          dataset={dataset}
          topic={guideTopic}
          setTopic={setGuideTopic}
          onClose={() => setGuideTopic(null)}
        />
      )}
    </div>
  );
}

type AgenticWorkspaceStatus = "idle" | "running" | "complete";
type AgenticWorkspacePanel = "contract" | "search" | "winner";

function AgenticSearchConfidencePanel({
  confidence,
}: {
  confidence: ReturnType<typeof agenticSearchConfidence>;
}) {
  const plotted = confidence.frontier.filter(
    (point): point is typeof point & { score: number } =>
      point.score !== null,
  );
  const maximumEvaluation = Math.max(
    ...confidence.frontier.map((point) => point.evaluation),
    1,
  );
  const chartPoints = plotted
    .map((point) => {
      const x = 12 + (point.evaluation / maximumEvaluation) * 276;
      const y = 78 - (point.score / 100) * 64;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <section className={`card agentic-confidence-card ${confidence.level}`}>
      <div className="agentic-confidence-heading">
        <div>
          <span className="eyebrow">Search confidence</span>
          <h2>
            {confidence.level === "not-established"
              ? "Best-known model is not yet stable"
              : `${confidence.score}/100 · ${confidence.level} confidence`}
          </h2>
          <p>{confidence.summary}</p>
        </div>
        <span className="agentic-confidence-badge">
          {confidence.level.replace("-", " ")}
        </span>
      </div>
      <div className="agentic-confidence-body">
        <div className="agentic-convergence-chart">
          <div>
            <b>Eligible frontier</b>
            <small>Best score discovered by evaluation</small>
          </div>
          <svg viewBox="0 0 300 90" role="img" aria-label="Best eligible score over evaluated candidates">
            <line x1="12" y1="14" x2="12" y2="78" />
            <line x1="12" y1="78" x2="288" y2="78" />
            {chartPoints && <polyline points={chartPoints} />}
            {!chartPoints && (
              <text x="150" y="48" textAnchor="middle">
                No eligible frontier yet
              </text>
            )}
          </svg>
        </div>
        <div className="agentic-confidence-metrics">
          <div>
            <span>Search-space coverage</span>
            <b>{Math.round(confidence.structuralCoverage * 100)}%</b>
            <small>Families, channel response, Advanced assumptions</small>
          </div>
          <div>
            <span>Restart agreement</span>
            <b>
              {confidence.level === "not-established"
                ? "—"
                : `${Math.round(confidence.restartAgreement * 100)}%`}
            </b>
            <small>{confidence.restartCount}/3 eligible starts</small>
          </div>
          <div>
            <span>Local challenges</span>
            <b>{confidence.localChallengeCount}</b>
            <small>{confidence.localImprovements} improved the prior champion</small>
          </div>
          <div>
            <span>Frontier plateau</span>
            <b>
              {confidence.level === "not-established"
                ? "—"
                : confidence.evaluationsSinceImprovement}
            </b>
            <small>
              {confidence.level === "not-established"
                ? "No eligible frontier exists"
                : "trials since a ≥0.25 point gain"}
            </small>
          </div>
        </div>
      </div>
      <div className="agentic-restart-strip">
        {confidence.restartWinners.map((winner) => (
          <div key={winner.restart}>
            <span>Start {winner.restart}</span>
            <b>{winner.candidateId ?? "No eligible model"}</b>
            <small>
              {winner.score === undefined
                ? "—"
                : `${winner.score.toFixed(1)} · ${winner.family}`}
            </small>
          </div>
        ))}
      </div>
      {confidence.boundaryParameters.length > 0 && (
        <p className="agentic-confidence-warning">
          <i>!</i>
          Winning parameters touch declared bounds: {confidence.boundaryParameters.join(", ")}.
        </p>
      )}
    </section>
  );
}

function AgenticView({
  dataset,
  contract,
  setContract,
  config,
  advancedConfig,
  experiments,
  industryPriorChannels,
  benchmarkScreeningEnabled,
  runs,
  status,
  stopReason,
  promotedFingerprint,
  onStart,
  onInspect,
  onPromote,
  onForcePromote,
}: {
  dataset: Dataset;
  contract: AgenticSearchContract;
  setContract: (contract: AgenticSearchContract) => void;
  config: ModelConfig;
  advancedConfig: AdvancedModelConfig;
  experiments: Experiment[];
  industryPriorChannels: string[];
  benchmarkScreeningEnabled: boolean;
  runs: AgenticCandidateRun[];
  status: AgenticWorkspaceStatus;
  stopReason?: string;
  promotedFingerprint?: string;
  onStart: () => void;
  onInspect: (run: AgenticCandidateRun) => void;
  onPromote: (run: AgenticCandidateRun) => void;
  onForcePromote: (run: AgenticCandidateRun) => void;
}) {
  const [panel, setPanel] = useState<AgenticWorkspacePanel>(
    status === "idle" ? "contract" : status === "running" ? "search" : "winner",
  );
  const [inspectedSpecification, setInspectedSpecification] =
    useState<AgenticCandidateRun>();
  const [candidateFilter, setCandidateFilter] = useState<
    "all" | ValidationModelKind
  >("all");
  const [confirmForcePromotion, setConfirmForcePromotion] = useState(false);
  const candidateCount = contract.candidateBudget;
  const seedCount = agenticSeedBudget(contract);
  const advancedChallengeCount = agenticAdvancedChallengeBudget(contract);
  const searchCapabilities = {
    likelihoodCalibration: experiments.length > 0,
    mediaColumns: dataset.mediaColumns,
  };
  const adaptiveMinimum = agenticAdaptiveMinimumBudget(contract, searchCapabilities);
  const adaptiveMaximum = agenticAdaptiveMaximumBudget(contract, searchCapabilities);
  const localChallengeBudget = agenticLocalChallengeBudget(contract);
  const responseChallengeCount = agenticChannelResponseBudget(
    contract,
    searchCapabilities,
  );
  const scoredRuns = runs.map((run) =>
    run.model
      ? {
          ...run,
          roiGuardrailViolations: findAgenticRoiGuardrailViolations(
            run.model,
            experiments,
            benchmarkScreeningEnabled,
            dataset,
          ),
        }
      : run,
  );
  const responseCoverage = agenticChannelResponseCoverage(
    scoredRuns,
    contract,
    searchCapabilities,
  );
  const completed = scoredRuns.filter(
    (run) => run.state === "complete" && run.spec.searchPhase !== "rescue",
  ).length;
  const rescueCount = scoredRuns.filter(
    (run) => run.state === "complete" && run.spec.searchPhase === "rescue",
  ).length;
  const errors = scoredRuns.filter(
    (run) => run.state === "error" && run.spec.searchPhase !== "rescue",
  ).length;
  const gateEligible = scoredRuns.filter(
    passesAgenticEligibility,
  ).length;
  const restored = scoredRuns.filter((run) => run.restoredFromCache).length;
  const ranked = rankAgenticCandidates(scoredRuns);
  const champion = selectAgenticWinner(scoredRuns);
  const comparisonLeader = champion ?? ranked[0];
  const comparisonIndustryPriorChannels = comparisonLeader
    ? comparisonLeader.spec.family === "frequentist"
      ? []
      : comparisonLeader.spec.evidencePriorChannels ?? []
    : industryPriorChannels;
  const forceFailedGates = comparisonLeader?.validation?.gates.filter(
    (gate) => gate.applicable && !gate.passed,
  ) ?? [];
  const forceRoiViolations = comparisonLeader?.roiGuardrailViolations ?? [];
  const boundaryParameters = comparisonLeader
    ? agenticBoundaryParameters(comparisonLeader.spec)
    : [];
  const adaptiveCount = scoredRuns.filter(
    (run) => run.spec.searchPhase === "adaptive",
  ).length;
  const advancedCoverage = agenticAdvancedCoverage(
    scoredRuns,
    contract,
    searchCapabilities,
  );
  const searchConfidence = agenticSearchConfidence(
    scoredRuns,
    contract,
    searchCapabilities,
  );
  const activeRun = scoredRuns.find((run) => run.state === "running");
  const searchStage =
    status === "complete"
      ? "select"
      : agenticSearchStage(
          completed + errors,
          candidateCount,
          seedCount,
          advancedChallengeCount,
          localChallengeBudget,
          responseChallengeCount,
        );
  const activeFamilies = (
    ["frequentist", "bayesian", "advanced"] as const
  ).filter((family) => contract.families[family]);
  const candidateRows = scoredRuns.length
    ? scoredRuns
    : generateAgenticSeeds(
        config,
        advancedConfig,
        contract,
        searchCapabilities,
      ).map((spec): AgenticCandidateRun => ({
        spec,
        state: "queued" as const,
      }));
  const visibleCandidateRows = candidateRows.filter(
    (run) => candidateFilter === "all" || run.spec.family === candidateFilter,
  );

  useEffect(() => {
    const nextPanel =
      status === "running"
        ? "search"
        : status === "complete"
          ? "winner"
          : undefined;
    if (!nextPanel) return;
    const frame = window.requestAnimationFrame(() => setPanel(nextPanel));
    return () => window.cancelAnimationFrame(frame);
  }, [status]);

  const toggleFamily = (family: ValidationModelKind) => {
    const nextFamilies = {
      ...contract.families,
      [family]: !contract.families[family],
    };
    if (!Object.values(nextFamilies).some(Boolean)) return;
    setContract({ ...contract, families: nextFamilies });
  };

  return (
    <div className="view agentic-view">
      <section className="page-heading compact-heading agentic-heading">
        <div>
          <span className="kicker">Objective model search</span>
          <h1>Agentic modeler</h1>
          <p>
            Find the best-known eligible specification through independent
            starts, feasibility-aware refinement, and local champion challenges.
          </p>
        </div>
        <span className="agentic-protocol-pill">Global channel search · active V6 score</span>
      </section>

      <section className="agentic-workspace-tabs" aria-label="Agentic search stages">
        {([
          ["contract", "1", "Search contract"],
          ["search", "2", "Live search"],
          ["winner", "3", "Winner"],
        ] as const).map(([id, number, label]) => (
          <button
            key={id}
            className={panel === id ? "active" : ""}
            onClick={() => setPanel(id)}
            disabled={id === "winner" && !ranked.length}
          >
            <span>{number}</span>
            {label}
          </button>
        ))}
      </section>

      {panel === "contract" && (
        <section className="agentic-contract-layout">
          <article className="card agentic-contract-card">
            <div className="card-heading">
              <div>
                <span className="eyebrow">Searchable space</span>
                <h2>Model families</h2>
              </div>
              <span className="subtle">{candidateCount} candidates</span>
            </div>
            <p className="agentic-card-copy">
              Cover every model structure, refine several promising basins
              across three starts, then challenge the champion locally.
            </p>
            <div className="agentic-family-grid">
              {([
                ["frequentist", "F", "Naive reference"],
                ["bayesian", "B", "Calibrated core"],
                ["advanced", "A", "Opt-in complexity"],
              ] as const).map(([family, glyph, detail]) => (
                <button
                  key={family}
                  className={contract.families[family] ? "selected" : ""}
                  onClick={() => toggleFamily(family)}
                  disabled={status === "running"}
                  aria-pressed={contract.families[family]}
                >
                  <span>{glyph}</span>
                  <p>
                    <b>{agenticFamilyLabel(family)}</b>
                    <small>{detail}</small>
                  </p>
                  <i>{contract.families[family] ? "Included" : "Off"}</i>
                </button>
              ))}
            </div>
            <div className="agentic-budget-control">
              <div>
                <b>Candidate budget</b>
                <small>
                  {seedCount} global seeds
                  {responseChallengeCount
                    ? ` + ${responseChallengeCount} channel-response grid challenges`
                    : ""}
                  {advancedChallengeCount
                    ? ` + ${advancedChallengeCount} paired Advanced challenges`
                    : ""}
                  {adaptiveMaximum > 0
                    ? ` + ${adaptiveMinimum} restart-balanced refinements${adaptiveMaximum > adaptiveMinimum ? ` + ${adaptiveMaximum - adaptiveMinimum} additional feasibility-led proposals` : ""}`
                    : "."}
                  {localChallengeBudget
                    ? ` + ${localChallengeBudget} champion-neighborhood challenges.`
                    : ""}
                </small>
              </div>
              <div className="segmented-control">
                {([96, 192, 384] as const).map((budget) => (
                  <button
                    key={budget}
                    className={
                      contract.candidateBudget === budget ? "active" : ""
                    }
                    onClick={() =>
                      setContract({ ...contract, candidateBudget: budget })
                    }
                    disabled={status === "running"}
                  >
                    {budget}
                  </button>
                ))}
              </div>
            </div>
          </article>

          <article className="card agentic-contract-card locked">
            <div className="card-heading">
              <div>
                <span className="eyebrow">Immutable contract</span>
                <h2>What the agent cannot change</h2>
              </div>
              <span className="agentic-lock">◆ Locked</span>
            </div>
            <div className="agentic-contract-list">
              <div>
                <span>Objective</span>
                <b>
                  {ACTIVE_SCORE_CONTRACT.kind === "learned"
                    ? "Learned diagnostic decision-loss score · V6"
                    : "Audited heuristic fallback"}
                </b>
              </div>
              <div>
                <span>Ranking weights</span>
                <b>{activeScoreFormula().replace("100 × ", "")}</b>
              </div>
              <div>
                <span>ROI plausibility</span>
                <b>Two-sided coherence gate · immutable</b>
              </div>
              <div>
                <span>Experiments</span>
                <b>
                  {experiments.length} fixed stud
                  {experiments.length === 1 ? "y" : "ies"}
                </b>
              </div>
              <div>
                <span>Industry guardrails</span>
                <b>
                  {industryPriorChannels.length
                    ? industryPriorChannels
                        .map((channel) => cleanChannel(channel))
                        .join(", ")
                    : "None active"}
                </b>
              </div>
              <div>
                <span>Decision-grade rule</span>
                <b>Score ≥75 + qualified anchor + gates</b>
              </div>
              <div>
                <span>Evidence challenge</span>
                <b>Paired refit for material implausible channels</b>
              </div>
              <div>
                <span>Selection rule</span>
                <b>
                  Highest evidence-coherent score passing gates
                  {benchmarkScreeningEnabled ? " + ROI screen" : ""}
                </b>
              </div>
              <div>
                <span>ROI plausibility</span>
                <b>
                  {benchmarkScreeningEnabled
                    ? "Material unanchored estimates outside P05–P95 require review"
                    : "Off in Calibration"}
                </b>
              </div>
              <div>
                <span>Parameter domain</span>
                <b>Predeclared bounds · immutable during search</b>
              </div>
              <div>
                <span>Advanced coverage</span>
                <b>
                  {advancedChallengeCount
                    ? `${advancedChallengeCount} paired challengers required before convergence`
                    : "Advanced family excluded"}
                </b>
              </div>
              <div>
                <span>Channel-response coverage</span>
                <b>
                  {responseChallengeCount} covering-array candidates across{" "}
                  {dataset.mediaColumns.length} channels
                </b>
              </div>
              <div>
                <span>Adaptive coverage</span>
                <b>
                  {adaptiveMinimum
                    ? `${adaptiveMinimum} minimum refinements · 3 independent starts`
                    : "Coverage screen only · no adaptive refinement"}
                </b>
              </div>
              <div>
                <span>Local optimality</span>
                <b>
                  {localChallengeBudget
                    ? `${localChallengeBudget} global + channel-level champion perturbations reserved`
                    : "No local refinement reserved"}
                </b>
              </div>
              <div>
                <span>Inference engine</span>
                <b>Fast analytic MAP/Laplace · no sampling</b>
              </div>
            </div>
            <button
              className="button primary full"
              onClick={() => {
                setConfirmForcePromotion(false);
                onStart();
              }}
              disabled={status === "running" || !activeFamilies.length}
            >
              {status === "complete" ? "Start a fresh search" : "Start objective search"} →
            </button>
          </article>
        </section>
      )}

      {panel === "search" && (
        <>
          <section className="agentic-stat-grid">
            <article className="card">
              <span>Evaluated</span>
              <strong>
                {completed + errors} / {candidateCount}
              </strong>
              <small>
                {restored} restored from cache
                {rescueCount ? ` · ${rescueCount} evidence rescue${rescueCount === 1 ? "" : "s"}` : ""}
              </small>
            </article>
            <article className="card">
              <span>Eligible</span>
              <strong>{gateEligible}</strong>
              <small>Validation + ROI plausibility</small>
            </article>
            <article className="card">
              <span>Current leader</span>
              <strong>
                {comparisonLeader?.validation?.finalScore === null ||
                comparisonLeader?.validation?.finalScore === undefined
                  ? "—"
                  : comparisonLeader.validation.finalScore.toFixed(1)}
              </strong>
              <small>
                {comparisonLeader?.validation
                  ? `Evidence grade ${comparisonLeader.validation.evidenceGrade}`
                  : "Awaiting first score"}
              </small>
            </article>
            <article className="card">
              <span>Search confidence</span>
              <strong>
                {searchConfidence.level === "not-established"
                  ? "—"
                  : searchConfidence.score}
              </strong>
              <small>
                {searchConfidence.level === "not-established"
                  ? "Awaiting eligible restart winners"
                  : `${searchConfidence.level} · ${searchConfidence.restartCount}/3 starts`}
              </small>
            </article>
          </section>

          <section className="agentic-pipeline" aria-label="Agentic search progress">
            {([
              ["seed", "Screen", "Balanced family coverage"],
              ["response", "Responses", "Per-channel covering array"],
              ["advanced", "Challenge", "Paired Advanced coverage"],
              ["adaptive", "Refine", "3-start constrained search"],
              ["local", "Polish", "Champion neighborhood"],
              ["select", "Select", "Rank candidates"],
            ] as const).map(([id, label, detail]) => {
              const order = ["seed", "response", "advanced", "adaptive", "local", "select"];
              const stateIndex = order.indexOf(searchStage);
              const itemIndex = order.indexOf(id);
              return (
                <div
                  key={id}
                  className={
                    itemIndex < stateIndex
                      ? "complete"
                      : itemIndex === stateIndex
                        ? "active"
                        : ""
                  }
                >
                  <i />
                  <b>{label}</b>
                  <small>{detail}</small>
                </div>
              );
            })}
          </section>

          {responseChallengeCount > 0 && (
            <details
              className={`card agentic-coverage-card ${responseCoverage.complete ? "complete" : "running"}`}
            >
              <summary>
                <div>
                  <span className="eyebrow">Mandatory channel-response grid</span>
                  <h2>
                    {responseCoverage.completedChallenges}/
                    {responseCoverage.requiredChallenges} response challengers
                  </h2>
                  <p>
                    Every material media channel receives distinct carryover,
                    saturation, half-saturation, and normalization challenges.
                  </p>
                </div>
                <div className="agentic-coverage-progress">
                  <span><i style={{ width: `${Math.min(100, responseCoverage.completedChallenges / Math.max(responseCoverage.requiredChallenges, 1) * 100)}%` }} /></span>
                  <b>{responseCoverage.complete ? "Coverage complete ✓" : "Convergence locked"}</b>
                </div>
              </summary>
              <div className="agentic-coverage-grid">
                {responseCoverage.channels.map((channel) => (
                  <div key={channel.channel} className={channel.complete ? "complete" : "pending"}>
                    <span>{cleanChannel(channel.channel)}</span>
                    <b>{channel.adstockFamilies.join(" · ") || "Awaiting response tests"}</b>
                    <small>
                      {channel.complete
                        ? "Carryover · saturation · half-saturation · normalization covered"
                        : "Required response profiles remain incomplete"}
                    </small>
                  </div>
                ))}
              </div>
            </details>
          )}

          {advancedChallengeCount > 0 && (
            <details
              className={`card agentic-coverage-card ${advancedCoverage.complete ? "complete" : "running"}`}
            >
              <summary>
                <div>
                  <span className="eyebrow">Mandatory Advanced challenge</span>
                  <h2>
                    {advancedCoverage.completedChallenges}/
                    {advancedCoverage.requiredChallenges} paired challengers
                  </h2>
                  <p>
                    The strongest base response is held constant while Advanced
                    assumptions receive a controlled, auditable challenge.
                  </p>
                </div>
                <div className="agentic-coverage-progress">
                  <span>
                    <i
                      style={{
                        width: `${Math.min(
                          100,
                          (advancedCoverage.completedChallenges /
                            Math.max(advancedCoverage.requiredChallenges, 1)) *
                            100,
                        )}%`,
                      }}
                    />
                  </span>
                  <b>
                    {advancedCoverage.complete
                      ? "Coverage complete ✓"
                      : "Convergence locked"}
                  </b>
                </div>
              </summary>
              <div className="agentic-coverage-grid">
                {advancedCoverage.dimensions.map((dimension) => (
                  <div
                    key={dimension.id}
                    className={dimension.complete ? "complete" : "pending"}
                  >
                    <span>{dimension.label}</span>
                    <b>{dimension.tested.join(" · ") || "Awaiting test"}</b>
                    <small>
                      {dimension.complete
                        ? "Required levels covered"
                        : `Needs ${dimension.required
                            .filter(
                              (value) => !dimension.tested.includes(value),
                            )
                            .join(", ")}`}
                    </small>
                  </div>
                ))}
              </div>
            </details>
          )}

          <section className="card agentic-leaderboard-card">
            <div className="card-heading">
              <div>
                <span className="eyebrow">Objective leaderboard</span>
                <h2>Candidate evidence</h2>
              </div>
              <div className="agentic-leaderboard-actions">
                <div className="agentic-family-filter" aria-label="Filter candidates by model family">
                  {(["all", ...activeFamilies] as const).map((family) => (
                    <button
                      key={family}
                      className={candidateFilter === family ? "active" : ""}
                      onClick={() => setCandidateFilter(family)}
                    >
                      {family === "all" ? "All" : agenticFamilyLabel(family)}
                      <span>
                        {family === "all"
                          ? candidateRows.length
                          : candidateRows.filter(
                              (run) => run.spec.family === family,
                            ).length}
                      </span>
                    </button>
                  ))}
                </div>
                <JobPill
                  status={
                    status === "running"
                      ? "running"
                      : status === "complete"
                        ? "complete"
                        : "idle"
                  }
                />
              </div>
            </div>
            <div className="agentic-leaderboard">
              <div className="agentic-candidate-row head">
                <span>Candidate</span>
                <span>Agent hypothesis</span>
                <span>G</span>
                <span>S</span>
                <span>C</span>
                <span>D</span>
                <span>Score</span>
                <span>Gates</span>
              </div>
              {visibleCandidateRows.map((run) => {
                const result = run.validation;
                const passes = passesAgenticEligibility(run);
                const roiReview = run.roiGuardrailViolations?.[0];
                const evidenceReview = result?.evidenceCoherence;
                return (
                  <div
                    className={`agentic-candidate-row ${run.state}`}
                    key={run.spec.id}
                    role="button"
                    tabIndex={0}
                    aria-label={`View all parameters for ${run.spec.label}`}
                    onClick={() => setInspectedSpecification(run)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        setInspectedSpecification(run);
                      }
                    }}
                  >
                    <strong>
                      <i>{run.spec.id}</i>
                      <span>
                        {run.spec.label}
                        <small>{run.spec.summary}</small>
                        <em>View specification →</em>
                      </span>
                    </strong>
                    <p>{run.spec.hypothesis}</p>
                    {(["generalization", "structure", "causal", "decision"] as const).map(
                      (layer) => (
                        <span key={layer} className="agentic-layer-score">
                          {result
                            ? Math.round(result.layers[layer].score)
                            : run.state === "running"
                              ? "…"
                              : "—"}
                        </span>
                      ),
                    )}
                    <span className="agentic-final-score">
                      {result?.finalScore === null ||
                      result?.finalScore === undefined
                        ? run.state === "running"
                          ? "Running"
                          : "—"
                        : result.finalScore.toFixed(1)}
                    </span>
                    <span
                      className={`agentic-gate-pill ${
                        result ? (passes ? "pass" : "review") : run.state
                      }`}
                      title={
                        roiReview
                          ? `${cleanChannel(roiReview.channel)} ROI ${roiReview.roi.toFixed(2)}× is ${roiReview.tail === "low" ? "below P05" : "above P95"} for ${roiReview.benchmarkLabel}.`
                          : evidenceReview?.summary
                            ? evidenceReview.summary
                          : undefined
                      }
                    >
                      {run.state === "running"
                        ? "Testing"
                        : run.state === "error"
                          ? "Error"
                          : result
                            ? passes
                              ? "Pass"
                              : evidenceReview?.blockingChannels.length
                                ? "Evidence fail"
                                : roiReview
                                  ? "ROI review"
                              : "Review"
                            : "Queued"}
                    </span>
                  </div>
                );
              })}
              {!visibleCandidateRows.length && (
                <p className="agentic-filter-empty">
                  No {candidateFilter === "all" ? "" : agenticFamilyLabel(candidateFilter)} candidates have been queued yet.
                </p>
              )}
            </div>
          </section>

          <section className="agentic-next-grid">
            <article className="card">
              <span className="eyebrow">Agent reasoning</span>
              <h2>{activeRun ? `Testing ${activeRun.spec.label}` : status === "complete" ? "Search complete" : "Ready to search"}</h2>
              <p>
                {activeRun?.progress?.detail ??
                  activeRun?.spec.hypothesis ??
                  (status === "complete"
                    ? champion
                      ? "The highest-scoring gate-passing candidate is ready for review."
                      : "No candidate passed every validation and ROI plausibility check. Review the strongest challenger."
                    : "Approve the search contract to begin.")}
              </p>
            </article>
            <article className="card">
              <span className="eyebrow">Stopping rule</span>
              <h2>Up to {contract.candidateBudget} bounded candidates</h2>
              <p>
                {stopReason ??
                  `The search completes ${seedCount} global seeds, ${responseChallengeCount} channel-response grid challenges, ${advancedChallengeCount} Advanced structure challenges, ${adaptiveMaximum} feasibility-aware proposals, and ${localChallengeBudget} local champion tests before ranking the best-known eligible model.`}
              </p>
            </article>
          </section>
        </>
      )}

      {panel === "winner" && (
        <section className="agentic-winner-workspace">
          {comparisonLeader ? (
            <>
              <div className="agentic-winner-grid">
                <article className="card agentic-champion-card">
                  <div className="card-heading">
                    <div>
                      <span className="eyebrow">
                        {champion
                          ? "Best-known eligible specification"
                          : "Strongest candidate · gates unresolved"}
                      </span>
                      <h2>{comparisonLeader.spec.label}</h2>
                    </div>
                    <span
                      className={`agentic-gate-pill ${
                        champion ? "pass" : "review"
                      }`}
                    >
                      {champion ? "Eligible" : "Review"}
                    </span>
                  </div>
                  <div className="agentic-champion-score">
                    <strong>
                      {comparisonLeader.validation?.finalScore?.toFixed(1) ??
                        "—"}
                    </strong>
                    <span>
                      Flux score
                      <small>
                        Evidence grade{" "}
                        {comparisonLeader.validation?.evidenceGrade ?? "—"}
                      </small>
                    </span>
                  </div>
                  <p>{comparisonLeader.spec.summary}</p>
                  <div className="agentic-champion-spec">
                    <span>{comparisonLeader.spec.config.adstockType} adstock</span>
                    {Object.keys(
                      comparisonLeader.spec.config.channelResponses ?? {},
                    ).length > 0 && (
                      <span>
                        {Object.keys(
                          comparisonLeader.spec.config.channelResponses ?? {},
                        ).length}{" "}
                        channel response profiles
                      </span>
                    )}
                    <span>
                      Hill {comparisonLeader.spec.config.saturation.toFixed(2)}
                    </span>
                    <span>
                      {comparisonLeader.spec.config.fourierOrder} Fourier pairs
                    </span>
                    {comparisonLeader.spec.family === "advanced" && (
                      <span>
                        {comparisonLeader.spec.advancedConfig
                          .likelihoodDistribution}{" "}
                        likelihood
                      </span>
                    )}
                    {comparisonLeader.spec.family !== "advanced" && (
                      <span title="Time-varying coefficients, planning intensity, and selectable distributions were evaluated only in competing Advanced candidates.">
                        Base estimator · Advanced knobs inactive
                      </span>
                    )}
                    <span
                      className={boundaryParameters.length ? "boundary" : ""}
                      title={
                        boundaryParameters.length
                          ? `${boundaryParameters.join(", ")} reached a declared search limit.`
                          : undefined
                      }
                    >
                      {boundaryParameters.length
                        ? `${boundaryParameters.length} parameter bound${boundaryParameters.length === 1 ? "" : "s"} active`
                        : "Interior solution"}
                    </span>
                  </div>
                </article>
                <article className="card agentic-why-card">
                  <span className="eyebrow">Why it won</span>
                  <h2>Objective selection receipt</h2>
                  <div>
                    <p>
                      <i>{champion ? "✓" : "!"}</i>
                      <span>
                        <b>{champion ? "Highest eligible score" : "Highest observed score"}</b>
                        <small>
                          {champion
                            ? "Ranked after validation, material-channel evidence, and ROI plausibility gates were evaluated. Benchmark agreement used in calibration earned no extra points."
                            : `No candidate passed every eligibility check. This candidate retains ${forceFailedGates.length} failed validation gate${forceFailedGates.length === 1 ? "" : "s"} and ${forceRoiViolations.length} ROI plausibility review${forceRoiViolations.length === 1 ? "" : "s"}.`}
                        </small>
                      </span>
                    </p>
                    <p>
                      <i>✓</i>
                      <span>
                        <b>Locked evidence</b>
                        <small>
                          {experiments.length} experiments and{" "}
                          {comparisonIndustryPriorChannels.length} benchmark guardrail
                          {comparisonIndustryPriorChannels.length === 1 ? "" : "s"}
                          {" "}were unchanged.
                        </small>
                      </span>
                    </p>
                    {advancedChallengeCount > 0 && (
                      <p>
                        <i>{advancedCoverage.complete ? "✓" : "△"}</i>
                        <span>
                          <b>Advanced challenge coverage</b>
                          <small>
                            {advancedCoverage.completedChallenges}/
                            {advancedCoverage.requiredChallenges} paired
                            challengers completed across dynamic effects,
                            planning, calibration, and distributions.
                          </small>
                        </span>
                      </p>
                    )}
                    <p>
                      <i>→</i>
                      <span>
                        <b>Winning hypothesis</b>
                        <small>{comparisonLeader.spec.hypothesis}</small>
                      </span>
                    </p>
                    <p>
                      <i>↗</i>
                      <span>
                        <b>Adaptive convergence</b>
                        <small>
                          {completed} models evaluated · {adaptiveCount} TPE
                          proposals across three starts · {searchConfidence.localChallengeCount}{" "}
                          local challenges. {stopReason}
                        </small>
                      </span>
                    </p>
                  </div>
                </article>
              </div>

              <AgenticSearchConfidencePanel confidence={searchConfidence} />

              <section className="card agentic-finalists-card">
                <div className="card-heading">
                  <div>
                    <span className="eyebrow">Final comparison</span>
                    <h2>Champion and challengers</h2>
                  </div>
                  <span className="subtle">Same validation contract</span>
                </div>
                <div className="agentic-finalists">
                  <div className="agentic-finalist-row head">
                    <span>Candidate</span>
                    <span>G</span>
                    <span>S</span>
                    <span>C</span>
                    <span>D</span>
                    <span>Score</span>
                    <span>Decision</span>
                  </div>
                  {ranked.slice(0, 4).map((run, index) => {
                    const result = run.validation!;
                    const passes = passesAgenticEligibility(run);
                    const roiReview = run.roiGuardrailViolations?.[0];
                    return (
                      <div className="agentic-finalist-row" key={run.spec.id}>
                        <strong>
                          <i>{index + 1}</i>
                          <span>
                            {run.spec.label}
                            <small>{run.spec.summary}</small>
                          </span>
                        </strong>
                        <span>{Math.round(result.layers.generalization.score)}</span>
                        <span>{Math.round(result.layers.structure.score)}</span>
                        <span>{Math.round(result.layers.causal.score)}</span>
                        <span>{Math.round(result.layers.decision.score)}</span>
                        <b>
                          {result.finalScore === null
                            ? "—"
                            : result.finalScore.toFixed(1)}
                        </b>
                        <span
                          className={`agentic-gate-pill ${
                            passes ? "pass" : "review"
                          }`}
                        >
                          {run === champion
                            ? "Selected"
                            : passes
                              ? "Runner-up"
                              : result.evidenceCoherence.blockingChannels.length
                                ? "Evidence fail"
                                : roiReview
                                  ? "ROI review"
                              : "Gate review"}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </section>

              <div className="agentic-winner-actions">
                <button
                  className="button secondary"
                  onClick={() => setInspectedSpecification(comparisonLeader)}
                >
                  View all {specificationParameterCount(comparisonLeader)} parameters
                </button>
                <button
                  className="button secondary"
                  onClick={() => onInspect(comparisonLeader)}
                >
                  Inspect validation
                </button>
                {champion ? (
                  <button
                    className="button primary"
                    onClick={() => onPromote(comparisonLeader)}
                  >
                    {promotedFingerprint ===
                    comparisonLeader.model?.fingerprint
                      ? "Winning specification promoted ✓"
                      : comparisonLeader.spec.family === "advanced"
                        ? "Promote & open Advanced"
                        : "Promote & open Models"}{" "}
                    →
                  </button>
                ) : (
                  <button
                    className="button force-promote"
                    onClick={() => setConfirmForcePromotion(true)}
                    disabled={promotedFingerprint === comparisonLeader.model?.fingerprint}
                  >
                    {promotedFingerprint === comparisonLeader.model?.fingerprint
                      ? "Forced specification promoted ✓"
                      : "Force promote strongest candidate →"}
                  </button>
                )}
              </div>
              {!champion && confirmForcePromotion && (
                <section className="agentic-force-confirm" role="alert">
                  <span>!</span>
                  <div>
                    <b>Promote with unresolved evidence?</b>
                    <p>
                      This preserves the {comparisonLeader.validation?.finalScore?.toFixed(1) ?? "—"}
                      {" "}score, {forceFailedGates.length} failed validation gate
                      {forceFailedGates.length === 1 ? "" : "s"}, and{" "}
                      {forceRoiViolations.length} ROI plausibility review
                      {forceRoiViolations.length === 1 ? "" : "s"}. The model
                      will be marked as a forced override and Production will
                      remain “Needs attention.”
                    </p>
                  </div>
                  <div>
                    <button
                      className="button secondary"
                      onClick={() => setConfirmForcePromotion(false)}
                    >
                      Cancel
                    </button>
                    <button
                      className="button force-promote confirm"
                      onClick={() => {
                        setConfirmForcePromotion(false);
                        onForcePromote(comparisonLeader);
                      }}
                    >
                      Confirm force promotion
                    </button>
                  </div>
                </section>
              )}
            </>
          ) : (
            <article className="card validation-empty">
              <span>✦</span>
              <h2>No completed search yet</h2>
              <p>
                Define the search contract and run the candidate set before
                selecting a winner.
              </p>
              <button
                className="button primary"
                onClick={() => setPanel("contract")}
              >
                Configure search →
              </button>
            </article>
          )}
        </section>
      )}
      {inspectedSpecification && (
        <SpecificationInspector
          run={inspectedSpecification}
          experiments={experiments}
          industryPriorChannels={
            inspectedSpecification.spec.family === "frequentist"
              ? []
              : inspectedSpecification.spec.evidencePriorChannels ?? []
          }
          onClose={() => setInspectedSpecification(undefined)}
        />
      )}
    </div>
  );
}

function approximationAssessment(channel: SamplingChannelPosterior): {
  id: "compatible" | "center" | "shape" | "center-shape";
  label: string;
  detail: string;
} {
  const id = channel.assessment ?? (channel.materialShift ? "center" : "compatible");
  if (id === "center-shape") {
    return {
      id,
      label: "Center & shape differ",
      detail: "Both the typical ROI and uncertainty geometry changed.",
    };
  }
  if (id === "center") {
    return {
      id,
      label: "Center differs",
      detail: "The analytic screen and full posterior center are separated.",
    };
  }
  if (id === "shape") {
    return {
      id,
      label: "Shape differs",
      detail: "The centers are compatible, but interval width or tails differ.",
    };
  }
  return {
    id,
    label: "Compatible",
    detail: "Center and uncertainty are broadly consistent.",
  };
}

function approximationNarrative(channel: SamplingChannelPosterior): string {
  const explanation = channel.explanation;
  const channelLabel = cleanChannel(channel.channel);
  if (!explanation) {
    return `${channelLabel} was sampled before explanatory diagnostics were added. Re-run this contract to inspect the posterior geometry.`;
  }
  const assessment = approximationAssessment(channel);
  const widthDescription =
    channel.uncertaintyRatio < 0.67
      ? `the screening interval is ${(1 / Math.max(channel.uncertaintyRatio, 1e-6)).toFixed(1)}× wider`
      : channel.uncertaintyRatio > 1.5
        ? `MCMC is ${channel.uncertaintyRatio.toFixed(1)}× wider`
        : "the interval widths are similar";
  const overlap = `${Math.round(explanation.intervalOverlap * 100)}% interval overlap`;
  if (assessment.id === "compatible") {
    return `${channelLabel}: the centers are compatible, ${widthDescription}, and the central intervals have ${overlap}.`;
  }
  if (assessment.id === "shape") {
    return `${channelLabel}: the centers remain compatible, but ${widthDescription}. The comparison keeps the promoted structural and evidence contract while Production samples additional continuous uncertainty.`;
  }
  if (assessment.id === "center") {
    return `${channelLabel}: the posterior center moved ${explanation.locationShiftSd.toFixed(2)} screening standard deviations while ${widthDescription}.`;
  }
  return `${channelLabel}: the posterior center moved ${explanation.locationShiftSd.toFixed(2)} standard deviations and ${widthDescription}; inspect both location and tail behavior.`;
}

type SamplingWorkspacePanel =
  | "overview"
  | "convergence"
  | "posterior"
  | "explain";

function SamplingView({
  promotion,
  contract,
  setContract,
  result,
  history,
  status,
  progress,
  serviceReady,
  serviceDetail,
  onRun,
  onApplyRetry,
  onNewContract,
  onOpenAgentic,
  onSelectHistory,
}: {
  promotion?: PromotedAgenticSpecification;
  contract: SamplingContract;
  setContract: (contract: SamplingContract) => void;
  result?: SamplingResult;
  history: SamplingResult[];
  status: JobStatus;
  progress?: SamplingJobProgress;
  serviceReady: boolean | null;
  serviceDetail?: string;
  onRun: () => void;
  onApplyRetry: (contract: SamplingContract) => void;
  onNewContract: () => void;
  onOpenAgentic: () => void;
  onSelectHistory: (result: SamplingResult) => void;
}) {
  const [panel, setPanel] = useState<SamplingWorkspacePanel>("overview");
  const [parameterIndex, setParameterIndex] = useState(0);
  const [channelIndex, setChannelIndex] = useState(0);
  const running = status === "running" || status === "queued";
  const promotedRun = promotion?.run;
  const forcedPromotion = promotion?.override;
  const supported =
    promotedRun?.spec.family === "bayesian" ||
    promotedRun?.spec.family === "advanced";
  const completed = progress?.completed ?? 0;
  const total = Math.max(progress?.total ?? 1, 1);
  const progressPercent = Math.min(100, (completed / total) * 100);
  const selectedParameter = result?.parameters[parameterIndex];
  const currentPreset = (Object.keys(SAMPLING_PRESETS) as Array<
    keyof typeof SAMPLING_PRESETS
  >).find((preset) =>
    sameSamplingContract(contract, SAMPLING_PRESETS[preset]),
  );
  const approximationReviews = result?.channels.filter(
    (channel) => approximationAssessment(channel).id !== "compatible",
  ).length ?? 0;
  const selectedChannel = result?.channels[channelIndex] ?? result?.channels[0];
  const selectedExplanation = selectedChannel?.explanation;
  const selectedAssessment = selectedChannel
    ? approximationAssessment(selectedChannel)
    : undefined;
  const selectedSkew = selectedExplanation
    ? Math.abs(selectedExplanation.posteriorSkewness) < 0.35
      ? "Nearly symmetric"
      : selectedExplanation.posteriorSkewness > 0
        ? "Right-skewed"
        : "Left-skewed"
    : "Unavailable";

  const updateCustom = <Key extends keyof SamplingContract>(
    key: Key,
    value: SamplingContract[Key],
  ) => setContract({ ...contract, preset: "custom", [key]: value });

  if (!promotion) {
    return (
      <div className="view sampling-view">
        <section className="page-heading compact-heading">
          <div>
            <span className="kicker">Production inference</span>
            <h1>Posterior sampling</h1>
            <p>
              Confirm the uncertainty of a validated winner with full NUTS
              sampling.
            </p>
          </div>
        </section>
        <section className="card sampling-empty-state">
          <span className="sampling-empty-orbit"><i /><i /><i /></span>
          <div>
            <span className="eyebrow">No promoted specification</span>
            <h2>Start from a model that has earned the right to be sampled</h2>
            <p>
              Run the bounded Agentic search, review its validation evidence,
              and promote the winning specification. Sampling will inherit its
              exact data, assumptions, calibration route, and guardrails.
            </p>
          </div>
          <button className="button primary" onClick={onOpenAgentic}>
            Open Agentic search →
          </button>
        </section>
      </div>
    );
  }

  if (!supported) {
    return (
      <div className="view sampling-view">
        <section className="page-heading compact-heading">
          <div>
            <span className="kicker">Production inference</span>
            <h1>Posterior sampling</h1>
            <p>Sampling requires a coherent probabilistic specification.</p>
          </div>
        </section>
        <section className="card sampling-empty-state frequentist">
          <span className="sampling-empty-orbit"><i /><i /><i /></span>
          <div>
            <span className="eyebrow">Frequentist winner promoted</span>
            <h2>Create a Bayesian counterpart before MCMC</h2>
            <p>
              {promotedRun?.spec.id} supplies a strong structural specification,
              but it has no posterior to sample. Flux will not silently relabel
              the frequentist fit as Bayesian. Promote a validated Bayesian or
              Advanced counterpart using the same structural settings.
            </p>
          </div>
          <button className="button primary" onClick={onOpenAgentic}>
            Review candidates →
          </button>
        </section>
      </div>
    );
  }

  return (
    <div className="view sampling-view">
      <section className="page-heading compact-heading sampling-heading">
        <div>
          <span className="kicker">Production inference</span>
          <h1>Posterior sampling</h1>
          <p>
            Turn the promoted family into a full posterior by sampling continuous
            response, coefficient, residual, and optional planning-factor uncertainty.
          </p>
        </div>
        <div className="sampling-heading-actions">
          {result && !running && (
            <button
              className="button secondary"
              onClick={() => {
                setPanel("overview");
                onNewContract();
              }}
            >
              New sampling contract
            </button>
          )}
          <span className={`sampling-readiness-pill ${result?.status ?? "pending"}`}>
            <i />
            {result
              ? result.status === "ready"
                ? "Production ready"
                : "Needs attention"
              : running
                ? "Sampling"
                : "Awaiting run"}
          </span>
        </div>
      </section>

      <section className="sampling-provenance-strip">
        <div>
          <span>{forcedPromotion ? "Force-promoted candidate" : "Promoted winner"}</span>
          <b>{promotedRun?.spec.id} · {agenticFamilyLabel(promotedRun!.spec.family)}</b>
        </div>
        <div>
          <span>Validation score</span>
          <b>{promotedRun?.validation?.finalScore?.toFixed(1) ?? "—"}</b>
        </div>
        <div>
          <span>Model contract</span>
          <b>{forcedPromotion ? "Override · warnings preserved" : "Family fixed · parameters sampled"}</b>
        </div>
        <button onClick={() => setPanel("posterior")}>View provenance →</button>
      </section>

      {forcedPromotion && (
        <section className="sampling-force-override" role="status">
          <span>!</span>
          <div>
            <b>Manual force-promotion override</b>
            <p>
              This candidate did not pass the normal promotion contract. Its
              validation score, {forcedPromotion.failedGates.length} failed gate
              {forcedPromotion.failedGates.length === 1 ? "" : "s"}, and{" "}
              {forcedPromotion.roiGuardrailViolations.length} ROI review
              {forcedPromotion.roiGuardrailViolations.length === 1 ? "" : "s"}
              {" "}remain attached. Sampling may run and be stored, but cannot
              erase these warnings.
            </p>
          </div>
          <strong>Needs attention</strong>
        </section>
      )}

      <nav className="sampling-panel-nav" aria-label="Posterior workspace">
        {([
          ["overview", "Overview", "Screening vs MCMC"],
          ["convergence", "Convergence", "Chains & gates"],
          ["posterior", "Posterior", "Channels & artifact"],
          ["explain", "Explain", "Why ROI moved"],
        ] as const).map(([id, label, detail]) => (
          <button
            key={id}
            className={panel === id ? "active" : ""}
            onClick={() => setPanel(id)}
          >
            <span>{label}</span>
            <small>{detail}</small>
          </button>
        ))}
      </nav>

      {!result && panel === "overview" && (
        <section className="sampling-setup-grid">
          <article className="card sampling-contract-card">
            <div className="card-heading">
              <div>
                <span className="eyebrow">Sampling contract</span>
                <h2>Choose the computational depth</h2>
              </div>
              <span className="immutable-tag">◆ Family locked</span>
            </div>
            <p className="sampling-contract-intro">
              Presets change only how thoroughly NUTS explores the posterior.
              They cannot alter the winning family, discrete structure, or evidence;
              continuous response parameters are inferred around the promoted values.
            </p>
            <div className="sampling-preset-grid">
              {(
                [
                  ["production", "Production", "Balanced default", "4 × 1,000"],
                  ["robust", "Robust retry", "Longer adaptation", "4 × 2,000"],
                  ["diagnostic", "Deep diagnostic", "More independent chains", "6 × 2,000"],
                ] as const
              ).map(([preset, label, detail, count]) => (
                <button
                  key={preset}
                  className={currentPreset === preset ? "active" : ""}
                  disabled={running}
                  onClick={() => setContract(SAMPLING_PRESETS[preset])}
                >
                  <span><i />{label}</span>
                  <b>{count}</b>
                  <small>{detail}</small>
                </button>
              ))}
            </div>
            <details className="sampling-expert-settings">
              <summary>
                <span>Expert settings</span>
                <small>Bounded controls · use diagnostics to guide changes</small>
              </summary>
              <div className="sampling-control-grid">
                <label>
                  <span>Independent chains</span>
                  <select
                    value={contract.chains}
                    disabled={running}
                    onChange={(event) =>
                      updateCustom(
                        "chains",
                        Number(event.target.value) as SamplingContract["chains"],
                      )
                    }
                  >
                    {[4, 6, 8].map((value) => <option key={value}>{value}</option>)}
                  </select>
                  <small>At least four for production diagnostics.</small>
                </label>
                <label>
                  <span>Warmup per chain</span>
                  <select
                    value={contract.tune}
                    disabled={running}
                    onChange={(event) =>
                      updateCustom(
                        "tune",
                        Number(event.target.value) as SamplingContract["tune"],
                      )
                    }
                  >
                    {[500, 1000, 2000, 3000, 4000].map((value) => <option key={value}>{value.toLocaleString()}</option>)}
                  </select>
                  <small>Adapts step size and mass matrix.</small>
                </label>
                <label>
                  <span>Retained draws</span>
                  <select
                    value={contract.draws}
                    disabled={running}
                    onChange={(event) =>
                      updateCustom(
                        "draws",
                        Number(event.target.value) as SamplingContract["draws"],
                      )
                    }
                  >
                    {[500, 1000, 2000, 4000].map((value) => <option key={value}>{value.toLocaleString()}</option>)}
                  </select>
                  <small>Raises ESS only when chains mix well.</small>
                </label>
                <label>
                  <span>Target acceptance</span>
                  <select
                    value={contract.targetAccept}
                    disabled={running}
                    onChange={(event) =>
                      updateCustom(
                        "targetAccept",
                        Number(event.target.value) as SamplingContract["targetAccept"],
                      )
                    }
                  >
                    {[0.8, 0.9, 0.95, 0.99].map((value) => <option key={value}>{value.toFixed(2)}</option>)}
                  </select>
                  <small>Higher values use smaller, slower steps.</small>
                </label>
                <label>
                  <span>Maximum tree depth</span>
                  <select
                    value={contract.maxTreeDepth}
                    disabled={running}
                    onChange={(event) =>
                      updateCustom(
                        "maxTreeDepth",
                        Number(event.target.value) as SamplingContract["maxTreeDepth"],
                      )
                    }
                  >
                    {[10, 12, 14, 16].map((value) => <option key={value}>{value}</option>)}
                  </select>
                  <small>Increase only after checking divergences.</small>
                </label>
                <label>
                  <span>Reproducible seed</span>
                  <input
                    type="number"
                    min="1"
                    max="999999999"
                    value={contract.seed}
                    disabled={running}
                    onChange={(event) =>
                      updateCustom(
                        "seed",
                        Math.max(1, Number(event.target.value) || 1),
                      )
                    }
                  />
                  <small>Independent streams are derived per chain.</small>
                </label>
              </div>
            </details>
          </article>

          <article className="card sampling-run-card">
            <span className="sampling-run-graphic"><i /><i /><i /><b>β</b></span>
            <span className="eyebrow">Confirmatory posterior</span>
            <h2>{samplingContractLabel(contract)}</h2>
            <p>
              The UI remains interactive while the local worker compiles,
              warms up each chain, retains draws, and computes diagnostics.
            </p>
            <div className="sampling-run-facts">
              <span><b>{(contract.chains * contract.draws).toLocaleString()}</b> retained draws</span>
              <span><b>{contract.targetAccept.toFixed(2)}</b> target acceptance</span>
              <span><b>PyMC NUTS</b> local engine</span>
            </div>
            {serviceReady === false && (
              <div className="sampling-service-warning">
                <b>Production sampler unavailable</b>
                <span>{serviceDetail ?? "Restart the local Flux server to activate production sampling."}</span>
              </div>
            )}
            <button
              className="button primary full"
              onClick={onRun}
              disabled={running || serviceReady === false}
            >
              {running ? "Sampling posterior…" : "Run production sampling →"}
            </button>
            <small className="sampling-run-note">
              Every distinct contract is fingerprinted and stored separately.
            </small>
          </article>
        </section>
      )}

      {running && (
        <section className="card sampling-progress-card">
          <div className="sampling-progress-heading">
            <span className="run-orbit" />
            <div>
              <span className="eyebrow">{progress?.stage ?? "Queued"}</span>
              <h2>{progress?.detail ?? "Preparing the sampling job."}</h2>
            </div>
            <strong>{progressPercent.toFixed(0)}%</strong>
          </div>
          <div className="sampling-progress-track"><i style={{ width: `${progressPercent}%` }} /></div>
          <div className="sampling-chain-progress">
            {Array.from({ length: contract.chains }, (_, index) => {
              const activeChain = progress?.chain ?? 0;
              const complete = activeChain > index + 1 || progress?.stage === "complete";
              const active = activeChain === index + 1;
              return (
                <span className={complete ? "complete" : active ? "active" : ""} key={index}>
                  <i>{complete ? "✓" : index + 1}</i>
                  <b>Chain {index + 1}</b>
                  <small>{complete ? "Complete" : active ? progress?.stage : "Waiting"}</small>
                </span>
              );
            })}
          </div>
        </section>
      )}

      {result && panel === "overview" && !running && (
        <>
          <section className={`sampling-readiness-card ${result.status}`}>
            <div>
              <span className="eyebrow">Sampling verdict</span>
              <h2>
                {result.status === "ready"
                  ? "The production posterior passed every gate"
                  : "The posterior needs attention before decision use"}
              </h2>
              <p>
                {result.status === "ready"
                  ? `${result.contract.chains} chains reached the same posterior with sufficient effective samples and no invalid NUTS transitions.`
                  : "The artifact remains stored for diagnosis. Flux will not silently mark it production-ready or overwrite it with a retry."}
              </p>
            </div>
            <div className="sampling-readiness-score">
              <strong>{result.gates.filter((gate) => gate.passed).length}/{result.gates.length}</strong>
              <span>gates passed</span>
            </div>
          </section>

          {result.retryRecommendation && (
            <section className="sampling-retry-banner">
              <span>↻</span>
              <div>
                <b>{result.retryRecommendation.title}</b>
                <p>{result.retryRecommendation.detail}</p>
              </div>
              {!result.retryRecommendation.title.startsWith("Return to") && (
                <button
                  className="button secondary"
                  onClick={() => onApplyRetry(result.retryRecommendation!.contract)}
                >
                  Apply recommended retry →
                </button>
              )}
            </section>
          )}

          <section className="sampling-overview-grid">
            <article className="card sampling-chart-card wide">
              <div className="card-heading">
                <div>
                  <span className="eyebrow">Uncertainty comparison</span>
                  <h2>Analytic screening versus MCMC channel ROI</h2>
                </div>
                <span className="subtle">95% intervals</span>
              </div>
              <SamplingChart
                result={result}
                mode="roi"
                label="Analytic screening and MCMC posterior ROI intervals by channel"
                height={Math.max(260, result.channels.length * 45)}
              />
              <div className="sampling-chart-footnote">
                <span><i className="violet" />MCMC posterior</span>
                <span><i className="orange" />Analytic screening interval</span>
                <button className="sampling-explain-link" onClick={() => setPanel("explain")}>
                  {approximationReviews} channel{approximationReviews === 1 ? "" : "s"} need approximation review →
                </button>
              </div>
            </article>
            <article className="card sampling-gate-summary">
              <div className="card-heading">
                <div>
                  <span className="eyebrow">Production gates</span>
                  <h2>Convergence at a glance</h2>
                </div>
                <button className="text-button" onClick={() => setPanel("convergence")}>Inspect →</button>
              </div>
              <div className="sampling-gate-list">
                {result.gates.map((gate) => (
                  <div className={gate.passed ? "pass" : "review"} key={gate.id}>
                    <i>{gate.passed ? "✓" : "!"}</i>
                    <span><b>{gate.label}</b><small>{gate.threshold}</small></span>
                    <strong>{gate.value}</strong>
                  </div>
                ))}
              </div>
            </article>
          </section>

          <section className="card sampling-predictive-card">
            <div className="card-heading">
              <div>
                <span className="eyebrow">Posterior predictive</span>
                <h2>Replicated outcomes and expected response</h2>
              </div>
              <span className="subtle">Hover for date-level intervals</span>
            </div>
            <SamplingChart
              result={result}
              mode="predictive"
              label="Observed outcome and posterior predictive interval over time"
              height={310}
            />
          </section>
        </>
      )}

      {result && panel === "convergence" && !running && (
        <section className="sampling-convergence-workspace">
          <article className="card sampling-diagnostic-header">
            <div>
              <span className="eyebrow">Parameter diagnostics</span>
              <h2>Inspect one parameter without losing the system view</h2>
              <p>
                Rank plots should overlap across chains; trace plots should look
                stationary and mix without persistent separation.
              </p>
            </div>
            <label>
              <span>Parameter</span>
              <select
                value={parameterIndex}
                onChange={(event) => setParameterIndex(Number(event.target.value))}
              >
                {result.parameters.map((parameter, index) => (
                  <option value={index} key={parameter.name}>{parameter.label}</option>
                ))}
              </select>
            </label>
          </article>

          <section className="sampling-diagnostic-metrics">
            <article><span>R-hat</span><strong>{selectedParameter?.rhat.toFixed(3)}</strong><small>≤ 1.01</small></article>
            <article><span>Bulk ESS</span><strong>{selectedParameter?.bulkEss.toFixed(0)}</strong><small>≥ 400</small></article>
            <article><span>Tail ESS</span><strong>{selectedParameter?.tailEss.toFixed(0)}</strong><small>≥ 400</small></article>
            <article><span>MCSE / SD</span><strong>{selectedParameter ? `${((selectedParameter.mcse / Math.max(selectedParameter.standardDeviation, 1e-12)) * 100).toFixed(1)}%` : "—"}</strong><small>&lt; 5%</small></article>
          </section>

          <section className="sampling-chain-charts">
            {(["trace", "rank"] as SamplingChartMode[]).map((mode) => (
              <article className="card" key={mode}>
                <div className="card-heading">
                  <div>
                    <span className="eyebrow">{mode === "trace" ? "Chain history" : "Rank uniformity"}</span>
                    <h2>{mode === "trace" ? "Trace plot" : "Rank plot"}</h2>
                  </div>
                  <span className="subtle">{selectedParameter?.label}</span>
                </div>
                <SamplingChart
                  result={result}
                  mode={mode}
                  parameterIndex={parameterIndex}
                  label={`${mode} plot for ${selectedParameter?.label ?? "selected parameter"}`}
                  height={280}
                />
              </article>
            ))}
          </section>

          <details className="card sampling-all-diagnostics">
            <summary>
              <span><b>All production gates</b><small>Threshold, result, and interpretation</small></span>
              <i>{result.gates.filter((gate) => gate.passed).length}/{result.gates.length} passed</i>
            </summary>
            <div className="sampling-diagnostic-table">
              {result.gates.map((gate) => (
                <div key={gate.id}>
                  <i className={gate.passed ? "pass" : "review"}>{gate.passed ? "✓" : "!"}</i>
                  <span><b>{gate.label}</b><small>{gate.detail}</small></span>
                  <span>{gate.threshold}</span>
                  <strong>{gate.value}</strong>
                </div>
              ))}
            </div>
          </details>
        </section>
      )}

      {result && panel === "posterior" && !running && (
        <section className="sampling-posterior-workspace">
          <article className="card sampling-posterior-table-card">
            <div className="card-heading">
              <div>
                <span className="eyebrow">Decision posterior</span>
                <h2>Channel ROI comparison</h2>
              </div>
              <span className="subtle">Shared contract · screening vs full posterior</span>
            </div>
            <div className="sampling-posterior-table">
              <div className="sampling-posterior-row head">
                <span>Channel</span><span>Analytic screen</span><span>MCMC median</span><span>95% HDI</span><span>Uncertainty</span><span>Assessment</span>
              </div>
              {result.channels.map((channel, index) => {
                const assessment = approximationAssessment(channel);
                return (
                <div className="sampling-posterior-row" key={channel.channel}>
                  <strong>{cleanChannel(channel.channel)}</strong>
                  <span>{channel.mapRoi.toFixed(2)}×<small>{channel.mapLow.toFixed(2)}–{channel.mapHigh.toFixed(2)}</small></span>
                  <b>{channel.posteriorMedian.toFixed(2)}×<small>{channel.difference >= 0 ? "+" : ""}{channel.difference.toFixed(2)}× vs screen</small></b>
                  <span>{channel.posteriorLow.toFixed(2)}–{channel.posteriorHigh.toFixed(2)}×</span>
                  <span>{channel.uncertaintyRatio.toFixed(2)}×<small>interval width</small></span>
                  <button
                    className={`sampling-assessment ${assessment.id}`}
                    onClick={() => {
                      setChannelIndex(index);
                      setPanel("explain");
                    }}
                    aria-label={`Explain ${assessment.label.toLowerCase()} for ${cleanChannel(channel.channel)}`}
                  >
                    {assessment.label} →
                  </button>
                </div>
                );
              })}
            </div>
          </article>

          <section className="sampling-artifact-grid">
            <article className="card">
              <span className="eyebrow">Frozen provenance</span>
              <h2>{promotedRun?.spec.id} · {promotedRun?.spec.label}</h2>
              <div className="sampling-artifact-facts">
                <span><b>Estimator</b>{agenticFamilyLabel(promotedRun!.spec.family)}</span>
                <span><b>Promotion</b>{forcedPromotion ? "Forced override" : "Eligible winner"}</span>
                <span><b>Calibration</b>{promotedRun?.spec.family === "advanced" ? promotedRun.spec.advancedConfig.calibrationMode : "prior"}</span>
                <span><b>Prior family</b>{promotedRun?.spec.family === "advanced" ? promotedRun.spec.advancedConfig.priorDistribution : "Half-normal media effects"}</span>
                <span><b>Likelihood</b>{promotedRun?.spec.family === "advanced" ? promotedRun.spec.advancedConfig.likelihoodDistribution : "gaussian"}</span>
                <span><b>Adstock</b>{promotedRun?.spec.config.adstockType}</span>
                <span><b>Response parameters</b>Sampled around promoted values</span>
                <span><b>Planning factor</b>{promotedRun?.spec.family === "advanced" && promotedRun.spec.advancedConfig.planningIntensity ? "Probabilistic latent factor" : "Not included"}</span>
                <span><b>Validation</b>{promotedRun?.validation?.finalScore?.toFixed(1) ?? "—"}</span>
              </div>
            </article>
            <article className="card sampling-artifact-card">
              <span className="eyebrow">Stored artifact</span>
              <h2>{result.artifact.retainedDraws.toLocaleString()} posterior draws</h2>
              <p>
                Full chains and sampler statistics are compressed locally;
                lightweight summaries are cached with the model run.
              </p>
              <div><span>Engine</span><b>{result.engine}</b></div>
              <div><span>Runtime</span><b>{result.runtimeSeconds.toFixed(1)} seconds</b></div>
              <div><span>Fingerprint</span><b>{result.fingerprint.slice(0, 12)}…</b></div>
              <div><span>Run</span><b>{new Date(result.runAt).toLocaleString()}</b></div>
            </article>
          </section>
        </section>
      )}

      {result && panel === "explain" && !running && (
        <section className="sampling-explain-workspace">
          {!selectedExplanation || !selectedChannel || !selectedAssessment ? (
            <article className="card sampling-explain-stale">
              <span className="eyebrow">New diagnostic contract</span>
              <h2>Re-run this stored artifact to explain its posterior geometry</h2>
              <p>
                This result predates the channel-density and attribution
                summaries. The new sampling version stores them automatically
                without changing the promoted model specification.
              </p>
              <button
                className="button primary"
                onClick={() => {
                  setPanel("overview");
                  onNewContract();
                }}
              >
                Prepare fresh sampling run →
              </button>
            </article>
          ) : (
            <>
              <article className="card sampling-explain-intro">
                <div>
                  <span className="eyebrow">Approximation lens</span>
                  <h2>Why did the ROI move?</h2>
                  <p>
                    The analytic screen finds a local solution with a local
                    interval. MCMC explores the full promoted probabilistic
                    specification—including continuous response uncertainty,
                    skew, boundaries, and attribution trade-offs.
                  </p>
                </div>
                <label>
                  <span>Channel to explain</span>
                  <select
                    value={channelIndex}
                    onChange={(event) => setChannelIndex(Number(event.target.value))}
                  >
                    {result.channels.map((channel, index) => (
                      <option value={index} key={channel.channel}>
                        {cleanChannel(channel.channel)}
                      </option>
                    ))}
                  </select>
                </label>
                <span className={`sampling-explain-verdict ${selectedAssessment.id}`}>
                  {selectedAssessment.label}
                </span>
              </article>

              <article className="card sampling-density-card">
                <div className="card-heading">
                  <div>
                    <span className="eyebrow">Posterior shape · {cleanChannel(selectedChannel.channel)}</span>
                    <h2>Local approximation against sampled probability</h2>
                  </div>
                  <span className="subtle">
                    {selectedExplanation.density.scale === "log1p"
                      ? "Log ROI scale reveals the long tail"
                      : "Shared ROI scale"}
                  </span>
                </div>
                <SamplingChart
                  result={result}
                  mode="density"
                  channelIndex={channelIndex}
                  label={`Analytic screening approximation and MCMC posterior density for ${cleanChannel(selectedChannel.channel)}`}
                  height={340}
                />
                <div className="sampling-density-narrative">
                  <i className={selectedAssessment.id} />
                  <p>{approximationNarrative(selectedChannel)}</p>
                </div>
              </article>

              <section className="sampling-explain-signals">
                <article className="card">
                  <span>Center</span>
                  <strong>{selectedExplanation.locationShiftSd.toFixed(2)} SD</strong>
                  <p>
                    {selectedExplanation.locationShiftSd < 0.75
                      ? "The point movement is small relative to posterior uncertainty."
                      : "The typical posterior ROI is materially separated from the MAP peak."}
                  </p>
                </article>
                <article className="card">
                  <span>Shape</span>
                  <strong>{selectedSkew}</strong>
                  <p>
                    MCMC width is {selectedChannel.uncertaintyRatio.toFixed(2)}×
                    the screening interval with {Math.round(selectedExplanation.intervalOverlap * 100)}%
                    central-interval overlap.
                  </p>
                </article>
                <article className="card">
                  <span>Attribution trade-off</span>
                  <strong>
                    {selectedExplanation.topTradeoff
                      ? cleanChannel(selectedExplanation.topTradeoff.label)
                      : "No strong peer"}
                  </strong>
                  <p>
                    {selectedExplanation.topTradeoff
                      ? `${selectedExplanation.topTradeoff.kind === "baseline" ? "Baseline component" : "Media channel"} · ${selectedExplanation.topTradeoff.correlation < 0 ? "competing" : "moving together"} attribution · ρ ${selectedExplanation.topTradeoff.correlation.toFixed(2)}.`
                      : "No channel or baseline component has a material posterior correlation with this ROI."}
                  </p>
                </article>
              </section>

              <details className="card sampling-explain-method">
                <summary>
                  <span>
                    <b>How the two estimates are constructed</b>
                    <small>Simple intuition and mathematical contract</small>
                  </span>
                  <i>Learn the difference</i>
                </summary>
                <div className="sampling-explain-steps">
                  <article>
                    <i>1</i>
                    <span><b>Screening finds a local solution</b><small>The fast estimator locates one high-density parameter combination.</small></span>
                    <code>ROIₘₐₚ = g(θₘₐₚ)</code>
                  </article>
                  <article>
                    <i>2</i>
                    <span><b>The interval looks locally</b><small>A Gaussian approximation summarizes curvature near that screening solution.</small></span>
                    <code>q(θ) ≈ 𝒩(θₘₐₚ, H⁻¹)</code>
                  </article>
                  <article>
                    <i>3</i>
                    <span><b>MCMC maps the landscape</b><small>Samples follow all plausible parameter combinations, not only the peak neighborhood.</small></span>
                    <code>ROI = median g(θ⁽ˢ⁾)</code>
                  </article>
                </div>
                <p className="sampling-explain-caveat">
                  Both stages use the promoted family, evidence, likelihood, and
                  ROI definition. Screening uses a local analytic approximation;
                  Production samples continuous response parameters and the optional
                  latent planning factor. Convergence validates computation, while
                  posterior plausibility and validation determine decision readiness.
                </p>
              </details>
            </>
          )}
        </section>
      )}

      {history.length > 1 && !running && (
        <section className="sampling-history-strip">
          <span><b>Sampling history</b><small>Retries remain separate</small></span>
          <div>
            {history.map((item, index) => (
              <button
                key={item.fingerprint}
                className={item.fingerprint === result?.fingerprint ? "active" : ""}
                onClick={() => onSelectHistory(item)}
              >
                <i>{index + 1}</i>
                <span>{item.contract.preset === "custom" ? "Custom retry" : item.contract.preset}<small>{item.status} · {item.diagnostics.maxRhat.toFixed(3)} R-hat</small></span>
              </button>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

const BUDGET_SCENARIO_COPY: Record<
  BudgetScenarioType,
  { label: string; short: string; title: string; question: string; glyph: string }
> = {
  fixed: {
    label: "Fixed budget",
    short: "Plan a budget",
    title: "Maximize response at a chosen budget",
    question: "How should this total budget be allocated?",
    glyph: "⇄",
  },
  target: {
    label: "Outcome target",
    short: "Hit a target",
    title: "Find the minimum supported budget",
    question: "What budget and mix give us the best chance of hitting the target?",
    glyph: "◎",
  },
  economic: {
    label: "Economic ceiling",
    short: "Find the ceiling",
    title: "Stop where another dollar no longer pays",
    question: "How much should we spend before marginal investment becomes unattractive?",
    glyph: "∩",
  },
};

function budgetOutcomeIsCurrency(dataset: Dataset): boolean {
  return /revenue|sales|gmv|value/i.test(dataset.outcomeColumn);
}

function budgetValue(value: number, currency: boolean): string {
  return formatCompact(value, currency);
}

function BudgetOptimizationGuide({
  contract,
  reference,
  result,
  currency,
  onClose,
}: {
  contract: BudgetOptimizationContract;
  reference: ReturnType<typeof createBudgetReference>;
  result?: BudgetOptimizationResult;
  currency: boolean;
  onClose: () => void;
}) {
  const copy = BUDGET_SCENARIO_COPY[contract.scenario];
  const target =
    contract.targetOutcome > 0
      ? contract.targetOutcome
      : reference.defaultTargetOutcome;
  const breakEvenRoi = 1 / Math.max(contract.grossMargin, 0.01);
  const equations = {
    fixed: {
      objective: "maximize ρ(R(b))",
      constraint: "subject to Σᵢ Bᵢ = B and Lᵢ ≤ Bᵢ ≤ Uᵢ",
      plain:
        "The total budget stays fixed. The solver moves spend toward higher marginal response until no feasible transfer improves the plan.",
    },
    target: {
      objective: "minimize Σᵢ Bᵢ",
      constraint: "subject to P(Baseline⁽ˢ⁾ + R⁽ˢ⁾(b) ≥ Target) ≥ p",
      plain:
        "The solver moves along the efficient frontier and selects the smallest budget whose posterior target probability reaches the requested confidence.",
    },
    economic: {
      objective: "maximize E[m · R⁽ˢ⁾(b) − Σᵢ Bᵢ]",
      constraint: "interior optimum: marginal iROAS = 1 / gross margin",
      plain:
        "Revenue is converted to contribution using gross margin, media cost is subtracted, and the highest supported profit point is selected.",
    },
  }[contract.scenario];
  const pipeline = [
    ["1", "Freeze evidence", "Use the promoted MCMC posterior without refitting the model."],
    ["2", "Create the future plan", `Distribute each channel budget over ${reference.periods} ${reference.periodUnit}s using its recent flighting pattern.`],
    ["3", "Apply response curves", "Propagate geometric or Weibull carryover, the frozen Hill curve, and posterior ROI uncertainty."],
    ["4", "Search feasible allocations", "Compare multiple starting mixes while respecting locks and channel bounds."],
    ["5", "Verify the recommendation", "Check solver stability, historical support, posterior risk, and model readiness."],
  ];

  return (
    <DrawerShell
      className="budget-guide-drawer"
      labelledBy="budget-guide-title"
      onClose={onClose}
    >
      <div className="assumptions-header budget-guide-header">
        <div>
          <span className="kicker">
            Optimization guide · {result ? "Your recommendation" : "Illustration"}
          </span>
          <h2 id="budget-guide-title">How {copy.label.toLowerCase()} optimization works</h2>
          <p>{copy.question}</p>
        </div>
        <button className="drawer-close" onClick={onClose} aria-label="Close budget optimization guide">×</button>
      </div>

      <section className="budget-guide-verdict">
        <span>{copy.glyph}</span>
        <div>
          <b>{copy.title}</b>
          <p>{equations.plain}</p>
        </div>
        <i>{result ? "Your plan" : "Illustration"}</i>
      </section>

      <section className="budget-guide-plot">
        <div className="card-heading">
          <div>
            <span className="eyebrow">
              {contract.scenario === "economic" ? "Profit frontier" : "Efficient frontier"}
            </span>
            <h3>
              {contract.scenario === "fixed"
                ? "One budget, the strongest feasible mix"
                : contract.scenario === "target"
                  ? "The first point that clears the target"
                  : "The point where contribution profit peaks"}
            </h3>
          </div>
          <span className="subtle">Hover to inspect</span>
        </div>
        <BudgetChart
          result={result}
          reference={reference}
          contract={contract}
          mode="guide"
          label={`${copy.label} optimization explanation`}
          height={285}
          currency={currency}
        />
      </section>

      {contract.scenario === "fixed" && (
        <section className="budget-guide-signal">
          <span className="eyebrow">Marginal-return stopping rule</span>
          <h3>Interior channels move toward comparable marginal iROAS</h3>
          <div>
            {(result?.channels ?? []).slice(0, 4).map((channel) => (
              <p key={channel.channel}>
                <b>{cleanChannel(channel.channel)}</b>
                <span><i style={{ width: `${Math.min(channel.marginalRoi / Math.max(result?.marginalRoi ?? 1, 0.01), 1.5) * 64}%` }} /></span>
                <strong>{channel.marginalRoi.toFixed(2)}×</strong>
              </p>
            ))}
            {!result && (
              <p className="illustrative"><b>Channels</b><span><i style={{ width: "64%" }} /></span><strong>converge</strong></p>
            )}
          </div>
          <small>Channels at a lower, upper, or locked constraint can legitimately finish at a different marginal return.</small>
        </section>
      )}

      {contract.scenario === "target" && (
        <section className="budget-guide-target">
          <div>
            <span>Outcome target</span>
            <strong>{budgetValue(target, currency)}</strong>
          </div>
          <div>
            <span>Required confidence</span>
            <strong>{Math.round(contract.targetProbability * 100)}%</strong>
          </div>
          <p>
            The posterior mean crossing the target is not enough. At least {Math.round(contract.targetProbability * 100)}% of sampled outcome scenarios must reach it.
          </p>
        </section>
      )}

      {contract.scenario === "economic" && (
        <section className="budget-guide-target economic">
          <div>
            <span>Gross margin</span>
            <strong>{Math.round(contract.grossMargin * 100)}%</strong>
          </div>
          <div>
            <span>Break-even marginal iROAS</span>
            <strong>{breakEvenRoi.toFixed(2)}×</strong>
          </div>
          <p>Spending stops when another dollar is expected to create less contribution margin than it costs.</p>
        </section>
      )}

      <section className="budget-guide-pipeline">
        <div className="card-heading">
          <div><span className="eyebrow">Optimization pipeline</span><h3>Five steps, one frozen model</h3></div>
        </div>
        {pipeline.map(([number, title, detail]) => (
          <article key={number}>
            <i>{number}</i>
            <span><b>{title}</b><small>{detail}</small></span>
          </article>
        ))}
      </section>

      <details className="budget-guide-disclosure">
        <summary><span><b>Show the mathematics</b><small>Objective, constraints, and response transformation</small></span><i>+</i></summary>
        <div>
          <code>{equations.objective}</code>
          <code>{equations.constraint}</code>
          <p><b>Carryover</b><span>Aᵢ,ₜ = bᵢ,ₜ + θᵢ Aᵢ,ₜ₋₁, or a Weibull-weighted lag convolution.</span></p>
          <p><b>Saturation</b><span>Hᵢ(A) = Aᵅ / (Aᵅ + Kᵅ). K is frozen from training data and cannot move with the proposed budget.</span></p>
          <p><b>Posterior response</b><span>R⁽ˢ⁾(b) evaluates every aligned channel ROI draw against the same proposed allocation.</span></p>
        </div>
      </details>

      <details className="budget-guide-disclosure">
        <summary><span><b>Assumptions and limitations</b><small>What the optimizer can and cannot claim</small></span><i>+</i></summary>
        <div className="budget-guide-assumptions">
          <p>Response curves remain relevant over the selected planning window.</p>
          <p>Recent channel flighting is the default future pacing pattern.</p>
          <p>Time-varying effects use a bounded recent-stable coefficient level.</p>
          <p>The planning-intensity factor is frozen; the solver cannot manipulate it.</p>
          <p>New channels without outcome evidence are excluded.</p>
          <p>Force-promoted or non-converged models remain exploratory.</p>
        </div>
      </details>
    </DrawerShell>
  );
}

function BudgetView({
  dataset,
  promotion,
  sampling,
  contract,
  setContract,
  result,
  status,
  progress,
  onRun,
  onOpenProduction,
}: {
  dataset: Dataset;
  promotion?: PromotedAgenticSpecification;
  sampling?: SamplingResult;
  contract: BudgetOptimizationContract;
  setContract: (contract: BudgetOptimizationContract) => void;
  result?: BudgetOptimizationResult;
  status: JobStatus;
  progress?: BudgetOptimizationProgress;
  onRun: () => void;
  onOpenProduction: () => void;
}) {
  const [guideOpen, setGuideOpen] = useState(false);
  const running = status === "queued" || status === "running";
  const currency = budgetOutcomeIsCurrency(dataset);
  const reference = useMemo(
    () =>
      promotion
        ? createBudgetReference(dataset, promotion.run, contract)
        : {
            periods:
              contract.horizon === "year"
                ? Math.round(dataset.periodsPerYear)
                : dataset.modelCadence === "monthly"
                  ? 3
                  : 13,
            periodUnit: dataset.periodUnit,
            currentBudget: 0,
            baselineOutcome: 0,
            currentIncrementalOutcome: 0,
            currentTotalOutcome: 0,
            defaultTargetOutcome: 0,
          },
    [contract, dataset, promotion],
  );
  const activeConstraints = dataset.mediaColumns.map(
    (channel) =>
      contract.constraints.find(
        (constraint) => constraint.channel.toLowerCase() === channel.toLowerCase(),
      ) ?? defaultBudgetConstraints(dataset).find((item) => item.channel === channel)!,
  );
  const effectiveTarget =
    contract.targetOutcome > 0
      ? contract.targetOutcome
      : reference.defaultTargetOutcome;
  const lockedCount = activeConstraints.filter(
    (constraint) => constraint.locked || constraint.excluded,
  ).length;
  const updateConstraint = (
    channel: string,
    update: Partial<BudgetChannelConstraint>,
  ) => {
    const constraints = activeConstraints.map((constraint) =>
      constraint.channel === channel ? { ...constraint, ...update } : constraint,
    );
    setContract({ ...contract, constraints });
  };

  if (!promotion || !sampling) {
    return (
      <div className="view budget-view">
        <section className="page-heading compact-heading">
          <div><span className="kicker">Decision planning</span><h1>Budget optimization</h1><p>Turn a converged production posterior into a constrained media plan.</p></div>
        </section>
        <section className="card sampling-empty-state budget-empty-state">
          <span className="budget-empty-mark">$</span>
          <div>
            <span className="eyebrow">Production posterior required</span>
            <h2>Budget recommendations begin after posterior sampling</h2>
            <p>Promote a Bayesian specification, run Production sampling, and return here. The optimizer will inherit the exact model, posterior ROI uncertainty, adstock, saturation, and validation receipt.</p>
          </div>
          <button className="button primary" onClick={onOpenProduction}>Open Production →</button>
        </section>
      </div>
    );
  }

  return (
    <div className="view budget-view">
      <section className="page-heading compact-heading budget-heading">
        <div>
          <span className="kicker">Decision planning</span>
          <h1>Budget planner</h1>
          <p>Turn the production posterior into a transparent, constrained media recommendation.</p>
        </div>
        <div className="budget-heading-actions">
          <button className="button secondary" onClick={() => setGuideOpen(true)}>How optimization works ↗</button>
          <span className={`budget-status-pill ${sampling.status === "ready" && !promotion.override ? "ready" : "review"}`}>
            <i />{promotion.override ? "Exploratory override" : sampling.status === "ready" ? "Production posterior" : "Posterior needs attention"}
          </span>
        </div>
      </section>

      {(promotion.override || sampling.status !== "ready") && (
        <section className="budget-evidence-warning">
          <span>!</span>
          <div><b>Recommendation will remain exploratory</b><p>{promotion.override ? "This specification was force promoted with unresolved evidence." : "The Production posterior has at least one unresolved convergence or validation gate."} Optimization will preserve the warning and will not label the plan decision-grade.</p></div>
        </section>
      )}

      <section className="budget-provenance-strip">
        <div><span>Production artifact</span><b>{sampling.promotedId} · {sampling.fingerprint.slice(0, 9)}…</b></div>
        <div><span>Planning window</span><b>{reference.periods} {reference.periodUnit}s</b></div>
        <div><span>Reference budget</span><b>{budgetValue(reference.currentBudget, true)}</b></div>
        <div><span>Constraints</span><b>{lockedCount ? `${lockedCount} locked / excluded` : "Default ±50%"}</b></div>
      </section>

      <section className="budget-scenario-grid" aria-label="Budget optimization scenario">
        {(Object.keys(BUDGET_SCENARIO_COPY) as BudgetScenarioType[]).map((scenario) => {
          const copy = BUDGET_SCENARIO_COPY[scenario];
          return (
            <button
              key={scenario}
              className={contract.scenario === scenario ? "active" : ""}
              onClick={() => setContract({ ...contract, scenario })}
            >
              <i>{copy.glyph}</i>
              <span><b>{copy.short}</b><small>{copy.question}</small></span>
              <strong>{contract.scenario === scenario ? "Selected" : "Choose"}</strong>
            </button>
          );
        })}
      </section>

      <section className="budget-setup-grid">
        <article className="card budget-contract-card">
          <div className="card-heading">
            <div><span className="eyebrow">{BUDGET_SCENARIO_COPY[contract.scenario].label}</span><h2>{BUDGET_SCENARIO_COPY[contract.scenario].title}</h2></div>
            <button className="text-button" onClick={() => setGuideOpen(true)}>Learn with a visual →</button>
          </div>

          <div className="budget-shared-controls">
            <label>
              <span>Planning window</span>
              <div className="segmented-control">
                {(["quarter", "year"] as const).map((horizon) => (
                  <button key={horizon} className={contract.horizon === horizon ? "active" : ""} onClick={() => setContract({ ...contract, horizon })}>{horizon === "quarter" ? "Next quarter" : "Next year"}</button>
                ))}
              </div>
            </label>
            <label>
              <span>Risk posture</span>
              <div className="segmented-control">
                {(["expected", "balanced", "conservative"] as const).map((riskMode) => (
                  <button key={riskMode} className={contract.riskMode === riskMode ? "active" : ""} onClick={() => setContract({ ...contract, riskMode })}>{riskMode[0].toUpperCase() + riskMode.slice(1)}</button>
                ))}
              </div>
            </label>
          </div>

          {contract.scenario === "fixed" && (
            <section className="budget-primary-control">
              <div><span>Total budget change</span><strong>{contract.budgetChange >= 0 ? "+" : ""}{Math.round(contract.budgetChange * 100)}%</strong></div>
              <input type="range" min="-50" max="50" step="1" value={Math.round(contract.budgetChange * 100)} onChange={(event) => setContract({ ...contract, budgetChange: Number(event.target.value) / 100 })} />
              <div className="budget-range-labels"><span>−50%</span><span>Current</span><span>+50%</span></div>
              <label className="budget-linked-input"><span>Linked total budget</span><input type="number" min="0" step="1000" value={Math.round(reference.currentBudget * (1 + contract.budgetChange))} onChange={(event) => setContract({ ...contract, budgetChange: Math.max(-0.5, Math.min(0.5, Number(event.target.value) / Math.max(reference.currentBudget, 1) - 1)) })} /></label>
            </section>
          )}

          {contract.scenario === "target" && (
            <section className="budget-primary-control target">
              <label><span>{currency ? "Revenue target" : `${dataset.outcomeColumn} target`}</span><input type="number" min="0" step="1000" value={Math.round(effectiveTarget)} onChange={(event) => setContract({ ...contract, targetOutcome: Math.max(0, Number(event.target.value)) })} /></label>
              <div className="budget-target-decomposition"><span><b>Recent comparable baseline</b>{budgetValue(reference.baselineOutcome, currency)}</span><i>+</i><span><b>Required media lift</b>{budgetValue(Math.max(effectiveTarget - reference.baselineOutcome, 0), currency)}</span><i>=</i><span><b>Target</b>{budgetValue(effectiveTarget, currency)}</span></div>
              <label><span>Probability required</span><div className="segmented-control">{([0.7, 0.8, 0.9] as const).map((probability) => <button key={probability} className={contract.targetProbability === probability ? "active" : ""} onClick={() => setContract({ ...contract, targetProbability: probability })}>{Math.round(probability * 100)}%</button>)}</div></label>
            </section>
          )}

          {contract.scenario === "economic" && (
            <section className="budget-primary-control economic">
              <div><span>Gross contribution margin</span><strong>{Math.round(contract.grossMargin * 100)}%</strong></div>
              <input type="range" min="5" max="90" step="1" value={Math.round(contract.grossMargin * 100)} onChange={(event) => setContract({ ...contract, grossMargin: Number(event.target.value) / 100 })} />
              <div className="budget-economic-rule"><span>Break-even marginal iROAS</span><strong>{(1 / Math.max(contract.grossMargin, 0.01)).toFixed(2)}×</strong><small>The optimizer stops where another dollar no longer covers its contribution cost.</small></div>
            </section>
          )}

          <details className="budget-constraints">
            <summary><span><b>Channel constraints</b><small>{dataset.mediaColumns.length} channels · bounds scale with the selected total budget</small></span><i>{lockedCount ? `${lockedCount} custom` : "Default ±50%"}</i></summary>
            <div className="budget-constraint-table">
              <div className="head"><span>Channel</span><span>Current</span><span>Lower</span><span>Upper</span><span>Lock</span><span>Off</span></div>
              {activeConstraints.map((constraint) => {
                const channelReference = dataset.rows.slice(-reference.periods).reduce((total, row) => total + Math.max(toNumber(row[constraint.channel]), 0), 0);
                return (
                  <div key={constraint.channel}>
                    <strong>{cleanChannel(constraint.channel)}</strong>
                    <span>{budgetValue(channelReference, true)}</span>
                    <select value={constraint.lowerMultiplier} disabled={constraint.locked || constraint.excluded} onChange={(event) => updateConstraint(constraint.channel, { lowerMultiplier: Number(event.target.value) })}>{[0, 0.25, 0.5, 0.75, 1].map((value) => <option key={value} value={value}>{Math.round(value * 100)}%</option>)}</select>
                    <select value={constraint.upperMultiplier} disabled={constraint.locked || constraint.excluded} onChange={(event) => updateConstraint(constraint.channel, { upperMultiplier: Number(event.target.value) })}>{[1, 1.25, 1.5, 2, 3].map((value) => <option key={value} value={value}>{Math.round(value * 100)}%</option>)}</select>
                    <input type="checkbox" checked={constraint.locked} disabled={constraint.excluded} aria-label={`Lock ${cleanChannel(constraint.channel)}`} onChange={(event) => updateConstraint(constraint.channel, { locked: event.target.checked })} />
                    <input type="checkbox" checked={constraint.excluded} aria-label={`Exclude ${cleanChannel(constraint.channel)}`} onChange={(event) => updateConstraint(constraint.channel, { excluded: event.target.checked, locked: false })} />
                  </div>
                );
              })}
            </div>
          </details>
        </article>

        <article className="card budget-run-card">
          <span className="budget-run-graphic"><i /><i /><i /><b>$</b></span>
          <span className="eyebrow">Posterior decision engine</span>
          <h2>{contract.scenario === "fixed" ? budgetValue(reference.currentBudget * (1 + contract.budgetChange), true) : contract.scenario === "target" ? `${Math.round(contract.targetProbability * 100)}% target confidence` : `${Math.round(contract.grossMargin * 100)}% contribution margin`}</h2>
          <p>The frontier compiles the frozen Production posterior, recent flighting, carryover, saturation, and every active channel constraint.</p>
          <div className="budget-run-facts"><span><b>{sampling.channels[0]?.posteriorSamples?.length ?? 256}</b> posterior scenarios</span><span><b>25</b> frontier budgets</span><span><b>Multi-start</b> constrained search</span></div>
          <button className="button primary full" disabled={running} onClick={onRun}>{running ? "Optimizing plan…" : result ? "Re-run recommendation →" : "Optimize plan →"}</button>
          <small>Identical scenario contracts restore from the local artifact cache.</small>
        </article>
      </section>

      {running && (
        <section className="card budget-progress-card">
          <div><span className="run-orbit" /><span><b>{progress?.stage ?? "Preparing"}</b><small>{progress?.detail ?? "Preparing the posterior frontier."}</small></span><strong>{Math.round(((progress?.completed ?? 0) / Math.max(progress?.total ?? 1, 1)) * 100)}%</strong></div>
          <span><i style={{ width: `${((progress?.completed ?? 0) / Math.max(progress?.total ?? 1, 1)) * 100}%` }} /></span>
        </section>
      )}

      {result && !running && (
        <>
          <section className={`budget-recommendation-hero ${result.status}`}>
            <div><span className="eyebrow">Recommendation · {BUDGET_SCENARIO_COPY[result.scenario].label}</span><h2>{result.headline}</h2><p>{result.status === "supported" ? "The posterior, solver, feasibility, and model evidence checks support this planning scenario." : "The plan is stored for scenario analysis, with unresolved evidence clearly preserved below."}</p></div>
            <span className="budget-decision-status"><i />{result.status === "supported" ? "Supported" : "Review required"}</span>
          </section>

          <section className="budget-metric-grid">
            <article><span>Recommended budget</span><strong>{budgetValue(result.recommendedBudget, true)}</strong><small className={result.budgetChange >= 0 ? "up" : "down"}>{result.budgetChange >= 0 ? "+" : ""}{Math.round(result.budgetChange * 100)}% vs reference</small></article>
            <article><span>Expected incremental outcome</span><strong>{budgetValue(result.expectedIncrementalOutcome, currency)}</strong><small>{result.incrementalLift >= 0 ? "+" : ""}{budgetValue(result.incrementalLift, currency)} vs current plan</small></article>
            <article><span>Overall incremental ROI</span><strong>{result.overallRoi.toFixed(2)}×</strong><small>{result.marginalRoi.toFixed(2)}× marginal</small></article>
            <article><span>{result.scenario === "target" ? "Target probability" : "Expected contribution profit"}</span><strong>{result.scenario === "target" ? `${Math.round(result.targetProbability * 100)}%` : budgetValue(result.expectedProfit, true)}</strong><small>{result.scenario === "target" ? `${Math.round(result.contract.targetProbability * 100)}% required` : `${Math.round(result.contract.grossMargin * 100)}% margin`}</small></article>
          </section>

          <section className="budget-result-grid">
            <article className="card budget-frontier-card">
              <div className="card-heading"><div><span className="eyebrow">Efficient frontier</span><h2>{result.scenario === "economic" ? "Budget against contribution profit" : "Budget against expected outcome"}</h2></div><button className="text-button" onClick={() => setGuideOpen(true)}>Explain this recommendation →</button></div>
              <BudgetChart result={result} reference={reference} contract={result.contract} mode="frontier" label="Budget efficient frontier and recommended plan" height={320} currency={currency} />
            </article>
            <article className="card budget-allocation-card">
              <div className="card-heading"><div><span className="eyebrow">Allocation shift</span><h2>Current versus recommended mix</h2></div><span className="subtle">Channel totals</span></div>
              <BudgetChart result={result} reference={reference} contract={result.contract} mode="allocation" label="Current and recommended channel budgets" height={320} />
            </article>
          </section>

          <section className="card budget-channel-card">
            <div className="card-heading"><div><span className="eyebrow">Execution plan</span><h2>Channel recommendation and marginal logic</h2></div><span className="subtle">Expand response explanations in the guide</span></div>
            <div className="budget-channel-table">
              <div className="head"><span>Channel</span><span>Current</span><span>Recommended</span><span>Change</span><span>Avg iROAS</span><span>Marginal</span><span>Saturation</span><span>Constraint</span></div>
              {result.channels.map((channel) => (
                <div key={channel.channel}>
                  <strong>{cleanChannel(channel.channel)}</strong>
                  <span>{budgetValue(channel.currentBudget, true)}</span>
                  <b>{budgetValue(channel.recommendedBudget, true)}</b>
                  <span className={channel.change >= 0 ? "up" : "down"}>{channel.change >= 0 ? "+" : ""}{Math.round(channel.change * 100)}%</span>
                  <span>{channel.averageRoi.toFixed(2)}×</span>
                  <span>{channel.marginalRoi.toFixed(2)}×</span>
                  <span><i className="budget-saturation"><b style={{ width: `${channel.saturation * 100}%` }} /></i>{Math.round(channel.saturation * 100)}%</span>
                  <span className={`budget-bound ${channel.constraint}`}>{channel.constraint}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="budget-receipt-grid">
            <article className="card">
              <span className="eyebrow">Optimization receipt</span><h2>{result.solver.converged ? "Independent starts reached a stable plan" : "Solver stability needs review"}</h2>
              <div><span>Solver</span><b>{result.solver.method}</b></div><div><span>Stable starts</span><b>{result.solver.stableStarts}/{result.solver.starts}</b></div><div><span>Frontier</span><b>{result.solver.frontierPoints} budgets</b></div><div><span>Posterior scenarios</span><b>{result.solver.posteriorSamples}</b></div>
            </article>
            <article className={`card budget-warning-card ${result.warnings.length ? "review" : "pass"}`}>
              <span className="eyebrow">Decision guardrails</span><h2>{result.warnings.length ? `${result.warnings.length} item${result.warnings.length === 1 ? "" : "s"} to resolve` : "Every recommendation gate passed"}</h2>
              {result.warnings.length ? result.warnings.map((warning) => <p key={warning}><i>!</i><span>{warning}</span></p>) : <p><i>✓</i><span>Model evidence, posterior, feasibility, and solver stability support this scenario.</span></p>}
            </article>
          </section>
        </>
      )}

      {guideOpen && (
        <BudgetOptimizationGuide contract={result?.contract ?? contract} reference={reference} result={result} currency={currency} onClose={() => setGuideOpen(false)} />
      )}
    </div>
  );
}

function CalibrationView({
  dataset,
  experiments,
  setExperiments,
  guardrailMode,
  onGuardrailModeChange,
  industryPriorChannels,
  onIndustryChannelToggle,
  screeningRois,
  bayesianResult,
  onRun,
}: {
  dataset: Dataset;
  experiments: Experiment[];
  setExperiments: (experiments: Experiment[]) => void;
  guardrailMode: BenchmarkGuardrailMode;
  onGuardrailModeChange: (mode: BenchmarkGuardrailMode) => void;
  industryPriorChannels: string[];
  onIndustryChannelToggle: (channel: string, enabled: boolean) => void;
  screeningRois: Record<string, number>;
  bayesianResult?: ModelResult;
  onRun: () => void;
}) {
  const [showForm, setShowForm] = useState(false);
  const defaultStartDate = String(
    dataset.rows[0]?.[dataset.dateColumn] ?? "2018-01-01",
  ).slice(0, 10);
  const defaultEndDate = String(
    dataset.rows[Math.min(8, Math.max(dataset.rows.length - 1, 0))]?.[
      dataset.dateColumn
    ] ?? defaultStartDate,
  ).slice(0, 10);
  const [draft, setDraft] = useState({
    channel: dataset.mediaColumns[0] ?? "",
    startDate: defaultStartDate,
    endDate: defaultEndDate,
    outcomeEndDate: defaultEndDate,
    incrementalOutcome: "50000",
    incrementalSpend: "20000",
    standardError: "0.35",
    source: "Lift experiment",
  });
  const recommendations = dataset.mediaColumns.map((channel) => ({
    channel,
    experiments: channelExperiments(experiments, channel),
    recommendation: inferIndustryPrior(channel),
    screeningRoi: screeningRois[channel],
    active: industryPriorChannels.some(
      (selected) => selected.toLowerCase() === channel.toLowerCase(),
    ),
  }));
  const experimentChannelCount = recommendations.filter(
    (item) => item.experiments.length > 0,
  ).length;
  const industryChannelCount = recommendations.filter(
    (item) => item.active && item.experiments.length === 0,
  ).length;
  const extremeChannelCount = recommendations.filter(
    (item) =>
      !item.experiments.length &&
      item.screeningRoi !== undefined &&
      isHighlyImprobableIndustryRoi(
        item.screeningRoi,
        item.recommendation,
      ),
  ).length;
  const weakChannelCount =
    dataset.mediaColumns.length -
    experimentChannelCount -
    industryChannelCount;
  const hasScreening = Object.keys(screeningRois).length > 0;

  return (
    <div className="view">
      <section className="page-heading compact-heading">
        <div><span className="kicker">Ground-truth first</span><h1>Calibration evidence</h1><p>Register experimental ground truth, then optionally fill evidence gaps with conservative DTC benchmarks.</p></div>
        <button className="button primary" onClick={() => setShowForm(!showForm)}>+ Add experiment</button>
      </section>
      <section className="calibration-banner">
        <div className="calibration-graphic"><span>Experiment evidence</span><b>Lift ROI ± uncertainty</b><i>→</i><span>Model routing</span><b>Prior or likelihood</b></div>
        <div><span className="eyebrow">Evidence routing</span><h2>Choose where experiment evidence enters</h2><p>Standard Bayesian models use experiments as informative priors. Advanced models can route experiments through the prior or likelihood. Industry benchmark fallbacks always remain priors.</p></div>
      </section>
      {showForm && (
        <form className="card experiment-form" onSubmit={(event) => {
          event.preventDefault();
          setExperiments([...experiments, {
            channel: draft.channel,
            startDate: draft.startDate,
            endDate: draft.endDate,
            outcomeEndDate:
              draft.outcomeEndDate && draft.outcomeEndDate !== draft.endDate
                ? draft.outcomeEndDate
                : undefined,
            incrementalOutcome: Number(draft.incrementalOutcome),
            incrementalSpend: Number(draft.incrementalSpend),
            standardError: Number(draft.standardError),
            confidence: 0.95,
            scope: "immediate",
            source: draft.source,
          }]);
          setShowForm(false);
        }}>
          <label><span>Channel</span><select value={draft.channel} onChange={(event) => setDraft({ ...draft, channel: event.target.value })}>{dataset.mediaColumns.map((channel) => <option key={channel}>{channel}</option>)}</select></label>
          <label><span>Campaign start</span><input type="date" required value={draft.startDate} onChange={(event) => setDraft({ ...draft, startDate: event.target.value })} /></label>
          <label><span>Campaign end</span><input type="date" required min={draft.startDate} value={draft.endDate} onChange={(event) => setDraft({ ...draft, endDate: event.target.value, outcomeEndDate: event.target.value > draft.outcomeEndDate ? event.target.value : draft.outcomeEndDate })} /></label>
          <label><span>Outcome observed through <small>optional carryover</small></span><input type="date" min={draft.endDate} value={draft.outcomeEndDate} onChange={(event) => setDraft({ ...draft, outcomeEndDate: event.target.value })} /></label>
          <label><span>Incremental outcome</span><input type="number" value={draft.incrementalOutcome} onChange={(event) => setDraft({ ...draft, incrementalOutcome: event.target.value })} /></label>
          <label><span>Incremental spend</span><input type="number" value={draft.incrementalSpend} onChange={(event) => setDraft({ ...draft, incrementalSpend: event.target.value })} /></label>
          <label><span>ROI standard error</span><input type="number" step="0.01" value={draft.standardError} onChange={(event) => setDraft({ ...draft, standardError: event.target.value })} /></label>
          <label><span>Source</span><input value={draft.source} onChange={(event) => setDraft({ ...draft, source: event.target.value })} /></label>
          <button className="button primary" type="submit">Save evidence</button>
          <p className="experiment-form-note">Flux compares the reported lift with model-attributed ROI from campaign-window spend. Extending the outcome date captures declared carryover without adding post-test spend to the denominator.</p>
        </form>
      )}
      <section className="card">
        <div className="card-heading"><div><span className="eyebrow">Evidence registry</span><h2>{experiments.length} calibration studies</h2></div><a className="text-button" href="/data/robyn_experiments.csv" download>↓ Schema template</a></div>
        <div className="experiment-table">
          <div className="experiment-row experiment-head"><span>Channel</span><span>Study window</span><span>Incremental outcome</span><span>Spend</span><span>Observed ROI</span><span>Confidence</span><span /></div>
          {experiments.map((experiment, index) => (
            <div className="experiment-row" key={`${experiment.channel}-${index}`}>
              <strong>{cleanChannel(experiment.channel)}<small>{experiment.source}</small></strong>
              <span>
                {experiment.startDate.slice(0, 7)} → {experiment.endDate.slice(0, 7)}
                {experiment.outcomeEndDate &&
                experiment.outcomeEndDate !== experiment.endDate ? (
                  <small>Outcomes through {experiment.outcomeEndDate.slice(0, 7)}</small>
                ) : null}
              </span>
              <span>{formatFull(experiment.incrementalOutcome, true)}</span>
              <span>{formatFull(experiment.incrementalSpend, true)}</span>
              <b>{(experiment.incrementalOutcome / Math.max(experiment.incrementalSpend, 1)).toFixed(2)}×</b>
              <span className="confidence-pill">{Math.round(experiment.confidence * 100)}%</span>
              <button aria-label={`Remove ${experiment.channel} experiment`} onClick={() => setExperiments(experiments.filter((_, itemIndex) => itemIndex !== index))}>×</button>
            </div>
          ))}
        </div>
      </section>
      <section
        className={`card benchmark-prior-card ${
          guardrailMode !== "off" ? "is-active" : ""
        }`}
      >
        <div className="benchmark-prior-heading">
          <div>
            <span className="eyebrow">Post-fit plausibility screen</span>
            <h2>Industry plausibility guardrails</h2>
            <p>
              Screen experiment-free ROI estimates against conservative DTC
              benchmarks, then choose exactly which channels to regularize.
            </p>
          </div>
          <div
            className="guardrail-mode-control"
            role="group"
            aria-label="Industry guardrail mode"
          >
            <button
              className={guardrailMode === "suggest" ? "active" : ""}
              onClick={() => onGuardrailModeChange("suggest")}
            >
              <b>Suggest only</b>
              <small>Recommended</small>
            </button>
            <button
              className={guardrailMode === "auto" ? "active" : ""}
              onClick={() => onGuardrailModeChange("auto")}
            >
              <b>Auto extreme</b>
              <small>Outside P05–P95</small>
            </button>
            <button
              className={guardrailMode === "off" ? "active" : ""}
              onClick={() => onGuardrailModeChange("off")}
            >
              <b>Off</b>
              <small>No benchmark</small>
            </button>
          </div>
        </div>
        <div className="benchmark-guardrail">
          <span>i</span>
          <p>
            <b>Experiments always win.</b>
            A guardrail selected after seeing the initial fit is recorded as a
            sensitivity specification, so the original screening result remains
            visible for comparison.
          </p>
        </div>
        <div className="benchmark-summary">
          <div>
            <b>{experimentChannelCount}</b>
            <span>Experiment anchored</span>
          </div>
          <div>
            <b>{hasScreening ? extremeChannelCount : "—"}</b>
            <span>Extreme ROI flags</span>
          </div>
          <div>
            <b>{industryChannelCount}</b>
            <span>Guardrails active</span>
          </div>
        </div>
        <div className="benchmark-table-wrap">
          <div className="benchmark-table">
            <div className="benchmark-row benchmark-head">
              <span>Dataset channel</span>
              <span>Initial ROI</span>
              <span>DTC benchmark</span>
              <span>80% range</span>
              <span>Assessment</span>
              <span>Guardrail</span>
            </div>
            {recommendations.map(
              ({
                channel,
                experiments: evidence,
                recommendation,
                screeningRoi,
                active,
              }) => {
                const experiment = evidence[0];
                const experimentRoi = experiment
                  ? experiment.incrementalOutcome /
                    Math.max(experiment.incrementalSpend, 1)
                  : undefined;
                const percentile =
                  screeningRoi === undefined
                    ? undefined
                    : industryPriorPercentile(
                        screeningRoi,
                        recommendation,
                      );
                const isExtreme =
                  percentile !== undefined &&
                  (percentile <= 0.05 || percentile >= 0.95);
                const guardedPosterior = bayesianResult?.channels.find(
                  (item) =>
                    item.channel === channel &&
                    item.priorSource === "industry",
                )?.roi;
                return (
                  <div
                    className={`benchmark-row ${
                      isExtreme ? "is-extreme" : ""
                    }`}
                    key={channel}
                  >
                    <strong>
                      {cleanChannel(channel)}
                      <small>{recommendation.matchQuality}</small>
                    </strong>
                    <span className="benchmark-roi-comparison">
                      <b>
                        {experimentRoi !== undefined
                          ? `${experimentRoi.toFixed(2)}×`
                          : screeningRoi !== undefined
                            ? `${screeningRoi.toFixed(2)}×`
                            : "—"}
                      </b>
                      <small>
                        {experiment
                          ? experiment.source
                          : guardedPosterior !== undefined
                            ? `→ ${guardedPosterior.toFixed(2)}× guardrailed`
                            : hasScreening
                              ? "Experiment-only screening"
                              : "Run screening model"}
                      </small>
                    </span>
                    <span className="benchmark-evidence-cell">
                      <b>
                        {experiment
                          ? "Experiment"
                          : `${recommendation.median.toFixed(2)}× center`}
                      </b>
                      <small>
                        {experiment
                          ? "External causal evidence"
                          : recommendation.label}
                      </small>
                    </span>
                    <span>
                      {experiment
                        ? "Experiment uncertainty"
                        : `${recommendation.low.toFixed(2)}–${recommendation.high.toFixed(2)}×`}
                    </span>
                    <span
                      className={`benchmark-status ${
                        experiment
                          ? "experiment"
                          : percentile === undefined
                            ? "pending"
                            : isExtreme
                              ? "extreme"
                              : "plausible"
                      }`}
                    >
                      {experiment
                        ? "Experiment wins"
                        : percentile === undefined
                          ? "Awaiting screening"
                          : isExtreme
                            ? percentile <= 0.05
                              ? `Highly improbable low · P${Math.max(1, Math.round(percentile * 100))}`
                              : `Highly improbable high · P${Math.min(99, Math.round(percentile * 100))}`
                            : `Plausible · P${Math.round(percentile * 100)}`}
                    </span>
                    <label
                      className={`channel-guardrail-toggle ${
                        active ? "active" : ""
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={active}
                        disabled={
                          Boolean(experiment) || guardrailMode === "off"
                        }
                        onChange={(event) =>
                          onIndustryChannelToggle(
                            channel,
                            event.target.checked,
                          )
                        }
                      />
                      <span aria-hidden="true"><i /></span>
                      <small>
                        {experiment
                          ? "Locked"
                          : guardrailMode === "off"
                            ? "Disabled"
                            : active
                              ? "Active"
                              : isExtreme
                                ? "Recommended"
                                : "Off"}
                      </small>
                    </label>
                  </div>
                );
              },
            )}
          </div>
        </div>
        <details className="benchmark-library">
          <summary>
            <span>Browse the full DTC benchmark library</span>
            <small>{INDUSTRY_BENCHMARKS.length} channel and tactic priors</small>
          </summary>
          <div className="benchmark-library-table">
            <div className="benchmark-library-row benchmark-head">
              <span>Channel / tactic</span>
              <span>Prior center</span>
              <span>80% range</span>
              <span>Confidence</span>
            </div>
            {INDUSTRY_BENCHMARKS.map((benchmark) => (
              <div className="benchmark-library-row" key={benchmark.id}>
                <strong>
                  {benchmark.label}
                  <small>{benchmark.family}</small>
                </strong>
                <b>{benchmark.median.toFixed(2)}×</b>
                <span>
                  {benchmark.low.toFixed(2)}–{benchmark.high.toFixed(2)}×
                </span>
                <span
                  className={`benchmark-confidence ${benchmark.confidence
                    .toLowerCase()
                    .replace(" ", "-")}`}
                >
                  {benchmark.confidence}
                </span>
              </div>
            ))}
          </div>
          <p className="benchmark-source-note">
            Scoped to large US DTC ecommerce and owned-site revenue. Values are
            conservative log-normal prior centers synthesized from 2025–26
            causal geo-test and DTC portfolio evidence—not guaranteed channel
            returns.
          </p>
        </details>
      </section>
      <section className="calibration-grid">
        <article className="card">
          <div className="card-heading"><div><span className="eyebrow">Prior predictive check</span><h2>ROI prior map</h2></div>{bayesianResult ? <span className="pass-tag">✓ Passed</span> : <span className="subtle">Awaiting model</span>}</div>
          <div className="prior-list">
            {dataset.mediaColumns.map((channel) => {
              const experiment = channelExperiments(
                experiments,
                channel,
              )[0];
              const benchmark = activeIndustryPrior(
                channel,
                experiments,
                industryPriorChannels,
              );
              const result = bayesianResult?.channels.find((item) => item.channel === channel);
              const screeningRoi = screeningRois[channel];
              const priorCenter = experiment
                ? experiment.incrementalOutcome /
                  Math.max(experiment.incrementalSpend, 1)
                : benchmark?.median;
              return (
                <div key={channel}>
                  <span>{cleanChannel(channel)}</span>
                  <div><i className={`prior-band ${benchmark ? "industry" : ""}`} style={{ width: `${experiment ? 58 : benchmark ? 44 : 28}%` }} /><b style={{ left: `${experiment ? 48 : benchmark ? 34 : 20}%` }} /></div>
                  <em>
                    {priorCenter
                      ? `${priorCenter.toFixed(2)}× ${experiment ? "experiment" : "benchmark"}`
                      : screeningRoi !== undefined
                        ? `${screeningRoi.toFixed(2)}× screening`
                        : "Weak prior"}
                  </em>
                  {result && (
                    <strong>
                      {screeningRoi !== undefined &&
                      result.priorSource === "industry"
                        ? `${screeningRoi.toFixed(2)} → ${result.roi.toFixed(2)}×`
                        : `${result.roi.toFixed(2)}× posterior`}
                    </strong>
                  )}
                </div>
              );
            })}
          </div>
        </article>
        <article className="card calibration-action">
          <div className="insight-mark">β</div>
          <span className="eyebrow">Ready to estimate</span>
          <h2>
            {experimentChannelCount} experiment prior
            {experimentChannelCount === 1 ? "" : "s"} +{" "}
            {industryChannelCount} selected guardrail
            {industryChannelCount === 1 ? "" : "s"}.
          </h2>
          <p>
            {!hasScreening
              ? "Start with an experiment-only screening run. Industry benchmarks will assess plausibility without entering that first fit."
              : industryChannelCount
                ? "The next run is stored as a benchmark-guardrailed sensitivity model, preserving the initial ROI for comparison."
                : extremeChannelCount
                  ? `${extremeChannelCount} extreme estimate${extremeChannelCount === 1 ? " is" : "s are"} ready for channel-level review.`
                  : `${weakChannelCount} unanchored channel${weakChannelCount === 1 ? " remains" : "s remain"} plausible under the current DTC benchmark screen.`}
          </p>
          <button className="button primary full" onClick={onRun}>
            {!hasScreening
              ? "Run experiment-only screening"
              : industryChannelCount
                ? "Run guardrailed Bayesian model"
                : "Re-run screening model"}{" "}
            →
          </button>
        </article>
      </section>
    </div>
  );
}

export function MmmWorkbench({ demoMode = false }: { demoMode?: boolean }) {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [eda, setEda] = useState<EdaResult | null>(null);
  const [edaStatus, setEdaStatus] = useState<JobStatus>("queued");
  const [view, setView] = useState<View>("overview");
  const [experiments, setExperiments] = useState<Experiment[]>(() =>
    defaultExperimentsForDataset("robyn-demo"),
  );
  const [guardrailMode, setGuardrailMode] =
    useState<BenchmarkGuardrailMode>("suggest");
  const [industryPriorChannels, setIndustryPriorChannels] = useState<string[]>(
    [],
  );
  const [benchmarkScreeningRois, setBenchmarkScreeningRois] = useState<
    Record<string, number>
  >({});
  const [config, setConfig] = useState<ModelConfig>(DEFAULT_CONFIG);
  const [advancedConfig, setAdvancedConfig] = useState<AdvancedModelConfig>(
    DEFAULT_ADVANCED_CONFIG,
  );
  const [models, setModels] = useState<Partial<Record<"frequentist" | "bayesian", ModelResult>>>({});
  const [modelStatuses, setModelStatuses] = useState<Record<"frequentist" | "bayesian", JobStatus>>({ frequentist: "idle", bayesian: "idle" });
  const [advancedResult, setAdvancedResult] = useState<ModelResult>();
  const [advancedStatus, setAdvancedStatus] = useState<JobStatus>("idle");
  const [validationResults, setValidationResults] = useState<
    Partial<Record<ValidationModelKind, ModelValidationResult>>
  >({});
  const [validationStatuses, setValidationStatuses] = useState<
    Record<ValidationModelKind, JobStatus>
  >({
    frequentist: "idle",
    bayesian: "idle",
    advanced: "idle",
  });
  const [validationProgress, setValidationProgress] = useState<
    Partial<Record<ValidationModelKind, ValidationProgress>>
  >({});
  const [agenticContract, setAgenticContract] =
    useState<AgenticSearchContract>(DEFAULT_AGENTIC_SEARCH_CONTRACT);
  const [agenticRuns, setAgenticRuns] = useState<AgenticCandidateRun[]>([]);
  const [agenticStatus, setAgenticStatus] =
    useState<AgenticWorkspaceStatus>("idle");
  const [agenticStopReason, setAgenticStopReason] = useState<string>();
  const [
    promotedAgenticSpecification,
    setPromotedAgenticSpecification,
  ] = useState<PromotedAgenticSpecification>();
  const [samplingContract, setSamplingContract] =
    useState<SamplingContract>(DEFAULT_SAMPLING_CONTRACT);
  const [samplingResult, setSamplingResult] = useState<SamplingResult>();
  const [samplingHistory, setSamplingHistory] = useState<SamplingResult[]>([]);
  const [samplingStatus, setSamplingStatus] = useState<JobStatus>("idle");
  const [samplingProgress, setSamplingProgress] =
    useState<SamplingJobProgress>();
  const [samplingServiceReady, setSamplingServiceReady] = useState<
    boolean | null
  >(null);
  const [samplingServiceDetail, setSamplingServiceDetail] = useState<string>();
  const [budgetContract, setBudgetContract] = useState<BudgetOptimizationContract>(
    DEFAULT_BUDGET_CONTRACT,
  );
  const [budgetResult, setBudgetResult] = useState<BudgetOptimizationResult>();
  const [budgetStatus, setBudgetStatus] = useState<JobStatus>("idle");
  const [budgetProgress, setBudgetProgress] =
    useState<BudgetOptimizationProgress>();
  const [
    anchorIndependenceConfirmed,
    setAnchorIndependenceConfirmed,
  ] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [workspaceReady, setWorkspaceReady] = useState(false);
  const [workspaceSaveStatus, setWorkspaceSaveStatus] =
    useState<WorkspaceSaveStatus>("restoring");
  const [workspaceSavedAt, setWorkspaceSavedAt] = useState<string>();
  const fileRef = useRef<HTMLInputElement>(null);
  const generationRef = useRef(0);
  const agenticGenerationRef = useRef(0);
  const samplingGenerationRef = useRef(0);
  const budgetGenerationRef = useRef(0);
  const workspaceCheckpointRef = useRef<
    WorkspaceCheckpointEnvelope<WorkspaceCheckpointPayload>
  >(undefined);
  const workspaceCheckpointRevisionRef = useRef(0);
  const workspaceCheckpointDirtyRef = useRef(false);
  const workspaceCheckpointSavingRef = useRef(false);

  const flushWorkspaceCheckpoint = useEffectEvent(async () => {
    const checkpoint = workspaceCheckpointRef.current;
    if (
      !checkpoint ||
      !workspaceCheckpointDirtyRef.current ||
      workspaceCheckpointSavingRef.current
    ) {
      return;
    }
    const revision = workspaceCheckpointRevisionRef.current;
    workspaceCheckpointDirtyRef.current = false;
    workspaceCheckpointSavingRef.current = true;
    const stored = await persistWorkspaceCheckpoint(checkpoint);
    workspaceCheckpointSavingRef.current = false;
    if (revision !== workspaceCheckpointRevisionRef.current) return;
    if (!stored) {
      setWorkspaceSaveStatus("unavailable");
      return;
    }
    setWorkspaceSavedAt(checkpoint.savedAt);
    setWorkspaceSaveStatus("saved");
  });

  const resetBudget = useCallback(() => {
    budgetGenerationRef.current += 1;
    setBudgetResult(undefined);
    setBudgetStatus("idle");
    setBudgetProgress(undefined);
  }, []);

  const resetSampling = useCallback((clearHistory = true) => {
    samplingGenerationRef.current += 1;
    setSamplingResult(undefined);
    if (clearHistory) setSamplingHistory([]);
    setSamplingStatus("idle");
    setSamplingProgress(undefined);
    resetBudget();
  }, [resetBudget]);

  const resetAgenticSearch = useCallback(() => {
    agenticGenerationRef.current += 1;
    setAgenticRuns([]);
    setAgenticStatus("idle");
    setAgenticStopReason(undefined);
  }, []);

  const loadCsv = useCallback(async (
    rawCsv: string,
    name: string,
    origin: DatasetExperimentOrigin = "advertiser-upload",
  ) => {
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    const parsed = parseCsv(rawCsv);
    const nextDataset = await createDataset(name, rawCsv, parsed.columns, parsed.rows);
    const nextValidation = validateDataset(nextDataset);
    const nextExperiments = defaultExperimentsForDataset(origin);
    const storedBenchmarkPolicy = readStoredBenchmarkPolicy(
      nextDataset,
      nextExperiments,
    );
    setDataset(nextDataset);
    setValidation(nextValidation);
    setConfig(defaultConfigForDataset(nextDataset));
    setAdvancedConfig(DEFAULT_ADVANCED_CONFIG);
    setExperiments(nextExperiments);
    setEda(null);
    setModels({});
    setModelStatuses({ frequentist: "idle", bayesian: "idle" });
    setAdvancedResult(undefined);
    setAdvancedStatus("idle");
    setValidationResults({});
    setValidationStatuses({
      frequentist: "idle",
      bayesian: "idle",
      advanced: "idle",
    });
    setValidationProgress({});
    setAnchorIndependenceConfirmed(false);
    setGuardrailMode(storedBenchmarkPolicy?.mode ?? "suggest");
    setIndustryPriorChannels(storedBenchmarkPolicy?.channels ?? []);
    setBenchmarkScreeningRois({});
    setPromotedAgenticSpecification(undefined);
    resetSampling();
    resetAgenticSearch();
    setEdaStatus("queued");

    if (nextValidation.status === "blocked") {
      setEdaStatus("idle");
      return nextValidation;
    }

    void persistDataset(nextDataset);
    await delay(180);
    if (generationRef.current !== generation) return nextValidation;
    setEdaStatus("running");
    await delay(320);
    if (generationRef.current !== generation) return nextValidation;
    setEda(runEda(nextDataset));
    setEdaStatus("complete");
    return nextValidation;
  }, [resetAgenticSearch, resetSampling]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      setWorkspaceSaveStatus("restoring");
      const checkpoint =
        await readLatestWorkspaceCheckpoint<WorkspaceCheckpointPayload>();
      if (cancelled) return;
      if (checkpoint && checkpoint.payload.demoMode === demoMode) {
        const restored = checkpoint.payload;
        const interruptedAgentic = restored.agenticStatus === "running";
        const interruptedSampling =
          restored.samplingStatus === "running" ||
          restored.samplingStatus === "queued";
        const interruptedBudget =
          restored.budgetStatus === "running" ||
          restored.budgetStatus === "queued";
        setDataset(restored.dataset);
        setValidation(restored.validation);
        setEda(restored.eda);
        setEdaStatus(restored.eda ? "complete" : "idle");
        setExperiments(restored.experiments);
        setGuardrailMode(restored.guardrailMode);
        setIndustryPriorChannels(restored.industryPriorChannels);
        setBenchmarkScreeningRois(restored.benchmarkScreeningRois);
        setConfig(restored.config);
        setAdvancedConfig(restored.advancedConfig);
        setModels(restored.models);
        setModelStatuses({
          frequentist: restored.models.frequentist
            ? restored.models.frequentist.cached
              ? "cached"
              : "complete"
            : "idle",
          bayesian: restored.models.bayesian
            ? restored.models.bayesian.cached
              ? "cached"
              : "complete"
            : "idle",
        });
        setAdvancedResult(restored.advancedResult);
        setAdvancedStatus(
          restored.advancedResult
            ? restored.advancedResult.cached
              ? "cached"
              : "complete"
            : "idle",
        );
        setValidationResults(restored.validationResults);
        setValidationStatuses({
          frequentist: restored.validationResults.frequentist
            ? restored.validationResults.frequentist.cached
              ? "cached"
              : "complete"
            : "idle",
          bayesian: restored.validationResults.bayesian
            ? restored.validationResults.bayesian.cached
              ? "cached"
              : "complete"
            : "idle",
          advanced: restored.validationResults.advanced
            ? restored.validationResults.advanced.cached
              ? "cached"
              : "complete"
            : "idle",
        });
        setValidationProgress({});
        setAnchorIndependenceConfirmed(
          restored.anchorIndependenceConfirmed,
        );
        setAgenticContract(restored.agenticContract);
        setAgenticRuns(interruptedAgentic ? [] : restored.agenticRuns);
        setAgenticStatus(interruptedAgentic ? "idle" : restored.agenticStatus);
        setAgenticStopReason(
          interruptedAgentic
            ? "The previous search was interrupted by a browser reload. Start it again to preserve the full eligibility contract."
            : restored.agenticStopReason,
        );
        setPromotedAgenticSpecification(
          restored.promotedAgenticSpecification,
        );
        setSamplingContract(restored.samplingContract);
        setSamplingResult(restored.samplingResult);
        setSamplingHistory(restored.samplingHistory);
        setSamplingStatus(
          restored.samplingResult
            ? restored.samplingResult.cached
              ? "cached"
              : "complete"
            : "idle",
        );
        setSamplingProgress(undefined);
        setBudgetContract(restored.budgetContract);
        setBudgetResult(restored.budgetResult);
        setBudgetStatus(
          restored.budgetResult
            ? restored.budgetResult.cached
              ? "cached"
              : "complete"
            : "idle",
        );
        setBudgetProgress(undefined);
        const restoredView =
          restored.view === "budget" && !restored.samplingResult
            ? restored.promotedAgenticSpecification
              ? "sampling"
              : "agentic"
            : restored.view === "sampling" &&
                !restored.promotedAgenticSpecification
              ? "agentic"
              : restored.view;
        setView(restoredView);
        setWorkspaceSavedAt(checkpoint.savedAt);
        setWorkspaceSaveStatus("saved");
        setWorkspaceReady(true);
        setToast(
          interruptedAgentic || interruptedSampling || interruptedBudget
            ? "Workspace restored. A job that was running during sleep was safely returned to an idle state."
            : "Local workspace restored from the latest checkpoint.",
        );
        return;
      }

      try {
        const response = await fetch("/data/robyn_weekly.csv");
        const csv = await response.text();
        await loadCsv(
          csv,
          "Robyn Public weekly data demo.csv",
          "robyn-demo",
        );
      } catch {
        setToast("The sample dataset could not be loaded.");
      } finally {
        if (!cancelled) setWorkspaceReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [demoMode, loadCsv]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 3600);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (!dataset) return;
    writeStoredBenchmarkPolicy(dataset, {
      mode: guardrailMode,
      channels:
        guardrailMode === "off"
          ? []
          : eligibleStoredBenchmarkChannels(
              dataset,
              experiments,
              industryPriorChannels,
            ),
    });
  }, [dataset, experiments, guardrailMode, industryPriorChannels]);

  const checkpointAgenticRuns =
    agenticStatus === "running"
      ? EMPTY_AGENTIC_CHECKPOINT_RUNS
      : agenticRuns;

  useEffect(() => {
    if (!workspaceReady || !dataset || !validation) return;
    workspaceCheckpointRef.current = createWorkspaceCheckpoint(dataset.hash, {
      demoMode,
      view,
      dataset,
      validation,
      eda,
      experiments,
      guardrailMode,
      industryPriorChannels,
      benchmarkScreeningRois,
      config,
      advancedConfig,
      models,
      advancedResult,
      validationResults,
      anchorIndependenceConfirmed,
      agenticContract,
      agenticRuns: checkpointAgenticRuns,
      agenticStatus,
      agenticStopReason,
      promotedAgenticSpecification,
      samplingContract,
      samplingResult,
      samplingHistory,
      samplingStatus,
      budgetContract,
      budgetResult,
      budgetStatus,
    });
    workspaceCheckpointRevisionRef.current += 1;
    workspaceCheckpointDirtyRef.current = true;
    const frame = window.requestAnimationFrame(() => {
      setWorkspaceSaveStatus("saving");
    });
    return () => window.cancelAnimationFrame(frame);
  }, [
    advancedConfig,
    advancedResult,
    agenticContract,
    checkpointAgenticRuns,
    agenticStatus,
    agenticStopReason,
    anchorIndependenceConfirmed,
    benchmarkScreeningRois,
    budgetContract,
    budgetResult,
    budgetStatus,
    config,
    dataset,
    demoMode,
    eda,
    experiments,
    guardrailMode,
    industryPriorChannels,
    models,
    promotedAgenticSpecification,
    samplingContract,
    samplingHistory,
    samplingResult,
    samplingStatus,
    validation,
    validationResults,
    view,
    workspaceReady,
  ]);

  useEffect(() => {
    if (!workspaceReady) return;
    const interval = window.setInterval(() => {
      void flushWorkspaceCheckpoint();
    }, 2500);
    const saveBeforeBackgrounding = () => {
      if (document.visibilityState === "hidden") {
        void flushWorkspaceCheckpoint();
      }
    };
    document.addEventListener("visibilitychange", saveBeforeBackgrounding);
    return () => {
      window.clearInterval(interval);
      document.removeEventListener(
        "visibilitychange",
        saveBeforeBackgrounding,
      );
    };
  }, [workspaceReady]);

  useEffect(() => {
    if (view !== "sampling") return;
    let cancelled = false;
    void samplingServiceHealth().then((health) => {
      if (!cancelled) {
        setSamplingServiceReady(health.ready);
        setSamplingServiceDetail(health.detail);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [view]);

  const handleDatasetChange = useCallback((nextDataset: Dataset) => {
    const nextValidation = validateDataset(nextDataset);
    setDataset(nextDataset);
    setValidation(nextValidation);
    setConfig((current) => {
      const validPeriods = cyclePeriodsForDataset(nextDataset);
      return validPeriods.some((period) => period === current.cyclePeriod)
        ? current
        : {
            ...current,
            cyclePeriod:
              nextDataset.modelCadence === "monthly"
                ? 6
                : DEFAULT_CONFIG.cyclePeriod,
          };
    });
    setModels({});
    setModelStatuses({ frequentist: "idle", bayesian: "idle" });
    setAdvancedResult(undefined);
    setAdvancedStatus("idle");
    setValidationResults({});
    setValidationStatuses({
      frequentist: "idle",
      bayesian: "idle",
      advanced: "idle",
    });
    setValidationProgress({});
    setAnchorIndependenceConfirmed(false);
    setIndustryPriorChannels((current) =>
      eligibleStoredBenchmarkChannels(nextDataset, experiments, current),
    );
    setBenchmarkScreeningRois({});
    setPromotedAgenticSpecification(undefined);
    resetSampling();
    resetAgenticSearch();
    if (nextValidation.status === "blocked") {
      setEda(null);
      setEdaStatus("idle");
      return;
    }
    void persistDataset(nextDataset);
    setEdaStatus("running");
    window.setTimeout(() => {
      setEda(runEda(nextDataset));
      setEdaStatus("complete");
    }, 240);
  }, [experiments, resetAgenticSearch, resetSampling]);

  const handleModelConfigChange = useCallback((nextConfig: ModelConfig) => {
    setConfig(nextConfig);
    setModels({});
    setModelStatuses({ frequentist: "idle", bayesian: "idle" });
    setAdvancedResult(undefined);
    setAdvancedStatus("idle");
    setValidationResults({});
    setValidationStatuses({
      frequentist: "idle",
      bayesian: "idle",
      advanced: "idle",
    });
    setValidationProgress({});
    setAnchorIndependenceConfirmed(false);
    setBenchmarkScreeningRois({});
    resetAgenticSearch();
  }, [resetAgenticSearch]);

  const handleAdvancedConfigChange = useCallback(
    (nextConfig: AdvancedModelConfig) => {
      setAdvancedConfig(nextConfig);
      setAdvancedResult(undefined);
      setAdvancedStatus("idle");
      setValidationResults((current) => {
        const next = { ...current };
        delete next.advanced;
        return next;
      });
      setValidationStatuses((current) => ({
        ...current,
        advanced: "idle",
      }));
      setValidationProgress((current) => {
        const next = { ...current };
        delete next.advanced;
        return next;
      });
      setAnchorIndependenceConfirmed(false);
      resetAgenticSearch();
    },
    [resetAgenticSearch],
  );

  const handleExperimentsChange = useCallback((nextExperiments: Experiment[]) => {
    setExperiments(nextExperiments);
    setIndustryPriorChannels((current) =>
      dataset
        ? eligibleStoredBenchmarkChannels(dataset, nextExperiments, current)
        : [],
    );
    setBenchmarkScreeningRois({});
    setModels((current) => ({ frequentist: current.frequentist }));
    setModelStatuses((current) => ({
      frequentist: current.frequentist,
      bayesian: "idle",
    }));
    setAdvancedResult(undefined);
    setAdvancedStatus("idle");
    setValidationResults({});
    setValidationStatuses({
      frequentist: "idle",
      bayesian: "idle",
      advanced: "idle",
    });
    setValidationProgress({});
    setAnchorIndependenceConfirmed(false);
    resetAgenticSearch();
  }, [dataset, resetAgenticSearch]);

  const invalidateGuardrailedArtifacts = useCallback(() => {
    setModels((current) => ({ frequentist: current.frequentist }));
    setModelStatuses((current) => ({
      frequentist: current.frequentist,
      bayesian: "idle",
    }));
    setAdvancedResult(undefined);
    setAdvancedStatus("idle");
    setValidationResults({});
    setValidationStatuses({
      frequentist: "idle",
      bayesian: "idle",
      advanced: "idle",
    });
    setValidationProgress({});
    setAnchorIndependenceConfirmed(false);
    resetAgenticSearch();
  }, [resetAgenticSearch]);

  const extremeChannelsFromScreening = useCallback(
    (screening: Record<string, number>) =>
      dataset?.mediaColumns.filter((channel) => {
        if (channelExperiments(experiments, channel).length) return false;
        const roi = screening[channel];
        return (
          roi !== undefined &&
          isHighlyImprobableIndustryRoi(
            roi,
            inferIndustryPrior(channel),
          )
        );
      }) ?? [],
    [dataset, experiments],
  );

  const handleGuardrailModeChange = useCallback(
    (mode: BenchmarkGuardrailMode) => {
      setGuardrailMode(mode);
      const nextChannels =
        mode === "off"
          ? []
          : mode === "auto"
            ? extremeChannelsFromScreening(benchmarkScreeningRois)
            : industryPriorChannels;
      const changed =
        [...nextChannels].sort().join("|") !==
        [...industryPriorChannels].sort().join("|");
      if (changed) {
        setIndustryPriorChannels(nextChannels);
        invalidateGuardrailedArtifacts();
      }
    },
    [
      benchmarkScreeningRois,
      extremeChannelsFromScreening,
      industryPriorChannels,
      invalidateGuardrailedArtifacts,
    ],
  );

  const handleIndustryChannelToggle = useCallback(
    (channel: string, enabled: boolean) => {
      if (channelExperiments(experiments, channel).length) return;
      setIndustryPriorChannels((current) => {
        const withoutChannel = current.filter(
          (item) => item.toLowerCase() !== channel.toLowerCase(),
        );
        return enabled ? [...withoutChannel, channel] : withoutChannel;
      });
      invalidateGuardrailedArtifacts();
    },
    [experiments, invalidateGuardrailedArtifacts],
  );

  const handleAnchorIndependenceChange = useCallback(
    (confirmed: boolean) => {
      setAnchorIndependenceConfirmed(confirmed);
      setValidationResults({});
      setValidationStatuses({
        frequentist: "idle",
        bayesian: "idle",
        advanced: "idle",
      });
      setValidationProgress({});
      resetAgenticSearch();
    },
    [resetAgenticSearch],
  );

  const registerBenchmarkScreening = useCallback(
    (result: ModelResult) => {
      const screening = Object.fromEntries(
        result.channels.map((channel) => [channel.channel, channel.roi]),
      );
      setBenchmarkScreeningRois(screening);
      if (guardrailMode !== "auto") return;
      const extremes = extremeChannelsFromScreening(screening);
      setIndustryPriorChannels(extremes);
      setAdvancedResult(undefined);
      setAdvancedStatus("idle");
      setValidationResults({});
      setValidationStatuses({
        frequentist: "idle",
        bayesian: "idle",
        advanced: "idle",
      });
      setValidationProgress({});
      setAnchorIndependenceConfirmed(false);
      resetAgenticSearch();
    },
    [extremeChannelsFromScreening, guardrailMode, resetAgenticSearch],
  );

  const runRequestedModel = useCallback(async (kind: "frequentist" | "bayesian") => {
    if (!dataset || !validation || validation.status === "blocked") {
      setToast("Resolve blocking schema issues before modeling.");
      setView("data");
      return;
    }
    const selectedIndustryChannels =
      kind === "bayesian" && guardrailMode !== "off"
        ? industryPriorChannels
        : [];
    setView("models");
    setModelStatuses((current) => ({ ...current, [kind]: "queued" }));
    const fingerprint = await modelFingerprint(
      dataset,
      config,
      experiments,
      kind,
      selectedIndustryChannels,
    );
    const cached = await getCachedModel(fingerprint);
    if (cached) {
      setModels((current) => ({ ...current, [kind]: cached }));
      setModelStatuses((current) => ({ ...current, [kind]: "cached" }));
      setValidationResults((current) => {
        const next = { ...current };
        delete next[kind];
        return next;
      });
      setValidationStatuses((current) => ({ ...current, [kind]: "idle" }));
      if (kind === "bayesian" && selectedIndustryChannels.length === 0) {
        registerBenchmarkScreening(cached);
      }
      setToast("Identical run restored from the artifact cache.");
      return;
    }

    setModelStatuses((current) => ({ ...current, [kind]: "running" }));
    await delay(kind === "bayesian" ? 1050 : 650);
    const result = await runModel(
      dataset,
      config,
      experiments,
      kind,
      fingerprint,
      selectedIndustryChannels,
    );
    setModels((current) => ({ ...current, [kind]: result }));
    setModelStatuses((current) => ({ ...current, [kind]: "complete" }));
    setValidationResults((current) => {
      const next = { ...current };
      delete next[kind];
      return next;
    });
    setValidationStatuses((current) => ({ ...current, [kind]: "idle" }));
    if (kind === "bayesian" && selectedIndustryChannels.length === 0) {
      registerBenchmarkScreening(result);
    }
    void persistModel(dataset.hash, result);
    setToast(
      kind === "frequentist"
        ? "Frequentist baseline run complete."
        : selectedIndustryChannels.length
          ? "Guardrailed Bayesian sensitivity run complete."
          : "Experiment-only ROI screening complete.",
    );
  }, [
    config,
    dataset,
    experiments,
    guardrailMode,
    industryPriorChannels,
    registerBenchmarkScreening,
    validation,
  ]);

  const runRequestedAdvancedModel = useCallback(async () => {
    if (!dataset || !validation || validation.status === "blocked") {
      setToast("Resolve blocking schema issues before modeling.");
      setView("data");
      return;
    }
    setView("advanced");
    setAdvancedStatus("queued");
    const fingerprint = await advancedModelFingerprint(
      dataset,
      config,
      advancedConfig,
      experiments,
      guardrailMode === "off" ? [] : industryPriorChannels,
    );
    const cached = await getCachedModel(fingerprint);
    if (cached?.kind === "advanced") {
      setAdvancedResult(cached);
      setAdvancedStatus("cached");
      setValidationResults((current) => {
        const next = { ...current };
        delete next.advanced;
        return next;
      });
      setValidationStatuses((current) => ({
        ...current,
        advanced: "idle",
      }));
      setToast("Identical advanced run restored from the artifact cache.");
      return;
    }

    setAdvancedStatus("running");
    await delay(1050);
    const result = await runAdvancedModel(
      dataset,
      config,
      advancedConfig,
      experiments,
      fingerprint,
      guardrailMode === "off" ? [] : industryPriorChannels,
    );
    setAdvancedResult(result);
    setAdvancedStatus("complete");
    setValidationResults((current) => {
      const next = { ...current };
      delete next.advanced;
      return next;
    });
    setValidationStatuses((current) => ({
      ...current,
      advanced: "idle",
    }));
    void persistModel(dataset.hash, result);
    setToast("Advanced model run complete.");
  }, [
    advancedConfig,
    config,
    dataset,
    experiments,
    guardrailMode,
    industryPriorChannels,
    validation,
  ]);

  const runRequestedValidation = useCallback(
    async (kind: ValidationModelKind) => {
      if (!dataset || !validation || validation.status === "blocked") {
        setToast("Resolve blocking schema issues before validation.");
        setView("data");
        return;
      }
      const model =
        kind === "advanced" ? advancedResult : models[kind];
      if (!model) {
        setToast(`Run the ${VALIDATION_MODEL_LABELS[kind]} model first.`);
        setView(kind === "advanced" ? "advanced" : "models");
        return;
      }
      const modelIndustryChannels = model.channels
        .filter((channel) => channel.priorSource === "industry")
        .map((channel) => channel.channel);
      setView("validation");
      setValidationStatuses((current) => ({
        ...current,
        [kind]: "queued",
      }));
      const fingerprint = await validationFingerprint(
        dataset,
        model,
        config,
        advancedConfig,
        experiments,
        {
          anchorIndependenceConfirmed,
          industryPriorChannels: modelIndustryChannels,
          industryBenchmarkScreeningEnabled: guardrailMode !== "off",
        },
      );
      const cached = await getCachedValidation(fingerprint);
      if (cached) {
        setValidationResults((current) => ({
          ...current,
          [kind]: cached,
        }));
        setValidationStatuses((current) => ({
          ...current,
          [kind]: "cached",
        }));
        setValidationProgress((current) => {
          const next = { ...current };
          delete next[kind];
          return next;
        });
        setToast(`${VALIDATION_MODEL_LABELS[kind]} validation restored from cache.`);
        return;
      }
      setValidationStatuses((current) => ({
        ...current,
        [kind]: "running",
      }));
      setValidationProgress((current) => ({
        ...current,
        [kind]: {
          stage: "structure",
          detail: "Checking the selected model assumptions…",
          layers: {},
        },
      }));
      await delay(180);
      const result = await runModelValidation(
        dataset,
        model,
        config,
        advancedConfig,
        experiments,
        fingerprint,
        (nextProgress) =>
          setValidationProgress((current) => ({
            ...current,
            [kind]: nextProgress,
          })),
        {
          anchorIndependenceConfirmed,
          industryPriorChannels: modelIndustryChannels,
          industryBenchmarkScreeningEnabled: guardrailMode !== "off",
        },
      );
      setValidationResults((current) => ({
        ...current,
        [kind]: result,
      }));
      setValidationStatuses((current) => ({
        ...current,
        [kind]: "complete",
      }));
      void persistValidation(dataset.hash, result);
      setToast(`${VALIDATION_MODEL_LABELS[kind]} validation complete.`);
    },
    [
      advancedConfig,
      advancedResult,
      anchorIndependenceConfirmed,
      config,
      dataset,
      experiments,
      guardrailMode,
      models,
      validation,
    ],
  );

  const runAllValidations = useCallback(() => {
    const available = ([
      "frequentist",
      "bayesian",
      "advanced",
    ] as const).filter((kind) =>
      kind === "advanced" ? Boolean(advancedResult) : Boolean(models[kind]),
    );
    void Promise.all(available.map((kind) => runRequestedValidation(kind)));
  }, [advancedResult, models, runRequestedValidation]);

  const handleAgenticContractChange = useCallback(
    (nextContract: AgenticSearchContract) => {
      setAgenticContract(nextContract);
      resetAgenticSearch();
    },
    [resetAgenticSearch],
  );

  const runAgenticSearch = useCallback(async () => {
    if (!dataset || !validation || validation.status === "blocked") {
      setToast("Resolve blocking schema issues before starting model search.");
      setView("data");
      return;
    }
    const generation = agenticGenerationRef.current + 1;
    agenticGenerationRef.current = generation;
    const capabilities = {
      likelihoodCalibration: experiments.length > 0,
      mediaColumns: dataset.mediaColumns,
    };
    const seedSpecifications = generateAgenticSeeds(
      config,
      advancedConfig,
      agenticContract,
      capabilities,
    );
    const advancedChallengeBudget =
      agenticAdvancedChallengeBudget(agenticContract);
    const responseChallengeBudget = agenticChannelResponseBudget(
      agenticContract,
      capabilities,
    );
    const localChallengeBudget =
      agenticLocalChallengeBudget(agenticContract);
    const localChallengeStart =
      agenticContract.candidateBudget - localChallengeBudget;
    const selectedIndustryChannels =
      guardrailMode === "off" ? [] : industryPriorChannels;
    setAgenticRuns(
      seedSpecifications.map((spec) => ({ spec, state: "queued" })),
    );
    setAgenticStatus("running");
    setAgenticStopReason(undefined);
    setPromotedAgenticSpecification(undefined);
    setView("agentic");
    const evaluatedRuns: AgenticCandidateRun[] = [];
    let finalStopReason: string | undefined;

    for (
      let attemptIndex = 0;
      attemptIndex < agenticContract.candidateBudget;
      attemptIndex += 1
    ) {
      if (agenticGenerationRef.current !== generation) return;
      let spec: AgenticCandidateRun["spec"];
      if (attemptIndex < seedSpecifications.length) {
        spec = seedSpecifications[attemptIndex];
      } else if (
        attemptIndex < seedSpecifications.length + responseChallengeBudget
      ) {
        spec = generateAgenticChannelResponseChallenge(
          config,
          advancedConfig,
          agenticContract,
          attemptIndex + 1,
          capabilities,
        );
        setAgenticRuns((current) => [
          ...current,
          { spec, state: "queued" },
        ]);
      } else if (
        attemptIndex <
        seedSpecifications.length + responseChallengeBudget + advancedChallengeBudget
      ) {
        spec = generateAgenticAdvancedChallenge(
          config,
          agenticContract,
          evaluatedRuns,
          attemptIndex + 1,
          capabilities,
        );
        setAgenticRuns((current) => [
          ...current,
          { spec, state: "queued" },
        ]);
      } else if (attemptIndex < localChallengeStart) {
        try {
          spec = proposeAgenticCandidate(
            config,
            advancedConfig,
            agenticContract,
            evaluatedRuns,
            attemptIndex + 1,
            capabilities,
          );
          setAgenticRuns((current) => [
            ...current,
            { spec, state: "queued" },
          ]);
        } catch (error) {
          finalStopReason =
            error instanceof Error
              ? error.message
              : "The bounded adaptive search space was exhausted.";
          break;
        }
      } else {
        try {
          spec = generateAgenticLocalChallenge(
            agenticContract,
            evaluatedRuns,
            attemptIndex + 1,
            capabilities,
          );
          setAgenticRuns((current) => [
            ...current,
            { spec, state: "queued" },
          ]);
        } catch (error) {
          finalStopReason =
            error instanceof Error
              ? error.message
              : "The local champion challenge could not be generated.";
          break;
        }
      }
      const fitIndustryChannels =
        spec.family === "frequentist"
          ? []
          : spec.evidencePriorChannels ?? selectedIndustryChannels;
      spec = {
        ...spec,
        evidencePriorChannels: [...fitIndustryChannels],
      };
      setAgenticRuns((current) =>
        current.map((run) =>
          run.spec.id === spec.id
            ? { ...run, spec, state: "running", error: undefined }
            : run,
        ),
      );
      await delay(24);

      try {
        const modelKey =
          spec.family === "advanced"
            ? await advancedModelFingerprint(
                dataset,
                spec.config,
                spec.advancedConfig,
                experiments,
                fitIndustryChannels,
              )
            : await modelFingerprint(
                dataset,
                spec.config,
                experiments,
                spec.family,
                fitIndustryChannels,
              );
        let model = await getCachedModel(modelKey);
        const modelWasCached = Boolean(model);
        if (!model) {
          model =
            spec.family === "advanced"
              ? await runAdvancedModel(
                  dataset,
                  spec.config,
                  spec.advancedConfig,
                  experiments,
                  modelKey,
                  fitIndustryChannels,
                )
              : await runModel(
                  dataset,
                  spec.config,
                  experiments,
                  spec.family,
                  modelKey,
                  fitIndustryChannels,
                );
          void persistModel(dataset.hash, model);
        }

        const roiGuardrailViolations =
          findAgenticRoiGuardrailViolations(
            model,
            experiments,
            guardrailMode !== "off",
            dataset,
          );
        const validationOptions = {
          anchorIndependenceConfirmed,
          industryPriorChannels: fitIndustryChannels,
          industryBenchmarkScreeningEnabled: guardrailMode !== "off",
        };
        const validationKey = await validationFingerprint(
          dataset,
          model,
          spec.config,
          spec.advancedConfig,
          experiments,
          validationOptions,
        );
        let candidateValidation =
          await getCachedValidation(validationKey);
        const validationWasCached = Boolean(candidateValidation);
        if (!candidateValidation) {
          candidateValidation = await runModelValidation(
            dataset,
            model,
            spec.config,
            spec.advancedConfig,
            experiments,
            validationKey,
            (progress) => {
              if (agenticGenerationRef.current !== generation) return;
              setAgenticRuns((current) =>
                current.map((run) =>
                  run.spec.id === spec.id
                    ? { ...run, progress }
                    : run,
                ),
              );
            },
            validationOptions,
          );
          void persistValidation(dataset.hash, candidateValidation);
        }
        if (agenticGenerationRef.current !== generation) return;
        const completedRun: AgenticCandidateRun = {
          spec,
          state: "complete",
          model,
          validation: candidateValidation,
          roiGuardrailViolations,
          restoredFromCache: modelWasCached && validationWasCached,
        };
        evaluatedRuns.push(completedRun);
        setAgenticRuns((current) =>
          current.map((run) =>
            run.spec.id === spec.id
              ? { ...completedRun, progress: undefined }
              : run,
          ),
        );
        const stopping = agenticStoppingDecision(
          evaluatedRuns,
          agenticContract,
          capabilities,
        );
        if (
          attemptIndex + 1 >= seedSpecifications.length &&
          stopping.shouldStop
        ) {
          finalStopReason = stopping.reason;
          break;
        }
      } catch (error) {
        if (agenticGenerationRef.current !== generation) return;
        const failedRun: AgenticCandidateRun = {
          spec,
          state: "error",
          error:
            error instanceof Error
              ? error.message
              : "Candidate evaluation failed.",
        };
        evaluatedRuns.push(failedRun);
        setAgenticRuns((current) =>
          current.map((run) =>
            run.spec.id === spec.id
              ? { ...failedRun, progress: undefined }
              : run,
          ),
        );
      }
    }

    if (agenticGenerationRef.current !== generation) return;
    const rescueParents = rankAgenticCandidates(evaluatedRuns)
      .filter(
        (run) =>
          run.state === "complete" &&
          run.spec.family !== "frequentist" &&
          Boolean(run.model) &&
          findAgenticBenchmarkRescueRecommendations(
            run.model!,
            experiments,
            guardrailMode !== "off",
            dataset,
            run.spec.evidencePriorChannels ?? [],
          ).length > 0,
      )
      .slice(0, 3);

    for (const parent of rescueParents) {
      if (agenticGenerationRef.current !== generation) return;
      const recommendations = findAgenticBenchmarkRescueRecommendations(
        parent.model!,
        experiments,
        guardrailMode !== "off",
        dataset,
        parent.spec.evidencePriorChannels ?? [],
      );
      const additionalChannels = recommendations.map(
        (recommendation) => recommendation.channel,
      );
      const channelBenchmarks = recommendations.filter(
        (recommendation) => recommendation.role === "channel-benchmark",
      );
      const weakFallbacks = recommendations.filter(
        (recommendation) => recommendation.role === "weak-fallback",
      );
      const rescueIndustryChannels = Array.from(
        new Set([
          ...(parent.spec.evidencePriorChannels ?? []),
          ...additionalChannels,
        ]),
      );
      const rescueSpec: AgenticCandidateRun["spec"] = {
        ...parent.spec,
        id: `${parent.spec.id}R`,
        label: `${parent.spec.label} · evidence rescue`,
        summary: `Paired evidence refit of ${parent.spec.id} for ${additionalChannels.map(cleanChannel).join(", ")}`,
        hypothesis:
          `${channelBenchmarks.length ? `Channel benchmark: ${channelBenchmarks.map((item) => cleanChannel(item.channel)).join(", ")}. ` : ""}${weakFallbacks.length ? `Weak broad fallback: ${weakFallbacks.map((item) => cleanChannel(item.channel)).join(", ")}. ` : ""}The paired refit must improve the independent V6 score; benchmark agreement itself earns no validation credit.`,
        searchPhase: "rescue",
        rescueOf: parent.spec.id,
        evidencePriorChannels: rescueIndustryChannels,
        proposal: {
          method: "evidence-rescue",
          reason: `Triggered because ${parent.spec.id} placed material unanchored ROI outside its external plausibility range. The original fit remains unchanged for a non-circular paired comparison.`,
        },
      };
      setAgenticRuns((current) => [
        ...current,
        { spec: rescueSpec, state: "running" },
      ]);
      try {
        const modelKey =
          rescueSpec.family === "advanced"
            ? await advancedModelFingerprint(
                dataset,
                rescueSpec.config,
                rescueSpec.advancedConfig,
                experiments,
                rescueIndustryChannels,
              )
            : await modelFingerprint(
                dataset,
                rescueSpec.config,
                experiments,
                "bayesian",
                rescueIndustryChannels,
              );
        let rescueModel = await getCachedModel(modelKey);
        const modelWasCached = Boolean(rescueModel);
        if (!rescueModel) {
          rescueModel =
            rescueSpec.family === "advanced"
              ? await runAdvancedModel(
                  dataset,
                  rescueSpec.config,
                  rescueSpec.advancedConfig,
                  experiments,
                  modelKey,
                  rescueIndustryChannels,
                )
              : await runModel(
                  dataset,
                  rescueSpec.config,
                  experiments,
                  "bayesian",
                  modelKey,
                  rescueIndustryChannels,
                );
          void persistModel(dataset.hash, rescueModel);
        }
        const validationOptions = {
          anchorIndependenceConfirmed,
          industryPriorChannels: rescueIndustryChannels,
          industryBenchmarkScreeningEnabled: guardrailMode !== "off",
        };
        const validationKey = await validationFingerprint(
          dataset,
          rescueModel,
          rescueSpec.config,
          rescueSpec.advancedConfig,
          experiments,
          validationOptions,
        );
        let rescueValidation = await getCachedValidation(validationKey);
        const validationWasCached = Boolean(rescueValidation);
        if (!rescueValidation) {
          rescueValidation = await runModelValidation(
            dataset,
            rescueModel,
            rescueSpec.config,
            rescueSpec.advancedConfig,
            experiments,
            validationKey,
            undefined,
            validationOptions,
          );
          void persistValidation(dataset.hash, rescueValidation);
        }
        const rescueRun: AgenticCandidateRun = {
          spec: rescueSpec,
          state: "complete",
          model: rescueModel,
          validation: rescueValidation,
          roiGuardrailViolations: findAgenticRoiGuardrailViolations(
            rescueModel,
            experiments,
            guardrailMode !== "off",
            dataset,
          ),
          restoredFromCache: modelWasCached && validationWasCached,
        };
        evaluatedRuns.push(rescueRun);
        setAgenticRuns((current) =>
          current.map((run) =>
            run.spec.id === rescueSpec.id ? rescueRun : run,
          ),
        );
      } catch (error) {
        const failedRescue: AgenticCandidateRun = {
          spec: rescueSpec,
          state: "error",
          error:
            error instanceof Error
              ? error.message
              : "Evidence rescue refit failed.",
        };
        evaluatedRuns.push(failedRescue);
        setAgenticRuns((current) =>
          current.map((run) =>
            run.spec.id === rescueSpec.id ? failedRescue : run,
          ),
        );
      }
    }

    const scoredCount = evaluatedRuns.filter(
      (run) => run.validation?.finalScore !== null &&
        run.validation?.finalScore !== undefined,
    ).length;
    setAgenticStopReason(
      finalStopReason ??
        `Search budget closed after ${evaluatedRuns.length} attempted specifications (${scoredCount} scored).`,
    );
    setAgenticStatus("complete");
    setToast("Adaptive Agentic search complete. The best eligible candidate is ready.");
  }, [
    advancedConfig,
    agenticContract,
    anchorIndependenceConfirmed,
    config,
    dataset,
    experiments,
    guardrailMode,
    industryPriorChannels,
    validation,
  ]);

  const loadAgenticCandidate = useCallback(
    (
      run: AgenticCandidateRun,
      options: { navigate: boolean; promote: boolean; force?: boolean },
    ) => {
      if (!run.model || !run.validation) return;
      if (options.promote && !options.force && !passesAgenticEligibility(run)) {
        setToast("This candidate is not eligible. Use Force Promote to preserve an explicit override receipt.");
        return;
      }
      setConfig(run.spec.config);
      if (run.spec.family === "advanced") {
        setAdvancedConfig(run.spec.advancedConfig);
      }
      setValidationResults({ [run.spec.family]: run.validation });
      setValidationStatuses({
        frequentist:
          run.spec.family === "frequentist"
            ? run.validation.cached
              ? "cached"
              : "complete"
            : "idle",
        bayesian:
          run.spec.family === "bayesian"
            ? run.validation.cached
              ? "cached"
              : "complete"
            : "idle",
        advanced:
          run.spec.family === "advanced"
            ? run.validation.cached
              ? "cached"
              : "complete"
            : "idle",
      });
      setValidationProgress({});
      if (run.spec.family === "advanced") {
        setModels({});
        setModelStatuses({ frequentist: "idle", bayesian: "idle" });
        setAdvancedResult(run.model);
        setAdvancedStatus(run.model.cached ? "cached" : "complete");
      } else {
        setModels({ [run.spec.family]: run.model });
        setModelStatuses({
          frequentist:
            run.spec.family === "frequentist"
              ? run.model.cached
                ? "cached"
                : "complete"
              : "idle",
          bayesian:
            run.spec.family === "bayesian"
              ? run.model.cached
                ? "cached"
                : "complete"
              : "idle",
        });
        setAdvancedResult(undefined);
        setAdvancedStatus("idle");
      }
      if (options.promote) {
        const promotedIndustryChannels =
          run.spec.family === "frequentist"
            ? []
            : [...(run.spec.evidencePriorChannels ?? [])];
        setIndustryPriorChannels(promotedIndustryChannels);
        resetSampling();
        setSamplingContract(DEFAULT_SAMPLING_CONTRACT);
        setPromotedAgenticSpecification({
          run,
          datasetHash: dataset?.hash ?? "",
          experiments: [...experiments],
          industryPriorChannels: promotedIndustryChannels,
          override: options.force
            ? createAgenticForcePromotionAudit(run)
            : undefined,
        });
        setView(run.spec.family === "advanced" ? "advanced" : "models");
        setToast(
          options.force
            ? `${run.spec.label} force promoted. Failed gates and ROI warnings remain attached.`
            : `${run.spec.label} promoted with its exact ${
                run.spec.family === "advanced"
                  ? "base and advanced"
                  : "base"
              } settings.`,
        );
      }
      if (options.navigate && !options.promote) setView("validation");
    },
    [
      dataset,
      experiments,
      resetSampling,
    ],
  );

  const inspectAgenticCandidate = useCallback(
    (run: AgenticCandidateRun) =>
      loadAgenticCandidate(run, { navigate: true, promote: false }),
    [loadAgenticCandidate],
  );

  const promoteAgenticCandidate = useCallback(
    (run: AgenticCandidateRun) =>
      loadAgenticCandidate(run, { navigate: false, promote: true }),
    [loadAgenticCandidate],
  );

  const forcePromoteAgenticCandidate = useCallback(
    (run: AgenticCandidateRun) =>
      loadAgenticCandidate(run, {
        navigate: false,
        promote: true,
        force: true,
      }),
    [loadAgenticCandidate],
  );

  const runProductionSampling = useCallback(async () => {
    if (!dataset || !promotedAgenticSpecification?.run.model) {
      setToast("Promote a validated Agentic winner before sampling.");
      setView("agentic");
      return;
    }
    if (
      promotedAgenticSpecification.run.spec.family !== "bayesian" &&
      promotedAgenticSpecification.run.spec.family !== "advanced"
    ) {
      setToast(
        "Production sampling requires a Bayesian or Advanced counterpart.",
      );
      return;
    }
    const health = await samplingServiceHealth();
    setSamplingServiceReady(health.ready);
    setSamplingServiceDetail(health.detail);
    if (!health.ready) {
      setToast(
        health.contractVersion
          ? "The workspace and MCMC service are on different versions. Refresh Flux before starting Production sampling."
          : "The local MCMC service is unavailable. Restart Flux to activate it.",
      );
      return;
    }

    resetBudget();

    const generation = samplingGenerationRef.current + 1;
    samplingGenerationRef.current = generation;
    setSamplingStatus("queued");
    setSamplingProgress({
      stage: "queued",
      completed: 0,
      total:
        samplingContract.chains *
        (samplingContract.tune + samplingContract.draws),
      chain: 0,
      detail: "Fingerprinting the promoted probabilistic specification.",
    });
    setView("sampling");

    try {
      const compiled = compileSamplingModel(
        dataset,
        promotedAgenticSpecification.run,
        promotedAgenticSpecification.experiments,
        promotedAgenticSpecification.industryPriorChannels,
      );
      const fingerprint = await samplingFingerprint(
        compiled,
        samplingContract,
      );
      const cached = await getCachedSampling(fingerprint);
      if (cached) {
        if (samplingGenerationRef.current !== generation) return;
        setSamplingResult(cached);
        setSamplingHistory((current) => [
          ...current.filter((item) => item.fingerprint !== cached.fingerprint),
          cached,
        ]);
        setSamplingStatus("cached");
        setSamplingProgress(undefined);
        setToast("Identical posterior restored from the artifact cache.");
        return;
      }

      setSamplingStatus("running");
      const started = await startSamplingJob(
        fingerprint,
        compiled,
        samplingContract,
      );
      let snapshot = await getSamplingJob(started.id);
      while (
        snapshot.status === "queued" ||
        snapshot.status === "running"
      ) {
        if (samplingGenerationRef.current !== generation) return;
        setSamplingProgress(snapshot.progress);
        await delay(650);
        snapshot = await getSamplingJob(started.id);
      }
      if (samplingGenerationRef.current !== generation) return;
      setSamplingProgress(snapshot.progress);
      if (snapshot.status === "error" || !snapshot.result) {
        throw new Error(
          snapshot.error ?? "Sampling ended without a posterior artifact.",
        );
      }
      const result = snapshot.result;
      setSamplingResult(result);
      setSamplingHistory((current) => [
        ...current.filter((item) => item.fingerprint !== result.fingerprint),
        result,
      ]);
      setSamplingStatus(result.cached ? "cached" : "complete");
      setSamplingProgress(undefined);
      void persistSampling(dataset.hash, result);
      setToast(
        result.status === "ready"
          ? "Production posterior passed every convergence gate."
          : "Sampling complete. Review the recommended diagnostic action.",
      );
    } catch (error) {
      if (samplingGenerationRef.current !== generation) return;
      setSamplingStatus("idle");
      setSamplingProgress(undefined);
      setToast(
        error instanceof Error
          ? error.message
          : "Production sampling could not be completed.",
      );
    }
  }, [dataset, promotedAgenticSpecification, resetBudget, samplingContract]);

  const applyRecommendedSamplingRetry = useCallback(
    (nextContract: SamplingContract) => {
      setSamplingContract(nextContract);
      resetSampling(false);
      setView("sampling");
      setToast("Recommended retry settings loaded; the original artifact is preserved.");
    },
    [resetSampling],
  );

  const prepareNewSamplingRun = useCallback(() => {
    resetSampling(false);
    setView("sampling");
  }, [resetSampling]);

  const handleBudgetContractChange = useCallback(
    (nextContract: BudgetOptimizationContract) => {
      budgetGenerationRef.current += 1;
      setBudgetContract(nextContract);
      setBudgetResult(undefined);
      setBudgetStatus("idle");
      setBudgetProgress(undefined);
    },
    [],
  );

  const runBudgetPlan = useCallback(async () => {
    if (
      !dataset ||
      !samplingResult ||
      !promotedAgenticSpecification?.run.model
    ) {
      setToast("Complete Production sampling before optimizing a budget plan.");
      setView("sampling");
      return;
    }
    const generation = budgetGenerationRef.current + 1;
    budgetGenerationRef.current = generation;
    setView("budget");
    setBudgetStatus("queued");
    setBudgetProgress({
      stage: "preparing",
      completed: 0,
      total: 25,
      detail: "Fingerprinting the posterior planning contract.",
    });
    try {
      const fingerprint = await budgetFingerprint(
        dataset,
        samplingResult,
        budgetContract,
      );
      const cached = await getCachedBudget(fingerprint);
      if (cached) {
        if (budgetGenerationRef.current !== generation) return;
        setBudgetResult(cached);
        setBudgetStatus("cached");
        setBudgetProgress(undefined);
        setToast("Identical budget recommendation restored from the artifact cache.");
        return;
      }
      setBudgetStatus("running");
      const optimized = await runBudgetOptimization(
        dataset,
        promotedAgenticSpecification.run,
        samplingResult,
        budgetContract,
        fingerprint,
        (nextProgress) => {
          if (budgetGenerationRef.current === generation) {
            setBudgetProgress(nextProgress);
          }
        },
      );
      if (budgetGenerationRef.current !== generation) return;
      setBudgetResult(optimized);
      setBudgetStatus("complete");
      setBudgetProgress(undefined);
      void persistBudget(dataset.hash, optimized);
      setToast(
        optimized.status === "supported"
          ? "Budget recommendation passed every decision guardrail."
          : "Budget scenario complete. Review the preserved decision warnings.",
      );
    } catch (error) {
      if (budgetGenerationRef.current !== generation) return;
      setBudgetStatus("idle");
      setBudgetProgress(undefined);
      setToast(
        error instanceof Error
          ? error.message
          : "The budget recommendation could not be completed.",
      );
    }
  }, [
    budgetContract,
    dataset,
    promotedAgenticSpecification,
    samplingResult,
  ]);

  const upload = useCallback(() => {
    if (!demoMode) fileRef.current?.click();
  }, [demoMode]);

  const activeContent = useMemo(() => {
    if (!dataset || !validation) return <EmptyState />;
    if (view === "data") {
      return <DataView dataset={dataset} validation={validation} onDatasetChange={handleDatasetChange} onUpload={upload} demoMode={demoMode} />;
    }
    if (view === "eda") {
      return <EdaView dataset={dataset} eda={eda} status={edaStatus} />;
    }
    if (view === "models") {
      return (
        <ModelsView
          dataset={dataset}
          config={config}
          setConfig={handleModelConfigChange}
          results={models}
          statuses={modelStatuses}
          promotion={promotedAgenticSpecification}
          experiments={experiments}
          industryPriorChannels={
            guardrailMode === "off" ? [] : industryPriorChannels
          }
          onRun={runRequestedModel}
        />
      );
    }
    if (view === "advanced") {
      return (
        <AdvancedModelerView
          dataset={dataset}
          config={config}
          setConfig={handleModelConfigChange}
          advancedConfig={advancedConfig}
          setAdvancedConfig={handleAdvancedConfigChange}
          result={advancedResult}
          status={advancedStatus}
          experiments={experiments}
          industryPriorChannels={
            guardrailMode === "off" ? [] : industryPriorChannels
          }
          promotion={promotedAgenticSpecification}
          onRun={runRequestedAdvancedModel}
        />
      );
    }
    if (view === "agentic") {
      return (
        <AgenticView
          contract={agenticContract}
          setContract={handleAgenticContractChange}
          config={config}
          advancedConfig={advancedConfig}
          experiments={experiments}
          industryPriorChannels={
            guardrailMode === "off" ? [] : industryPriorChannels
          }
          benchmarkScreeningEnabled={guardrailMode !== "off"}
          dataset={dataset}
          runs={agenticRuns}
          status={agenticStatus}
          stopReason={agenticStopReason}
          promotedFingerprint={
            promotedAgenticSpecification?.run.model?.fingerprint
          }
          onStart={() => void runAgenticSearch()}
          onInspect={inspectAgenticCandidate}
          onPromote={promoteAgenticCandidate}
          onForcePromote={forcePromoteAgenticCandidate}
        />
      );
    }
    if (view === "scorelab") {
      return <ScoreLabView />;
    }
    if (view === "sampling") {
      return (
        <SamplingView
          promotion={promotedAgenticSpecification}
          contract={samplingContract}
          setContract={setSamplingContract}
          result={samplingResult}
          history={samplingHistory}
          status={samplingStatus}
          progress={samplingProgress}
          serviceReady={samplingServiceReady}
          serviceDetail={samplingServiceDetail}
          onRun={() => void runProductionSampling()}
          onApplyRetry={applyRecommendedSamplingRetry}
          onNewContract={prepareNewSamplingRun}
          onOpenAgentic={() => setView("agentic")}
          onSelectHistory={(selected) => {
            resetBudget();
            setSamplingResult(selected);
            setSamplingContract(selected.contract);
            setSamplingStatus(selected.cached ? "cached" : "complete");
          }}
        />
      );
    }
    if (view === "budget") {
      return (
        <BudgetView
          dataset={dataset}
          promotion={promotedAgenticSpecification}
          sampling={samplingResult}
          contract={budgetContract}
          setContract={handleBudgetContractChange}
          result={budgetResult}
          status={budgetStatus}
          progress={budgetProgress}
          onRun={() => void runBudgetPlan()}
          onOpenProduction={() => setView("sampling")}
        />
      );
    }
    if (view === "validation") {
      return (
        <ValidationView
          dataset={dataset}
          models={{
            frequentist: models.frequentist,
            bayesian: models.bayesian,
            advanced: advancedResult,
          }}
          results={validationResults}
          statuses={validationStatuses}
          progress={validationProgress}
          anchorQualificationAvailable={
            assessExternalAnchorEligibility(experiments).status ===
            "needs-confirmation"
          }
          anchorIndependenceConfirmed={anchorIndependenceConfirmed}
          onAnchorIndependenceChange={handleAnchorIndependenceChange}
          onRun={(kind) => void runRequestedValidation(kind)}
          onRunAll={runAllValidations}
          onNavigate={setView}
        />
      );
    }
    if (view === "calibration") {
      return (
        <CalibrationView
          dataset={dataset}
          experiments={experiments}
          setExperiments={handleExperimentsChange}
          guardrailMode={guardrailMode}
          onGuardrailModeChange={handleGuardrailModeChange}
          industryPriorChannels={
            guardrailMode === "off" ? [] : industryPriorChannels
          }
          onIndustryChannelToggle={handleIndustryChannelToggle}
          screeningRois={benchmarkScreeningRois}
          bayesianResult={models.bayesian}
          onRun={() => runRequestedModel("bayesian")}
        />
      );
    }
    return <OverviewView dataset={dataset} validation={validation} eda={eda} edaStatus={edaStatus} models={models} onNavigate={setView} />;
  }, [advancedConfig, advancedResult, advancedStatus, agenticContract, agenticRuns, agenticStatus, agenticStopReason, anchorIndependenceConfirmed, applyRecommendedSamplingRetry, benchmarkScreeningRois, budgetContract, budgetProgress, budgetResult, budgetStatus, config, dataset, demoMode, eda, edaStatus, experiments, forcePromoteAgenticCandidate, guardrailMode, handleAdvancedConfigChange, handleAgenticContractChange, handleAnchorIndependenceChange, handleBudgetContractChange, handleDatasetChange, handleExperimentsChange, handleGuardrailModeChange, handleIndustryChannelToggle, handleModelConfigChange, industryPriorChannels, inspectAgenticCandidate, modelStatuses, models, prepareNewSamplingRun, promoteAgenticCandidate, promotedAgenticSpecification, resetBudget, runAgenticSearch, runAllValidations, runBudgetPlan, runProductionSampling, runRequestedAdvancedModel, runRequestedModel, runRequestedValidation, samplingContract, samplingHistory, samplingProgress, samplingResult, samplingServiceDetail, samplingServiceReady, samplingStatus, upload, validation, validationProgress, validationResults, validationStatuses, view]);

  if (!dataset || !validation) return <EmptyState />;

  return (
    <div className="app-shell">
      <Sidebar view={view} setView={setView} validation={validation} />
      <div className="workspace">
        <Header
          dataset={dataset}
          onUpload={upload}
          demoMode={demoMode}
          saveStatus={workspaceSaveStatus}
          savedAt={workspaceSavedAt}
        />
        {activeContent}
      </div>
      {!demoMode && (
        <input
          ref={fileRef}
          className="hidden-input"
          type="file"
          accept=".csv,text/csv"
          onChange={async (event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            try {
              const uploadValidation = await loadCsv(await file.text(), file.name);
              setView("data");
              setToast(
                uploadValidation.status === "blocked"
                  ? `${file.name} loaded. Resolve the blocking data-contract issues before modeling.`
                  : uploadValidation.automaticRepairs.length > 0
                    ? `${file.name} uploaded and repaired automatically: ${uploadValidation.automaticRepairs.map((repair) => repair.title.toLowerCase()).join(", ")}. Bundled demo experiments were cleared.`
                    : `${file.name} uploaded and validated. Bundled demo experiments were cleared.`,
              );
            } catch (error) {
              setToast(error instanceof Error ? error.message : "The CSV could not be parsed.");
            }
            event.target.value = "";
          }}
        />
      )}
      {toast && <div className="toast"><span>✓</span>{toast}</div>}
    </div>
  );
}
