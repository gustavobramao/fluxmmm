# Reproduce the default four-year weekly AMSS vignette scenario.
#
# This is a Step 2 interface smoke test. The low-repetition ROAS calculations
# below verify the truth/counterfactual API; they are not V10 audit labels.

options(stringsAsFactors = FALSE, width = 120)

required_packages <- c("amss", "assertthat", "data.table")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    "Missing R packages: ",
    paste(missing_packages, collapse = ", "),
    ". Set R_LIBS_USER to the temporary Step 2 library."
  )
}

library(amss)
library(data.table)

set.seed(1L)

n.years <- 4L
time.n <- n.years * 52L
budget.index <- rep(seq_len(n.years), each = 52L)

activity.transition <- matrix(
  c(
    0.60, 0.30, 0.10,
    0.60, 0.30, 0.10,
    0.60, 0.30, 0.10
  ),
  nrow = length(kActivityStates),
  byrow = TRUE
)
favorability.transition <- matrix(
  rep(c(0.03, 0.07, 0.65, 0.20, 0.05), 5L),
  nrow = length(kFavorabilityStates),
  byrow = TRUE
)

market.rate.nonoise <- SimulateSinusoidal(
  time.n,
  52,
  vert.trans = 0.6,
  amplitude = 0.25
)
market.rate.seas <- pmax(
  0,
  pmin(
    1,
    market.rate.nonoise *
      SimulateAR1(length(market.rate.nonoise), 1, 0.1, 0.3)
  )
)

nat.mig.params <- list(
  population = 2.4e8,
  market.rate.trend = 0.68,
  market.rate.seas = market.rate.seas,
  prop.activity = c(0.375, 0.425, 0.2),
  prop.favorability = c(0.03, 0.07, 0.65, 0.20, 0.05),
  prop.loyalty = c(1, 0, 0),
  transition.matrices = list(
    activity = activity.transition,
    favorability = favorability.transition
  )
)

tv.flighting <- pmax(
  0,
  market.rate.seas +
    SimulateAR1(length(market.rate.seas), -0.7, 0.7, -0.7)
)
tv.flighting <- tv.flighting[c(6:length(tv.flighting), 1:5)]
tv.activity.trans.mat <- diag(length(kActivityStates))
tv.favorability.trans.mat <- matrix(
  c(
    0.4, 0.0, 0.4, 0.2, 0.0,
    0.0, 0.9, 0.1, 0.0, 0.0,
    0.0, 0.0, 0.6, 0.4, 0.0,
    0.0, 0.0, 0.0, 0.8, 0.2,
    0.0, 0.0, 0.0, 0.0, 1.0
  ),
  nrow = length(kFavorabilityStates),
  byrow = TRUE
)
params.tv <- list(
  audience.membership = list(activity = rep(0.4, 3)),
  budget = rep(c(545e5, 475e5, 420e5, 455e5), length = n.years),
  budget.index = budget.index,
  flighting = tv.flighting,
  unit.cost = 0.005,
  hill.ec = 1.56,
  hill.slope = 1,
  transition.matrices = list(
    activity = tv.activity.trans.mat,
    favorability = tv.favorability.trans.mat
  )
)

spend.cap.fn <- function(time.index, budget, budget.index) {
  if ((time.index %% 13) > 1) Inf else 0
}
bid.fn <- function(time.index, per.capita.budget, budget.index) 1.1
kwl.fn <- function(time.index, per.capita.budget, budget.index) {
  4.5 * per.capita.budget
}
search.activity.trans.mat <- matrix(
  c(
    0.05, 0.95, 0.00,
    0.00, 0.85, 0.15,
    0.00, 0.00, 1.00
  ),
  nrow = length(kActivityStates),
  byrow = TRUE
)
search.favorability.trans.mat <- diag(length(kFavorabilityStates))
params.search <- list(
  audience.membership = list(activity = c(0.01, 0.3, 0.4)),
  budget = (2.4e7 / n.years) * seq_len(n.years),
  budget.index = budget.index,
  spend.cap.fn = spend.cap.fn,
  bid.fn = bid.fn,
  kwl.fn = kwl.fn,
  query.rate = 1,
  cpc.min = 0.8,
  cpc.max = 1.1,
  ctr = list(activity = c(0.005, 0.08, 0.10)),
  relative.effectiveness = c(0, 0.1, 1),
  transition.matrices = list(
    activity = search.activity.trans.mat,
    favorability = search.favorability.trans.mat
  )
)

sales.params <- list(
  competitor.demand.max = list(loyalty = c(0.8, 0, 0.8)),
  advertiser.demand.slope = list(favorability = rep(0, 5)),
  advertiser.demand.intercept = list(
    favorability = c(0.014, 0, 0.2, 0.3, 0.9)
  ),
  price = 80
)

sim.data <- SimulateAMSS(
  time.n = time.n,
  nat.mig.params = nat.mig.params,
  media.names = c("tv", "search"),
  media.modules = c(
    DefaultTraditionalMediaModule,
    DefaultSearchMediaModule
  ),
  media.params = list(params.tv, params.search),
  sales.params = sales.params,
  ping = time.n
)

burn.in.length <- 52L
final.year.start <- time.n - 51L
observed.data <- copy(sim.data$data[(burn.in.length + 1L):time.n])
observed.data[, market.rate := market.rate.seas[(burn.in.length + 1L):time.n]]

get_budget <- getFromNamespace("GetBudget", "amss")
generate_counterfactual <- getFromNamespace(
  "GenerateDataUnderNewBudget",
  "amss"
)
original.budget <- get_budget(sim.data)
counterfactual.budget <- original.budget
counterfactual.budget[n.years, "tv"] <-
  counterfactual.budget[n.years, "tv"] * 0.8
counterfactual.budget[n.years, "search"] <-
  counterfactual.budget[n.years, "search"] * 1.2

set.seed(101L)
original.counterfactual.data <- generate_counterfactual(
  sim.data,
  new.budget = original.budget,
  reps = 2L,
  t.start = final.year.start,
  t.end = time.n
)
set.seed(101L)
counterfactual.data <- generate_counterfactual(
  sim.data,
  new.budget = counterfactual.budget,
  reps = 2L,
  t.start = final.year.start,
  t.end = time.n
)
summarize_counterfactual <- function(data, scenario) {
  data[time.index >= final.year.start, .(
    revenue = sum(revenue),
    total.spend = sum(total.spend),
    tv.spend = sum(tv.spend),
    search.spend = sum(search.spend)
  ), by = rep.index][, scenario := scenario]
}
counterfactual.summary <- rbindlist(list(
  summarize_counterfactual(original.counterfactual.data, "original_budget"),
  summarize_counterfactual(counterfactual.data, "tv_minus_20_search_plus_20")
))
setcolorder(counterfactual.summary, c("scenario", "rep.index", "revenue", "total.spend", "tv.spend", "search.spend"))

# Small-population, two-replicate truth calls prove the interface only. The V10
# protocol will predeclare its own precision and seed contract before labels run.
truth_rows <- vector("list", 4L)
truth_specs <- data.table(
  channel = c("tv", "search", "tv", "search"),
  estimand = c("average_roas", "average_roas", "mroas_5pct_decrease", "mroas_5pct_decrease"),
  budget.proportion = c(0, 0, 0.95, 0.95)
)
for (i in seq_len(nrow(truth_specs))) {
  set.seed(200L + i)
  value <- CalculateROAS(
    sim.data,
    media.names = truth_specs$channel[i],
    budget.periods = n.years,
    budget.proportion = truth_specs$budget.proportion[i],
    t.start = final.year.start,
    t.end = time.n,
    scaled.pop.size = 2.4e9,
    min.reps = 2L,
    max.coef.var = Inf,
    max.margin.error = Inf,
    max.time = 0,
    verbose = TRUE
  )
  truth_rows[[i]] <- data.table(
    channel = truth_specs$channel[i],
    estimand = truth_specs$estimand[i],
    estimate = value$roas,
    margin.error.95 = value$margin.error,
    coefficient.variation = value$coef.var,
    repetitions = length(value$sample),
    research.label = FALSE
  )
}
truth.metrics <- rbindlist(truth_rows)

required.observed.columns <- c(
  "time.index",
  "tv.volume",
  "tv.spend",
  "search.clicks",
  "search.imps",
  "search.spend",
  "total.spend",
  "brand.sales",
  "revenue",
  "profit"
)
stopifnot(
  nrow(sim.data$data) == 208L,
  nrow(observed.data) == 156L,
  nrow(sim.data$data.full[[1]]) == 198L,
  identical(sim.data$params$media.names, c("tv", "search")),
  all(required.observed.columns %in% names(observed.data)),
  all(is.finite(as.matrix(observed.data[, ..required.observed.columns]))),
  max(abs(observed.data$profit - (observed.data$revenue - observed.data$total.spend))) < 1e-6,
  all(is.finite(truth.metrics$estimate)),
  all(!truth.metrics$research.label),
  nrow(counterfactual.summary) == 4L,
  length(unique(counterfactual.summary$scenario)) == 2L
)

artifact.root <- file.path(
  "research",
  "svi_score_v10_amss",
  "step2-artifacts"
)
dir.create(artifact.root, recursive = TRUE, showWarnings = FALSE)
fwrite(observed.data, file.path(artifact.root, "default-observed.csv"))
fwrite(truth.metrics, file.path(artifact.root, "truth-interface-smoke.csv"))
budget.table <- as.data.table(original.budget)
budget.table[, budget.period := seq_len(.N)]
setcolorder(budget.table, c("budget.period", setdiff(names(budget.table), "budget.period")))
fwrite(
  budget.table,
  file.path(artifact.root, "default-budget.csv")
)
fwrite(
  counterfactual.summary,
  file.path(artifact.root, "counterfactual-interface-smoke.csv")
)

cat("AMSS_VERSION=", as.character(packageVersion("amss")), "\n", sep = "")
cat("TIME_POINTS=", nrow(sim.data$data), "\n", sep = "")
cat("OBSERVED_AFTER_BURN_IN=", nrow(observed.data), "\n", sep = "")
cat("OBSERVED_COLUMNS=", paste(names(observed.data), collapse = "|"), "\n", sep = "")
cat("HIDDEN_STATE_ROWS_PER_WEEK=", nrow(sim.data$data.full[[1]]), "\n", sep = "")
cat("HIDDEN_COLUMNS=", paste(names(sim.data$data.full[[1]]), collapse = "|"), "\n", sep = "")
cat("MEDIA_NAMES=", paste(sim.data$params$media.names, collapse = "|"), "\n", sep = "")
cat("BUDGET_DIMENSIONS=", paste(dim(original.budget), collapse = "x"), "\n", sep = "")
cat("TRUTH_API=CalculateROAS\n")
cat("COUNTERFACTUAL_API=GenerateDataUnderNewBudget\n")
cat("OPTIMIZATION_API=OptimizeSpend\n")
cat("STEP2_STATUS=PASS\n")
