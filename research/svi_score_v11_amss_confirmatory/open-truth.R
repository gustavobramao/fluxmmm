options(stringsAsFactors = FALSE, width = 120)

required.packages <- c("amss", "data.table", "jsonlite")
missing.packages <- required.packages[
  !vapply(required.packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing.packages) > 0L) {
  stop("Missing R packages: ", paste(missing.packages, collapse = ", "))
}

library(data.table)
generate_counterfactual <- getFromNamespace("GenerateDataUnderNewBudget", "amss")

argument_value <- function(name, fallback = NA_character_) {
  prefix <- paste0("--", name, "=")
  match <- commandArgs(trailingOnly = TRUE)[startsWith(commandArgs(trailingOnly = TRUE), prefix)]
  if (length(match) == 0L) fallback else substring(match[[1L]], nchar(prefix) + 1L)
}

root <- file.path(".flux-artifacts", "svi-score-v11-amss-confirmatory")
sealed.root <- file.path(root, "sealed-cohort")
observable.root <- file.path(root, "observable-freeze")
truth.root <- file.path(root, "opened-truth")
part.root <- file.path(truth.root, "parts")
receipt.path <- file.path(truth.root, "truth-open-receipt.json")
actions.path <- file.path(observable.root, "candidate-actions.csv")
metadata.path <- file.path(sealed.root, "business-metadata.csv")

worker.index <- as.integer(argument_value("worker", "0"))
worker.count <- as.integer(argument_value("workers", "1"))
if (!file.exists(receipt.path)) stop("Truth-open receipt is absent; refusing to deserialize hidden AMSS state.")
receipt <- jsonlite::read_json(receipt.path, simplifyVector = TRUE)
if (!identical(receipt$status, "truth-opened-for-frozen-action-evaluation")) {
  stop("Truth-open receipt has an invalid status.")
}
if (is.na(worker.index) || is.na(worker.count) || worker.count < 1L ||
    worker.index < 0L || worker.index >= worker.count) {
  stop("Invalid truth worker index/count.")
}

dir.create(part.root, recursive = TRUE, showWarnings = FALSE)
actions <- fread(actions.path, encoding = "UTF-8")
metadata <- fread(metadata.path, encoding = "UTF-8")
required.action.columns <- c(
  "business_id", "candidate_id", "scenario",
  "meta_acquisition_spend", "google_search_nonbrand_spend", "ctv_spend"
)
stopifnot(
  nrow(actions) == 19200L,
  all(required.action.columns %in% names(actions)),
  nrow(metadata) == 100L,
  !anyDuplicated(metadata$business_id)
)

scenario.ids <- c(
  "budget-reduction", "fixed-budget-mix", "budget-growth", "economic-ceiling"
)
business.ids <- sort(unique(actions$business_id))
selected.businesses <- business.ids[(seq_along(business.ids) - 1L) %% worker.count == worker.index]

translate_budget <- function(hidden, metadata.row, action.row) {
  budget <- hidden$baseline_budget
  flux.to.raw <- c(
    meta_acquisition_spend = "paid_social_proxy",
    google_search_nonbrand_spend = "search",
    ctv_spend = "tv"
  )
  utilization <- c(
    meta_acquisition_spend = as.numeric(metadata.row$paid_social_utilization),
    google_search_nonbrand_spend = as.numeric(metadata.row$search_utilization),
    ctv_spend = as.numeric(metadata.row$tv_utilization)
  )
  # Reconstruct the declared observable utilization as an independent runtime
  # assertion. The sealed metadata already contains this same named ratio.
  observed.spend <- c(
    meta_acquisition_spend = as.numeric(metadata.row$year4_paid_social_spend),
    google_search_nonbrand_spend = as.numeric(metadata.row$year4_search_spend),
    ctv_spend = as.numeric(metadata.row$year4_tv_spend)
  )
  declared.budget <- c(
    meta_acquisition_spend = as.numeric(metadata.row$year4_paid_social_budget),
    google_search_nonbrand_spend = as.numeric(metadata.row$year4_search_budget),
    ctv_spend = as.numeric(metadata.row$year4_tv_budget)
  )
  reconstructed <- observed.spend / declared.budget
  missing.utilization <- !is.finite(utilization)
  utilization[missing.utilization] <- reconstructed[missing.utilization]
  utilization <- setNames(
    pmax(0.05, pmin(2, utilization)),
    names(utilization)
  )
  stopifnot(all(is.finite(utilization)))
  for (flux.channel in names(flux.to.raw)) {
    raw.channel <- flux.to.raw[[flux.channel]]
    budget[7L, raw.channel] <- as.numeric(action.row[[flux.channel]]) /
      max(0.05, min(2, utilization[[flux.channel]]))
  }
  budget[8L, ] <- 0
  budget
}

evaluate_action <- function(hidden, metadata.row, action.row, seed) {
  new.budget <- translate_budget(hidden, metadata.row, action.row)
  set.seed(seed)
  result <- generate_counterfactual(
    hidden$simulation,
    new.budget = new.budget,
    reps = 4L,
    t.start = 209L,
    t.end = 268L
  )
  gross.margin <- as.numeric(metadata.row$gross_margin)
  by.replicate <- result[
    time.index >= 209L & time.index <= 268L,
    .(
      revenue = sum(revenue),
      realized_spend = sum(total.spend)
    ),
    by = rep.index
  ]
  by.replicate[, realized_profit := gross.margin * revenue - realized_spend]
  list(
    profit = mean(by.replicate$realized_profit),
    profit.se = sd(by.replicate$realized_profit) / sqrt(nrow(by.replicate)),
    revenue = mean(by.replicate$revenue),
    spend = mean(by.replicate$realized_spend)
  )
}

output <- list()
output.index <- 0L
for (business.id in selected.businesses) {
  hidden.path <- file.path(sealed.root, "hidden", paste0(business.id, ".rds"))
  hidden <- readRDS(hidden.path)
  metadata.row <- metadata[business_id == business.id]
  business.actions <- actions[business_id == business.id]
  stopifnot(nrow(metadata.row) == 1L, nrow(business.actions) == 192L)

  baseline <- data.table(
    meta_acquisition_spend = as.numeric(metadata.row$year4_paid_social_spend),
    google_search_nonbrand_spend = as.numeric(metadata.row$year4_search_spend),
    ctv_spend = as.numeric(metadata.row$year4_tv_spend)
  )

  for (scenario.index in seq_along(scenario.ids)) {
    scenario.id <- scenario.ids[[scenario.index]]
    scenario.actions <- business.actions[scenario == scenario.id]
    stopifnot(nrow(scenario.actions) == 48L)
    unique.actions <- unique(scenario.actions[, .(
      meta_acquisition_spend,
      google_search_nonbrand_spend,
      ctv_spend
    )])
    unique.actions[, action_key := seq_len(.N)]
    mapped <- merge(
      scenario.actions,
      unique.actions,
      by = c(
        "meta_acquisition_spend",
        "google_search_nonbrand_spend",
        "ctv_spend"
      ),
      all.x = TRUE,
      sort = FALSE
    )
    scenario.seed <- as.integer(metadata.row$truth_seed) + scenario.index * 10000L
    values <- vector("list", nrow(unique.actions))
    for (action.index in seq_len(nrow(unique.actions))) {
      action.row <- unique.actions[action.index]
      measured <- evaluate_action(hidden, metadata.row, action.row, scenario.seed)
      values[[action.index]] <- data.table(
        action_key = action.row$action_key,
        realized_profit = measured$profit,
        realized_profit_se = measured$profit.se,
        realized_revenue = measured$revenue,
        realized_spend = measured$spend
      )
    }
    measured.actions <- merge(mapped, rbindlist(values), by = "action_key", sort = FALSE)
    measured.actions[, `:=`(
      truth_replicates = 4L,
      common_random_seed = scenario.seed
    )]

    baseline.value <- evaluate_action(hidden, metadata.row, baseline, scenario.seed)
    measured.actions[, `:=`(
      baseline_profit = baseline.value$profit,
      baseline_profit_se = baseline.value$profit.se,
      baseline_revenue = baseline.value$revenue,
      baseline_realized_spend = baseline.value$spend
    )]
    output.index <- output.index + 1L
    output[[output.index]] <- measured.actions[, .(
      business_id,
      candidate_id,
      scenario,
      meta_acquisition_spend,
      google_search_nonbrand_spend,
      ctv_spend,
      predicted_incremental_profit,
      realized_profit,
      realized_profit_se,
      realized_revenue,
      realized_spend,
      baseline_profit,
      baseline_profit_se,
      baseline_revenue,
      baseline_realized_spend,
      truth_replicates,
      common_random_seed
    )]
  }
  cat(sprintf(
    "V11 confirmatory AMSS truth worker %02d/%02d completed %s\n",
    worker.index + 1L,
    worker.count,
    business.id
  ))
}

part <- rbindlist(output)
expected.rows <- length(selected.businesses) * 48L * 4L
stopifnot(nrow(part) == expected.rows, all(is.finite(part$realized_profit)))
target <- file.path(part.root, sprintf("part-%03d-of-%03d.csv", worker.index, worker.count))
temporary <- paste0(target, ".tmp")
if (file.exists(target)) {
  existing <- fread(target)
  stopifnot(nrow(existing) == expected.rows)
} else {
  fwrite(part, temporary)
  if (!file.rename(temporary, target)) stop("Unable to seal truth worker part: ", target)
}

cat(jsonlite::toJSON(list(
  worker = worker.index,
  workers = worker.count,
  businesses = length(selected.businesses),
  rows = nrow(part),
  hidden_truth_dereferenced = TRUE
), auto_unbox = TRUE, pretty = TRUE), "\n")
