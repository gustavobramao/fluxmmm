options(stringsAsFactors = FALSE, width = 120)

required.packages <- c("amss", "data.table")
missing.packages <- required.packages[
  !vapply(required.packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing.packages) > 0L) {
  stop("Missing R packages: ", paste(missing.packages, collapse = ", "))
}

library(data.table)
source(file.path("research", "svi_score_v10_amss", "amss-cohort-scenario.R"))
get_budget <- getFromNamespace("GetBudget", "amss")
generate_counterfactual <- getFromNamespace("GenerateDataUnderNewBudget", "amss")

root <- file.path(".flux-artifacts", "svi-score-v10-amss")
sealed.root <- file.path(root, "sealed-cohort")
cohort.root <- file.path(root, "cohort-staging")
observed.root <- file.path(cohort.root, "observed")
hidden.root <- file.path(cohort.root, "hidden")
if (dir.exists(sealed.root) || dir.exists(cohort.root)) {
  stop("The V10 AMSS cohort or its staging directory already exists; immutable regeneration is forbidden.")
}
dir.create(observed.root, recursive = TRUE, showWarnings = FALSE)
dir.create(hidden.root, recursive = TRUE, showWarnings = FALSE)

design.path <- file.path(
  "research",
  "svi_score_v10_amss",
  "frozen-protocol",
  "cohort-design.csv"
)
design <- fread(design.path)
stopifnot(nrow(design) == 100L, !anyDuplicated(design$business_id))

metadata <- list()
experiments <- list()

for (business.index in seq_len(nrow(design))) {
  row <- as.list(design[business.index])
  scenario <- build_v10_amss_cohort_scenario(row)
  set.seed(as.integer(row$business_seed))
  simulation <- do.call(amss::SimulateAMSS, scenario$simulation.args)

  observed.raw <- copy(simulation$data[53L:208L])
  observed <- data.table(
    date = as.character(as.Date("2000-01-03") +
      7L * (observed.raw$time.index - 53L)),
    revenue = observed.raw$revenue,
    meta_acquisition_spend = observed.raw$paid_social_proxy.spend,
    google_search_nonbrand_spend = observed.raw$search.spend,
    ctv_spend = observed.raw$tv.spend
  )
  stopifnot(
    nrow(observed) == 156L,
    all(is.finite(as.matrix(observed[, -"date"]))),
    all(as.matrix(observed[, -c("date", "revenue")]) >= 0)
  )
  fwrite(observed, file.path(observed.root, paste0(row$business_id, ".csv")))

  budget.matrix <- get_budget(simulation)
  raw.order <- colnames(budget.matrix)
  year4.budget <- budget.matrix[4L, ] + budget.matrix[5L, ] + budget.matrix[6L, ]
  names(year4.budget) <- raw.order
  year4.rows <- simulation$data[time.index >= 157L & time.index <= 208L]
  year4.spend <- c(
    paid_social_proxy = sum(year4.rows$paid_social_proxy.spend),
    search = sum(year4.rows$search.spend),
    tv = sum(year4.rows$tv.spend)
  )
  utilization <- pmax(0.05, pmin(2, year4.spend / year4.budget[names(year4.spend)]))

  experiment.channel <- v10_flux_channel_for_group(row$evidence_group)
  if (!is.na(experiment.channel)) {
    raw.channel <- v10_raw_channel_for_flux(experiment.channel)
    raw.index <- match(raw.channel, raw.order)
    control.budget <- budget.matrix
    control.budget[5L, raw.index] <- control.budget[5L, raw.index] * 0.8
    roi.values <- numeric(4L)
    spend.values <- numeric(4L)
    outcome.values <- numeric(4L)
    for (replicate.index in seq_len(4L)) {
      paired.seed <- as.integer(row$truth_seed) + replicate.index
      set.seed(paired.seed)
      treatment <- generate_counterfactual(
        simulation,
        new.budget = budget.matrix,
        reps = 1,
        t.start = 170L,
        t.end = 202L
      )
      set.seed(paired.seed)
      control <- generate_counterfactual(
        simulation,
        new.budget = control.budget,
        reps = 1,
        t.start = 170L,
        t.end = 202L
      )
      treatment.window <- treatment[time.index >= 170L & time.index <= 195L]
      control.window <- control[time.index >= 170L & time.index <= 195L]
      treatment.outcome <- treatment[time.index >= 170L & time.index <= 202L]
      control.outcome <- control[time.index >= 170L & time.index <= 202L]
      incremental.spend <-
        sum(treatment.window[[paste0(raw.channel, ".spend")]]) -
        sum(control.window[[paste0(raw.channel, ".spend")]])
      incremental.outcome <-
        sum(treatment.outcome$revenue) - sum(control.outcome$revenue)
      spend.values[replicate.index] <- incremental.spend
      outcome.values[replicate.index] <- incremental.outcome
      roi.values[replicate.index] <- incremental.outcome / max(incremental.spend, 1)
    }
    true.roi <- mean(roi.values)
    empirical.se <- if (length(roi.values) > 1L) sd(roi.values) / sqrt(length(roi.values)) else 0
    standard.error <- max(
      empirical.se,
      abs(true.roi) * 0.15,
      0.05
    )
    set.seed(as.integer(row$truth_seed) + 9L * 1000L)
    observed.roi <- max(
      0.05,
      true.roi * (1 + as.numeric(row$experiment_bias_share)) +
        rnorm(1L, 0, standard.error)
    )
    incremental.spend <- max(mean(spend.values), 1)
    experiments[[length(experiments) + 1L]] <- data.table(
      business_id = row$business_id,
      channel = experiment.channel,
      start_date = as.character(as.Date("2000-01-03") + 7L * (170L - 53L)),
      end_date = as.character(as.Date("2000-01-03") + 7L * (195L - 53L)),
      outcome_end_date = as.character(as.Date("2000-01-03") + 7L * (202L - 53L)),
      incremental_outcome = observed.roi * incremental.spend,
      incremental_spend = incremental.spend,
      standard_error = standard.error,
      confidence = 0.90,
      scope = "total",
      source = "AMSS paired period-matched external experiment",
      design_replicates = 4L,
      relative_control_reduction = 0.20,
      experiment_bias_share = as.numeric(row$experiment_bias_share)
    )
  }

  metadata[[business.index]] <- data.table(
    business_id = row$business_id,
    business_seed = as.integer(row$business_seed),
    truth_seed = as.integer(row$truth_seed),
    evidence_group = row$evidence_group,
    module_order = row$module_order,
    gross_margin = as.numeric(row$gross_margin),
    price = as.numeric(row$price),
    year4_paid_social_budget = year4.budget["paid_social_proxy"],
    year4_search_budget = year4.budget["search"],
    year4_tv_budget = year4.budget["tv"],
    year4_paid_social_spend = year4.spend["paid_social_proxy"],
    year4_search_spend = year4.spend["search"],
    year4_tv_spend = year4.spend["tv"],
    paid_social_utilization = utilization["paid_social_proxy"],
    search_utilization = utilization["search"],
    tv_utilization = utilization["tv"],
    observed_start_week = 53L,
    observed_end_week = 208L,
    future_start_week = 209L,
    future_end_week = 260L,
    carryover_end_week = 268L
  )

  saveRDS(
    list(
      business_id = row$business_id,
      design = row,
      scenario = scenario,
      simulation = simulation,
      baseline_budget = budget.matrix
    ),
    file.path(hidden.root, paste0(row$business_id, ".rds")),
    compress = "xz"
  )
  cat(sprintf("V10 AMSS cohort %03d/100 %s\n", business.index, row$business_id))
}

fwrite(rbindlist(metadata), file.path(cohort.root, "business-metadata.csv"))
if (length(experiments) > 0L) {
  fwrite(rbindlist(experiments), file.path(cohort.root, "experiments.csv"))
} else {
  fwrite(data.table(), file.path(cohort.root, "experiments.csv"))
}
writeLines(
  c(
    "artifact_id=flux-v10-amss-sealed-cohort-v1",
    "businesses=100",
    "observed_weeks_per_business=156",
    "experiments=75",
    "hidden_future_truth_opened=FALSE"
  ),
  file.path(cohort.root, "generation-receipt.txt")
)
