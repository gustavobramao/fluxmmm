options(stringsAsFactors = FALSE, width = 120)

required.packages <- c("amss", "data.table")
missing.packages <- required.packages[
  !vapply(required.packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing.packages) > 0L) {
  stop("Missing R packages: ", paste(missing.packages, collapse = ", "))
}

library(data.table)
source(file.path(
  "research",
  "svi_score_v10_amss",
  "amss-three-channel-scenario.R"
))

scenario <- build_v10_amss_three_channel_scenario(seed = 31001L)
sim.data <- do.call(amss::SimulateAMSS, scenario$simulation.args)

burn.in <- 52L
observed.raw <- copy(sim.data$data[(burn.in + 1L):scenario$simulation.args$time.n])
observed.flux <- data.table(
  date = as.character(as.Date("2000-01-03") +
    7L * (observed.raw$time.index - observed.raw$time.index[1L])),
  revenue = observed.raw$revenue,
  meta_acquisition_spend = observed.raw$paid_social_proxy.spend,
  google_search_nonbrand_spend = observed.raw$search.spend,
  ctv_spend = observed.raw$tv.spend
)

required.raw <- c(
  "paid_social_proxy.volume",
  "paid_social_proxy.spend",
  "search.spend",
  "tv.spend",
  "revenue",
  "profit"
)
stopifnot(
  nrow(observed.raw) == 156L,
  nrow(observed.flux) == 156L,
  nrow(sim.data$data.full[[1L]]) == 198L,
  identical(
    sim.data$params$media.names,
    c("paid_social_proxy", "search", "tv")
  ),
  all(required.raw %in% names(observed.raw)),
  all(is.finite(as.matrix(observed.raw[, ..required.raw]))),
  all(is.finite(as.matrix(observed.flux[, -"date"]))),
  all(observed.flux$meta_acquisition_spend >= 0),
  all(observed.flux$google_search_nonbrand_spend >= 0),
  all(observed.flux$ctv_spend >= 0),
  !("market.rate" %in% names(observed.flux))
)

artifact.root <- file.path(
  "research",
  "svi_score_v10_amss",
  "step3-artifacts"
)
dir.create(artifact.root, recursive = TRUE, showWarnings = FALSE)
fwrite(
  observed.flux,
  file.path(artifact.root, "three-channel-flux-input-smoke.csv")
)
fwrite(
  data.table(
    raw.channel = c("paid_social_proxy", "search", "tv"),
    amss.module = c(
      "DefaultTraditionalMediaModule",
      "DefaultSearchMediaModule",
      "DefaultTraditionalMediaModule"
    ),
    frozen.v9.role = c(
      "meta_acquisition_spend",
      "google_search_nonbrand_spend",
      "ctv_spend"
    ),
    interpretation = c(
      "paid-social exposure stress proxy; not a platform auction",
      "paid-search demand-harvesting archetype",
      "long-memory reach-media archetype; not literal CTV"
    )
  ),
  file.path(artifact.root, "channel-role-map.csv")
)

cat("AMSS_VERSION=", as.character(packageVersion("amss")), "\n", sep = "")
cat("RAW_MEDIA_NAMES=", paste(sim.data$params$media.names, collapse = "|"), "\n", sep = "")
cat("FROZEN_V9_ROLES=meta_acquisition_spend|google_search_nonbrand_spend|ctv_spend\n")
cat("OBSERVED_ROWS=", nrow(observed.flux), "\n", sep = "")
cat("PERFECT_MARKET_RATE_EXPOSED=FALSE\n")
cat("TRUTH_CALCULATED=FALSE\n")
cat("RESEARCH_LABELS_CREATED=FALSE\n")
cat("STEP3_CHANNEL_SMOKE_STATUS=PASS\n")
