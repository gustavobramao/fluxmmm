# Parameterized AMSS scenario used only by the fresh V11 confirmatory audit.

v11c_phase_index <- function() {
  c(
    rep(1L, 52L),
    rep(2L, 52L),
    rep(3L, 52L),
    rep(4L, 13L),
    rep(5L, 26L),
    rep(6L, 13L),
    rep(7L, 52L),
    rep(8L, 8L)
  )
}

v11c_phase_budgets <- function(base, growth) {
  years <- base * (1 + growth) ^ c(-1, 0, 1, 2, 3)
  c(
    years[1],
    years[2],
    years[3],
    years[4] * 13 / 52,
    years[4] * 26 / 52,
    years[4] * 13 / 52,
    years[5],
    0
  )
}

build_v11c_amss_cohort_scenario <- function(design) {
  if (!requireNamespace("amss", quietly = TRUE)) {
    stop("The pinned AMSS package is required.")
  }

  set.seed(as.integer(design$business_seed))
  time.n <- 268L
  budget.index <- v11c_phase_index()
  phase.length <- tabulate(budget.index, nbins = 8L)

  activity.transition <- matrix(
    rep(c(0.60, 0.30, 0.10), 3L),
    nrow = length(amss::kActivityStates),
    byrow = TRUE
  )
  favorability.transition <- matrix(
    rep(c(0.03, 0.07, 0.65, 0.20, 0.05), 5L),
    nrow = length(amss::kFavorabilityStates),
    byrow = TRUE
  )
  market.rate.nonoise <- amss::SimulateSinusoidal(
    time.n,
    52,
    vert.translation = as.numeric(design$market_trend),
    amplitude = as.numeric(design$season_amplitude)
  )
  market.rate.seas <- pmax(
    0,
    pmin(
      1,
      market.rate.nonoise * amss::SimulateAR1(time.n, 1, 0.1, 0.3)
    )
  )
  nat.mig.params <- list(
    population = as.numeric(design$population),
    market.rate.trend = as.numeric(design$market_trend),
    market.rate.seas = market.rate.seas,
    prop.activity = c(0.375, 0.425, 0.2),
    prop.favorability = c(0.03, 0.07, 0.65, 0.20, 0.05),
    prop.loyalty = c(1, 0, 0),
    transition.matrices = list(
      activity = activity.transition,
      favorability = favorability.transition
    )
  )

  paid.social.planning <- as.numeric(amss::SimulateCorrelated(
    market.rate.seas,
    cor.vx = as.numeric(design$social_planning_correlation),
    mu.x = 1,
    sigma.x = 0.30
  ))
  paid.social.pacing <- amss::SimulateAR1(time.n, 0, 0.18, 0.60)
  paid.social.flighting <- pmax(0.05, paid.social.planning + paid.social.pacing)
  paid.social.activity.trans.mat <- matrix(
    c(
      0.45, 0.50, 0.05,
      0.00, 0.70, 0.30,
      0.00, 0.00, 1.00
    ),
    nrow = length(amss::kActivityStates),
    byrow = TRUE
  )
  paid.social.favorability.trans.mat <- matrix(
    c(
      0.55, 0.00, 0.30, 0.12, 0.03,
      0.00, 0.80, 0.15, 0.05, 0.00,
      0.00, 0.00, 0.65, 0.30, 0.05,
      0.00, 0.00, 0.00, 0.85, 0.15,
      0.00, 0.00, 0.00, 0.00, 1.00
    ),
    nrow = length(amss::kFavorabilityStates),
    byrow = TRUE
  )
  params.paid.social.proxy <- list(
    audience.membership = list(
      activity = c(0.15, 0.55, 0.45),
      favorability = c(0.25, 0.35, 0.65, 0.80, 0.90)
    ),
    budget = v11c_phase_budgets(
      as.numeric(design$social_budget),
      as.numeric(design$social_growth)
    ),
    budget.index = budget.index,
    flighting = paid.social.flighting,
    unit.cost = as.numeric(design$social_unit_cost),
    hill.ec = as.numeric(design$social_hill_ec),
    hill.slope = as.numeric(design$social_hill_slope),
    transition.matrices = list(
      activity = paid.social.activity.trans.mat,
      favorability = paid.social.favorability.trans.mat
    )
  )

  spend.cap.fn <- function(time.index, budget, budget.index) {
    if ((time.index %% 13) > 1) Inf else 0
  }
  bid.fn <- function(time.index, per.capita.budget, budget.index) 1.1
  kwl.fn <- function(time.index, per.capita.budget, budget.index) {
    phase <- budget.index[time.index]
    # AMSS computes a weighted average of this probability. Returning exactly
    # one can round that average above one and make base R's rbinom return NA.
    # The one-part-per-billion cap is a numerical boundary repair, not a
    # substantive change to search availability.
    min(1 - 1e-9, 4.5 * per.capita.budget * 52 / phase.length[phase])
  }
  search.activity.trans.mat <- matrix(
    c(
      0.05, 0.95, 0.00,
      0.00, 0.85, 0.15,
      0.00, 0.00, 1.00
    ),
    nrow = length(amss::kActivityStates),
    byrow = TRUE
  )
  params.search <- list(
    audience.membership = list(activity = c(0.01, 0.3, 0.4)),
    budget = v11c_phase_budgets(
      as.numeric(design$search_budget),
      as.numeric(design$search_growth)
    ),
    budget.index = budget.index,
    spend.cap.fn = spend.cap.fn,
    bid.fn = bid.fn,
    kwl.fn = kwl.fn,
    query.rate = 1,
    cpc.min = 0.8,
    cpc.max = 1.1,
    ctr = list(
      activity = c(0.005, 0.08, 0.10) * as.numeric(design$search_ctr_scale)
    ),
    relative.effectiveness = c(0, 0.1, 1),
    transition.matrices = list(
      activity = search.activity.trans.mat,
      favorability = diag(length(amss::kFavorabilityStates))
    )
  )

  tv.flighting <- pmax(
    0,
    market.rate.seas + amss::SimulateAR1(time.n, -0.7, 0.7, -0.7)
  )
  lag.weeks <- as.integer(design$tv_lag_weeks)
  if (lag.weeks > 0L) {
    tv.flighting <- tv.flighting[c(
      (lag.weeks + 1L):length(tv.flighting),
      seq_len(lag.weeks)
    )]
  }
  tv.favorability.trans.mat <- matrix(
    c(
      0.4, 0.0, 0.4, 0.2, 0.0,
      0.0, 0.9, 0.1, 0.0, 0.0,
      0.0, 0.0, 0.6, 0.4, 0.0,
      0.0, 0.0, 0.0, 0.8, 0.2,
      0.0, 0.0, 0.0, 0.0, 1.0
    ),
    nrow = length(amss::kFavorabilityStates),
    byrow = TRUE
  )
  params.tv <- list(
    audience.membership = list(activity = rep(0.4, 3)),
    budget = v11c_phase_budgets(
      as.numeric(design$tv_budget),
      as.numeric(design$tv_growth)
    ),
    budget.index = budget.index,
    flighting = tv.flighting,
    unit.cost = 0.005,
    hill.ec = as.numeric(design$tv_hill_ec),
    hill.slope = as.numeric(design$tv_hill_slope),
    transition.matrices = list(
      activity = diag(length(amss::kActivityStates)),
      favorability = tv.favorability.trans.mat
    )
  )

  demand.scale <- as.numeric(design$demand_scale)
  price <- as.numeric(design$price)
  gross.margin <- as.numeric(design$gross_margin)
  sales.params <- list(
    competitor.demand.max = list(loyalty = c(0.8, 0, 0.8)),
    advertiser.demand.slope = list(favorability = rep(0, 5)),
    advertiser.demand.intercept = list(
      favorability = pmin(0.98, c(0.014, 0, 0.2, 0.3, 0.9) * demand.scale)
    ),
    price = price,
    unit.cost = price * (1 - gross.margin)
  )

  media.names <- c("paid_social_proxy", "search", "tv")
  media.modules <- list(
    paid_social_proxy = amss::DefaultTraditionalMediaModule,
    search = amss::DefaultSearchMediaModule,
    tv = amss::DefaultTraditionalMediaModule
  )
  media.params <- list(
    paid_social_proxy = params.paid.social.proxy,
    search = params.search,
    tv = params.tv
  )
  declared.order <- strsplit(as.character(design$module_order), "|", fixed = TRUE)[[1L]]
  stopifnot(setequal(declared.order, media.names), length(declared.order) == 3L)

  list(
    simulation.args = list(
      time.n = time.n,
      nat.mig.params = nat.mig.params,
      media.names = declared.order,
      media.modules = unname(media.modules[declared.order]),
      media.params = unname(media.params[declared.order]),
      sales.params = sales.params,
      ping = time.n
    ),
    hidden.design = list(
      market.rate.seas = market.rate.seas,
      phase.index = budget.index,
      design = design
    )
  )
}

v11c_flux_channel_for_group <- function(evidence.group) {
  switch(
    as.character(evidence.group),
    "paid-social-experiment" = "meta_acquisition_spend",
    "search-experiment" = "google_search_nonbrand_spend",
    "tv-experiment" = "ctv_spend",
    "no-experiment" = NA_character_,
    stop("Unknown V11 confirmatory evidence group: ", evidence.group)
  )
}

v11c_raw_channel_for_flux <- function(flux.channel) {
  switch(
    as.character(flux.channel),
    "meta_acquisition_spend" = "paid_social_proxy",
    "google_search_nonbrand_spend" = "search",
    "ctv_spend" = "tv",
    stop("Unknown V11 confirmatory Flux channel: ", flux.channel)
  )
}
