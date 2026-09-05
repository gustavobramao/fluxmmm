export const SVI_SCORE_V5D_SVI_VERSION =
  "flux-svi-score-v5d-svi.1.0.0-public-research-release";

export const SVI_SCORE_V5D_SVI_CONTRACT = {
  purpose:
    "compare like-for-like SVI posterior features against SVI posterior economic-loss labels",
  development: {
    businesses: 420,
    candidatesPerBusiness: 48,
    outerBusinessFolds: 5,
    candidateRegions: 4,
    composition:
      "300 original training plus 120 previously opened validation businesses, all development",
    freshValidationBusinesses: 0,
    auditBusinesses: 0,
  },
  posterior: {
    engine: "PyMC 6.2 FullRankADVI",
    fitsAlreadyAvailable: 24_000,
    developmentFitsUsed: 20_160,
    refitRequired: false,
    features:
      "stored SVI convergence, predictive calibration, ROI plausibility, ROI precision, seed agreement, and channel posterior centers",
  },
  safety: {
    convergence:
      "finite posterior, stable ELBO, and cross-seed ROI agreement",
    predictiveCoverageMinimum: 0.8,
    maximumImplausibleProbability: 0.2,
    maximumRelativeRoiWidth: 3,
    fallback:
      "if no candidate passes every posterior gate, select from converged candidates for review only",
  },
  comparison: {
    baseline: "flux-svi-score-v5d.0.1-direct-economic-loss-ranker",
    unchanged:
      "economic-loss target, linear Huber/quantile/ranking learner, business folds, candidate regions, search algorithms, and evaluation budgets",
    changed:
      "posterior features and posterior-defined candidate safety pool",
  },
  forbiddenFeatures: [
    "business-id",
    "generator-family",
    "split-label",
    "candidate-id",
    "proposal-method",
    "search-phase",
    "synthetic-truth",
    "svi-decision-loss",
    "svi-profit-regret",
    "svi-roi-truth-error",
    "svi-contribution-truth-error",
    "map-to-svi-loss-shift",
  ],
  governance: {
    activation: "public-research-release",
    freshValidationAccessed: false,
    auditAccessed: false,
    publicResearchReleasePermitted: true,
    freshGeneralizationEstablished: false,
    sotaClaimPermitted: false,
    productionActivationPermitted: false,
  },
} as const;
