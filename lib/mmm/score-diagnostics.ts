import type { ValidationLayerId, ValidationLayerResult } from "./validation/types";

export const SCORE_DIAGNOSTIC_GROUPS = {
  generalization: [
    "rolling-oos",
    "predictive-coverage",
    "fold-stability",
    "spend-regimes",
  ],
  structure: [
    "residual-independence",
    "variance-structure",
    "likelihood-shape",
    "multicollinearity",
    "functional-form",
    "influence-stability",
    "non-negative-boundary",
  ],
  causal: [
    "anchor-recovery",
    "new-data-stability",
    "confounder-sensitivity",
    "future-media-placebo",
  ],
  decision: [
    "roi-posterior-plausibility",
    "roi-decision-stability",
    "roi-resolution",
    "roi-identification",
    "roi-economic-consistency",
  ],
} as const satisfies Record<ValidationLayerId, readonly string[]>;

export type ValidationScoreDiagnostic =
  (typeof SCORE_DIAGNOSTIC_GROUPS)[ValidationLayerId][number];

export type ValidationDiagnosticValues = Record<
  ValidationScoreDiagnostic,
  number | null
>;

export type ValidationDiagnosticWeights = Record<
  ValidationScoreDiagnostic,
  number
>;

export function validationDiagnosticValues(
  layers: Record<ValidationLayerId, ValidationLayerResult>,
): ValidationDiagnosticValues {
  const tests = Object.values(layers).flatMap((layer) => layer.tests);
  return Object.fromEntries(
    Object.values(SCORE_DIAGNOSTIC_GROUPS).flatMap((ids) =>
      ids.map((id) => {
        const diagnostic = tests.find((test) => test.id === id);
        return [
          id,
          !diagnostic || diagnostic.status === "incomplete"
            ? null
            : diagnostic.score,
        ];
      }),
    ),
  ) as ValidationDiagnosticValues;
}
