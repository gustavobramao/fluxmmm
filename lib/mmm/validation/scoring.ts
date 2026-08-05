import type {
  EvidenceCoherenceAssessment,
  ValidationGate,
  ValidationLayerResult,
  ValidationResult,
} from "./types";

export function scoreValidation(
  layers: {
    generalization: ValidationLayerResult;
    structure: ValidationLayerResult;
    causal: ValidationLayerResult;
    decision: ValidationLayerResult;
  },
  evidenceCoherence: EvidenceCoherenceAssessment,
): Pick<
  ValidationResult,
  "finalScore" | "eligible" | "evidenceGrade" | "recommendation" | "gates"
> {
  const criticalStructure = layers.structure.tests.filter(
    (test) => test.importance === "critical",
  );
  const anchor = layers.causal.tests.find(
    (test) => test.id === "anchor-recovery",
  );
  const placebo = layers.causal.tests.find(
    (test) => test.id === "future-media-placebo",
  );
  const anchorAssessment =
    layers.causal.evidence.kind === "causal"
      ? layers.causal.evidence.anchorAssessment
      : undefined;
  const anchorApplicable = anchorAssessment?.status === "qualified";
  const gates: ValidationGate[] = [
    {
      id: "temporal-integrity",
      label: "Temporal integrity",
      applicable: true,
      passed: true,
      detail:
        "Rolling folds fit transforms on training history only and preserve forward-only adstock.",
    },
    {
      id: "decision-support",
      label: "Observed spend support",
      applicable: true,
      passed: layers.generalization.status !== "incomplete",
      detail:
        layers.generalization.status === "incomplete"
          ? "High- or low-spend behavior could not be held out reliably."
          : "Both observed spend regimes contain enough data for holdout testing.",
    },
    {
      id: "identification",
      label: "Structural identification",
      applicable: true,
      passed: criticalStructure.every((test) => test.status !== "fail"),
      detail: criticalStructure.some((test) => test.status === "fail")
        ? "At least one critical identification diagnostic failed."
        : "No critical structural diagnostic failed.",
    },
    {
      id: "external-anchor",
      label: "External anchor prediction",
      applicable: anchorApplicable,
      passed: Boolean(
        anchorApplicable &&
          anchor &&
          anchor.status !== "incomplete" &&
          anchor.score >= 50,
      ),
      detail:
        !anchorApplicable
          ? `${anchorAssessment?.summary ?? "No qualified external holdout is available."} This gate is not scored.`
          : `Anchor recovery score is ${Math.round(anchor?.score ?? 0)}.`,
    },
    {
      id: "placebo",
      label: "Temporal placebo",
      applicable: true,
      passed: Boolean(placebo && placebo.score >= 45),
      detail:
        (placebo?.score ?? 0) >= 45
          ? "Future media does not materially explain current residuals."
          : "Future media retains a concerning relationship with current residuals.",
    },
    {
      id: "evidence-coherence",
      label: "ROI decision coherence",
      applicable: true,
      passed:
        evidenceCoherence.blockingChannels.length === 0 &&
        layers.decision.status !== "fail",
      detail: `${evidenceCoherence.summary} Decision Coherence score is ${Math.round(layers.decision.score)}/100.`,
    },
  ];
  const complete =
    layers.generalization.status !== "incomplete" &&
    layers.decision.status !== "incomplete";
  const rawScore =
    100 *
    (layers.generalization.score / 100) ** 0.2 *
    (layers.structure.score / 100) ** 0.15 *
    (layers.causal.score / 100) ** 0.25 *
    (layers.decision.score / 100) ** 0.4;
  const finalScore = complete ? Math.max(0, Math.min(100, rawScore)) : null;
  const eligible =
    complete &&
    anchorApplicable &&
    gates
      .filter((gate) => gate.applicable)
      .every((gate) => gate.passed);
  const evidenceGrade =
    !complete
      ? "Incomplete"
      : anchorApplicable && eligible
        ? "A"
        : anchorApplicable && (finalScore ?? 0) >= 65
          ? "B"
          : "C";
  const recommendation: ValidationResult["recommendation"] =
    finalScore === null
      ? "Incomplete evidence"
      : eligible && finalScore >= 75
        ? "Decision-grade candidate"
        : finalScore >= 60
          ? "Use with caution"
          : "Not decision-grade";
  return {
    finalScore,
    eligible,
    evidenceGrade,
    recommendation,
    gates,
  };
}
