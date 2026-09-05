import type { ModelResult } from "../types";
import { clampScore } from "./math";
import type {
  DecisionEvidence,
  EvidenceCoherenceAssessment,
  ValidationLayerResult,
  ValidationStatus,
  ValidationTest,
} from "./types";

function statusForScore(score: number): ValidationStatus {
  return score >= 75 ? "pass" : score >= 55 ? "review" : "fail";
}

function test(
  id: string,
  name: string,
  score: number,
  metric: string,
  detail: string,
  importance: ValidationTest["importance"],
  status?: ValidationStatus,
): ValidationTest {
  return {
    id,
    name,
    score: clampScore(score),
    status: status ?? statusForScore(score),
    metric,
    detail,
    importance,
  };
}

function normalCdf(value: number): number {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * x);
  const erf =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
      0.284496736) *
      t +
      0.254829592) *
      t) *
      Math.exp(-x * x);
  return 0.5 * (1 + sign * erf);
}

function posteriorMassInRange(
  roi: number,
  roiLow: number,
  roiHigh: number,
  evidenceLow?: number,
  evidenceHigh?: number,
): number | undefined {
  if (
    evidenceLow === undefined ||
    evidenceHigh === undefined ||
    ![roi, roiLow, roiHigh, evidenceLow, evidenceHigh].every(Number.isFinite)
  ) {
    return undefined;
  }
  const standardError = Math.max(
    Math.abs(roiHigh - roiLow) / (2 * 1.96),
    Math.abs(roi) * 0.02,
    0.01,
  );
  return Math.max(
    0,
    Math.min(
      1,
      normalCdf((evidenceHigh - roi) / standardError) -
        normalCdf((evidenceLow - roi) / standardError),
    ),
  );
}

function weightedGeometric(
  values: { score: number; weight: number }[],
): number {
  const totalWeight = values.reduce((total, item) => total + item.weight, 0);
  if (!totalWeight) return 0;
  const logScore = values.reduce(
    (total, item) =>
      total +
      (item.weight / totalWeight) *
        Math.log(Math.max(item.score / 100, 0.01)),
    0,
  );
  return clampScore(100 * Math.exp(logScore));
}

function portfolioScore(
  values: { score: number; spendShare: number }[],
): number {
  if (!values.length) return 0;
  const weighted = weightedGeometric(
    values.map((item) => ({
      score: item.score,
      weight: Math.min(Math.max(item.spendShare, 0.02), 0.3),
    })),
  );
  const vulnerableTail = Math.min(...values.map((item) => item.score));
  return clampScore(weighted * 0.4 + vulnerableTail * 0.6);
}

function stabilityScore(
  channel: string,
  causal: ValidationLayerResult,
  referenceRoi: number,
): number {
  if (causal.evidence.kind !== "causal") return 55;
  const path = causal.evidence.stability.find(
    (item) => item.channel.toLowerCase() === channel.toLowerCase(),
  );
  const earlier = path?.points.slice(0, -1) ?? [];
  if (!earlier.length) return 55;
  const averageDrift =
    earlier.reduce(
      (total, point) =>
        total +
        Math.abs(point.roi - referenceRoi) /
          Math.max(Math.abs(referenceRoi), 0.1),
      0,
    ) / earlier.length;
  return clampScore(100 - averageDrift * 170);
}

function identificationScore(
  model: ModelResult,
  channel: string,
  dependence?: DecisionEvidence["channels"][number]["evidenceDependence"],
): number {
  const attribution = model.numerical?.evidenceAttribution?.find(
    (item) => item.channel.toLowerCase() === channel.toLowerCase(),
  );
  if (!attribution) return 50;
  const shares = attribution.uncertaintyShare;
  const experimentQuality = dependence?.source === "experiment"
    ? dependence.quality
    : 0;
  const benchmarkQuality = dependence?.source === "benchmark"
    ? dependence.quality
    : 0;
  return clampScore(
    100 * (
      shares.observational * attribution.conditionalDataShare +
      shares.experiment * experimentQuality +
      shares.benchmark * benchmarkQuality
    ),
  );
}

function evidenceCompatibilityScore(
  dependence?: DecisionEvidence["channels"][number]["evidenceDependence"],
): number {
  if (!dependence) return 100;
  return clampScore(100 / (1 + dependence.standardizedConflict ** 2));
}

function evidenceDependenceScore(
  dependence?: DecisionEvidence["channels"][number]["evidenceDependence"],
): number {
  if (!dependence) return 100;
  const dependenceMagnitude = Math.min(
    1,
    Math.max(
      dependence.roiLocationShift / Math.log(2),
      dependence.contributionShift,
      dependence.allocationProbabilityShift * 2,
      dependence.marginalProfitIndexShift,
    ),
  );
  const conflictSeverity = Math.min(
    1,
    dependence.standardizedConflict / 3,
  );
  const qualifiedRisk =
    dependence.influence *
    dependenceMagnitude *
    ((1 - dependence.quality + conflictSeverity) / 2);
  return clampScore(100 * (1 - qualifiedRisk));
}

function resolutionScore(
  roi: number,
  roiLow: number,
  roiHigh: number,
  evidenceLow?: number,
  evidenceHigh?: number,
): number {
  const posteriorWidth = Math.max(roiHigh - roiLow, 0);
  const referenceWidth =
    evidenceLow !== undefined && evidenceHigh !== undefined
      ? Math.max(evidenceHigh - evidenceLow, 0.1)
      : Math.max(Math.abs(roi), 0.25);
  const ratio = posteriorWidth / referenceWidth;
  return clampScore(100 / (1 + Math.max(0, ratio - 1) * 1.5));
}

function economicScore(model: ModelResult, channel: string): number {
  const estimate = model.channels.find(
    (item) => item.channel.toLowerCase() === channel.toLowerCase(),
  );
  if (
    !estimate ||
    ![
      estimate.roi,
      estimate.roiLow,
      estimate.roiHigh,
      estimate.contribution,
      estimate.contributionShare,
    ].every(Number.isFinite) ||
    estimate.roi < 0 ||
    estimate.contribution < 0 ||
    estimate.contributionShare < 0
  ) {
    return 0;
  }
  const clipping = model.numerical?.clipping.channels.find(
    (item) => item.channel.toLowerCase() === channel.toLowerCase(),
  );
  if (!clipping) return 100;
  return model.numerical?.clipping.material || clipping.roiIntervalMaterial
    ? 35
    : 65;
}

export function runDecisionValidation(
  model: ModelResult,
  coherence: EvidenceCoherenceAssessment,
  causal: ValidationLayerResult,
): ValidationLayerResult {
  const channels: DecisionEvidence["channels"] = coherence.channels.map(
    (channel) => {
      const posteriorMass = posteriorMassInRange(
        channel.roi,
        channel.roiLow,
        channel.roiHigh,
        channel.evidenceLow,
        channel.evidenceHigh,
      );
      const plausibilityScore =
        posteriorMass === undefined
          ? 50
          : clampScore((posteriorMass / 0.6) * 100);
      const channelStabilityScore = stabilityScore(
        channel.channel,
        causal,
        channel.roi,
      );
      const channelResolutionScore = resolutionScore(
        channel.roi,
        channel.roiLow,
        channel.roiHigh,
        channel.evidenceLow,
        channel.evidenceHigh,
      );
      const channelIdentificationScore = identificationScore(
        model,
        channel.channel,
        channel.evidenceDependence,
      );
      const channelEconomicScore = economicScore(model, channel.channel);
      const components =
        channel.priorUse === "calibration"
          ? [
              { score: channelStabilityScore, weight: 40 },
              { score: channelResolutionScore, weight: 25 },
              { score: channelIdentificationScore, weight: 25 },
              { score: channelEconomicScore, weight: 10 },
            ]
          : [
              { score: plausibilityScore, weight: 45 },
              { score: channelStabilityScore, weight: 25 },
              { score: channelResolutionScore, weight: 15 },
              { score: channelIdentificationScore, weight: 10 },
              { score: channelEconomicScore, weight: 5 },
            ];
      const unconstrainedScore = weightedGeometric(components);
      const score = channel.blocking
        ? Math.min(unconstrainedScore, 20)
        : unconstrainedScore;
      return {
        channel: channel.channel,
        spendShare: channel.spendShare,
        material: channel.material,
        roi: channel.roi,
        roiLow: channel.roiLow,
        roiHigh: channel.roiHigh,
        evidenceLabel: channel.evidenceLabel,
        evidenceLow: channel.evidenceLow,
        evidenceHigh: channel.evidenceHigh,
        posteriorMass,
        priorUse: channel.priorUse,
        identification: channel.identification,
        evidenceDependence: channel.evidenceDependence,
        plausibilityScore,
        stabilityScore: channelStabilityScore,
        resolutionScore: channelResolutionScore,
        identificationScore: channelIdentificationScore,
        economicScore: channelEconomicScore,
        score,
        status: channel.blocking ? "fail" : statusForScore(score),
        blocking: channel.blocking,
      };
    },
  );
  const materialChannels = channels.filter((channel) => channel.material);
  const scoredChannels = materialChannels.length ? materialChannels : channels;
  const score = portfolioScore(scoredChannels);
  const status: ValidationStatus =
    coherence.status === "fail"
      ? "fail"
      : coherence.status === "review" && score >= 75
        ? "review"
        : statusForScore(score);
  const calibratedCount = scoredChannels.filter(
    (channel) => channel.priorUse === "calibration",
  ).length;
  const externallyInformedChannels = scoredChannels.filter(
    (channel) => channel.evidenceDependence,
  );
  const plausibility = portfolioScore(
    scoredChannels.map((channel) => ({
      score: channel.plausibilityScore,
      spendShare: channel.spendShare,
    })),
  );
  const stability = portfolioScore(
    scoredChannels.map((channel) => ({
      score: channel.stabilityScore,
      spendShare: channel.spendShare,
    })),
  );
  const resolution = portfolioScore(
    scoredChannels.map((channel) => ({
      score: channel.resolutionScore,
      spendShare: channel.spendShare,
    })),
  );
  const identification = portfolioScore(
    scoredChannels.map((channel) => ({
      score: channel.identificationScore,
      spendShare: channel.spendShare,
    })),
  );
  const sourceQuality = externallyInformedChannels.length
    ? portfolioScore(
        externallyInformedChannels.map((channel) => ({
          score: (channel.evidenceDependence?.quality ?? 0) * 100,
          spendShare: channel.spendShare,
        })),
      )
    : 50;
  const evidenceCompatibility = externallyInformedChannels.length
    ? portfolioScore(
        externallyInformedChannels.map((channel) => ({
          score: evidenceCompatibilityScore(channel.evidenceDependence),
          spendShare: channel.spendShare,
        })),
      )
    : 50;
  const decisionDependence = externallyInformedChannels.length
    ? portfolioScore(
        externallyInformedChannels.map((channel) => ({
          score: evidenceDependenceScore(channel.evidenceDependence),
          spendShare: channel.spendShare,
        })),
      )
    : 50;
  const economics = portfolioScore(
    scoredChannels.map((channel) => ({
      score: channel.economicScore,
      spendShare: channel.spendShare,
    })),
  );
  const tests: ValidationTest[] = [
    test(
      "roi-posterior-plausibility",
      "Posterior ROI plausibility",
      plausibility,
      coherence.blockingChannels.length
        ? `${coherence.blockingChannels.length} blocking channel${coherence.blockingChannels.length === 1 ? "" : "s"}`
        : `${Math.round(plausibility)}/100 portfolio alignment`,
      `${coherence.summary} Agreement with evidence already used for calibration is contract adherence and receives no independent score credit.`,
      "critical",
      coherence.status === "fail"
        ? "fail"
        : coherence.status === "review"
          ? "review"
          : "pass",
    ),
    test(
      "roi-decision-stability",
      "ROI decision stability",
      stability,
      `${Math.round(stability)}/100 across histories`,
      "Checks whether material-channel ROI remains usable as the available measurement history expands.",
      "critical",
    ),
    test(
      "roi-resolution",
      "Posterior resolution",
      resolution,
      `${Math.round(resolution)}/100 interval resolution`,
      "Penalizes posterior intervals that are too broad relative to the channel's registered evidence range.",
      "high",
    ),
    test(
      "roi-identification",
      "Evidence attribution and identification",
      identification,
      `${calibratedCount}/${scoredChannels.length} externally calibrated`,
      "Combines conditional observational information with continuous experiment and benchmark attribution. Regularization is reported but receives no identification credit.",
      "critical",
    ),
    test(
      "evidence-source-quality",
      "External evidence quality",
      sourceQuality,
      externallyInformedChannels.length
        ? `${externallyInformedChannels.length} source-informed channel${externallyInformedChannels.length === 1 ? "" : "s"}`
        : "Not applicable",
      "Reports independence, precision, relevance, and transportability separately from how influential the source became.",
      "critical",
      externallyInformedChannels.length ? undefined : "incomplete",
    ),
    test(
      "evidence-compatibility",
      "Evidence compatibility",
      evidenceCompatibility,
      externallyInformedChannels.length
        ? `${Math.round(evidenceCompatibility)}/100 standardized agreement`
        : "Not applicable",
      "Compares each external ROI source with the leave-source-out observational estimate on their joint uncertainty scale.",
      "critical",
      externallyInformedChannels.length ? undefined : "incomplete",
    ),
    test(
      "evidence-decision-dependence",
      "Evidence-qualified decision dependence",
      decisionDependence,
      externallyInformedChannels.length
        ? `${Math.round(decisionDependence)}/100 qualified robustness`
        : "Not applicable",
      "Measures ROI, contribution, marginal-profit, and incremental-allocation changes after removing a source; dependence is penalized only when influence is paired with weak quality or conflict.",
      "critical",
      externallyInformedChannels.length ? undefined : "incomplete",
    ),
    test(
      "roi-economic-consistency",
      "Economic consistency",
      economics,
      `${Math.round(economics)}/100 accounting integrity`,
      "Checks finite, non-negative ROI and contribution accounting and reports where coefficient clipping materially changes the decision estimate.",
      "high",
    ),
  ];
  return {
    id: "decision",
    score,
    status,
    summary:
      status === "pass"
        ? "Material-channel ROI is plausible, resolved, stable, and economically usable under the declared evidence contract."
        : status === "review"
          ? "ROI is usable only with the reported plausibility, stability, or identification qualifications."
          : "At least one material-channel ROI is not sufficiently coherent for production allocation.",
    tests,
    evidence: {
      kind: "decision",
      channels,
    },
  };
}
