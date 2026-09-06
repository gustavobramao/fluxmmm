import {
  normalizeMediaResponseConfig,
  responseTransform,
} from "../../lib/mmm/response";
import type { SamplingDecisionDraw } from "../../lib/mmm/sampling";
import {
  V11_AMSS_CONFIRMATORY_CHANNELS,
  V11_AMSS_CONFIRMATORY_DECISION_SCENARIOS,
} from "./contract";

export interface V11ConfirmatoryBusinessDecisionContext {
  grossMargin: number;
  baselineAnnualSpend: Record<string, number>;
  observedSpend: Record<string, number[]>;
}

export interface V11ConfirmatoryCandidateAction {
  scenario: (typeof V11_AMSS_CONFIRMATORY_DECISION_SCENARIOS)[number]["id"];
  annualSpend: Record<string, number>;
  totalAnnualSpend: number;
  predictedIncrementalProfit: number;
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function simplex(units: number): number[][] {
  const output: number[][] = [];
  for (let left = 0; left <= units; left += 1) {
    for (let middle = 0; middle <= units - left; middle += 1) {
      output.push([left / units, middle / units, (units - left - middle) / units]);
    }
  }
  return output;
}

const ALLOCATION_SHARES = simplex(20);

function linearGrid(low: number, high: number, count: number): number[] {
  if (count <= 1 || Math.abs(high - low) < 1e-12) return [low];
  return Array.from({ length: count }, (_, index) => low + index * (high - low) / (count - 1));
}

function expectedContributionSurface(
  context: V11ConfirmatoryBusinessDecisionContext,
  draws: readonly SamplingDecisionDraw[],
): (spend: Record<string, number>) => number {
  const selected = draws.length <= 64
    ? [...draws]
    : Array.from({ length: 64 }, (_, index) =>
      draws[Math.floor(((index + 0.5) / 64) * draws.length)]
    );
  const surfacePoints = 201;
  const maximumScale = 2.5;
  const surfaces = new Map(V11_AMSS_CONFIRMATORY_CHANNELS.map((channel) => {
    const spend = context.observedSpend[channel] ?? [];
    const recent = spend.slice(-52);
    const recentTotal = Math.max(sum(recent), 1);
    const values = Array.from({ length: surfacePoints }, () => 0);
    selected.forEach((draw) => {
      const sampled = draw.channels.find((item) =>
        item.channel.toLowerCase() === channel.toLowerCase()
      );
      if (!sampled) return;
      const response = normalizeMediaResponseConfig(sampled.response);
      const fitted = responseTransform(spend, sampled.response);
      const coefficient = sampled.contribution / Math.max(sum(fitted.transformed), 1e-9);
      const baseline = responseTransform(recent, response);
      const carryoverPowers = baseline.carryover.map((value) =>
        Math.max(value, 0) ** response.saturation
      );
      const halfPower = baseline.halfSaturation ** response.saturation;
      for (let index = 0; index < values.length; index += 1) {
        const scale = maximumScale * index / (values.length - 1);
        const scalePower = scale ** response.saturation;
        let transformed = 0;
        for (const carryoverPower of carryoverPowers) {
          const powered = carryoverPower * scalePower;
          transformed += powered / (powered + halfPower || 1);
        }
        values[index] += coefficient * transformed / selected.length;
      }
    });
    return [channel, { values, recentTotal }] as const;
  }));
  return (target) => V11_AMSS_CONFIRMATORY_CHANNELS.reduce((total, channel) => {
    const surface = surfaces.get(channel)!;
    const scale = Math.min(maximumScale, Math.max(0, (target[channel] ?? 0) / surface.recentTotal));
    const position = scale / maximumScale * (surface.values.length - 1);
    const lower = Math.floor(position);
    const upper = Math.min(surface.values.length - 1, Math.ceil(position));
    const weight = position - lower;
    return total + surface.values[lower] * (1 - weight) + surface.values[upper] * weight;
  }, 0);
}

function roundAction(
  values: Record<string, number>,
  baselineTotal: number,
): Record<string, number> {
  const step = Math.max(1, baselineTotal * 0.005);
  return Object.fromEntries(V11_AMSS_CONFIRMATORY_CHANNELS.map((channel) => [
    channel,
    Math.max(0, Math.round((values[channel] ?? 0) / step) * step),
  ]));
}

export function recommendV11ConfirmatoryCandidateActions(
  context: V11ConfirmatoryBusinessDecisionContext,
  draws: readonly SamplingDecisionDraw[],
): V11ConfirmatoryCandidateAction[] {
  if (!draws.length) throw new Error("V11 confirmatory decision construction requires posterior draws.");
  const expectedContribution = expectedContributionSurface(context, draws);
  const baselineTotal = sum(V11_AMSS_CONFIRMATORY_CHANNELS.map((channel) =>
    context.baselineAnnualSpend[channel] ?? 0
  ));
  const baselineContribution = expectedContribution(context.baselineAnnualSpend);
  return V11_AMSS_CONFIRMATORY_DECISION_SCENARIOS.map((scenario) => {
    let bestSpend = structuredClone(context.baselineAnnualSpend);
    let bestProfit = Number.NEGATIVE_INFINITY;
    const totalShares = linearGrid(
      scenario.minimumTotalShare,
      scenario.maximumTotalShare,
      21,
    );
    for (const totalShare of totalShares) {
      const targetTotal = baselineTotal * totalShare;
      for (const shares of ALLOCATION_SHARES) {
        const candidate = Object.fromEntries(V11_AMSS_CONFIRMATORY_CHANNELS.map((channel, index) => [
          channel,
          targetTotal * shares[index],
        ]));
        if (V11_AMSS_CONFIRMATORY_CHANNELS.some((channel) =>
          candidate[channel] > 2.5 * Math.max(context.baselineAnnualSpend[channel] ?? 0, 1)
        )) continue;
        const profit =
          (expectedContribution(candidate) - baselineContribution) * context.grossMargin -
          (targetTotal - baselineTotal);
        if (profit > bestProfit) {
          bestProfit = profit;
          bestSpend = candidate;
        }
      }
    }
    const annualSpend = roundAction(bestSpend, baselineTotal);
    return {
      scenario: scenario.id,
      annualSpend,
      totalAnnualSpend: sum(Object.values(annualSpend)),
      predictedIncrementalProfit: bestProfit,
    };
  });
}
