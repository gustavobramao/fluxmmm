export interface V10EconomicLoss {
  economicScale: number;
  capped: number;
  uncapped: number;
}

export function calculateV10EconomicLoss(input: {
  oracleProfit: number;
  realizedProfit: number;
  baselineProfit: number;
  baselineBudget: number;
  normalizedLossCap: number;
}): V10EconomicLoss {
  const values = Object.values(input);
  if (values.some((value) => !Number.isFinite(value))) {
    throw new Error("V10 economic-loss inputs must be finite.");
  }
  if (input.baselineBudget < 0 || input.normalizedLossCap <= 0) {
    throw new Error("V10 economic-loss budget and cap must be positive.");
  }
  const economicScale = Math.max(
    Math.abs(input.oracleProfit - input.baselineProfit),
    0.02 * input.baselineBudget,
    1,
  );
  const uncapped = Math.max(0, input.oracleProfit - input.realizedProfit) / economicScale;
  return {
    economicScale,
    uncapped,
    capped: Math.min(uncapped, input.normalizedLossCap),
  };
}
