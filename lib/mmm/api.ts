import type { Dataset, ModelResult } from "./types";
import type { SamplingResult } from "./sampling";
import type { BudgetOptimizationResult } from "./budget";
import type { ValidationResult } from "./validation";

export async function persistDataset(dataset: Dataset): Promise<boolean> {
  try {
    const response = await fetch("/api/datasets", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        hash: dataset.hash,
        name: dataset.name,
        csv: dataset.rawCsv,
        rows: dataset.rows.length,
        columns: dataset.columns.length,
      }),
    });
    return response.ok;
  } catch {
    return false;
  }
}

export async function getCachedModel(
  fingerprint: string,
): Promise<ModelResult | null> {
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(fingerprint)}`);
    if (!response.ok) return null;
    const payload = (await response.json()) as { result?: ModelResult };
    return payload.result ? { ...payload.result, cached: true } : null;
  } catch {
    return null;
  }
}

export async function persistModel(
  datasetHash: string,
  result: ModelResult,
): Promise<boolean> {
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        fingerprint: result.fingerprint,
        datasetHash,
        kind: result.kind,
        result,
      }),
    });
    return response.ok;
  } catch {
    return false;
  }
}

export async function getCachedValidation(
  fingerprint: string,
): Promise<ValidationResult | null> {
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(fingerprint)}`);
    if (!response.ok) return null;
    const payload = (await response.json()) as { result?: ValidationResult };
    return payload.result
      ? { ...payload.result, cached: true }
      : null;
  } catch {
    return null;
  }
}

export async function persistValidation(
  datasetHash: string,
  result: ValidationResult,
): Promise<boolean> {
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        fingerprint: result.fingerprint,
        datasetHash,
        kind: "validation",
        result,
      }),
    });
    return response.ok;
  } catch {
    return false;
  }
}

export async function getCachedSampling(
  fingerprint: string,
): Promise<SamplingResult | null> {
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(fingerprint)}`);
    if (!response.ok) return null;
    const payload = (await response.json()) as { result?: SamplingResult };
    return payload.result?.kind === "mcmc"
      ? { ...payload.result, cached: true }
      : null;
  } catch {
    return null;
  }
}

export async function persistSampling(
  datasetHash: string,
  result: SamplingResult,
): Promise<boolean> {
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        fingerprint: result.fingerprint,
        datasetHash,
        kind: "mcmc",
        result,
      }),
    });
    return response.ok;
  } catch {
    return false;
  }
}

export async function getCachedBudget(
  fingerprint: string,
): Promise<BudgetOptimizationResult | null> {
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(fingerprint)}`);
    if (!response.ok) return null;
    const payload = (await response.json()) as {
      result?: BudgetOptimizationResult;
    };
    return payload.result?.kind === "budget"
      ? { ...payload.result, cached: true }
      : null;
  } catch {
    return null;
  }
}

export async function persistBudget(
  datasetHash: string,
  result: BudgetOptimizationResult,
): Promise<boolean> {
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        fingerprint: result.fingerprint,
        datasetHash,
        kind: "budget",
        result,
      }),
    });
    return response.ok;
  } catch {
    return false;
  }
}
