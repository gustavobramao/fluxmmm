import type { SviInferenceContract } from "../../research/svi_score_v3/contract";
import type { SviApproximationResult } from "../../research/svi_score_v3/types";
import type { CompiledSamplingModel, SamplingJobProgress } from "./sampling";
import { SAMPLING_SERVICE_PORT } from "./sampling-api";

export const REGRETSET_V11_SERVICE_CONTRACT =
  "flux-fullrank-advi-v11-runtime-v1";
const SERVICE_URL = `http://127.0.0.1:${SAMPLING_SERVICE_PORT}`;

async function serviceError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { error?: string };
    return payload.error ?? `FullRankADVI service returned ${response.status}.`;
  } catch {
    return `FullRankADVI service returned ${response.status}.`;
  }
}

export async function regretSetV11ServiceHealth(): Promise<{
  ready: boolean;
  engine?: string;
  workers?: number;
  detail?: string;
}> {
  try {
    const response = await fetch(`${SERVICE_URL}/v1/svi/health`);
    if (!response.ok) return { ready: false };
    const health = (await response.json()) as {
      ready: boolean;
      engine?: string;
      contractVersion?: string;
      workers?: number;
    };
    return {
      ready:
        health.ready &&
        health.contractVersion === REGRETSET_V11_SERVICE_CONTRACT,
      engine: health.engine,
      workers: health.workers,
      detail:
        health.contractVersion === REGRETSET_V11_SERVICE_CONTRACT
          ? undefined
          : "The local FullRankADVI service does not match the frozen RegretSet-MMM runtime contract.",
    };
  } catch {
    return {
      ready: false,
      detail:
        "The local FullRankADVI service is unavailable. Restart Flux with pnpm dev after running pnpm mcmc:setup.",
    };
  }
}

export async function startRegretSetV11PosteriorJob(
  fingerprint: string,
  model: CompiledSamplingModel,
  contract: SviInferenceContract,
): Promise<{ id: string; status: string; cached?: boolean }> {
  let response: Response;
  try {
    response = await fetch(`${SERVICE_URL}/v1/svi`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ fingerprint, model, contract }),
    });
  } catch {
    throw new Error(
      "The local FullRankADVI service is unavailable. Restart Flux after installing the sampling requirements.",
    );
  }
  if (!response.ok) throw new Error(await serviceError(response));
  return (await response.json()) as { id: string; status: string; cached?: boolean };
}

export async function getRegretSetV11PosteriorJob(jobId: string): Promise<{
  id: string;
  status: "queued" | "running" | "complete" | "error";
  progress: SamplingJobProgress;
  result?: SviApproximationResult;
  error?: string;
}> {
  let response: Response;
  try {
    response = await fetch(`${SERVICE_URL}/v1/svi/${encodeURIComponent(jobId)}`);
  } catch {
    throw new Error("Connection to the local FullRankADVI service was interrupted.");
  }
  if (!response.ok) throw new Error(await serviceError(response));
  return (await response.json()) as {
    id: string;
    status: "queued" | "running" | "complete" | "error";
    progress: SamplingJobProgress;
    result?: SviApproximationResult;
    error?: string;
  };
}
