import {
  SAMPLING_ENGINE_VERSION,
  type CompiledSamplingModel,
  type SamplingContract,
  type SamplingJobSnapshot,
} from "./sampling";

const SERVICE_URL = "http://127.0.0.1:8789";

async function serviceError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { error?: string };
    return payload.error ?? `Sampling service returned ${response.status}.`;
  } catch {
    return `Sampling service returned ${response.status}.`;
  }
}

export async function samplingServiceHealth(): Promise<{
  ready: boolean;
  engine?: string;
  contractVersion?: string;
}> {
  try {
    const response = await fetch(`${SERVICE_URL}/health`);
    if (!response.ok) return { ready: false };
    const health = (await response.json()) as {
      ready: boolean;
      engine?: string;
      contractVersion?: string;
    };
    return {
      ...health,
      ready:
        health.ready && health.contractVersion === SAMPLING_ENGINE_VERSION,
    };
  } catch {
    return { ready: false };
  }
}

export async function startSamplingJob(
  fingerprint: string,
  model: CompiledSamplingModel,
  contract: SamplingContract,
): Promise<{ id: string; status: string; cached?: boolean }> {
  let response: Response;
  try {
    response = await fetch(`${SERVICE_URL}/v1/sampling`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ fingerprint, model, contract }),
    });
  } catch {
    throw new Error(
      "The local sampling service is unavailable. Restart the Flux development server after installing the MCMC requirements.",
    );
  }
  if (!response.ok) throw new Error(await serviceError(response));
  return (await response.json()) as {
    id: string;
    status: string;
    cached?: boolean;
  };
}

export async function getSamplingJob(
  jobId: string,
): Promise<SamplingJobSnapshot> {
  let response: Response;
  try {
    response = await fetch(
      `${SERVICE_URL}/v1/sampling/${encodeURIComponent(jobId)}`,
    );
  } catch {
    throw new Error("Connection to the local sampling service was interrupted.");
  }
  if (!response.ok) throw new Error(await serviceError(response));
  return (await response.json()) as SamplingJobSnapshot;
}
