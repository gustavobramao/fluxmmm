/** Cloudflare Worker entry point for Flux MMM. */
import {
  DEFAULT_DEVICE_SIZES,
  DEFAULT_IMAGE_SIZES,
  handleImageOptimization,
} from "vinext/server/image-optimization";
import handler from "vinext/server/app-router-entry";

interface Env {
  ASSETS: Fetcher;
  DB: D1Database;
  DATASETS: R2Bucket;
  IMAGES: {
    input(stream: ReadableStream): {
      transform(options: Record<string, unknown>): {
        output(options: {
          format: string;
          quality: number;
        }): Promise<{ response(): Response }>;
      };
    };
  };
}

interface ExecutionContext {
  waitUntil(promise: Promise<unknown>): void;
  passThroughOnException(): void;
}

async function ensureSchema(env: Env): Promise<void> {
  await env.DB.batch([
    env.DB.prepare(`
      CREATE TABLE IF NOT EXISTS datasets (
        hash TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        object_key TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        column_count INTEGER NOT NULL,
        created_at TEXT NOT NULL
      )
    `),
    env.DB.prepare(`
      CREATE TABLE IF NOT EXISTS model_runs (
        fingerprint TEXT PRIMARY KEY,
        dataset_hash TEXT NOT NULL,
        kind TEXT NOT NULL,
        result_json TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    `),
    env.DB.prepare(
      "CREATE INDEX IF NOT EXISTS model_runs_dataset_idx ON model_runs(dataset_hash)",
    ),
  ]);
}

function json(data: unknown, status = 200): Response {
  return Response.json(data, {
    status,
    headers: { "cache-control": "no-store" },
  });
}

async function handleApi(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  await ensureSchema(env);

  if (url.pathname === "/api/datasets" && request.method === "POST") {
    const payload = (await request.json()) as {
      hash?: string;
      name?: string;
      csv?: string;
      rows?: number;
      columns?: number;
    };
    if (!payload.hash || !payload.name || !payload.csv) {
      return json({ error: "Dataset hash, name, and CSV are required." }, 400);
    }
    if (payload.csv.length > 8_000_000) {
      return json({ error: "The v1 upload limit is 8 MB." }, 413);
    }

    const objectKey = `datasets/${payload.hash}.csv`;
    await env.DATASETS.put(objectKey, payload.csv, {
      httpMetadata: { contentType: "text/csv; charset=utf-8" },
      customMetadata: { originalName: payload.name },
    });
    await env.DB.prepare(
      `INSERT INTO datasets
        (hash, name, object_key, row_count, column_count, created_at)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(hash) DO UPDATE SET
        name = excluded.name,
        object_key = excluded.object_key,
        row_count = excluded.row_count,
        column_count = excluded.column_count`,
    )
      .bind(
        payload.hash,
        payload.name,
        objectKey,
        payload.rows ?? 0,
        payload.columns ?? 0,
        new Date().toISOString(),
      )
      .run();
    return json({ stored: true, hash: payload.hash });
  }

  if (url.pathname.startsWith("/api/runs/") && request.method === "GET") {
    const fingerprint = decodeURIComponent(
      url.pathname.slice("/api/runs/".length),
    );
    const row = await env.DB.prepare(
      "SELECT result_json, created_at FROM model_runs WHERE fingerprint = ?",
    )
      .bind(fingerprint)
      .first<{ result_json: string; created_at: string }>();
    if (!row) return json({ found: false }, 404);
    return json({
      found: true,
      result: JSON.parse(row.result_json),
      createdAt: row.created_at,
    });
  }

  if (url.pathname === "/api/runs" && request.method === "POST") {
    const payload = (await request.json()) as {
      fingerprint?: string;
      datasetHash?: string;
      kind?: string;
      result?: unknown;
    };
    if (
      !payload.fingerprint ||
      !payload.datasetHash ||
      !payload.kind ||
      !payload.result
    ) {
      return json({ error: "A complete model artifact is required." }, 400);
    }
    await env.DB.prepare(
      `INSERT INTO model_runs
        (fingerprint, dataset_hash, kind, result_json, created_at)
       VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(fingerprint) DO UPDATE SET
        result_json = excluded.result_json,
        created_at = excluded.created_at`,
    )
      .bind(
        payload.fingerprint,
        payload.datasetHash,
        payload.kind,
        JSON.stringify(payload.result),
        new Date().toISOString(),
      )
      .run();
    return json({ stored: true, fingerprint: payload.fingerprint });
  }

  return json({ error: "Not found" }, 404);
}

const worker = {
  async fetch(
    request: Request,
    env: Env,
    ctx: ExecutionContext,
  ): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname.startsWith("/api/")) {
      try {
        return await handleApi(request, env);
      } catch (error) {
        return json(
          {
            error:
              error instanceof Error ? error.message : "Unexpected API error",
          },
          500,
        );
      }
    }

    if (url.pathname === "/_vinext/image") {
      const allowedWidths = [...DEFAULT_DEVICE_SIZES, ...DEFAULT_IMAGE_SIZES];
      return handleImageOptimization(
        request,
        {
          fetchAsset: (path) =>
            env.ASSETS.fetch(new Request(new URL(path, request.url))),
          transformImage: async (body, { width, format, quality }) => {
            const result = await env.IMAGES.input(body)
              .transform(width > 0 ? { width } : {})
              .output({ format, quality });
            return result.response();
          },
        },
        allowedWidths,
      );
    }

    return handler.fetch(request, env, ctx);
  },
};

export default worker;
