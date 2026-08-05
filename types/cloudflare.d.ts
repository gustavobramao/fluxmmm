declare interface Fetcher {
  fetch(request: Request): Promise<Response>;
}

declare interface D1Result<T = unknown> {
  success: boolean;
  results?: T[];
}

declare interface D1PreparedStatement {
  bind(...values: unknown[]): D1PreparedStatement;
  run<T = unknown>(): Promise<D1Result<T>>;
  first<T = Record<string, unknown>>(): Promise<T | null>;
}

declare interface D1Database {
  prepare(query: string): D1PreparedStatement;
  batch<T = unknown>(
    statements: D1PreparedStatement[],
  ): Promise<D1Result<T>[]>;
}

declare interface R2PutOptions {
  httpMetadata?: { contentType?: string };
  customMetadata?: Record<string, string>;
}

declare interface R2Bucket {
  put(
    key: string,
    value: string | ArrayBuffer | ReadableStream,
    options?: R2PutOptions,
  ): Promise<unknown>;
}

declare module "cloudflare:workers" {
  export const env: {
    DB: D1Database;
    [key: string]: unknown;
  };
}
