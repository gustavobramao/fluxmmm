const WORKSPACE_DATABASE = "fluxmmm-local-workspace";
const WORKSPACE_STORE = "checkpoints";
const LATEST_CHECKPOINT_KEY = "latest";

export const WORKSPACE_CHECKPOINT_VERSION =
  "fluxmmm-workspace-checkpoint-v1";

export interface WorkspaceCheckpointEnvelope<T> {
  id: typeof LATEST_CHECKPOINT_KEY;
  version: typeof WORKSPACE_CHECKPOINT_VERSION;
  savedAt: string;
  datasetHash: string;
  payload: T;
}

export function createWorkspaceCheckpoint<T>(
  datasetHash: string,
  payload: T,
  savedAt = new Date().toISOString(),
): WorkspaceCheckpointEnvelope<T> {
  return {
    id: LATEST_CHECKPOINT_KEY,
    version: WORKSPACE_CHECKPOINT_VERSION,
    savedAt,
    datasetHash,
    payload,
  };
}

export function isCompatibleWorkspaceCheckpoint<T>(
  value: unknown,
): value is WorkspaceCheckpointEnvelope<T> {
  if (!value || typeof value !== "object") return false;
  const checkpoint = value as Partial<WorkspaceCheckpointEnvelope<T>>;
  return Boolean(
    checkpoint.id === LATEST_CHECKPOINT_KEY &&
      checkpoint.version === WORKSPACE_CHECKPOINT_VERSION &&
      typeof checkpoint.savedAt === "string" &&
      typeof checkpoint.datasetHash === "string" &&
      checkpoint.payload &&
      typeof checkpoint.payload === "object",
  );
}

function openWorkspaceDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(WORKSPACE_DATABASE, 1);
    request.onerror = () => reject(request.error);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(WORKSPACE_STORE)) {
        database.createObjectStore(WORKSPACE_STORE, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
  });
}

export async function readLatestWorkspaceCheckpoint<T>(): Promise<
  WorkspaceCheckpointEnvelope<T> | null
> {
  if (typeof window === "undefined" || !window.indexedDB) return null;
  try {
    const database = await openWorkspaceDatabase();
    const value = await new Promise<unknown>((resolve, reject) => {
      const transaction = database.transaction(WORKSPACE_STORE, "readonly");
      const request = transaction.objectStore(WORKSPACE_STORE).get(
        LATEST_CHECKPOINT_KEY,
      );
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);
    });
    database.close();
    return isCompatibleWorkspaceCheckpoint<T>(value) ? value : null;
  } catch {
    return null;
  }
}

export async function persistWorkspaceCheckpoint<T>(
  checkpoint: WorkspaceCheckpointEnvelope<T>,
): Promise<boolean> {
  if (typeof window === "undefined" || !window.indexedDB) return false;
  try {
    const database = await openWorkspaceDatabase();
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(WORKSPACE_STORE, "readwrite");
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
      transaction.oncomplete = () => resolve();
      transaction.objectStore(WORKSPACE_STORE).put(checkpoint);
    });
    database.close();
    return true;
  } catch {
    return false;
  }
}

export async function clearWorkspaceCheckpoint(): Promise<boolean> {
  if (typeof window === "undefined" || !window.indexedDB) return false;
  try {
    const database = await openWorkspaceDatabase();
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(WORKSPACE_STORE, "readwrite");
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
      transaction.oncomplete = () => resolve();
      transaction.objectStore(WORKSPACE_STORE).delete(LATEST_CHECKPOINT_KEY);
    });
    database.close();
    return true;
  } catch {
    return false;
  }
}
