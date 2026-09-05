import { readFile, rename, writeFile } from "node:fs/promises";

/**
 * Recover complete top-level JSON objects from an interrupted pretty-printed
 * array. Nested objects and braces inside strings are handled explicitly; an
 * incomplete trailing object is ignored.
 */
export function recoverCompleteJsonObjects<T>(source: string): T[] {
  const recovered: T[] = [];
  let objectStart = -1;
  let objectDepth = 0;
  let inString = false;
  let escaped = false;

  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    if (inString) {
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === '"') inString = false;
      continue;
    }
    if (character === '"') {
      inString = true;
      continue;
    }
    if (character === "{") {
      if (objectDepth === 0) objectStart = index;
      objectDepth += 1;
      continue;
    }
    if (character !== "}" || objectDepth === 0) continue;
    objectDepth -= 1;
    if (objectDepth === 0 && objectStart >= 0) {
      recovered.push(JSON.parse(source.slice(objectStart, index + 1)) as T);
      objectStart = -1;
    }
  }

  return recovered;
}

export async function recoverCheckpoint<T>(
  inputPath: string,
  outputPath: string,
): Promise<T[]> {
  const recovered = recoverCompleteJsonObjects<T>(await readFile(inputPath, "utf8"));
  const temporaryPath = `${outputPath}.tmp`;
  await writeFile(temporaryPath, JSON.stringify(recovered, null, 2));
  await rename(temporaryPath, outputPath);
  return recovered;
}
