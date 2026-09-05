export function integerArgument(
  arguments_: readonly string[],
  name: string,
  fallback: number,
  minimum = 1,
): number {
  const prefix = `--${name}=`;
  const value = arguments_.find((argument) => argument.startsWith(prefix));
  if (!value) return fallback;
  const parsed = Number.parseInt(value.slice(prefix.length), 10);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Invalid integer argument --${name}.`);
  }
  return Math.max(minimum, parsed);
}
