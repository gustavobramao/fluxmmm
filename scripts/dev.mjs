import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const python = path.join(root, ".venv", "bin", "python");
const vinext = path.join(root, "node_modules", ".bin", "vinext");
const webPort = process.env.FLUX_PORT || "3003";
const samplerPort = process.env.FLUX_MCMC_PORT || "8790";
const children = [];

function launch(command, args, label) {
  const child = spawn(command, args, {
    cwd: root,
    env: {
      ...process.env,
      WRANGLER_LOG_PATH: ".wrangler/wrangler.log",
    },
    stdio: "inherit",
  });
  child.on("exit", (code, signal) => {
    if (signal || code === 0) return;
    console.error(`${label} stopped with exit code ${code}.`);
  });
  children.push(child);
  return child;
}

if (existsSync(python)) {
  launch(python, ["scripts/mcmc_server.py", "--port", samplerPort], "MCMC service");
} else {
  console.warn(
    "MCMC service is not installed. Run `pnpm mcmc:setup` before production sampling.",
  );
}

const web = launch(vinext, ["dev", "--port", webPort], "Flux web app");

function shutdown(signal) {
  for (const child of children) {
    if (!child.killed) child.kill(signal);
  }
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
web.on("exit", () => shutdown("SIGTERM"));
