#!/usr/bin/env python3
"""Run one Flux NUTS payload synchronously for offline score research."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "flux_mcmc_batch_server",
    ROOT / "scripts" / "mcmc_server.py",
)
if not SPEC or not SPEC.loader:
    raise RuntimeError("Unable to load the Flux MCMC implementation.")
MCMC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MCMC
SPEC.loader.exec_module(MCMC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload_path = Path(args.payload).resolve()
    output_path = Path(args.output).resolve()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    fingerprint = str(payload.get("fingerprint", ""))
    cached = MCMC.cached_result(fingerprint)
    if cached:
        result = cached
    else:
        job = MCMC.Job(id=fingerprint, fingerprint=fingerprint)
        MCMC.run_sampling(job, payload)
        if job.status != "complete" or not job.result:
            raise RuntimeError(job.error or "Sampling did not produce a posterior result.")
        result = job.result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "status": result.get("status"),
                "runtimeSeconds": result.get("runtimeSeconds"),
                "cached": result.get("cached", False),
            }
        )
    )


if __name__ == "__main__":
    main()
