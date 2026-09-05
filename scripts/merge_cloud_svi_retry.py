#!/usr/bin/env python3
"""Validate staged cloud SVI results and immutably promote them into cache."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def load_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}.")
    return payload


def validate_result(
    result: dict[str, Any],
    payload: dict[str, Any],
    fingerprint: str,
) -> None:
    if result.get("fingerprint") != fingerprint:
        raise ValueError(f"Result fingerprint mismatch: {fingerprint}.")
    if payload.get("fingerprint") != fingerprint:
        raise ValueError(f"Payload fingerprint mismatch: {fingerprint}.")
    if result.get("contract") != payload.get("contract"):
        raise ValueError(f"Inference contract mismatch: {fingerprint}.")
    if result.get("executionBackend") != "cvm-retry":
        raise ValueError(f"Unexpected execution backend: {fingerprint}.")
    if result.get("engine") != "PyMC 6.2 · FullRankADVI":
        raise ValueError(f"Unexpected inference engine: {fingerprint}.")
    if result.get("status") not in {"labelled", "review"}:
        raise ValueError(f"Result was not retained: {fingerprint}.")
    if not isinstance(result.get("channels"), list) or not result["channels"]:
        raise ValueError(f"Result has no channel posterior: {fingerprint}.")
    if not isinstance(result.get("decisionDraws"), list) or not result["decisionDraws"]:
        raise ValueError(f"Result has no decision draws: {fingerprint}.")
    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"Result has no diagnostics receipt: {fingerprint}.")


def promote_results(
    manifest_path: Path,
    staged_directory: Path,
    cache_directory: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != "flux-svi-cvm-retry-v1":
        raise ValueError("Unsupported retry manifest version.")
    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != manifest.get("failureCount"):
        raise ValueError("Retry manifest count is inconsistent.")
    expected_names = {
        f"{item['fingerprint']}.result.json.gz"
        for item in items
        if isinstance(item, dict) and isinstance(item.get("fingerprint"), str)
    }
    staged_names = {path.name for path in staged_directory.glob("*.result.json.gz")}
    if staged_names != expected_names:
        raise ValueError(
            f"Staged result set mismatch: expected {len(expected_names)}, "
            f"received {len(staged_names)}."
        )

    cache_directory.mkdir(parents=True, exist_ok=True)
    promoted = 0
    existing = 0
    statuses = {"labelled": 0, "review": 0}
    runtimes: list[float] = []
    for item in items:
        fingerprint = item["fingerprint"]
        staged_path = staged_directory / f"{fingerprint}.result.json.gz"
        payload_path = cache_directory / f"{fingerprint}.payload.json.gz"
        target_path = cache_directory / staged_path.name
        staged_result = load_gzip_json(staged_path)
        payload = load_gzip_json(payload_path)
        validate_result(staged_result, payload, fingerprint)
        statuses[staged_result["status"]] += 1
        runtime = staged_result.get("runtimeSeconds")
        if isinstance(runtime, (int, float)):
            runtimes.append(float(runtime))
        if target_path.exists():
            cached_result = load_gzip_json(target_path)
            validate_result(cached_result, payload, fingerprint)
            existing += 1
            continue
        with target_path.open("xb") as destination:
            destination.write(staged_path.read_bytes())
        promoted += 1

    sorted_runtimes = sorted(runtimes)
    return {
        "validated": len(items),
        "promoted": promoted,
        "existingImmutableResults": existing,
        "statuses": statuses,
        "minimumRuntimeSeconds": sorted_runtimes[0] if sorted_runtimes else None,
        "medianRuntimeSeconds": (
            sorted_runtimes[len(sorted_runtimes) // 2] if sorted_runtimes else None
        ),
        "maximumRuntimeSeconds": sorted_runtimes[-1] if sorted_runtimes else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--staged-results", required=True)
    parser.add_argument("--cache", default=".flux-artifacts/svi-score-v3")
    arguments = parser.parse_args()
    receipt = promote_results(
        Path(arguments.manifest).resolve(),
        Path(arguments.staged_results).resolve(),
        Path(arguments.cache).resolve(),
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
