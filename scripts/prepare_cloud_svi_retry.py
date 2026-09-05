#!/usr/bin/env python3
"""Build an immutable GCP retry manifest from technical-error SVI rows."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def prepare_manifest(
    checkpoint_path: Path,
    cache_directory: Path,
    output_directory: Path,
    remote_prefix: str,
) -> tuple[Path, Path, dict[str, Any]]:
    records = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    failures = [record for record in records if record.get("status") == "error"]
    fingerprints = [record.get("inferenceFingerprint") for record in failures]
    if any(not isinstance(value, str) or len(value) != 64 for value in fingerprints):
        raise ValueError("Every technical-error row must have an inference fingerprint.")
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Technical-error inference fingerprints must be unique.")

    normalized_prefix = remote_prefix.strip("/")
    items: list[dict[str, str]] = []
    payload_paths: list[str] = []
    staged_payload_directory = output_directory / "payloads"
    staged_payload_directory.mkdir(parents=True, exist_ok=True)
    for record, fingerprint in zip(failures, fingerprints, strict=True):
        payload_path = cache_directory / f"{fingerprint}.payload.json.gz"
        payload_bytes = payload_path.read_bytes()
        payload = json.loads(gzip.decompress(payload_bytes))
        if payload.get("fingerprint") != fingerprint:
            raise ValueError(f"Payload fingerprint mismatch: {payload_path}.")
        items.append(
            {
                "fingerprint": fingerprint,
                "businessId": str(record["businessId"]),
                "candidateId": str(record["candidateId"]),
                "payloadObject": f"{normalized_prefix}/payloads/{payload_path.name}",
                "payloadSha256": hashlib.sha256(payload_bytes).hexdigest(),
                "outputObject": (
                    f"{normalized_prefix}/results/{fingerprint}.result.json.gz"
                ),
            }
        )
        payload_paths.append(str(payload_path.resolve()))
        shutil.copyfile(payload_path, staged_payload_directory / payload_path.name)

    manifest: dict[str, Any] = {
        "version": "flux-svi-cvm-retry-v1",
        "sourceCheckpoint": checkpoint_path.name,
        "failureCount": len(items),
        "remotePrefix": normalized_prefix,
        "items": items,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    payload_list_path = output_directory / "payload-paths.txt"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    payload_list_path.write_text("\n".join(payload_paths) + "\n", encoding="utf-8")
    return manifest_path, payload_list_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=".flux-artifacts/svi-score-v3/design-pilot-records.json",
    )
    parser.add_argument("--cache", default=".flux-artifacts/svi-score-v3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--remote-prefix", required=True)
    arguments = parser.parse_args()
    manifest_path, payload_list_path, manifest = prepare_manifest(
        Path(arguments.checkpoint).resolve(),
        Path(arguments.cache).resolve(),
        Path(arguments.output).resolve(),
        arguments.remote_prefix,
    )
    print(
        json.dumps(
            {
                "failureCount": manifest["failureCount"],
                "manifest": str(manifest_path),
                "payloadList": str(payload_list_path),
            }
        )
    )


if __name__ == "__main__":
    main()
