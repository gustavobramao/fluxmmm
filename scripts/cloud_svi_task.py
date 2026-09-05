#!/usr/bin/env python3
"""Execute one immutable Flux SVI retry task from a GCS manifest."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


METADATA_TOKEN_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "service-accounts/default/token"
)
FINGERPRINT_PATTERN = re.compile(r"^[a-f0-9]{64}$")


def access_token() -> str:
    request = Request(METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"})
    with urlopen(request, timeout=15) as response:
        payload = json.loads(response.read())
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("The Batch VM did not return a usable access token.")
    return token


def object_url(bucket: str, object_name: str) -> str:
    encoded_bucket = quote(bucket, safe="")
    encoded_object = quote(object_name, safe="")
    return (
        f"https://storage.googleapis.com/download/storage/v1/b/{encoded_bucket}/"
        f"o/{encoded_object}?alt=media"
    )


def download_object(bucket: str, object_name: str, *, missing_ok: bool = False) -> bytes | None:
    request = Request(
        object_url(bucket, object_name),
        headers={"Authorization": f"Bearer {access_token()}"},
    )
    try:
        with urlopen(request, timeout=120) as response:
            return response.read()
    except HTTPError as error:
        if missing_ok and error.code == 404:
            return None
        raise


def upload_object_immutably(
    bucket: str,
    object_name: str,
    payload: bytes,
    content_type: str,
) -> bool:
    parameters = urlencode(
        {
            "uploadType": "media",
            "name": object_name,
            "ifGenerationMatch": "0",
        }
    )
    request = Request(
        f"https://storage.googleapis.com/upload/storage/v1/b/"
        f"{quote(bucket, safe='')}/o?{parameters}",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {access_token()}",
            "Content-Type": content_type,
        },
    )
    try:
        with urlopen(request, timeout=120) as response:
            response.read()
        return True
    except HTTPError as error:
        if error.code == 412:
            return False
        raise


def manifest_entry(manifest: dict[str, Any], task_index: int) -> dict[str, str]:
    if manifest.get("version") != "flux-svi-cvm-retry-v1":
        raise ValueError("Unsupported retry manifest version.")
    items = manifest.get("items")
    if not isinstance(items, list) or not 0 <= task_index < len(items):
        raise IndexError(f"Batch task index {task_index} is outside the retry manifest.")
    item = items[task_index]
    if not isinstance(item, dict):
        raise TypeError("Retry manifest entries must be objects.")
    required = ("fingerprint", "payloadObject", "payloadSha256", "outputObject")
    if any(not isinstance(item.get(key), str) or not item[key] for key in required):
        raise ValueError("Retry manifest entry is incomplete.")
    if not FINGERPRINT_PATTERN.fullmatch(item["fingerprint"]):
        raise ValueError("Retry manifest contains an invalid inference fingerprint.")
    return {key: item[key] for key in required}


def validate_payload(payload_bytes: bytes, entry: dict[str, str]) -> None:
    digest = hashlib.sha256(payload_bytes).hexdigest()
    if digest != entry["payloadSha256"]:
        raise ValueError(
            f"Payload digest mismatch for {entry['fingerprint']}: {digest}."
        )
    payload = json.loads(gzip.decompress(payload_bytes))
    if payload.get("fingerprint") != entry["fingerprint"]:
        raise ValueError("Payload fingerprint does not match the retry manifest.")


def validate_result(result_bytes: bytes, fingerprint: str) -> dict[str, Any]:
    result = json.loads(gzip.decompress(result_bytes))
    if result.get("fingerprint") != fingerprint:
        raise ValueError("SVI result fingerprint does not match its task.")
    if result.get("executionBackend") != "cvm-retry":
        raise ValueError("Cloud retry did not use the declared C/VM backend.")
    if result.get("status") not in {"labelled", "review"}:
        raise ValueError("Cloud retry did not produce a retained SVI status.")
    return result


def main() -> None:
    bucket = os.environ["FLUX_GCS_BUCKET"]
    manifest_object = os.environ["FLUX_MANIFEST_OBJECT"]
    task_index = int(os.environ["BATCH_TASK_INDEX"])
    task_attempt = int(os.environ.get("BATCH_TASK_RETRY_ATTEMPT", "0"))
    manifest_bytes = download_object(bucket, manifest_object)
    if manifest_bytes is None:
        raise FileNotFoundError(manifest_object)
    manifest = json.loads(manifest_bytes)
    entry = manifest_entry(manifest, task_index)
    fingerprint = entry["fingerprint"]

    existing = download_object(bucket, entry["outputObject"], missing_ok=True)
    if existing is not None:
        result = validate_result(existing, fingerprint)
        print(
            json.dumps(
                {
                    "event": "immutable-result-reused",
                    "taskIndex": task_index,
                    "fingerprint": fingerprint,
                    "status": result["status"],
                }
            )
        )
        return

    payload_bytes = download_object(bucket, entry["payloadObject"])
    if payload_bytes is None:
        raise FileNotFoundError(entry["payloadObject"])
    validate_payload(payload_bytes, entry)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"flux-svi-{task_index:03d}-") as directory:
        temporary = Path(directory)
        payload_path = temporary / f"{fingerprint}.payload.json.gz"
        result_path = temporary / f"{fingerprint}.result.json.gz"
        payload_path.write_bytes(payload_bytes)
        environment = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTENSOR_FLAGS": f"base_compiledir={temporary / 'pytensor'}",
        }
        subprocess.run(
            [
                sys.executable,
                "/app/scripts/svi_batch.py",
                "--payload",
                str(payload_path),
                "--output",
                str(result_path),
                "--backend",
                "c",
            ],
            check=True,
            env=environment,
        )
        result_bytes = result_path.read_bytes()
        result = validate_result(result_bytes, fingerprint)
        created = upload_object_immutably(
            bucket,
            entry["outputObject"],
            result_bytes,
            "application/gzip",
        )
        if not created:
            existing = download_object(bucket, entry["outputObject"])
            if existing is None:
                raise RuntimeError("Immutable result upload raced but no result exists.")
            result = validate_result(existing, fingerprint)

    print(
        json.dumps(
            {
                "event": "svi-retry-complete",
                "taskIndex": task_index,
                "taskAttempt": task_attempt,
                "fingerprint": fingerprint,
                "status": result["status"],
                "workerRuntimeSeconds": result.get("runtimeSeconds"),
                "taskRuntimeSeconds": time.perf_counter() - started,
                "created": created,
            }
        )
    )


if __name__ == "__main__":
    main()
