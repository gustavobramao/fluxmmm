#!/usr/bin/env python3
"""Run one immutable Flux SVI structure shard with persistent worker processes.

This module changes orchestration only. Every candidate is evaluated by the
frozen ``svi_batch.py`` runtime with backend ``c`` and its original payload,
fingerprint, inference contract, and random seeds.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import multiprocessing
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
MANIFEST_VERSION = "flux-svi-cvm-structure-shards-v2"
_TOKEN_LOCK = threading.Lock()
_TOKEN: tuple[str, float] | None = None
_SVI: Any = None


def access_token() -> str:
    """Return a cached VM service-account token, refreshing before expiry."""
    global _TOKEN
    with _TOKEN_LOCK:
        now = time.monotonic()
        if _TOKEN is not None and _TOKEN[1] > now + 60:
            return _TOKEN[0]
        request = Request(METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"})
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read())
        token = payload.get("access_token")
        expires_in = payload.get("expires_in", 3_000)
        if not isinstance(token, str) or not token:
            raise RuntimeError("The Batch VM did not return a usable access token.")
        _TOKEN = (token, now + max(float(expires_in), 120))
        return token


def object_url(bucket: str, object_name: str) -> str:
    return (
        "https://storage.googleapis.com/download/storage/v1/b/"
        f"{quote(bucket, safe='')}/o/{quote(object_name, safe='')}?alt=media"
    )


def download_object(
    bucket: str,
    object_name: str,
    *,
    missing_ok: bool = False,
) -> bytes | None:
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
    parameters = urlencode({
        "uploadType": "media",
        "name": object_name,
        "ifGenerationMatch": "0",
    })
    request = Request(
        "https://storage.googleapis.com/upload/storage/v1/b/"
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


def validate_entry(item: Any) -> dict[str, str]:
    if not isinstance(item, dict):
        raise TypeError("Shard candidate entries must be objects.")
    required = (
        "fingerprint",
        "businessId",
        "candidateId",
        "payloadObject",
        "payloadSha256",
        "outputObject",
    )
    if any(not isinstance(item.get(key), str) or not item[key] for key in required):
        raise ValueError("Shard candidate entry is incomplete.")
    if not FINGERPRINT_PATTERN.fullmatch(item["fingerprint"]):
        raise ValueError("Shard manifest contains an invalid inference fingerprint.")
    return {key: item[key] for key in required}


def manifest_shard(manifest: dict[str, Any], task_index: int) -> list[dict[str, Any]]:
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("Unsupported structure-shard manifest version.")
    group_by = manifest.get("groupBy")
    if group_by not in {"businessId", "candidateId", "fingerprint"}:
        raise ValueError("Shard manifest has an invalid grouping contract.")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not 0 <= task_index < len(shards):
        raise IndexError(f"Batch task index {task_index} is outside the shard manifest.")
    shard = shards[task_index]
    if not isinstance(shard, dict) or shard.get("shardIndex") != task_index:
        raise ValueError("Shard identity does not match the Batch task index.")
    groups = shard.get("workGroups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Shard contains no work groups.")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("groupId"), str):
            raise ValueError("Shard work-group entry is invalid.")
        items = [validate_entry(item) for item in group.get("items", [])]
        if not items or any(item[group_by] != group["groupId"] for item in items):
            raise ValueError("Shard candidates are missing or mis-grouped.")
        for item in items:
            if item["fingerprint"] in seen:
                raise ValueError("Shard contains a duplicate inference fingerprint.")
            seen.add(item["fingerprint"])
        normalized.append({"groupId": group["groupId"], "items": items})
    return normalized


def validate_payload(payload_bytes: bytes, entry: dict[str, str]) -> None:
    digest = hashlib.sha256(payload_bytes).hexdigest()
    if digest != entry["payloadSha256"]:
        raise ValueError(f"Payload digest mismatch for {entry['fingerprint']}: {digest}.")
    payload = json.loads(gzip.decompress(payload_bytes))
    if payload.get("fingerprint") != entry["fingerprint"]:
        raise ValueError("Payload fingerprint does not match the shard manifest.")


def validate_result(result_bytes: bytes, fingerprint: str) -> dict[str, Any]:
    result = json.loads(gzip.decompress(result_bytes))
    if result.get("fingerprint") != fingerprint:
        raise ValueError("SVI result fingerprint does not match its candidate.")
    if result.get("executionBackend") != "cvm-retry":
        raise ValueError("Cloud shard did not use the declared C/VM backend.")
    if result.get("status") not in {"labelled", "review"}:
        raise ValueError("Cloud shard did not produce a retained SVI status.")
    return result


def prepare_candidate(
    bucket: str,
    entry: dict[str, str],
) -> tuple[bytes | None, dict[str, Any] | None]:
    existing = download_object(bucket, entry["outputObject"], missing_ok=True)
    if existing is not None:
        result = validate_result(existing, entry["fingerprint"])
        return None, {
            "fingerprint": entry["fingerprint"],
            "status": result["status"],
            "created": False,
            "runtimeSeconds": 0.0,
        }
    payload_bytes = download_object(bucket, entry["payloadObject"])
    if payload_bytes is None:
        raise FileNotFoundError(entry["payloadObject"])
    validate_payload(payload_bytes, entry)
    return payload_bytes, None


def pytensor_flags(compiledir: str, inherited: str = "") -> str:
    """Preserve numerical-runtime flags while assigning the shared cache."""
    inherited_flags = inherited.strip(",")
    compiledir_flag = f"base_compiledir={compiledir}"
    return ",".join(
        flag for flag in (inherited_flags, compiledir_flag) if flag
    )


def initialize_worker(compiledir: str) -> None:
    """Load the frozen scientific runtime once in each persistent process."""
    global _SVI
    inherited_flags = os.environ.get("PYTENSOR_FLAGS", "").strip(",")
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTENSOR_FLAGS": pytensor_flags(compiledir, inherited_flags),
    })
    spec = importlib.util.spec_from_file_location(
        "flux_svi_shard_scientific_runtime",
        "/app/scripts/svi_batch.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the frozen SVI scientific runtime.")
    _SVI = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = _SVI
    spec.loader.exec_module(_SVI)


def fit_payload(payload_bytes: bytes) -> bytes:
    if _SVI is None:
        raise RuntimeError("Persistent SVI worker was not initialized.")
    payload = json.loads(gzip.decompress(payload_bytes))
    started = time.perf_counter()
    result = _SVI.run(payload, "c")
    result["orchestrationRuntimeSeconds"] = time.perf_counter() - started
    return gzip.compress(
        json.dumps(result, separators=(",", ":")).encode(),
        mtime=0,
    )


def retain_result(
    bucket: str,
    entry: dict[str, str],
    result_bytes: bytes,
) -> dict[str, Any]:
    result = validate_result(result_bytes, entry["fingerprint"])
    created = upload_object_immutably(
        bucket,
        entry["outputObject"],
        result_bytes,
        "application/gzip",
    )
    if not created:
        raced = download_object(bucket, entry["outputObject"])
        if raced is None:
            raise RuntimeError("Immutable upload raced but no result exists.")
        result = validate_result(raced, entry["fingerprint"])
    return {
        "fingerprint": entry["fingerprint"],
        "status": result["status"],
        "created": created,
        "runtimeSeconds": result.get("orchestrationRuntimeSeconds", 0.0),
    }


def main() -> None:
    bucket = os.environ["FLUX_GCS_BUCKET"]
    manifest_object = os.environ["FLUX_MANIFEST_OBJECT"]
    task_index = int(os.environ.get("BATCH_TASK_INDEX", os.environ.get("CLOUD_RUN_TASK_INDEX", "0")))
    concurrency = max(1, int(os.environ.get("FLUX_CANDIDATE_CONCURRENCY", "3")))
    manifest_bytes = download_object(bucket, manifest_object)
    if manifest_bytes is None:
        raise FileNotFoundError(manifest_object)
    groups = manifest_shard(json.loads(manifest_bytes), task_index)
    started = time.perf_counter()
    totals = {"created": 0, "reused": 0, "labelled": 0, "review": 0}

    with tempfile.TemporaryDirectory(prefix=f"flux-svi-shard-{task_index:02d}-") as directory:
        compiledir = Path(directory) / "pytensor-shared"
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=concurrency,
            mp_context=context,
            initializer=initialize_worker,
            initargs=(str(compiledir),),
            max_tasks_per_child=50,
        ) as executor:
            for group_index, group in enumerate(groups):
                group_started = time.perf_counter()
                prepared = [
                    (item, *prepare_candidate(bucket, item))
                    for item in group["items"]
                ]
                results = [
                    cached
                    for _, _, cached in prepared
                    if cached is not None
                ]
                pending = [
                    (item, payload)
                    for item, payload, cached in prepared
                    if cached is None and payload is not None
                ]
                if pending:
                    warm_item, warm_payload = pending.pop(0)
                    results.append(retain_result(
                        bucket,
                        warm_item,
                        executor.submit(fit_payload, warm_payload).result(),
                    ))
                    futures = {
                        executor.submit(fit_payload, payload): item
                        for item, payload in pending
                    }
                    for future in as_completed(futures):
                        results.append(retain_result(
                            bucket,
                            futures[future],
                            future.result(),
                        ))
                for result in results:
                    totals["created" if result["created"] else "reused"] += 1
                    totals[result["status"]] += 1
                print(json.dumps({
                    "event": "svi-structure-group-complete",
                    "taskIndex": task_index,
                    "groupIndex": group_index,
                    "groupsInShard": len(groups),
                    "groupId": group["groupId"],
                    "candidates": len(results),
                    "runtimeSeconds": time.perf_counter() - group_started,
                    "totals": totals,
                }), flush=True)

    print(json.dumps({
        "event": "svi-shard-complete",
        "taskIndex": task_index,
        "groups": len(groups),
        "candidates": sum(len(group["items"]) for group in groups),
        "runtimeSeconds": time.perf_counter() - started,
        "totals": totals,
    }), flush=True)


if __name__ == "__main__":
    main()
