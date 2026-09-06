#!/usr/bin/env python3
"""Evaluate one frozen AMSS business and seal its truth rows in GCS."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage


TASKS = 100
ROWS_PER_BUSINESS = 192
ROOT = Path(".flux-artifacts/svi-score-v11-amss-confirmatory")
INPUT_FILES = {
    "actions": ROOT / "observable-freeze/candidate-actions.csv",
    "metadata": ROOT / "sealed-cohort/business-metadata.csv",
    "receipt": ROOT / "opened-truth/truth-open-receipt.json",
    "cohort_manifest": ROOT / "sealed-cohort/manifest.json",
    "selections": ROOT / "observable-freeze/frozen-selections.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def download(bucket: storage.Bucket, object_name: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    bucket.blob(object_name).download_to_filename(target)


def verify_inputs() -> tuple[str, str]:
    receipt = json.loads(INPUT_FILES["receipt"].read_text(encoding="utf-8"))
    if receipt["status"] != "truth-opened-for-frozen-action-evaluation":
        raise RuntimeError("Truth-open receipt is not valid.")
    if receipt["workers"] != TASKS:
        raise RuntimeError("Truth-open receipt does not declare 100 cloud tasks.")
    checks = {
        "candidateActionsSha256": INPUT_FILES["actions"],
        "cohortManifestSha256": INPUT_FILES["cohort_manifest"],
        "frozenSelectionsSha256": INPUT_FILES["selections"],
    }
    for field, path in checks.items():
        actual = sha256(path)
        if actual != receipt[field]:
            raise RuntimeError(f"Immutable input hash mismatch for {field}: {actual}")
    return receipt["version"], "cbf5e7f6"


def verify_hidden_state(business_id: str, hidden_path: Path) -> None:
    manifest = json.loads(INPUT_FILES["cohort_manifest"].read_text(encoding="utf-8"))
    expected_name = f"{business_id}.rds"
    matches = [row for row in manifest["hiddenFiles"] if row["path"] == expected_name]
    if len(matches) != 1:
        raise RuntimeError(f"No unique sealed hidden-state record for {business_id}.")
    actual = sha256(hidden_path)
    if actual != matches[0]["sha256"]:
        raise RuntimeError(f"Hidden-state hash mismatch for {business_id}: {actual}")


def verify_output(path: Path, business_id: str) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != ROWS_PER_BUSINESS:
        raise RuntimeError(f"{business_id} produced {len(rows)}/{ROWS_PER_BUSINESS} rows.")
    if {row["business_id"] for row in rows} != {business_id}:
        raise RuntimeError(f"Truth output contains the wrong business for {business_id}.")
    expected_scenarios = {
        "budget-reduction", "fixed-budget-mix", "budget-growth", "economic-ceiling"
    }
    if {row["scenario"] for row in rows} != expected_scenarios:
        raise RuntimeError(f"Truth output has an invalid scenario set for {business_id}.")


def upload_immutable(bucket: storage.Bucket, object_name: str, source: Path) -> None:
    blob = bucket.blob(object_name)
    try:
        blob.upload_from_filename(source, if_generation_match=0, content_type="text/csv")
    except PreconditionFailed:
        with tempfile.NamedTemporaryFile(suffix=".csv") as handle:
            blob.download_to_filename(handle.name)
            if sha256(Path(handle.name)) != sha256(source):
                raise RuntimeError(f"Existing immutable result differs: gs://{bucket.name}/{object_name}")


def main() -> int:
    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "-1"))
    task_count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "-1"))
    if task_count != TASKS or task_index < 0 or task_index >= TASKS:
        raise RuntimeError(f"Expected a 100-task Cloud Run job; received {task_index}/{task_count}.")

    bucket_name = required_env("FLUX_GCS_BUCKET")
    input_prefix = required_env("FLUX_TRUTH_INPUT_PREFIX").rstrip("/")
    output_prefix = required_env("FLUX_TRUTH_OUTPUT_PREFIX").rstrip("/")
    business_id = f"amss-v11c-{task_index + 1:03d}"
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for name, target in INPUT_FILES.items():
        download(bucket, f"{input_prefix}/{target.name}", target)
    hidden = ROOT / f"sealed-cohort/hidden/{business_id}.rds"
    download(bucket, f"{input_prefix}/hidden/{business_id}.rds", hidden)

    version, amss_commit = verify_inputs()
    verify_hidden_state(business_id, hidden)
    subprocess.run(
        [
            "Rscript",
            "/app/open-truth.R",
            f"--worker={task_index}",
            f"--workers={TASKS}",
        ],
        check=True,
        env=os.environ,
    )

    part = ROOT / f"opened-truth/parts/part-{task_index:03d}-of-{TASKS:03d}.csv"
    verify_output(part, business_id)
    object_name = f"{output_prefix}/{part.name}"
    upload_immutable(bucket, object_name, part)
    print(json.dumps({
        "status": "sealed",
        "version": version,
        "amssCommit": amss_commit,
        "taskIndex": task_index,
        "businessId": business_id,
        "rows": ROWS_PER_BUSINESS,
        "sha256": sha256(part),
        "output": f"gs://{bucket_name}/{object_name}",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        raise
