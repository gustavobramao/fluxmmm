#!/usr/bin/env python3
"""Fit and freeze the deployable relative-log-regret selector.

This consumes only the existing development posterior-token table.  It does not
fit an MMM, read an audit cohort, or inspect hidden information at deployment.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / ".flux-artifacts/svi-score-v11-evidence-adaptive/development-dataset.json"
OUTPUT_PATH = ROOT / "research/regretset_relative_features/artifacts/regretset-relative-log-regret-selector.json"
PUBLIC_PATH = ROOT / "lib/mmm/artifacts/regretset-v11-selector.json"
ENSEMBLE_MEMBERS = 3
SEEDS = tuple(730_500 + member for member in range(ENSEMBLE_MEMBERS))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RELATIVE = load_module(
    "regretset_relative_development",
    ROOT / "research/regretset_relative_features/run_development.py",
)
BASELINES = RELATIVE.BASELINES
V11 = RELATIVE.V11


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialize(model: Any) -> dict[str, Any]:
    return {
        "kind": "context-no-gate-deepsets",
        "config": model.config.__dict__,
        "seed": model.seed,
        "pooling": True,
        "featureIndexes": model.feature_indexes.tolist(),
        "means": model.means.tolist(),
        "scales": model.scales.tolist(),
        "contextMeans": model.context_means.tolist(),
        "contextScales": model.context_scales.tolist(),
        "parameters": {
            name: value.tolist() for name, value in model.parameters.items()
        },
        "receipt": model.training_receipt,
    }


def main() -> None:
    source = DATASET_PATH.read_bytes()
    dataset = json.loads(source)
    if (
        dataset.get("activation") != "development-only"
        or dataset.get("provenance", {}).get("posteriorRefits") != 0
        or dataset.get("provenance", {}).get("amssConfirmatoryCohortAccessed")
        or dataset.get("provenance", {}).get("previousAuditAccessed")
        or len(dataset.get("rows", [])) != 20_160
        or len(dataset.get("featureNames", [])) != 232
        or len(dataset.get("contextNames", [])) != 63
    ):
        raise RuntimeError("Unexpected development artifact; selector freeze aborted")

    original = V11.prepare_arrays(dataset)
    relative = RELATIVE.candidate_relative_features(
        original["features"], original["valid"], original["present"]
    )
    combined = np.concatenate((original["features"], relative), axis=2)
    arrays = RELATIVE.arrays_with_features(original, combined)
    config = next(
        item for item in V11.MODEL_GRID if item.id == "evidence-adaptive-compact"
    )
    indexes = np.arange(combined.shape[2], dtype=int)
    all_sets = np.arange(combined.shape[0], dtype=int)
    started = time.time()
    models = BASELINES.fit_context_ensemble(
        config, arrays, indexes, True, all_sets, SEEDS
    )

    raw_names = [str(value) for value in dataset["featureNames"]]
    relative_names = [f"within-set-rank:{name}" for name in raw_names] + [
        f"within-set-robust-z:{name}" for name in raw_names
    ]
    artifact = {
        "artifactId": "regretset-relative-log-regret-production-v1",
        "version": "flux-regretset-relative-log-regret-1.0.0",
        "activation": "production-frozen-after-grouped-development",
        "architecture": "context-no-gate-deepsets",
        "rawFeatureNames": raw_names,
        "featureNames": raw_names + relative_names,
        "contextNames": [str(value) for value in dataset["contextNames"]],
        "relativeFeatureContract": {
            "rank": "within-advertiser percentile rank with average ties",
            "robustZ": "within-advertiser median/IQR z-score clipped to [-8, 8]",
            "usesOutcomeOrTruth": False,
        },
        "trainingTarget": "log1p uncapped excess economic loss",
        "policy": {
            "mean_weight": 0.65,
            "p90_weight": 0.35,
            "danger_penalty": 0.0,
            "uncertainty_penalty": 0.0,
        },
        "models": [serialize(model) for model in models],
        "development": {
            "businesses": 420,
            "regimeSets": int(combined.shape[0]),
            "candidateRows": 20_160,
            "rawCandidateTokens": 232,
            "relativeCandidateTokens": 464,
            "totalCandidateTokens": 696,
            "businessContextTokens": 63,
            "groupedCrossValidationObjective": 0.804591076778812,
        },
        "provenance": {
            "posteriorRefits": 0,
            "auditDataAccessed": False,
            "developmentDatasetSha256": hashlib.sha256(source).hexdigest(),
            "developmentComparisonSha256": sha256(
                ROOT / "research/regretset_relative_features/artifacts/development-comparison.json"
            ),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "trainingSeconds": time.time() - started,
        },
    }
    serialized = json.dumps(artifact, separators=(",", ":")) + "\n"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(serialized, encoding="utf-8")
    PUBLIC_PATH.write_text(serialized, encoding="utf-8")
    print(json.dumps({
        "artifact": str(OUTPUT_PATH),
        "publicArtifact": str(PUBLIC_PATH),
        "sha256": sha256(PUBLIC_PATH),
        "models": len(models),
        "candidateTokens": len(artifact["featureNames"]),
        "contextTokens": len(artifact["contextNames"]),
        "seconds": artifact["provenance"]["trainingSeconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
