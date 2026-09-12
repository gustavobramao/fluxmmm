#!/usr/bin/env python3
"""Grouped-CV ablation for deployment-safe candidate-relative features."""

from __future__ import annotations

import argparse
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
OUTPUT_PATH = ROOT / "research/regretset_relative_features/artifacts/development-comparison.json"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REPAIR = load_module(
    "relative_feature_rank_repair",
    ROOT / "research/regretset_rank_repair/run_development.py",
)
BASELINES = REPAIR.BASELINES
V11 = REPAIR.V11

VARIANTS = (
    "raw-log1p",
    "raw-plus-relative-log1p",
    "relative-only-log1p",
)
ENSEMBLE_MEMBERS = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_relative_features(
    features: np.ndarray,
    valid: np.ndarray,
    present: np.ndarray,
) -> np.ndarray:
    """Return within-set percentile rank and robust z-score for every token.

    Only candidates from the same advertiser/regime participate. Invalid or
    absent candidates are never used to calculate another candidate's feature.
    No outcome, economic-loss, or oracle field is accepted by this function.
    """
    if features.ndim != 3 or valid.shape != features.shape[:2] or present.shape != valid.shape:
        raise ValueError("Candidate feature and mask shapes disagree")
    if np.any(~np.isfinite(features)):
        raise ValueError("Candidate features must be finite")
    ranks = np.zeros_like(features, dtype=float)
    robust_z = np.zeros_like(features, dtype=float)
    for set_index in range(features.shape[0]):
        active = np.flatnonzero(valid[set_index] & present[set_index])
        if active.size == 0:
            raise ValueError(f"Candidate set {set_index} has no active candidates")
        values = features[set_index, active]
        for feature_index in range(features.shape[2]):
            column = values[:, feature_index]
            if active.size == 1:
                ranks[set_index, active, feature_index] = 0.5
            else:
                ranks[set_index, active, feature_index] = (
                    REPAIR.rankdata(column) - 1.0
                ) / (active.size - 1.0)
        median = np.median(values, axis=0)
        lower = np.quantile(values, 0.25, axis=0, method="linear")
        upper = np.quantile(values, 0.75, axis=0, method="linear")
        scale = upper - lower
        stable = scale > 1e-8
        z = np.zeros_like(values)
        z[:, stable] = (values[:, stable] - median[stable]) / scale[stable]
        robust_z[set_index, active] = np.clip(z, -8.0, 8.0)
    return np.concatenate((ranks, robust_z), axis=2)


def arrays_with_features(
    arrays: dict[str, np.ndarray],
    features: np.ndarray,
) -> dict[str, np.ndarray]:
    result = dict(arrays)
    result["features"] = features
    result["normalizedGaps"] = REPAIR.log_regret(arrays["excessRisk"])
    return result


def run(smoke: bool) -> dict[str, Any]:
    started = time.time()
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if (
        dataset.get("activation") != "development-only"
        or dataset.get("provenance", {}).get("posteriorRefits") != 0
        or dataset.get("provenance", {}).get("amssConfirmatoryCohortAccessed")
        or dataset.get("provenance", {}).get("previousAuditAccessed")
        or len(dataset.get("rows", [])) != 20_160
    ):
        raise RuntimeError("Relative-feature experiment requires the isolated development data")

    original = V11.prepare_arrays(dataset)
    relative = candidate_relative_features(
        original["features"], original["valid"], original["present"]
    )
    feature_sets = {
        "raw-log1p": original["features"],
        "raw-plus-relative-log1p": np.concatenate((original["features"], relative), axis=2),
        "relative-only-log1p": relative,
    }
    training_arrays = {
        name: arrays_with_features(original, values)
        for name, values in feature_sets.items()
    }
    config = next(item for item in V11.MODEL_GRID if item.id == "evidence-adaptive-compact")
    if smoke:
        config = V11.V9.ModelConfig(
            "relative-feature-smoke",
            config.candidate_width,
            config.decoder_width,
            1,
            config.learning_rate,
            config.weight_decay,
            config.temperature,
        )
    folds = sorted(np.unique(original["folds"][original["fullIndexes"]]).tolist())
    if smoke:
        folds = folds[:1]
    members = 1 if smoke else ENSEMBLE_MEMBERS
    policy = V11.V9.PolicyConfig(0.0, 0.0)
    selections = {name: [] for name in VARIANTS}
    predictions = {name: [] for name in VARIANTS}
    assessment_indexes: list[int] = []
    timings = {name: 0.0 for name in VARIANTS}

    for outer_fold in folds:
        training = np.flatnonzero(original["folds"] != outer_fold)
        assessment = original["fullIndexes"][
            original["folds"][original["fullIndexes"]] == outer_fold
        ]
        assessment_indexes.extend(int(value) for value in assessment)
        seeds = tuple(730_000 + int(outer_fold) * 100 + member for member in range(members))
        for name in VARIANTS:
            began = time.time()
            arrays = training_arrays[name]
            indexes = np.arange(arrays["features"].shape[2], dtype=int)
            fitted = BASELINES.fit_context_ensemble(
                config, arrays, indexes, True, training, seeds,
            )
            prediction = BASELINES.context_ensemble_predict(fitted, arrays, assessment)
            selections[name].extend(
                BASELINES.receipts(original, assessment, prediction, policy)
            )
            predictions[name].extend(
                prediction["risk"][local].copy() for local in range(len(assessment))
            )
            timings[name] += time.time() - began
            print(json.dumps({
                "fold": int(outer_fold),
                "variant": name,
                "businesses": len(assessment),
                "seconds": time.time() - began,
            }), flush=True)

    assessment_array = np.asarray(assessment_indexes, dtype=int)
    results = {
        name: {
            "metrics": REPAIR.evaluation_metrics(rows),
            "ranking": REPAIR.ranking_diagnostics(
                original, assessment_array, predictions[name]
            ),
        }
        for name, rows in selections.items()
    }
    comparisons = {
        name: REPAIR.paired_bootstrap_comparison(
            selections[name],
            selections["raw-log1p"],
            seed_offset=index,
            repeats=250 if smoke else REPAIR.BOOTSTRAP_REPEATS,
        )
        for index, name in enumerate(VARIANTS)
        if name != "raw-log1p"
    }
    ranking = sorted(
        ({
            "variant": name,
            "log1pObjective": float(value["metrics"]["log1p"]["objective65Mean35P90"]),
            "uncappedObjective": float(value["metrics"]["uncapped"]["objective65Mean35P90"]),
            "uncappedP95": float(value["metrics"]["uncapped"]["p95"]),
        } for name, value in results.items()),
        key=lambda row: (row["log1pObjective"], row["variant"]),
    )
    checks = {
        "allRequestedFoldsRun": len(folds) == (1 if smoke else 5),
        "allAssessmentBusinessesUnique": len(set(assessment_indexes)) == len(assessment_indexes),
        "allDevelopmentBusinessesCrossFitted": smoke or len(assessment_indexes) == 420,
        "relativeFeatureShapeCorrect": relative.shape[2] == 2 * original["features"].shape[2],
        "relativeFeaturesFinite": bool(np.all(np.isfinite(relative))),
        "noPosteriorRefits": True,
        "sealedAuditAccessed": False,
        "amssAccessed": False,
    }
    return {
        "artifactId": "regretset-candidate-relative-development-v1",
        "status": "smoke" if smoke else "development-only-grouped-cross-validation-complete",
        "method": {
            "modelName": "RegretSet-MMM — relative log-regret",
            "baseCandidateTokens": int(original["features"].shape[2]),
            "derivedTokenBlocks": [
                "within-advertiser candidate percentile rank with average ties",
                "within-advertiser candidate median/IQR robust z-score",
            ],
            "derivedTokens": int(relative.shape[2]),
            "rawPlusRelativeTokens": int(feature_sets["raw-plus-relative-log1p"].shape[2]),
            "trainingRegretTransform": "log1p(excess economic loss)",
            "outcomeOrTruthUsedInRelativeFeatures": False,
        },
        "data": {
            "businesses": len(assessment_indexes),
            "candidateFitsReused": 20_160,
            "outerGroupedFolds": len(folds),
        },
        "results": results,
        "ranking": ranking,
        "pairedComparisonsVersusRawLogRegret": comparisons,
        "checks": checks,
        "governance": {
            "developmentOnly": True,
            "posteriorRefits": 0,
            "productionChanged": False,
            "sealedAuditAccessed": False,
            "amssAccessed": False,
        },
        "timingsSeconds": timings,
        "runtimeSeconds": time.time() - started,
        "provenance": {
            "developmentDatasetSha256": sha256(DATASET_PATH),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.smoke)
    if not arguments.smoke:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "ranking": result["ranking"],
        "checks": result["checks"],
        "runtimeSeconds": result["runtimeSeconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
