#!/usr/bin/env python3
"""Outer-fold ablation of the posterior-decision token family.

This analysis decomposes the only broad pillar with robust incremental value in
the development cohort.  Every lens is removed, the selector is relearned on
outer-training advertisers, and economic loss is assessed on the untouched
outer fold.  The sealed audit is never read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_development_interpretability import pillar_for
from train_selector import (
    MODEL_GRID,
    POLICY_GRID,
    ensemble_predict,
    fit_ensemble,
    prepare_arrays,
    promoted_metrics,
    selection_receipts,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / ".flux-artifacts/svi-score-v9/development-dataset.json"
DEVELOPMENT_PATH = ROOT / "research/svi_score_v9/artifacts/svi-score-v9-development.json"
OUTPUT_PATH = ROOT / "research/svi_score_v9/artifacts/svi-score-v9-posterior-geometry.json"

LENS_ORDER = (
    "Inference reliability",
    "Predictive and plausibility safety",
    "ROI location and uncertainty",
    "Tail shape and approximation shift",
    "External-evidence presence",
    "Response-mechanics uncertainty",
    "Contribution and channel dependence",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lens_for(feature: str) -> str:
    if feature.startswith("svi:status:") or feature.startswith("svi:convergence:"):
        return "Inference reliability"
    if feature.startswith(("svi:predictive:", "svi:decision:", "svi:margin:")):
        return "Predictive and plausibility safety"
    if feature.startswith("svi:roi:"):
        return "ROI location and uncertainty"
    if feature.startswith("posterior:channel-set:"):
        field = feature.split("posterior:channel-set:", 1)[1]
        if field.startswith((
            "log-roi-median:", "log-roi-mean:", "log-roi-low:",
            "log-roi-high:", "log-relative-roi-width:", "log-roi-draw-sd:",
        )):
            return "ROI location and uncertainty"
        if field.startswith((
            "implausible-probability:", "near-zero-probability:",
            "posterior-skewness:", "location-shift-sd:", "interval-overlap:",
        )):
            return "Tail shape and approximation shift"
        if field.startswith("evidence-"):
            return "External-evidence presence"
        if field.startswith("response-"):
            return "Response-mechanics uncertainty"
        if field.startswith("log-contribution-cv:"):
            return "Contribution and channel dependence"
    if feature.startswith("posterior:channel-pair:"):
        return "Contribution and channel dependence"
    raise ValueError(f"Unassigned posterior token: {feature}")


def metric(receipts: list[dict]) -> float:
    return float(promoted_metrics(receipts)["selectionObjective"])


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text())
    development = json.loads(DEVELOPMENT_PATH.read_text())
    arrays = prepare_arrays(dataset["rows"])
    feature_names = list(dataset["featureNames"])
    posterior_indexes = np.asarray([
        index for index, name in enumerate(feature_names)
        if pillar_for(name) == "Posterior decision geometry"
    ], dtype=int)
    lens_indexes = {
        lens: np.asarray([
            index for index in posterior_indexes.tolist()
            if lens_for(feature_names[index]) == lens
        ], dtype=int)
        for lens in LENS_ORDER
    }
    assigned = sorted(index for values in lens_indexes.values() for index in values.tolist())
    if assigned != sorted(posterior_indexes.tolist()):
        raise RuntimeError("Posterior lens registry is not exhaustive and exclusive")

    folds = arrays["folds"]
    outer_by_fold = {
        int(receipt["fold"]): receipt
        for receipt in development["nestedCrossValidation"]["outerReceipts"]
    }
    baseline_fold_receipts: dict[int, list[dict]] = {}
    fold_policies = {}
    for outer_fold in sorted(np.unique(folds)):
        fold = int(outer_fold)
        receipt = outer_by_fold[fold]
        config = next(item for item in MODEL_GRID if item.id == receipt["selectedModel"])
        policy = next(item for item in POLICY_GRID if item.id == receipt["selectedPolicy"])
        train = np.flatnonzero(folds != fold)
        assessment = np.flatnonzero(folds == fold)
        models = fit_ensemble(
            config,
            arrays["features"],
            arrays["targets"],
            arrays["dangerous"],
            arrays["normalizedGaps"],
            arrays["valid"],
            train,
            tuple(100_000 + fold * 100 + seed for seed in range(3)),
            bootstrap=True,
        )
        fold_policies[fold] = policy
        predicted = ensemble_predict(models, arrays["features"][assessment])
        baseline_fold_receipts[fold] = selection_receipts(
            arrays["businessIds"][assessment],
            arrays["families"][assessment],
            arrays["candidateIds"][assessment],
            arrays["valid"][assessment],
            arrays["excessRisk"][assessment],
            arrays["normalizedGaps"][assessment],
            arrays["dangerous"][assessment].astype(bool),
            predicted,
            policy,
        )
    baseline_receipts = [
        row for fold in sorted(baseline_fold_receipts)
        for row in baseline_fold_receipts[fold]
    ]
    baseline = metric(baseline_receipts)

    results = []
    for lens in LENS_ORDER:
        removed = set(lens_indexes[lens].tolist())
        retained = np.asarray([
            index for index in range(len(feature_names)) if index not in removed
        ], dtype=int)
        reduced_features = arrays["features"][:, :, retained]
        refit_receipts = []
        fold_deltas = []
        for outer_fold in sorted(np.unique(folds)):
            fold = int(outer_fold)
            receipt = outer_by_fold[fold]
            config = next(item for item in MODEL_GRID if item.id == receipt["selectedModel"])
            train = np.flatnonzero(folds != fold)
            assessment = np.flatnonzero(folds == fold)
            models = fit_ensemble(
                config,
                reduced_features,
                arrays["targets"],
                arrays["dangerous"],
                arrays["normalizedGaps"],
                arrays["valid"],
                train,
                tuple(100_000 + fold * 100 + seed for seed in range(3)),
                bootstrap=True,
            )
            predicted = ensemble_predict(models, reduced_features[assessment])
            receipts = selection_receipts(
                arrays["businessIds"][assessment],
                arrays["families"][assessment],
                arrays["candidateIds"][assessment],
                arrays["valid"][assessment],
                arrays["excessRisk"][assessment],
                arrays["normalizedGaps"][assessment],
                arrays["dangerous"][assessment].astype(bool),
                predicted,
                fold_policies[fold],
            )
            refit_receipts.extend(receipts)
            fold_deltas.append(metric(receipts) - metric(baseline_fold_receipts[fold]))
        objective = metric(refit_receipts)
        results.append({
            "lens": lens,
            "tokenCount": int(lens_indexes[lens].size),
            "dropColumnRefitObjective": objective,
            "dropColumnRefitIncrease": objective - baseline,
            "dropColumnRelativeIncrease": (objective - baseline) / max(baseline, 1e-12),
            "dropColumnFoldIncreases": fold_deltas,
            "dropColumnPositiveFoldShare": float(np.mean(np.asarray(fold_deltas) > 0)),
        })

    output = {
        "artifactId": "flux-svi-score-v9-development-posterior-geometry-v1",
        "activation": "development-only",
        "estimand": "outer-fold economic-loss increase after removing one posterior lens and relearning the selector",
        "baselineObjective": baseline,
        "businesses": int(arrays["features"].shape[0]),
        "posteriorTokenCount": int(posterior_indexes.size),
        "lenses": results,
        "provenance": {
            "datasetSha256": sha256(DATASET_PATH),
            "developmentSha256": sha256(DEVELOPMENT_PATH),
            "sealedAuditDereferenced": False,
        },
        "interpretation": (
            "Positive values indicate incremental predictive value after the selector is relearned; "
            "negative values indicate conditional redundancy or finite-sample overfit, not scientific uselessness."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
