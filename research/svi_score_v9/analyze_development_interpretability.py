#!/usr/bin/env python3
"""Truth-blind interpretation receipts for the RegretSet-MMM development cohort.

Importance is estimated only on outer-fold advertisers.  For each practitioner
pillar, the entire pillar is permuted across the 48 candidates within an
advertiser, preserving its joint distribution while breaking its alignment
with the remaining evidence.  The reported estimand is the change in the
declared selected-model economic-risk objective, not a neural-weight magnitude.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

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
OUTPUT_PATH = ROOT / "research/svi_score_v9/artifacts/svi-score-v9-interpretability.json"
REPEATS = 20

PILLAR_ORDER = (
    "Predictive generalization",
    "Structural adequacy",
    "Causal and temporal robustness",
    "Decision and evidence coherence",
    "Posterior decision geometry",
    "Specification context",
)

PREDICTIVE_DIAGNOSTICS = {
    "rolling-oos",
    "predictive-coverage",
    "fold-stability",
    "spend-regimes",
}
STRUCTURAL_DIAGNOSTICS = {
    "residual-independence",
    "variance-structure",
    "likelihood-shape",
    "multicollinearity",
    "functional-form",
    "influence-stability",
    "non-negative-boundary",
}
CAUSAL_TEMPORAL_DIAGNOSTICS = {
    "anchor-recovery",
    "new-data-stability",
    "confounder-sensitivity",
    "future-media-placebo",
    "whole-flight-generalization",
    "carryover-support",
    "post-flight-residual-stability",
    "kernel-distinguishability",
}
DECISION_DIAGNOSTICS = {
    "roi-posterior-plausibility",
    "roi-decision-stability",
    "roi-resolution",
    "roi-identification",
    "evidence-source-quality",
    "evidence-compatibility",
    "evidence-decision-dependence",
    "roi-economic-consistency",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostic_name(feature: str) -> str | None:
    if feature.startswith("diagnostic:"):
        return feature.split(":", 1)[1]
    if feature.startswith("deficit:"):
        return feature.split(":", 2)[1]
    if feature.startswith("context:temporal:") and ":spec:" not in feature:
        return feature.split("context:temporal:", 1)[1]
    return None


def pillar_for(feature: str) -> str:
    # Declared model assumptions are kept separate from fit diagnostics.
    if feature.startswith("spec:") or feature.startswith("context:temporal:spec:"):
        return "Specification context"
    if feature.startswith("interaction:"):
        interaction = feature.split(":", 1)[1]
        if interaction.startswith(("student-t:", "log-normal:", "ridge:")):
            return "Structural adequacy"
        if interaction.startswith(("benchmark:", "saturation:")):
            return "Decision and evidence coherence"
        return "Causal and temporal robustness"

    name = diagnostic_name(feature)
    if name in PREDICTIVE_DIAGNOSTICS:
        return "Predictive generalization"
    if name in STRUCTURAL_DIAGNOSTICS:
        return "Structural adequacy"
    if name in CAUSAL_TEMPORAL_DIAGNOSTICS:
        return "Causal and temporal robustness"
    if name in DECISION_DIAGNOSTICS:
        return "Decision and evidence coherence"

    if feature.startswith("svi:predictive:") or feature.startswith("svi:margin:predictive-"):
        return "Predictive generalization"
    if feature.startswith("svi:margin:roi-"):
        return "Decision and evidence coherence"
    if feature.startswith("posterior:channel-set:evidence-"):
        return "Decision and evidence coherence"
    if feature.startswith("svi:") or feature.startswith("posterior:"):
        return "Posterior decision geometry"
    raise ValueError(f"Unassigned feature: {feature}")


def metric(receipts: list[dict]) -> float:
    return float(promoted_metrics(receipts)["selectionObjective"])


def permute_within_business(
    features: np.ndarray,
    indexes: np.ndarray,
    seed: int,
) -> np.ndarray:
    output = features.copy()
    random = np.random.default_rng(seed)
    for business in range(output.shape[0]):
        order = random.permutation(output.shape[1])
        output[business][:, indexes] = output[business][order][:, indexes]
    return output


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text())
    development = json.loads(DEVELOPMENT_PATH.read_text())
    arrays = prepare_arrays(dataset["rows"])
    feature_names = list(dataset["featureNames"])
    pillar_indexes = {
        pillar: np.asarray(
            [index for index, name in enumerate(feature_names) if pillar_for(name) == pillar],
            dtype=int,
        )
        for pillar in PILLAR_ORDER
    }
    assigned = sorted(index for values in pillar_indexes.values() for index in values.tolist())
    if assigned != list(range(len(feature_names))):
        raise RuntimeError("Interpretability registry is not exhaustive and exclusive")

    folds = arrays["folds"]
    fold_models: dict[int, list] = {}
    fold_policies = {}
    baseline_fold_receipts: dict[int, list[dict]] = {}
    outer_by_fold = {
        int(receipt["fold"]): receipt
        for receipt in development["nestedCrossValidation"]["outerReceipts"]
    }
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
        fold_models[fold] = models
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
        row
        for fold in sorted(baseline_fold_receipts)
        for row in baseline_fold_receipts[fold]
    ]
    baseline = metric(baseline_receipts)
    results = []
    for pillar_index, pillar in enumerate(PILLAR_ORDER):
        deltas = []
        objectives = []
        for repeat in range(REPEATS):
            repeated_receipts = []
            for fold in sorted(fold_models):
                assessment = np.flatnonzero(folds == fold)
                perturbed = permute_within_business(
                    arrays["features"][assessment],
                    pillar_indexes[pillar],
                    700_000 + pillar_index * 10_000 + repeat * 100 + fold,
                )
                predicted = ensemble_predict(fold_models[fold], perturbed)
                repeated_receipts.extend(selection_receipts(
                    arrays["businessIds"][assessment],
                    arrays["families"][assessment],
                    arrays["candidateIds"][assessment],
                    arrays["valid"][assessment],
                    arrays["excessRisk"][assessment],
                    arrays["normalizedGaps"][assessment],
                    arrays["dangerous"][assessment].astype(bool),
                    predicted,
                    fold_policies[fold],
                ))
            objective = metric(repeated_receipts)
            objectives.append(objective)
            deltas.append(objective - baseline)
        delta = np.asarray(deltas)
        # Drop-column refitting is the primary importance estimand.  Unlike
        # permutation, it lets the selector relearn from the remaining token
        # families and therefore avoids scoring deliberately mismatched token
        # combinations.
        refit_receipts = []
        refit_fold_deltas = []
        retained = np.asarray(
            [index for index in range(len(feature_names)) if index not in set(pillar_indexes[pillar])],
            dtype=int,
        )
        reduced_features = arrays["features"][:, :, retained]
        for fold in sorted(fold_models):
            assessment = np.flatnonzero(folds == fold)
            train = np.flatnonzero(folds != fold)
            receipt = outer_by_fold[fold]
            config = next(item for item in MODEL_GRID if item.id == receipt["selectedModel"])
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
            fold_receipts = selection_receipts(
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
            refit_receipts.extend(fold_receipts)
            refit_fold_deltas.append(
                metric(fold_receipts) - metric(baseline_fold_receipts[fold])
            )
        refit_objective = metric(refit_receipts)
        results.append({
            "pillar": pillar,
            "tokenCount": int(pillar_indexes[pillar].size),
            "dropColumnRefitObjective": refit_objective,
            "dropColumnRefitIncrease": refit_objective - baseline,
            "dropColumnRelativeIncrease": (refit_objective - baseline) / max(baseline, 1e-12),
            "dropColumnFoldIncreases": refit_fold_deltas,
            "dropColumnPositiveFoldShare": float(np.mean(np.asarray(refit_fold_deltas) > 0)),
            "meanObjectiveIncrease": float(np.mean(delta)),
            "medianObjectiveIncrease": float(np.median(delta)),
            "p10ObjectiveIncrease": float(np.quantile(delta, 0.10)),
            "p90ObjectiveIncrease": float(np.quantile(delta, 0.90)),
            "positiveRepeatShare": float(np.mean(delta > 0)),
            "meanRelativeIncrease": float(np.mean(delta) / max(baseline, 1e-12)),
            "repeatObjectives": objectives,
        })
    positive_total = sum(max(item["dropColumnRefitIncrease"], 0) for item in results)
    for item in results:
        item["positiveImportanceShare"] = (
            max(item["dropColumnRefitIncrease"], 0) / max(positive_total, 1e-12)
        )

    output = {
        "artifactId": "flux-svi-score-v9-development-interpretability-v1",
        "activation": "development-only",
        "estimand": "outer-fold change in 65%-mean/35%-P90 selected-model excess-loss objective",
        "primaryMethod": "drop one pillar, refit on outer-training advertisers, and assess outer-fold economic loss",
        "sensitivityMethod": "20 candidate-alignment permutations per pillar within each held-out advertiser",
        "baselineObjective": baseline,
        "businesses": int(arrays["features"].shape[0]),
        "candidatesPerBusiness": int(arrays["features"].shape[1]),
        "featureCount": len(feature_names),
        "repeats": REPEATS,
        "pillars": results,
        "provenance": {
            "datasetSha256": sha256(DATASET_PATH),
            "developmentSha256": sha256(DEVELOPMENT_PATH),
            "sealedAuditDereferenced": False,
        },
        "interpretation": (
            "A positive drop-column value means that removing the pillar and relearning from the "
            "remaining tokens increased held-out economic loss. Permutation is a secondary sensitivity "
            "check. Both are conditional predictive importance, not causal effects or hand-assigned weights."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
