#!/usr/bin/env python3
"""Grouped-CV drop-column importance for the final RegretSet-MMM selector.

The estimand is conditional information value: remove a complete practitioner
pillar (raw values plus its percentile-rank and robust-z counterparts), refit
the selector on outer-training advertisers, and assess untouched advertisers.
No posterior is refitted and no audit cohort is accessed.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from run_development import BASELINES, DATASET_PATH, REPAIR, V11, arrays_with_features, candidate_relative_features


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "research/regretset_relative_features/artifacts/final-selector-importance.json"
ENSEMBLE_MEMBERS = 3

PILLAR_ORDER = (
    "Predictive generalization",
    "Structural adequacy",
    "Causal and temporal robustness",
    "Decision and evidence coherence",
    "Posterior decision geometry",
    "Specification context",
)

PREDICTIVE_DIAGNOSTICS = {
    "rolling-oos", "predictive-coverage", "fold-stability", "spend-regimes",
}
STRUCTURAL_DIAGNOSTICS = {
    "residual-independence", "variance-structure", "likelihood-shape",
    "multicollinearity", "functional-form", "influence-stability",
    "non-negative-boundary",
}
CAUSAL_TEMPORAL_DIAGNOSTICS = {
    "anchor-recovery", "new-data-stability", "confounder-sensitivity",
    "future-media-placebo", "whole-flight-generalization", "carryover-support",
    "post-flight-residual-stability", "kernel-distinguishability",
}
DECISION_DIAGNOSTICS = {
    "roi-posterior-plausibility", "roi-decision-stability", "roi-resolution",
    "roi-identification", "evidence-source-quality", "evidence-compatibility",
    "evidence-decision-dependence", "roi-economic-consistency",
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
    if feature.startswith("svi:margin:roi-") or feature.startswith("posterior:channel-set:evidence-"):
        return "Decision and evidence coherence"
    if feature.startswith("svi:") or feature.startswith("posterior:"):
        return "Posterior decision geometry"
    raise ValueError(f"Unassigned feature: {feature}")


def objective(receipts: list[dict]) -> float:
    return float(REPAIR.evaluation_metrics(receipts)["log1p"]["objective65Mean35P90"])


def run() -> dict:
    started = time.time()
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    original = V11.prepare_arrays(dataset)
    relative = candidate_relative_features(original["features"], original["valid"], original["present"])
    combined = np.concatenate((original["features"], relative), axis=2)
    arrays = arrays_with_features(original, combined)
    raw_names = list(dataset["featureNames"])
    raw_dim = len(raw_names)
    if raw_dim != 232 or combined.shape[2] != 3 * raw_dim:
        raise RuntimeError("Unexpected final token contract")

    raw_pillars = {
        pillar: np.asarray([index for index, name in enumerate(raw_names) if pillar_for(name) == pillar], dtype=int)
        for pillar in PILLAR_ORDER
    }
    assigned = sorted(index for indexes in raw_pillars.values() for index in indexes.tolist())
    if assigned != list(range(raw_dim)):
        raise RuntimeError("Pillar registry is not exhaustive and exclusive")
    combined_pillars = {
        pillar: np.concatenate((indexes, raw_dim + indexes, 2 * raw_dim + indexes))
        for pillar, indexes in raw_pillars.items()
    }

    config = next(item for item in V11.MODEL_GRID if item.id == "evidence-adaptive-compact")
    policy = V11.V9.PolicyConfig(0.0, 0.0)
    folds = sorted(np.unique(original["folds"][original["fullIndexes"]]).tolist())
    variants: dict[str, np.ndarray] = {"all": np.arange(combined.shape[2], dtype=int)}
    for pillar, indexes in combined_pillars.items():
        removed = set(indexes.tolist())
        variants[f"without:{pillar}"] = np.asarray(
            [index for index in range(combined.shape[2]) if index not in removed], dtype=int
        )

    selections = {name: [] for name in variants}
    fold_objectives = {name: [] for name in variants}
    timings = {name: 0.0 for name in variants}
    for outer_fold in folds:
        training = np.flatnonzero(original["folds"] != outer_fold)
        assessment = original["fullIndexes"][original["folds"][original["fullIndexes"]] == outer_fold]
        # Reuse the exact final-selector ensemble seeds so the all-token
        # reference reproduces the published cross-fitted benchmark.
        seeds = tuple(730_000 + int(outer_fold) * 100 + member for member in range(ENSEMBLE_MEMBERS))
        for name, indexes in variants.items():
            began = time.time()
            fitted = BASELINES.fit_context_ensemble(config, arrays, indexes, True, training, seeds)
            prediction = BASELINES.context_ensemble_predict(fitted, arrays, assessment)
            receipts = BASELINES.receipts(original, assessment, prediction, policy)
            selections[name].extend(receipts)
            fold_objectives[name].append(objective(receipts))
            timings[name] += time.time() - began
            print(json.dumps({
                "fold": int(outer_fold), "variant": name,
                "businesses": len(assessment), "seconds": time.time() - began,
            }), flush=True)

    baseline = objective(selections["all"])
    pillars = []
    for pillar in PILLAR_ORDER:
        name = f"without:{pillar}"
        removed_objective = objective(selections[name])
        comparison = REPAIR.paired_bootstrap_comparison(
            selections[name], selections["all"], seed_offset=PILLAR_ORDER.index(pillar) + 80,
            repeats=REPAIR.BOOTSTRAP_REPEATS,
        )
        pillars.append({
            "pillar": pillar,
            "rawTokenCount": int(raw_pillars[pillar].size),
            "finalTokenCount": int(combined_pillars[pillar].size),
            "allTokenObjective": baseline,
            "dropColumnObjective": removed_objective,
            "absoluteIncrease": removed_objective - baseline,
            "relativeIncrease": (removed_objective - baseline) / baseline,
            "positiveFoldShare": float(np.mean(
                np.asarray(fold_objectives[name]) > np.asarray(fold_objectives["all"])
            )),
            "pairedComparison": comparison,
        })

    output = {
        "artifactId": "regretset-final-selector-importance-v1",
        "status": "development-only-grouped-cross-validation-complete",
        "estimand": "change in cross-fitted 65%-mean/35%-P90 log1p excess-loss objective after removing a pillar and refitting",
        "method": "drop raw tokens and their within-advertiser percentile-rank and robust-z counterparts; refit on outer-training advertisers",
        "data": {"businesses": 420, "candidateFitsReused": 20_160, "groupedFolds": 5},
        "finalSelector": {
            "rawTokens": raw_dim, "relativeTokens": 2 * raw_dim,
            "candidateTokens": 3 * raw_dim, "contextTokens": 63,
            "trainingRegretTransform": "log1p(excess economic loss)",
            "postPredictionPenalty": "none",
        },
        "baselineObjective": baseline,
        "pillars": pillars,
        "timingsSeconds": timings,
        "runtimeSeconds": time.time() - started,
        "checks": {
            "allBusinessesCrossFitted": all(len(rows) == 420 for rows in selections.values()),
            "pillarRegistryExhaustive": assigned == list(range(raw_dim)),
            "allDerivedCounterpartsRemoved": all(
                combined_pillars[pillar].size == 3 * raw_pillars[pillar].size for pillar in PILLAR_ORDER
            ),
            "noPosteriorRefits": True,
            "auditCohortsAccessed": False,
        },
        "provenance": {"developmentDatasetSha256": sha256(DATASET_PATH)},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    output = run()
    print(json.dumps({
        "status": output["status"], "baselineObjective": output["baselineObjective"],
        "pillars": [{"pillar": row["pillar"], "relativeIncrease": row["relativeIncrease"]} for row in output["pillars"]],
        "checks": output["checks"], "runtimeSeconds": output["runtimeSeconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
