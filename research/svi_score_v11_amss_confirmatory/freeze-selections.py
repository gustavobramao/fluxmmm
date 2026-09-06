#!/usr/bin/env python3
"""Freeze V11, V9, and prediction-only choices before AMSS truth is opened."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / ".flux-artifacts/svi-score-v11-amss-confirmatory/observable-freeze/v11-observable-input.json"
OBSERVABLE_MANIFEST_PATH = ROOT / ".flux-artifacts/svi-score-v11-amss-confirmatory/observable-freeze/manifest.json"
V11_SELECTOR_PATH = ROOT / "research/svi_score_v11_amss_confirmatory/frozen-selectors/v11-selector.json"
V9_SELECTOR_PATH = ROOT / "research/svi_score_v11_amss_confirmatory/frozen-selectors/v9-selector.json"
V11_SOURCE = ROOT / "research/svi_score_v11_evidence_adaptive/train_selector.py"
V9_SOURCE = ROOT / "research/svi_score_v10_amss/frozen-v9/source/research/svi_score_v9/train_selector.py"
FROZEN_V11_SOURCE = ROOT / "research/svi_score_v11_amss_confirmatory/frozen-selectors/v11-train-selector.py"
FROZEN_V9_SOURCE = ROOT / "research/svi_score_v11_amss_confirmatory/frozen-selectors/v9-train-selector.py"
OUTPUT_PATH = ROOT / ".flux-artifacts/svi-score-v11-amss-confirmatory/observable-freeze/frozen-selections.json"


def module_from(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load frozen selector source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if V11_SOURCE.read_bytes() != FROZEN_V11_SOURCE.read_bytes():
    raise RuntimeError("V11 selector runtime differs from its pre-cohort source freeze")
if V9_SOURCE.read_bytes() != FROZEN_V9_SOURCE.read_bytes():
    raise RuntimeError("V9 comparator runtime differs from its pre-cohort source freeze")
V11 = module_from(V11_SOURCE, "flux_v11_confirmatory_selector")
V9 = module_from(V9_SOURCE, "flux_v9_confirmatory_comparator")


def sha256(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def restore_v11(serialized: dict[str, Any]) -> Any:
    config = V11.V9.ModelConfig(**serialized["config"])
    expert_indices = {
        name: np.asarray(values, dtype=int)
        for name, values in serialized["expertIndices"].items()
    }
    model = V11.EvidenceAdaptiveSelector(config, int(serialized["seed"]), expert_indices)
    model.means = np.asarray(serialized["means"], dtype=float)
    model.scales = np.asarray(serialized["scales"], dtype=float)
    model.context_means = np.asarray(serialized["contextMeans"], dtype=float)
    model.context_scales = np.asarray(serialized["contextScales"], dtype=float)
    model.parameters = {
        name: np.asarray(value, dtype=float)
        for name, value in serialized["parameters"].items()
    }
    model.training_receipt = dict(serialized.get("receipt", {}))
    return model


def restore_v9(serialized: dict[str, Any]) -> Any:
    model = V9.DeepSetSelector(V9.ModelConfig(**serialized["config"]), int(serialized["seed"]))
    model.means = np.asarray(serialized["means"], dtype=float)
    model.scales = np.asarray(serialized["scales"], dtype=float)
    model.parameters = {
        name: np.asarray(value, dtype=float)
        for name, value in serialized["parameters"].items()
    }
    return model


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError("Confirmatory selections already exist; reselection is forbidden")
    input_source = INPUT_PATH.read_bytes()
    manifest_source = OBSERVABLE_MANIFEST_PATH.read_bytes()
    v11_source = V11_SELECTOR_PATH.read_bytes()
    v9_source = V9_SELECTOR_PATH.read_bytes()
    dataset = json.loads(input_source)
    manifest = json.loads(manifest_source)
    v11_selector = json.loads(v11_source)
    v9_selector = json.loads(v9_source)
    if (
        dataset.get("status") != "truth-blind-input-prepared-for-confirmatory-selection"
        or dataset.get("provenance", {}).get("hiddenTruthReadByPreparation")
        or dataset.get("provenance", {}).get("posteriorRefits") != 0
        or not dataset.get("provenance", {}).get("cloudComputeUsed")
        or manifest.get("truthOpened")
        or len(dataset.get("rows", [])) != 4_800
        or len(dataset.get("sets", [])) != 100
        or dataset.get("featureNames") != v11_selector.get("featureNames")
        or dataset.get("featureNames") != v9_selector.get("featureNames")
        or dataset.get("contextNames") != v11_selector.get("contextNames")
    ):
        raise RuntimeError("Truth-blind confirmatory selection inputs violate the freeze contract")

    rows_by_key = {
        (str(row["businessId"]), str(row["candidateId"])): row
        for row in dataset["rows"]
    }
    sets = sorted(dataset["sets"], key=lambda row: str(row["businessId"]))
    feature_count = len(dataset["featureNames"])
    context_count = len(dataset["contextNames"])
    features = np.zeros((100, 48, feature_count), dtype=float)
    contexts = np.zeros((100, context_count), dtype=float)
    valid = np.zeros((100, 48), dtype=bool)
    candidate_ids = np.full((100, 48), "", dtype=object)
    business_ids = np.empty(100, dtype=object)
    evidence_groups = np.empty(100, dtype=object)
    for business_index, item in enumerate(sets):
        business_id = str(item["businessId"])
        contexts[business_index] = np.asarray(item["context"], dtype=float)
        business_ids[business_index] = business_id
        evidence_groups[business_index] = str(item["evidenceGroup"])
        for candidate_index, candidate_id in enumerate(item["candidateIds"]):
            row = rows_by_key[(business_id, str(candidate_id))]
            features[business_index, candidate_index] = np.asarray(row["features"], dtype=float)
            valid[business_index, candidate_index] = bool(row["valid"])
            candidate_ids[business_index, candidate_index] = str(candidate_id)
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(contexts)):
        raise RuntimeError("Confirmatory observable arrays contain non-finite values")
    if np.any(np.sum(valid, axis=1) < 2):
        raise RuntimeError("A confirmatory business has fewer than two valid candidates")

    v11_models = [restore_v11(item) for item in v11_selector["models"]]
    arrays = {"features": features, "contexts": contexts, "valid": valid}
    v11_predicted = V11.ensemble_predict(v11_models, arrays, np.arange(100))
    v11_danger = v11_predicted["heads"][..., V11.DANGER_HEAD]
    v11_adjusted = (
        v11_predicted["risk"]
        + float(v11_selector["policy"]["danger_penalty"]) * v11_danger
        + float(v11_selector["policy"]["uncertainty_penalty"]) * v11_predicted["uncertainty"]
    )

    v9_members = []
    for item in v9_selector["models"]:
        model = restore_v9(item)
        v9_members.append(model.predict(features))
    v9_heads = np.stack(v9_members, axis=0)
    v9_member_risk = 0.65 * v9_heads[..., V9.MEAN_HEAD] + 0.35 * v9_heads[..., V9.P90_HEAD]
    v9_risk = np.mean(v9_member_risk, axis=0)
    v9_uncertainty = np.std(v9_member_risk, axis=0, ddof=1)
    v9_adjusted = v9_risk + float(v9_selector["policy"]["uncertainty_penalty"]) * v9_uncertainty
    rolling_index = dataset["featureNames"].index("diagnostic:rolling-oos")

    selections = []
    for business_index in range(100):
        active = np.flatnonzero(valid[business_index])
        v11_choice = int(active[np.argmin(v11_adjusted[business_index, active])])
        v9_choice = int(active[np.argmin(v9_adjusted[business_index, active])])
        prediction_choice = int(active[np.argmax(features[business_index, active, rolling_index])])
        selections.append({
            "businessId": str(business_ids[business_index]),
            "evidenceGroup": str(evidence_groups[business_index]),
            "v11CandidateId": str(candidate_ids[business_index, v11_choice]),
            "v11PredictedRisk": float(v11_predicted["risk"][business_index, v11_choice]),
            "v11PredictedDanger": float(v11_danger[business_index, v11_choice]),
            "v11EnsembleUncertainty": float(v11_predicted["uncertainty"][business_index, v11_choice]),
            "v11SelectionObjective": float(v11_adjusted[business_index, v11_choice]),
            "v9CandidateId": str(candidate_ids[business_index, v9_choice]),
            "v9PredictedRisk": float(v9_risk[business_index, v9_choice]),
            "v9EnsembleUncertainty": float(v9_uncertainty[business_index, v9_choice]),
            "v9SelectionObjective": float(v9_adjusted[business_index, v9_choice]),
            "predictionOnlyCandidateId": str(candidate_ids[business_index, prediction_choice]),
            "evidenceGate": {
                expert: float(v11_predicted["gates"][business_index, position])
                for position, expert in enumerate(V11.EXPERTS)
            },
            "evidenceGateUncertainty": {
                expert: float(v11_predicted["gateUncertainty"][business_index, position])
                for position, expert in enumerate(V11.EXPERTS)
            },
            "validCandidates": int(active.size),
        })
    if len(selections) != 100 or any(abs(sum(row["evidenceGate"].values()) - 1) > 1e-8 for row in selections):
        raise RuntimeError("Confirmatory selection completeness checks failed")

    artifact = {
        "artifactId": "flux-v11-amss-confirmatory-frozen-selections-v1",
        "version": dataset["version"],
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "all-comparator-selections-frozen-before-amss-truth-open",
        "provenance": {
            "observableInputSha256": sha256(input_source),
            "observableManifestSha256": sha256(manifest_source),
            "frozenV11SelectorSha256": sha256(v11_source),
            "frozenV9SelectorSha256": sha256(v9_source),
            "hiddenEconomicOutcomesReadBySelection": False,
            "selectorRetrained": False,
            "posteriorRefitsAfterFreeze": 0,
        },
        "policies": {"v11": v11_selector["policy"], "v9": v9_selector["policy"]},
        "businesses": 100,
        "candidates": 4_800,
        "selections": selections,
    }
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frozen": True, "businesses": 100, "truthOpened": False,
        "selectionSha256": sha256(OUTPUT_PATH.read_bytes())}, indent=2))


if __name__ == "__main__":
    main()
