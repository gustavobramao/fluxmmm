#!/usr/bin/env python3
"""Apply the frozen V11 selector to cached truth-blind AMSS observables."""

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
V11_SPEC = importlib.util.spec_from_file_location(
    "flux_v11_selector_for_amss_retrospective",
    ROOT / "research/svi_score_v11_evidence_adaptive/train_selector.py",
)
assert V11_SPEC and V11_SPEC.loader
V11 = importlib.util.module_from_spec(V11_SPEC)
sys.modules[V11_SPEC.name] = V11
V11_SPEC.loader.exec_module(V11)

INPUT_PATH = (
    ROOT
    / ".flux-artifacts/svi-score-v11-evidence-adaptive/amss-retrospective/observable-input.json"
)
SELECTOR_PATH = (
    ROOT
    / ".flux-artifacts/svi-score-v11-evidence-adaptive/development-selector.json"
)
OUTPUT_PATH = (
    ROOT
    / ".flux-artifacts/svi-score-v11-evidence-adaptive/amss-retrospective/frozen-selections.json"
)


def sha256(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def restore_model(serialized: dict[str, Any]) -> Any:
    config = V11.V9.ModelConfig(**serialized["config"])
    expert_indices = {
        name: np.asarray(values, dtype=int)
        for name, values in serialized["expertIndices"].items()
    }
    model = V11.EvidenceAdaptiveSelector(
        config,
        int(serialized["seed"]),
        expert_indices,
    )
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


def main() -> None:
    input_source = INPUT_PATH.read_bytes()
    selector_source = SELECTOR_PATH.read_bytes()
    dataset = json.loads(input_source)
    selector = json.loads(selector_source)
    if (
        dataset.get("status")
        != "truth-blind-input-prepared-for-retrospective-selection"
        or dataset.get("provenance", {}).get("hiddenTruthReadByPreparation")
        or dataset.get("provenance", {}).get("posteriorRefits") != 0
        or dataset.get("provenance", {}).get("cloudComputeUsed")
        or selector.get("activation") != "development-only"
        or selector.get("provenance", {}).get("posteriorRefits") != 0
        or len(dataset.get("rows", [])) != 4_800
        or len(dataset.get("sets", [])) != 100
        or len(selector.get("models", [])) != 5
        or dataset.get("featureNames") != selector.get("featureNames")
        or dataset.get("contextNames") != selector.get("contextNames")
    ):
        raise RuntimeError("V11 retrospective selection inputs violate the freeze contract")

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
            features[business_index, candidate_index] = np.asarray(
                row["features"], dtype=float
            )
            valid[business_index, candidate_index] = bool(row["valid"])
            candidate_ids[business_index, candidate_index] = str(candidate_id)
    if (
        not np.all(np.isfinite(features))
        or not np.all(np.isfinite(contexts))
        or np.any(np.sum(valid, axis=1) < 2)
    ):
        raise RuntimeError("AMSS observable arrays are incomplete or non-finite")

    models = [restore_model(item) for item in selector["models"]]
    arrays = {"features": features, "contexts": contexts, "valid": valid}
    predicted = V11.ensemble_predict(models, arrays, np.arange(100))
    policy = selector["policy"]
    danger = predicted["heads"][..., V11.DANGER_HEAD]
    adjusted = (
        predicted["risk"]
        + float(policy["danger_penalty"]) * danger
        + float(policy["uncertainty_penalty"]) * predicted["uncertainty"]
    )
    rolling_index = dataset["featureNames"].index("diagnostic:rolling-oos")
    selections: list[dict[str, Any]] = []
    for business_index in range(100):
        active = np.flatnonzero(valid[business_index])
        order = active[
            np.argsort(adjusted[business_index, active], kind="stable")
        ]
        selected = int(order[0])
        prediction_only = int(
            active[
                np.argmax(features[business_index, active, rolling_index])
            ]
        )
        selections.append(
            {
                "businessId": str(business_ids[business_index]),
                "evidenceGroup": str(evidence_groups[business_index]),
                "v11CandidateId": str(candidate_ids[business_index, selected]),
                "v11PredictedRisk": float(
                    predicted["risk"][business_index, selected]
                ),
                "v11PredictedDanger": float(danger[business_index, selected]),
                "v11EnsembleUncertainty": float(
                    predicted["uncertainty"][business_index, selected]
                ),
                "v11SelectionObjective": float(
                    adjusted[business_index, selected]
                ),
                "predictionOnlyCandidateId": str(
                    candidate_ids[business_index, prediction_only]
                ),
                "evidenceGate": {
                    expert: float(predicted["gates"][business_index, position])
                    for position, expert in enumerate(V11.EXPERTS)
                },
                "evidenceGateUncertainty": {
                    expert: float(
                        predicted["gateUncertainty"][business_index, position]
                    )
                    for position, expert in enumerate(V11.EXPERTS)
                },
                "validCandidates": int(active.size),
                "candidatePredictions": [
                    {
                        "candidateId": str(candidate_ids[business_index, candidate]),
                        "predictedRisk": float(
                            predicted["risk"][business_index, candidate]
                        ),
                        "predictedDanger": float(danger[business_index, candidate]),
                        "ensembleUncertainty": float(
                            predicted["uncertainty"][business_index, candidate]
                        ),
                        "selectionObjective": float(
                            adjusted[business_index, candidate]
                        ),
                    }
                    for candidate in active
                ],
            }
        )
    if (
        len(selections) != 100
        or len({row["businessId"] for row in selections}) != 100
        or any(abs(sum(row["evidenceGate"].values()) - 1) > 1e-8 for row in selections)
    ):
        raise RuntimeError("V11 retrospective selection failed completeness checks")

    artifact = {
        "artifactId": "flux-v11-amss-retrospective-frozen-selections-v1",
        "version": dataset["version"],
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "v11-selections-frozen-before-retrospective-scoring",
        "scientificLabel": "retrospective external benchmark; not confirmatory",
        "provenance": {
            "observableInputSha256": sha256(input_source),
            "frozenV11SelectorSha256": sha256(selector_source),
            "hiddenEconomicOutcomesReadBySelection": False,
            "selectorRetrained": False,
            "posteriorRefits": 0,
            "cloudComputeUsed": False,
        },
        "policy": policy,
        "businesses": 100,
        "candidates": 4_800,
        "selections": selections,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(artifact, separators=(",", ":")))
    temporary.replace(OUTPUT_PATH)
    print(
        json.dumps(
            {
                "path": str(OUTPUT_PATH),
                "businesses": len(selections),
                "posteriorRefits": 0,
                "cloudComputeUsed": False,
                "hiddenEconomicOutcomesReadBySelection": False,
                "selectionSha256": sha256(OUTPUT_PATH.read_bytes()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
