#!/usr/bin/env python3
"""Apply the frozen V9 selector to truth-blind AMSS candidates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
V10_ROOT = ROOT / ".flux-artifacts/svi-score-v10-amss"
OBSERVABLE_PATH = V10_ROOT / "observable-freeze/observable-candidates.json"
OBSERVABLE_MANIFEST_PATH = V10_ROOT / "observable-freeze/manifest.json"
SELECTOR_PATH = ROOT / "research/svi_score_v10_amss/frozen-v9/selector.json"
SELECTOR_SOURCE = (
    ROOT
    / "research/svi_score_v10_amss/frozen-v9/source/research/svi_score_v9/train_selector.py"
)
OUTPUT_PATH = V10_ROOT / "observable-freeze/frozen-selections.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selector_module() -> Any:
    spec = importlib.util.spec_from_file_location("flux_v10_frozen_v9", SELECTOR_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load the frozen V9 selector implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def member_prediction(module: Any, member: dict[str, Any], features: np.ndarray) -> np.ndarray:
    selector = module.DeepSetSelector(
        module.ModelConfig(**member["config"]), int(member["seed"])
    )
    selector.means = np.asarray(member["means"], dtype=float)
    selector.scales = np.asarray(member["scales"], dtype=float)
    selector.parameters = {
        key: np.asarray(value, dtype=float)
        for key, value in member["parameters"].items()
    }
    return selector.predict(features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    if OUTPUT_PATH.exists():
        raise RuntimeError("V10 truth-blind selections are already frozen")
    observable = json.loads(OBSERVABLE_PATH.read_text())
    manifest = json.loads(OBSERVABLE_MANIFEST_PATH.read_text())
    model = json.loads(SELECTOR_PATH.read_text())
    if (
        observable.get("activation") != "external-audit-truth-sealed"
        or observable.get("provenance", {}).get("hiddenTruthDereferenced")
        or manifest.get("truthOpened")
        or len(observable.get("rows", [])) != 4_800
        or observable.get("featureNames") != model.get("featureNames")
    ):
        raise RuntimeError("V10 observable set or selector freeze is invalid")

    module = load_selector_module()
    by_business: dict[str, list[dict[str, Any]]] = {}
    for row in observable["rows"]:
        by_business.setdefault(str(row["businessId"]), []).append(row)
    selections: list[dict[str, Any]] = []
    for business_id in sorted(by_business):
        rows = sorted(by_business[business_id], key=lambda row: str(row["candidateId"]))
        if len(rows) != 48:
            raise RuntimeError(f"{business_id} has {len(rows)}/48 candidates")
        features = np.asarray([[row["features"] for row in rows]], dtype=float)
        predictions = np.stack(
            [member_prediction(module, member, features) for member in model["models"]],
            axis=0,
        )
        member_risk = (
            0.65 * predictions[..., module.MEAN_HEAD]
            + 0.35 * predictions[..., module.P90_HEAD]
        )
        risk = np.mean(member_risk, axis=0)[0]
        uncertainty = np.std(member_risk, axis=0, ddof=1)[0]
        objective = risk + float(model["policy"]["uncertainty_penalty"]) * uncertainty
        valid = np.asarray([bool(row["valid"]) for row in rows])
        if not np.any(valid):
            raise RuntimeError(f"{business_id} has no finite SVI candidate")
        masked = np.where(valid, objective, np.inf)
        v9_index = int(np.argmin(masked))
        heuristic = np.asarray(
            [float(row["heuristicScore"]) if row["valid"] else -np.inf for row in rows]
        )
        heuristic_index = int(np.argmax(heuristic))
        selections.append(
            {
                "businessId": business_id,
                "evidenceGroup": rows[0]["evidenceGroup"],
                "v9CandidateId": rows[v9_index]["candidateId"],
                "v9PredictedRisk": float(risk[v9_index]),
                "v9EnsembleUncertainty": float(uncertainty[v9_index]),
                "v9SelectionObjective": float(objective[v9_index]),
                "heuristicCandidateId": rows[heuristic_index]["candidateId"],
                "heuristicScore": float(rows[heuristic_index]["heuristicScore"]),
                "validCandidates": int(np.sum(valid)),
                "candidatePredictions": [
                    {
                        "candidateId": row["candidateId"],
                        "valid": bool(row["valid"]),
                        "predictedRisk": float(risk[index]),
                        "ensembleUncertainty": float(uncertainty[index]),
                        "selectionObjective": float(objective[index]),
                    }
                    for index, row in enumerate(rows)
                ],
            }
        )
    artifact = {
        "artifactId": "flux-v10-amss-frozen-truth-blind-selections-v1",
        "version": observable["version"],
        "status": "frozen-before-amss-truth-open",
        "provenance": {
            "observableDatasetSha256": digest(OBSERVABLE_PATH),
            "observableManifestSha256": digest(OBSERVABLE_MANIFEST_PATH),
            "frozenV9SelectorSha256": digest(SELECTOR_PATH),
            "frozenV9Retrained": False,
            "hiddenTruthDereferenced": False,
        },
        "policy": model["policy"],
        "businesses": len(selections),
        "selections": selections,
    }
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "frozen": True,
                "businesses": len(selections),
                "selector": "frozen V9 DeepSets ensemble",
                "hiddenTruthDereferenced": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
