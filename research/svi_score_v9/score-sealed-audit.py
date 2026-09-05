#!/usr/bin/env python3
"""Score the V9 sealed audit once with frozen V9 and V8 selectors."""

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
V9_DIR = Path(__file__).resolve().parent
V8_DIR = ROOT / "research/svi_score_v8"


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Unable to load frozen selector module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


v9 = load_module("flux_v9_audit_selector", V9_DIR / "train_selector.py")
v8 = load_module("flux_v8_audit_selector", V8_DIR / "train_selector.py")

AUDIT_ROOT = ROOT / ".flux-artifacts/svi-score-v9/audit"
AUDIT_PATH = AUDIT_ROOT / "sealed-audit-dataset.json"
V9_MODEL_PATH = AUDIT_ROOT / "frozen-v9-selector.json"
V8_MODEL_PATH = AUDIT_ROOT / "frozen-v8-selector.json"
FREEZE_PATH = AUDIT_ROOT / "sealed-audit-protocol-freeze.json"
OPEN_PATH = AUDIT_ROOT / "sealed-audit-open-receipt.json"
OUTPUT_PATH = ROOT / "research/svi_score_v9/artifacts/svi-score-v9-sealed-audit.json"
DEVELOPMENT_DATASET_PATH = ROOT / ".flux-artifacts/svi-score-v9/development-dataset.json"
DEVELOPMENT_V9_MODEL_PATH = ROOT / ".flux-artifacts/svi-score-v9/development-selector.json"
DEVELOPMENT_V8_MODEL_PATH = (
    ROOT / ".flux-artifacts/svi-score-v8/frozen-development-selector.json"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialized_v8_predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    means = np.asarray(model["means"], dtype=float)
    scales = np.asarray(model["scales"], dtype=float)
    family = str(model["kind"])
    if family == "boosted":
        transformed = np.clip((features - means) / scales, -8, 8)
        raw = np.tile(np.asarray(model["initial"], dtype=float), (features.shape[0], 1))
        rate = float(model["config"]["learning_rate"])
        for head, trees in enumerate(model["trees"]):
            for stump in trees:
                raw[:, head] += rate * v8.stump_predict(transformed, stump)
    else:
        basis = v8.transform_basis(features, means, scales, family)
        raw = basis @ np.asarray(model["weights"], dtype=float)
    output = raw.copy()
    output[:, :6] = np.maximum(output[:, :6], 0)
    output[:, v8.DANGER_HEAD] = v8.sigmoid(raw[:, v8.DANGER_HEAD])
    return output


def serialized_v9_predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    selector = v9.DeepSetSelector(
        v9.ModelConfig(**model["config"]), int(model["seed"])
    )
    selector.means = np.asarray(model["means"], dtype=float)
    selector.scales = np.asarray(model["scales"], dtype=float)
    selector.parameters = {
        key: np.asarray(value, dtype=float) for key, value in model["parameters"].items()
    }
    return selector.predict(features)


def v9_predictions(model: dict[str, Any], features: np.ndarray) -> dict[str, np.ndarray]:
    members = np.stack(
        [serialized_v9_predict(member, features) for member in model["models"]], axis=0
    )
    heads = np.mean(members, axis=0)
    member_risk = (
        v9.MEAN_RISK_WEIGHT * members[..., v9.MEAN_HEAD]
        + v9.P90_RISK_WEIGHT * members[..., v9.P90_HEAD]
    )
    return {
        "heads": heads,
        "risk": np.mean(member_risk, axis=0),
        "uncertainty": np.std(member_risk, axis=0, ddof=1),
    }


def v8_predictions(model: dict[str, Any], features: np.ndarray) -> dict[str, np.ndarray]:
    businesses, candidates, _ = features.shape
    flattened = features[:, :, :136].reshape(businesses * candidates, 136)
    members = np.stack(
        [serialized_v8_predict(member, flattened) for member in model["models"]], axis=0
    )
    member_risk = (
        v8.MEAN_RISK_WEIGHT * members[..., v8.MEAN_HEAD]
        + v8.P90_RISK_WEIGHT * members[..., v8.P90_HEAD]
    )
    heads = np.zeros((businesses, candidates, v9.HEAD_COUNT), dtype=float)
    heads[..., v9.DANGER_HEAD] = np.mean(members[..., v8.DANGER_HEAD], axis=0).reshape(
        businesses, candidates
    )
    return {
        "heads": heads,
        "risk": np.mean(member_risk, axis=0).reshape(businesses, candidates),
        "uncertainty": np.std(member_risk, axis=0, ddof=1).reshape(
            businesses, candidates
        ),
    }


def select(
    arrays: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray],
    policy: v9.PolicyConfig,
) -> list[dict[str, Any]]:
    return v9.selection_receipts(
        arrays["businessIds"],
        arrays["families"],
        arrays["candidateIds"],
        arrays["valid"],
        arrays["excessRisk"],
        arrays["normalizedGaps"],
        arrays["dangerous"].astype(bool),
        predicted,
        policy,
    )


def primary_objective(receipts: list[dict[str, Any]]) -> float:
    return float(v9.promoted_metrics(receipts)["selectionObjective"])


def stratified_paired_bootstrap(
    v9_receipts: list[dict[str, Any]],
    v8_receipts: list[dict[str, Any]],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    v8_by_business = {row["businessId"]: row for row in v8_receipts}
    pairs = [(row, v8_by_business[row["businessId"]]) for row in v9_receipts]
    family_indexes: dict[str, list[int]] = {}
    for index, (row, _) in enumerate(pairs):
        family_indexes.setdefault(str(row["family"]), []).append(index)
    random = np.random.default_rng(seed)
    differences = np.zeros(resamples, dtype=float)
    for draw in range(resamples):
        selected: list[int] = []
        for indexes in family_indexes.values():
            selected.extend(random.choice(indexes, size=len(indexes), replace=True).tolist())
        left = [pairs[index][0] for index in selected]
        right = [pairs[index][1] for index in selected]
        differences[draw] = primary_objective(left) - primary_objective(right)
    point = primary_objective(v9_receipts) - primary_objective(v8_receipts)
    return {
        "estimand": "V9 minus V8 full-coverage selection objective",
        "pointDifference": point,
        "twoSided95Interval": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "oneSided95UpperBound": float(np.quantile(differences, 0.95)),
        "resamples": resamples,
        "seed": seed,
        "familyStratified": True,
    }


def family_metrics(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({str(row["family"]) for row in receipts})
    return {
        family: v9.promoted_metrics(
            [row for row in receipts if str(row["family"]) == family]
        )
        for family in families
    }


def paired_outcomes(
    v9_receipts: list[dict[str, Any]], v8_receipts: list[dict[str, Any]]
) -> dict[str, float]:
    v8_by_business = {row["businessId"]: row for row in v8_receipts}
    difference = np.asarray(
        [
            float(row["actualExcessLoss"])
            - float(v8_by_business[row["businessId"]]["actualExcessLoss"])
            for row in v9_receipts
        ]
    )
    return {
        "v9LowerLossShare": float(np.mean(difference < -1e-12)),
        "equalLossShare": float(np.mean(np.abs(difference) <= 1e-12)),
        "v9HigherLossShare": float(np.mean(difference > 1e-12)),
        "meanPairedLossDifference": float(np.mean(difference)),
    }


def verify_freeze(freeze: dict[str, Any]) -> None:
    if freeze.get("status") != "frozen-sealed":
        raise RuntimeError("V9 audit protocol was not frozen")
    for source in freeze["sourceSha256"].values():
        path = ROOT / source["path"]
        if digest(path) != source["sha256"]:
            raise RuntimeError(f"Frozen V9 audit source changed: {source['path']}")
    if digest(V9_MODEL_PATH) != freeze["frozenV9SelectorSha256"]:
        raise RuntimeError("Frozen V9 selector changed")
    if digest(V8_MODEL_PATH) != freeze["frozenV8SelectorSha256"]:
        raise RuntimeError("Frozen V8 comparator changed")


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError("The V9 sealed audit has already been scored; reruns are forbidden.")
    audit = json.loads(AUDIT_PATH.read_text())
    v9_model = json.loads(V9_MODEL_PATH.read_text())
    v8_model = json.loads(V8_MODEL_PATH.read_text())
    freeze = json.loads(FREEZE_PATH.read_text())
    opened = json.loads(OPEN_PATH.read_text())
    verify_freeze(freeze)
    contract = freeze["contract"]
    if (
        audit.get("activation") != "sealed-audit-opened-unscored"
        or not opened.get("oneTimeOpening")
        or audit.get("provenance", {}).get("nutsUsed")
        or v9_model.get("provenance", {}).get("nutsUsed")
        or len(audit.get("rows", [])) != contract["cohort"]["candidateRows"]
        or len(audit.get("featureNames", [])) != contract["cohort"]["tokens"]
        or audit["featureNames"] != v9_model["featureNames"]
        or audit["featureNames"][:136] != v8_model["featureNames"]
    ):
        raise RuntimeError("V9 sealed-audit provenance or token contract is invalid")

    arrays = v9.prepare_arrays(audit["rows"])
    v9_predicted = v9_predictions(v9_model, arrays["features"])
    v8_predicted = v8_predictions(v8_model, arrays["features"])
    v9_policy = v9.PolicyConfig(**v9_model["policy"])
    v8_policy = v9.PolicyConfig(
        danger_penalty=float(v8_model["policy"]["danger_penalty"]),
        uncertainty_penalty=float(v8_model["policy"]["uncertainty_penalty"]),
    )
    selected_v9 = select(arrays, v9_predicted, v9_policy)
    selected_v8 = select(arrays, v8_predicted, v8_policy)
    v9_full = v9.promoted_metrics(selected_v9)
    v8_full = v9.promoted_metrics(selected_v8)
    v9_operational = v9.at_coverage(selected_v9, 0.70)
    v8_operational = v9.at_coverage(selected_v8, 0.70)
    v9_curve = v9.risk_coverage_curve(selected_v9)
    v8_curve = v9.risk_coverage_curve(selected_v8)
    inference_contract = contract["endpoints"]["primaryInference"]
    bootstrap = stratified_paired_bootstrap(
        selected_v9,
        selected_v8,
        int(inference_contract["resamples"]),
        int(inference_contract["seed"]),
    )
    checks = {
        "primarySuperiority": bootstrap["pointDifference"] < 0
        and bootstrap["oneSided95UpperBound"] < 0,
        "fullCoverageMeanImproved": v9_full["meanPromotedExcessLoss"]
        < v8_full["meanPromotedExcessLoss"],
        "fullCoverageP90Improved": v9_full["p90PromotedExcessLoss"]
        < v8_full["p90PromotedExcessLoss"],
        "fullCoverageP95NonInferior": v9_full["p95PromotedExcessLoss"]
        <= 1.02 * v8_full["p95PromotedExcessLoss"],
        "dangerousFalseChampionImproved": v9_full["dangerousFalseChampionShare"]
        < v8_full["dangerousFalseChampionShare"],
        "selectiveRiskAreaImproved": v9_curve["areaUnderSelectionRisk"]
        < v8_curve["areaUnderSelectionRisk"],
        "matchedOperationalObjectiveImproved": v9_operational["selectionObjective"]
        < v8_operational["selectionObjective"],
        "theoremVerified": v9_full["theoremViolations"] == 0,
        "nutsExcluded": True,
    }
    artifact = {
        "artifactId": "flux-svi-score-v9-sealed-audit-v1",
        "version": audit["version"],
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "activation": "sealed-audit-final",
        "stage": "one-time-confirmatory-synthetic-audit-complete",
        "provenance": {
            "auditDatasetSha256": digest(AUDIT_PATH),
            "frozenV9SelectorSha256": digest(V9_MODEL_PATH),
            "frozenV8SelectorSha256": digest(V8_MODEL_PATH),
            "protocolFreezeSha256": digest(FREEZE_PATH),
            "openReceiptSha256": digest(OPEN_PATH),
            "auditOpenedExactlyOnce": True,
            "postAuditTuningPermitted": False,
            "nutsUsed": False,
        },
        "contract": contract,
        "frozenPolicies": {"v9": v9_model["policy"], "v8": v8_model["policy"]},
        "primaryInference": bootstrap,
        "v9": {
            "at100PercentCoverage": v9_full,
            "at70PercentCoverage": v9_operational,
            "riskCoverageCurve": v9_curve,
            "byFamily": family_metrics(selected_v9),
        },
        "v8": {
            "at100PercentCoverage": v8_full,
            "at70PercentCoverage": v8_operational,
            "riskCoverageCurve": v8_curve,
            "byFamily": family_metrics(selected_v8),
        },
        "pairedBusinessOutcomes": paired_outcomes(selected_v9, selected_v8),
        "checks": checks,
        "conclusion": {
            "allConfirmatoryChecksPassed": all(checks.values()),
            "v9SyntheticGeneralizationConfirmed": checks["primarySuperiority"]
            and checks["theoremVerified"],
            "externalRealWorldGeneralizationEstablished": False,
            "sotaClaimPermitted": False,
            "productionActivationPermitted": False,
            "furtherTuningRequiresNewSealedCohort": True,
        },
        "selections": {"v9": selected_v9, "v8": selected_v8},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x") as handle:
        json.dump(artifact, handle, indent=2)
    print(
        json.dumps(
            {
                "path": str(OUTPUT_PATH),
                "v9": {
                    "at100PercentCoverage": v9_full,
                    "at70PercentCoverage": v9_operational,
                },
                "v8": {
                    "at100PercentCoverage": v8_full,
                    "at70PercentCoverage": v8_operational,
                },
                "primaryInference": bootstrap,
                "checks": checks,
                "conclusion": artifact["conclusion"],
            },
            indent=2,
        )
    )


def development_smoke() -> None:
    dataset = json.loads(DEVELOPMENT_DATASET_PATH.read_text())
    v9_model = json.loads(DEVELOPMENT_V9_MODEL_PATH.read_text())
    v8_model = json.loads(DEVELOPMENT_V8_MODEL_PATH.read_text())
    arrays = v9.prepare_arrays(dataset["rows"])
    left = select(
        arrays,
        v9_predictions(v9_model, arrays["features"]),
        v9.PolicyConfig(**v9_model["policy"]),
    )
    right = select(
        arrays,
        v8_predictions(v8_model, arrays["features"]),
        v9.PolicyConfig(
            danger_penalty=float(v8_model["policy"]["danger_penalty"]),
            uncertainty_penalty=float(v8_model["policy"]["uncertainty_penalty"]),
        ),
    )
    receipt = stratified_paired_bootstrap(left, right, 100, 9_031_771)
    print(
        json.dumps(
            {
                "developmentSmoke": True,
                "businesses": len(left),
                "v9Finite": bool(np.isfinite(primary_objective(left))),
                "v8Finite": bool(np.isfinite(primary_objective(right))),
                "bootstrapFinite": bool(
                    np.isfinite(receipt["oneSided95UpperBound"])
                ),
                "auditOpened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    if "--development-smoke" in sys.argv:
        development_smoke()
    else:
        main()
