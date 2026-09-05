#!/usr/bin/env python3
"""Score the once-opened V8 audit with the frozen development selector."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))
import train_selector as v8  # noqa: E402

AUDIT_PATH = ROOT / ".flux-artifacts/svi-score-v8/sealed-audit-dataset.json"
MODEL_PATH = ROOT / ".flux-artifacts/svi-score-v8/frozen-development-selector.json"
FREEZE_PATH = ROOT / ".flux-artifacts/svi-score-v8/sealed-audit-protocol-freeze.json"
OPEN_PATH = ROOT / ".flux-artifacts/svi-score-v8/sealed-audit-open-receipt.json"
OUTPUT_PATH = ROOT / "research/svi_score_v8/artifacts/svi-score-v8-sealed-audit.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialized_predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
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


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError("The V8 sealed audit has already been scored; reruns are forbidden.")
    audit = json.loads(AUDIT_PATH.read_text())
    model = json.loads(MODEL_PATH.read_text())
    freeze = json.loads(FREEZE_PATH.read_text())
    opened = json.loads(OPEN_PATH.read_text())
    if (
        audit.get("activation") != "sealed-audit-opened"
        or model.get("activation") != "development-only"
        or freeze.get("status") != "frozen-sealed"
        or not opened.get("oneTimeOpening")
        or audit.get("provenance", {}).get("nutsUsed")
        or model.get("provenance", {}).get("nutsUsed")
    ):
        raise RuntimeError("V8 audit provenance or SVI-only contract is invalid")
    rows: list[dict[str, Any]] = audit["rows"]
    if len(rows) != 3_840 or len(audit["featureNames"]) != 136:
        raise RuntimeError("V8 sealed audit must contain 80 x 48 rows and 136 tokens")
    if audit["featureNames"] != model["featureNames"]:
        raise RuntimeError("V8 sealed-audit token order does not match the frozen selector")

    features = np.asarray([row["features"] for row in rows], dtype=float)
    outputs = np.stack(
        [serialized_predict(member, features) for member in model["models"]], axis=0
    )
    heads = np.mean(outputs, axis=0)
    member_risk = (
        v8.MEAN_RISK_WEIGHT * outputs[:, :, v8.MEAN_HEAD]
        + v8.P90_RISK_WEIGHT * outputs[:, :, v8.P90_HEAD]
    )
    predicted = {
        "heads": heads,
        "risk": np.mean(member_risk, axis=0),
        "uncertainty": np.std(member_risk, axis=0, ddof=1),
    }
    indexes = np.arange(len(rows), dtype=int)
    business_ids = np.asarray([row["businessId"] for row in rows], dtype=object)
    families = np.asarray([row["family"] for row in rows], dtype=object)
    candidate_ids = np.asarray([row["candidateId"] for row in rows], dtype=object)
    valid = np.asarray([row["valid"] for row in rows], dtype=bool)
    excess = np.asarray([row["excessEconomicRisk"] for row in rows], dtype=float)
    normalized = np.asarray(
        [row["normalizedExcessEconomicRisk"] for row in rows], dtype=float
    )
    dangerous = np.asarray([row["dangerous"] for row in rows], dtype=bool)
    policy = v8.PolicyConfig(**model["policy"])
    selections = v8.selection_receipts(
        indexes,
        business_ids,
        families,
        candidate_ids,
        valid,
        excess,
        normalized,
        dangerous,
        predicted,
        policy,
    )
    result = v8.metrics(selections)
    baseline_predictions = [
        {
            "businessId": row["businessId"],
            "candidateId": row["candidateId"],
            "risk": row["v5dSviPredictedRisk"],
        }
        for row in rows
    ]
    baseline = v8.metrics(v8.baseline_receipts(rows, baseline_predictions))
    for key in (
        "theoremViolations",
        "maximumTheoremSlack",
        "meanIndicatorBound",
        "meanLogisticBound",
    ):
        baseline[key] = None
    checks = {
        "selectionObjectiveImproved": result["selectionObjective"]
        < baseline["selectionObjective"],
        "meanPolicyLossImproved": result["meanPolicyLoss"]
        < baseline["meanPolicyLoss"],
        "p90PolicyLossImproved": result["p90PolicyLoss"]
        < baseline["p90PolicyLoss"],
        "p95PolicyLossNonInferior": result["p95PolicyLoss"]
        <= 1.02 * baseline["p95PolicyLoss"],
        "dangerousFalseChampionImproved": (
            result["dangerousFalseChampionShare"] is not None
            and result["dangerousFalseChampionShare"]
            < baseline["dangerousFalseChampionShare"]
        ),
        "minimumPromotionCoverage": result["promotionCoverage"]
        >= v8.MINIMUM_PROMOTION_COVERAGE,
        "theoremVerified": result["theoremViolations"] == 0,
        "nutsExcluded": True,
    }
    artifact = {
        "artifactId": "flux-svi-score-v8-sealed-audit-v1",
        "version": audit["version"],
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "activation": "sealed-audit-final",
        "stage": "one-time-sealed-audit-complete",
        "provenance": {
            "auditDatasetSha256": digest(AUDIT_PATH),
            "frozenSelectorSha256": digest(MODEL_PATH),
            "protocolFreezeSha256": digest(FREEZE_PATH),
            "openReceiptSha256": digest(OPEN_PATH),
            "auditOpenedExactlyOnce": True,
            "postAuditTuningPermitted": False,
            "nutsUsed": False,
        },
        "frozenPolicy": model["policy"],
        "v8": result,
        "v5dSviComparator": baseline,
        "checks": checks,
        "conclusion": {
            "allAuditChecksPassed": all(checks.values()),
            "externalGeneralizationEstablished": False,
            "sotaClaimPermitted": False,
            "productionActivationPermitted": False,
            "furtherTuningRequiresNewSealedCohort": True,
        },
        "selections": selections,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x") as handle:
        json.dump(artifact, handle, indent=2)
    print(
        json.dumps(
            {
                "path": str(OUTPUT_PATH),
                "v8": result,
                "v5dSvi": baseline,
                "checks": checks,
                "conclusion": artifact["conclusion"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
