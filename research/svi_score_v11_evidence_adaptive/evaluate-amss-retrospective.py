#!/usr/bin/env python3
"""Score frozen V11 AMSS choices against the already-opened accepted outcomes."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = (
    ROOT / ".flux-artifacts/svi-score-v11-evidence-adaptive/amss-retrospective"
)
SELECTION_PATH = ARTIFACT_ROOT / "frozen-selections.json"
V9_SELECTION_PATH = (
    ROOT
    / ".flux-artifacts/svi-score-v10-amss/observable-freeze/frozen-selections.json"
)
CANDIDATE_LOSSES_PATH = (
    ROOT / ".flux-artifacts/svi-score-v10-amss/external-audit/candidate-losses.csv"
)
ACCEPTANCE_PATH = (
    ROOT / ".flux-artifacts/svi-score-v10-amss/external-audit/acceptance-receipt.json"
)
OUTPUT_PATH = ARTIFACT_ROOT / "result.json"
PUBLIC_RESULT_PATH = (
    ROOT
    / "research/svi_score_v11_evidence_adaptive/artifacts/v11-amss-retrospective-result.json"
)
BOOTSTRAP_SEED = 11_100_510
BOOTSTRAP_RESAMPLES = 10_000
MEAN_WEIGHT = 0.65
P90_WEIGHT = 0.35


def sha256(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def endpoint(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    risks = [float(row["risk"]) for row in rows]
    uncapped = [float(row["uncappedRisk"]) for row in rows]
    oracle = [float(row["oracleRisk"]) for row in rows]
    excess = [max(0.0, value - floor) for value, floor in zip(risks, oracle)]
    mean_risk = float(np.mean(risks))
    p90_risk = quantile(risks, 0.90)
    return {
        "businesses": len(rows),
        "mean": mean_risk,
        "p90": p90_risk,
        "p95": quantile(risks, 0.95),
        "selectionObjective": MEAN_WEIGHT * mean_risk + P90_WEIGHT * p90_risk,
        "uncappedMean": float(np.mean(uncapped)),
        "uncappedP90": quantile(uncapped, 0.90),
        "uncappedP95": quantile(uncapped, 0.95),
        "meanExcessOverInPoolOracle": float(np.mean(excess)),
        "oracleRecall": float(np.mean(np.asarray(excess) <= 1e-12)),
    }


def paired_comparison(
    v11: list[dict[str, Any]],
    comparator: list[dict[str, Any]],
) -> dict[str, Any]:
    comparator_by_business = {row["businessId"]: row for row in comparator}
    pairs = []
    by_group: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in v11:
        baseline = comparator_by_business[row["businessId"]]
        pair = (float(row["risk"]), float(baseline["risk"]))
        pairs.append(pair)
        by_group[str(row["evidenceGroup"])].append(pair)
    differences = np.asarray([left - right for left, right in pairs])
    point_objective = (
        endpoint(v11)["selectionObjective"]
        - endpoint(comparator)["selectionObjective"]
    )
    random = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_mean = np.zeros(BOOTSTRAP_RESAMPLES)
    bootstrap_objective = np.zeros(BOOTSTRAP_RESAMPLES)
    groups = list(by_group.values())
    for index in range(BOOTSTRAP_RESAMPLES):
        sampled = []
        for group in groups:
            selected = random.integers(0, len(group), size=len(group))
            sampled.extend(group[position] for position in selected)
        left = [pair[0] for pair in sampled]
        right = [pair[1] for pair in sampled]
        bootstrap_mean[index] = float(np.mean(np.asarray(left) - np.asarray(right)))
        bootstrap_objective[index] = (
            MEAN_WEIGHT * float(np.mean(left))
            + P90_WEIGHT * quantile(left, 0.90)
            - MEAN_WEIGHT * float(np.mean(right))
            - P90_WEIGHT * quantile(right, 0.90)
        )
    return {
        "estimand": "V11 minus comparator risk; negative favors V11",
        "meanDifference": float(np.mean(differences)),
        "meanDifferenceInterval95": [
            float(np.quantile(bootstrap_mean, 0.025)),
            float(np.quantile(bootstrap_mean, 0.975)),
        ],
        "selectionObjectiveDifference": float(point_objective),
        "selectionObjectiveDifferenceInterval95": [
            float(np.quantile(bootstrap_objective, 0.025)),
            float(np.quantile(bootstrap_objective, 0.975)),
        ],
        "v11LowerRisk": int(np.sum(differences < -1e-12)),
        "tied": int(np.sum(np.abs(differences) <= 1e-12)),
        "comparatorLowerRisk": int(np.sum(differences > 1e-12)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "stratifiedByEvidenceGroup": True,
    }


def main() -> None:
    selection_source = SELECTION_PATH.read_bytes()
    v9_source = V9_SELECTION_PATH.read_bytes()
    losses_source = CANDIDATE_LOSSES_PATH.read_bytes()
    acceptance_source = ACCEPTANCE_PATH.read_bytes()
    selections = json.loads(selection_source)
    v9_frozen = json.loads(v9_source)
    acceptance = json.loads(acceptance_source)
    if (
        selections.get("status")
        != "v11-selections-frozen-before-retrospective-scoring"
        or selections.get("provenance", {}).get("hiddenEconomicOutcomesReadBySelection")
        or selections.get("provenance", {}).get("selectorRetrained")
        or selections.get("provenance", {}).get("posteriorRefits") != 0
        or selections.get("provenance", {}).get("cloudComputeUsed")
        or v9_frozen.get("status") != "frozen-before-amss-truth-open"
        or v9_frozen.get("provenance", {}).get("hiddenTruthDereferenced")
        or acceptance.get("status")
        != "confirmatory-result-accepted-after-contract-verification"
        or acceptance.get("artifacts", {}).get("candidateLossesSha256")
        != sha256(losses_source)
    ):
        raise RuntimeError("Retrospective scoring inputs violate the accepted freeze chain")

    losses: dict[tuple[str, str], dict[str, Any]] = {}
    by_business: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reader = csv.DictReader(losses_source.decode("utf8").splitlines())
    for row in reader:
        item = {
            "businessId": row["business_id"],
            "evidenceGroup": row["evidence_group"],
            "candidateId": row["candidate_id"],
            "valid": row["valid"] == "true",
            "risk": float(row["risk"]),
            "uncappedRisk": float(row["uncapped_risk"]),
        }
        losses[(item["businessId"], item["candidateId"])] = item
        by_business[item["businessId"]].append(item)
    if len(losses) != 4_800 or len(by_business) != 100:
        raise RuntimeError("Accepted AMSS candidate losses are incomplete")
    oracle_by_business = {
        business: min(
            (row for row in rows if row["valid"]), key=lambda row: row["risk"]
        )
        for business, rows in by_business.items()
    }

    v9_ids = {
        row["businessId"]: row["v9CandidateId"]
        for row in v9_frozen["selections"]
    }
    v11_rows = []
    v9_rows = []
    prediction_rows = []
    oracle_rows = []
    selection_details = []
    for frozen in selections["selections"]:
        business = frozen["businessId"]
        oracle = oracle_by_business[business]

        def chosen(candidate_id: str) -> dict[str, Any]:
            row = dict(losses[(business, candidate_id)])
            row["oracleRisk"] = oracle["risk"]
            return row

        v11 = chosen(frozen["v11CandidateId"])
        v9 = chosen(v9_ids[business])
        prediction = chosen(frozen["predictionOnlyCandidateId"])
        oracle_row = dict(oracle)
        oracle_row["oracleRisk"] = oracle["risk"]
        v11_rows.append(v11)
        v9_rows.append(v9)
        prediction_rows.append(prediction)
        oracle_rows.append(oracle_row)
        selection_details.append(
            {
                "businessId": business,
                "evidenceGroup": frozen["evidenceGroup"],
                "v11CandidateId": v11["candidateId"],
                "v11Risk": v11["risk"],
                "v9CandidateId": v9["candidateId"],
                "v9Risk": v9["risk"],
                "predictionOnlyCandidateId": prediction["candidateId"],
                "predictionOnlyRisk": prediction["risk"],
                "oracleCandidateId": oracle["candidateId"],
                "oracleRisk": oracle["risk"],
                "evidenceGate": frozen["evidenceGate"],
            }
        )

    endpoints = {
        "evidenceAdaptiveV11": endpoint(v11_rows),
        "frozenV9": endpoint(v9_rows),
        "predictionOnly": endpoint(prediction_rows),
        "inPoolOracle": endpoint(oracle_rows),
    }
    comparisons = {
        "v11MinusFrozenV9": paired_comparison(v11_rows, v9_rows),
        "v11MinusPredictionOnly": paired_comparison(v11_rows, prediction_rows),
    }
    groups = sorted({row["evidenceGroup"] for row in v11_rows})
    subgroups = {
        group: {
            "evidenceAdaptiveV11": endpoint(
                [row for row in v11_rows if row["evidenceGroup"] == group]
            ),
            "frozenV9": endpoint(
                [row for row in v9_rows if row["evidenceGroup"] == group]
            ),
            "predictionOnly": endpoint(
                [row for row in prediction_rows if row["evidenceGroup"] == group]
            ),
        }
        for group in groups
    }
    v11_objective = float(endpoints["evidenceAdaptiveV11"]["selectionObjective"])
    v9_objective = float(endpoints["frozenV9"]["selectionObjective"])
    prediction_objective = float(endpoints["predictionOnly"]["selectionObjective"])
    screening_checks = {
        "lowerObjectiveThanFrozenV9": v11_objective < v9_objective,
        "noHigherObjectiveThanPredictionOnly": v11_objective <= prediction_objective,
    }
    artifact = {
        "artifactId": "flux-v11-amss-retrospective-result-v1",
        "version": selections["version"],
        "status": "complete-retrospective-external-benchmark-not-confirmatory",
        "scientificLabel": "retrospective reuse of previously opened AMSS outcomes",
        "provenance": {
            "frozenV11SelectionsSha256": sha256(selection_source),
            "frozenV9SelectionsSha256": sha256(v9_source),
            "acceptedCandidateLossesSha256": sha256(losses_source),
            "acceptanceReceiptSha256": sha256(acceptance_source),
            "selectorRetrained": False,
            "posteriorRefits": 0,
            "cloudComputeUsed": False,
        },
        "cohort": {
            "businesses": 100,
            "cachedSviCandidates": 4_800,
            "evidenceGroups": groups,
        },
        "primaryEndpoint": (
            "65% cohort mean candidate risk plus 35% cohort P90 candidate risk"
        ),
        "endpoints": endpoints,
        "pairedComparisons": comparisons,
        "subgroupsDescriptiveOnly": subgroups,
        "screeningRule": {
            "checks": screening_checks,
            "passed": all(screening_checks.values()),
            "freshConfirmatoryClaimPermitted": False,
        },
        "selections": selection_details,
        "governance": {
            "confirmatory": False,
            "previouslyOpenedAmssOutcomes": True,
            "postAmssSelectorDevelopment": True,
            "newPosteriorFits": 0,
            "cloudComputeUsed": False,
            "freshCohortRequiredForV11Claim": True,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for path in (OUTPUT_PATH, PUBLIC_RESULT_PATH):
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(artifact, indent=2))
        temporary.replace(path)
    print(
        json.dumps(
            {
                "resultPath": str(PUBLIC_RESULT_PATH),
                "endpoints": endpoints,
                "pairedComparisons": comparisons,
                "screeningRule": artifact["screeningRule"],
                "posteriorRefits": 0,
                "cloudComputeUsed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
