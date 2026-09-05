#!/usr/bin/env python3
"""Leave one simulator mechanism out of all V9 fitting and tuning."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "flux_v9_selector_runtime", ROOT / "research/svi_score_v9/train_selector.py"
)
assert SPEC and SPEC.loader
V9 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V9
SPEC.loader.exec_module(V9)

DATASET_PATH = ROOT / ".flux-artifacts/svi-score-v9/development-dataset.json"
V8_PREDICTIONS_PATH = ROOT / ".flux-artifacts/svi-score-v8/cross-fitted-predictions.json"
OUTPUT_PATH = ROOT / "research/svi_score_v9/artifacts/svi-score-v9-mechanism-holdout.json"


def digest(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def main() -> None:
    started = time.time()
    dataset_source = DATASET_PATH.read_bytes()
    v8_source = V8_PREDICTIONS_PATH.read_bytes()
    dataset = json.loads(dataset_source)
    if (
        dataset.get("activation") != "development-only"
        or dataset.get("provenance", {}).get("v8AuditDereferenced")
        or dataset.get("provenance", {}).get("nutsUsed")
    ):
        raise RuntimeError("V9 mechanism holdout requires development-only SVI data")
    arrays = V9.prepare_arrays(dataset["rows"])
    families = arrays["families"]
    folds = arrays["folds"]
    all_receipts = []
    family_receipts = []
    for family_index, held_family in enumerate(sorted(np.unique(families))):
        training = np.flatnonzero(families != held_family)
        assessment = np.flatnonzero(families == held_family)
        best = None
        candidate_receipts = []
        for config in V9.MODEL_GRID:
            inner_heads = np.full(
                (
                    arrays["features"].shape[0],
                    arrays["features"].shape[1],
                    V9.HEAD_COUNT,
                ),
                np.nan,
            )
            inner_risk = np.full(arrays["features"].shape[:2], np.nan)
            for inner_fold in sorted(np.unique(folds[training])):
                inner_train = training[folds[training] != inner_fold]
                inner_assessment = training[folds[training] == inner_fold]
                model = V9.fit_ensemble(
                    config,
                    arrays["features"],
                    arrays["targets"],
                    arrays["dangerous"],
                    arrays["normalizedGaps"],
                    arrays["valid"],
                    inner_train,
                    (120_000 + family_index * 1_000 + int(inner_fold),),
                    bootstrap=False,
                )[0]
                prediction = V9.ensemble_predict(
                    [model], arrays["features"][inner_assessment]
                )
                inner_heads[inner_assessment] = prediction["heads"]
                inner_risk[inner_assessment] = prediction["risk"]
            for policy in V9.POLICY_GRID:
                receipts = V9.selection_receipts(
                    arrays["businessIds"][training],
                    families[training],
                    arrays["candidateIds"][training],
                    arrays["valid"][training],
                    arrays["excessRisk"][training],
                    arrays["normalizedGaps"][training],
                    arrays["dangerous"][training].astype(bool),
                    {
                        "heads": inner_heads[training],
                        "risk": inner_risk[training],
                        "uncertainty": np.zeros_like(inner_risk[training]),
                    },
                    policy,
                )
                objective = V9.development_objective(receipts)
                candidate_receipts.append(
                    {"model": config.id, "policy": policy.id, "objective": objective}
                )
                key = (
                    objective,
                    config.candidate_width,
                    f"{config.id}/{policy.id}",
                    config,
                    policy,
                )
                if best is None or key[:3] < best[:3]:
                    best = key
        assert best is not None
        config, policy = best[3], best[4]
        models = V9.fit_ensemble(
            config,
            arrays["features"],
            arrays["targets"],
            arrays["dangerous"],
            arrays["normalizedGaps"],
            arrays["valid"],
            training,
            tuple(130_000 + family_index * 100 + seed for seed in range(3)),
            bootstrap=True,
        )
        prediction = V9.ensemble_predict(models, arrays["features"][assessment])
        receipts = V9.selection_receipts(
            arrays["businessIds"][assessment],
            families[assessment],
            arrays["candidateIds"][assessment],
            arrays["valid"][assessment],
            arrays["excessRisk"][assessment],
            arrays["normalizedGaps"][assessment],
            arrays["dangerous"][assessment].astype(bool),
            prediction,
            policy,
        )
        all_receipts.extend(receipts)
        result = {
            "heldOutFamily": str(held_family),
            "trainingBusinesses": int(training.size),
            "assessmentBusinesses": int(assessment.size),
            "selectedModel": config.id,
            "selectedPolicy": policy.id,
            "innerObjective": best[0],
            "at100PercentCoverage": V9.promoted_metrics(receipts),
            "at70PercentCoverage": V9.at_coverage(receipts, V9.OPERATIONAL_COVERAGE),
            "candidateConfigurations": candidate_receipts,
        }
        family_receipts.append(result)
        print(json.dumps(result), flush=True)

    v8 = V9.v8_receipts(json.loads(v8_source))
    v9_full = V9.promoted_metrics(all_receipts)
    v8_full = V9.promoted_metrics(v8)
    v9_curve = V9.risk_coverage_curve(all_receipts)
    v8_curve = V9.risk_coverage_curve(v8)
    checks = {
        "allFiveMechanismsHeldOutExactlyOnce": len(family_receipts) == 5
        and len(all_receipts) == 420,
        "overallMeanImprovedAgainstV8Reference": v9_full["meanPromotedExcessLoss"]
        < v8_full["meanPromotedExcessLoss"],
        "overallP90NonInferiorAgainstV8Reference": v9_full["p90PromotedExcessLoss"]
        <= 1.02 * v8_full["p90PromotedExcessLoss"],
        "theoremVerified": v9_full["theoremViolations"] == 0,
        "v8AuditUntouched": True,
        "nutsExcluded": True,
    }
    artifact = {
        "artifactId": "flux-svi-score-v9-leave-one-generator-family-out-v1",
        "version": dataset["version"],
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "activation": "development-only",
        "provenance": {
            "datasetSha256": digest(dataset_source),
            "v8PredictionsSha256": digest(v8_source),
            "v8AuditDereferenced": False,
            "nutsUsed": False,
        },
        "interpretation": {
            "v9": "Every business was assessed by a selector that saw no business from that simulator family during fitting or hyperparameter selection.",
            "v8Reference": "The V8 cross-fitted reference held out advertisers but not entire simulator families, so this is a conservative descriptive reference rather than a like-for-like mechanism-holdout comparator.",
        },
        "v9": {
            "at100PercentCoverage": v9_full,
            "at70PercentCoverage": V9.at_coverage(
                all_receipts, V9.OPERATIONAL_COVERAGE
            ),
            "riskCoverageCurve": v9_curve,
        },
        "v8DescriptiveReference": {
            "at100PercentCoverage": v8_full,
            "at70PercentCoverage": V9.at_coverage(v8, V9.OPERATIONAL_COVERAGE),
            "riskCoverageCurve": v8_curve,
        },
        "families": family_receipts,
        "checks": checks,
        "allChecksPassed": all(checks.values()),
        "runtimeSeconds": time.time() - started,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(artifact, indent=2))
    temporary.replace(OUTPUT_PATH)
    print(json.dumps({
        "artifactPath": str(OUTPUT_PATH),
        "v9": artifact["v9"],
        "v8DescriptiveReference": artifact["v8DescriptiveReference"],
        "checks": checks,
        "runtimeSeconds": artifact["runtimeSeconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
