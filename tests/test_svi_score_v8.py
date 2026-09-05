from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "flux_v8_selector",
    ROOT / "research/svi_score_v8/train_selector.py",
)
assert SPEC and SPEC.loader
V8 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V8
SPEC.loader.exec_module(V8)
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "flux_v8_audit",
    ROOT / "research/svi_score_v8/score-sealed-audit.py",
)
assert AUDIT_SPEC and AUDIT_SPEC.loader
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = AUDIT
AUDIT_SPEC.loader.exec_module(AUDIT)


class V8SelectorPolicyTest(unittest.TestCase):
    def test_risk_uses_declared_business_preference(self) -> None:
        outputs = np.zeros((2, V8.HEAD_COUNT))
        outputs[:, V8.MEAN_HEAD] = [1.0, 2.0]
        outputs[:, V8.P90_HEAD] = [3.0, 4.0]
        risk = (
            V8.MEAN_RISK_WEIGHT * outputs[:, V8.MEAN_HEAD]
            + V8.P90_RISK_WEIGHT * outputs[:, V8.P90_HEAD]
        )
        np.testing.assert_allclose(risk, [1.7, 2.7])

    def test_selection_theorem_bounds_selected_regret(self) -> None:
        indexes = np.arange(6)
        business_ids = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
        families = np.asarray(["f"] * 6, dtype=object)
        candidate_ids = np.asarray(["1", "2", "3", "1", "2", "3"], dtype=object)
        valid = np.ones(6, dtype=bool)
        excess = np.asarray([0.0, 0.4, 1.2, 0.0, 0.2, 0.8])
        normalized = np.minimum(excess, 1)
        dangerous = excess >= 1
        heads = np.zeros((6, V8.HEAD_COUNT))
        # Deliberately misrank candidate 2 for business a; the theorem sum must
        # still dominate the selected normalized regret.
        risk = np.asarray([0.3, 0.1, 0.6, 0.1, 0.2, 0.4])
        heads[:, V8.DANGER_HEAD] = 0.1
        receipts = V8.selection_receipts(
            indexes,
            business_ids,
            families,
            candidate_ids,
            valid,
            excess,
            normalized,
            dangerous,
            {"heads": heads, "risk": risk, "uncertainty": np.zeros(6)},
            V8.PolicyConfig(0, 0, 0.7, 1.5, 0),
        )
        self.assertEqual(len(receipts), 2)
        self.assertTrue(all(row["theoremSlack"] <= 1e-10 for row in receipts))
        self.assertEqual(V8.metrics(receipts)["theoremViolations"], 0)

    def test_abstention_is_charged_and_coverage_is_reported(self) -> None:
        indexes = np.arange(2)
        heads = np.zeros((2, V8.HEAD_COUNT))
        heads[:, V8.DANGER_HEAD] = [0.9, 0.8]
        receipts = V8.selection_receipts(
            indexes,
            np.asarray(["a", "a"], dtype=object),
            np.asarray(["f", "f"], dtype=object),
            np.asarray(["1", "2"], dtype=object),
            np.ones(2, dtype=bool),
            np.asarray([0.0, 0.4]),
            np.asarray([0.0, 0.4]),
            np.asarray([False, False]),
            {"heads": heads, "risk": np.asarray([0.1, 0.2]), "uncertainty": np.zeros(2)},
            V8.PolicyConfig(0, 0, 0.7, 1.5, 0),
        )
        result = V8.metrics(receipts)
        self.assertEqual(result["promotionCoverage"], 0)
        self.assertEqual(result["meanPolicyLoss"], V8.ABSTENTION_COST)
        self.assertGreater(result["selectionObjective"], 100)

    def test_frozen_linear_audit_model_replays_serialized_prediction(self) -> None:
        features = np.asarray([[2.0, -1.0], [0.0, 3.0]])
        model = {
            "kind": "linear",
            "config": {"learning_rate": 0.1},
            "means": [0.0, 0.0],
            "scales": [1.0, 1.0],
            "weights": np.zeros((3, V8.HEAD_COUNT)).tolist(),
        }
        model["weights"][0][V8.MEAN_HEAD] = 0.5
        model["weights"][1][V8.MEAN_HEAD] = 0.25
        model["weights"][0][V8.DANGER_HEAD] = 0.0
        prediction = AUDIT.serialized_predict(model, features)
        np.testing.assert_allclose(prediction[:, V8.MEAN_HEAD], [1.0, 0.5])
        np.testing.assert_allclose(prediction[:, V8.DANGER_HEAD], [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
