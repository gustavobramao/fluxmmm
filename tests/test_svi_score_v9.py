from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "flux_v9_selector", ROOT / "research/svi_score_v9/train_selector.py"
)
assert SPEC and SPEC.loader
V9 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V9
SPEC.loader.exec_module(V9)


class V9SetwiseSelectorTest(unittest.TestCase):
    def test_candidate_order_is_permutation_equivariant(self) -> None:
        random = np.random.default_rng(7)
        features = random.normal(size=(2, 5, 8))
        targets = np.abs(random.normal(size=(2, 5, 9)))
        dangerous = np.zeros((2, 5))
        model = V9.DeepSetSelector(
            V9.ModelConfig("test", 5, 4, 1, 0.001, 0.0, 0.35), 11
        )
        model.means = np.zeros(8)
        model.scales = np.ones(8)
        model.initialize(8, targets, dangerous)
        original = model.predict(features)
        permutation = np.asarray([3, 0, 4, 1, 2])
        permuted = model.predict(features[:, permutation])
        inverse = np.argsort(permutation)
        np.testing.assert_allclose(original, permuted[:, inverse], atol=1e-12)

    def test_selected_regret_is_bounded_by_oracle_misrankings(self) -> None:
        heads = np.zeros((1, 3, V9.HEAD_COUNT))
        heads[..., V9.DANGER_HEAD] = 0.1
        receipts = V9.selection_receipts(
            np.asarray(["business"], dtype=object),
            np.asarray(["family"], dtype=object),
            np.asarray([["a", "b", "c"]], dtype=object),
            np.ones((1, 3), dtype=bool),
            np.asarray([[0.0, 0.4, 1.2]]),
            np.asarray([[0.0, 0.4, 1.0]]),
            np.asarray([[False, False, True]]),
            {
                "heads": heads,
                "risk": np.asarray([[0.3, 0.1, 0.6]]),
                "uncertainty": np.zeros((1, 3)),
            },
            V9.PolicyConfig(0.0, 0.0),
        )
        self.assertEqual(receipts[0]["candidateId"], "b")
        self.assertLessEqual(receipts[0]["theoremSlack"], 1e-10)
        self.assertEqual(V9.promoted_metrics(receipts)["theoremViolations"], 0)

    def test_selective_curve_uses_equal_declared_coverages(self) -> None:
        receipts = []
        for index in range(20):
            receipts.append(
                {
                    "businessId": str(index),
                    "actualExcessLoss": float(index) / 10,
                    "normalizedExcessLoss": min(float(index) / 10, 1),
                    "dangerousFalseChampion": index >= 10,
                    "oracle": index == 0,
                    "confidenceRisk": float(index),
                    "theoremSlack": 0.0,
                }
            )
        curve = V9.risk_coverage_curve(receipts)
        self.assertEqual(len(curve["points"]), 9)
        self.assertAlmostEqual(curve["range"][0], 0.6)
        self.assertAlmostEqual(curve["range"][1], 1.0)
        self.assertLess(
            curve["points"][0]["meanPromotedExcessLoss"],
            curve["points"][-1]["meanPromotedExcessLoss"],
        )


if __name__ == "__main__":
    unittest.main()
