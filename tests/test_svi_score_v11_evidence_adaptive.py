from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "flux_v11_evidence_adaptive_selector",
    ROOT / "research/svi_score_v11_evidence_adaptive/train_selector.py",
)
assert SPEC and SPEC.loader
EA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EA
SPEC.loader.exec_module(EA)


class EvidenceAdaptiveSelectorTest(unittest.TestCase):
    def fixture(self):
        random = np.random.default_rng(17)
        features = random.normal(size=(2, 5, 8))
        contexts = random.normal(size=(2, 6))
        valid = np.ones((2, 5), dtype=bool)
        targets = np.abs(random.normal(size=(2, 5, 9)))
        dangerous = np.zeros((2, 5))
        indices = {
            expert: np.asarray([2 * index, 2 * index + 1])
            for index, expert in enumerate(EA.EXPERTS)
        }
        model = EA.EvidenceAdaptiveSelector(
            EA.V9.ModelConfig("test", 5, 4, 1, 0.001, 0.0, 0.35),
            11,
            indices,
        )
        model.means = np.zeros(8)
        model.scales = np.ones(8)
        model.context_means = np.zeros(6)
        model.context_scales = np.ones(6)
        model.initialize(8, 6, targets, dangerous)
        return model, features, contexts, valid

    def test_candidate_order_is_permutation_equivariant(self) -> None:
        model, features, contexts, valid = self.fixture()
        original, original_gate = model.predict(features, contexts, valid)
        permutation = np.asarray([3, 0, 4, 1, 2])
        permuted, permuted_gate = model.predict(
            features[:, permutation],
            contexts,
            valid[:, permutation],
        )
        inverse = np.argsort(permutation)
        np.testing.assert_allclose(original, permuted[:, inverse], atol=1e-12)
        np.testing.assert_allclose(original_gate, permuted_gate, atol=1e-12)

    def test_gate_is_business_level_and_sums_to_one(self) -> None:
        model, features, contexts, valid = self.fixture()
        _output, gate = model.predict(features, contexts, valid)
        self.assertEqual(gate.shape, (2, 4))
        np.testing.assert_allclose(np.sum(gate, axis=1), np.ones(2), atol=1e-12)
        self.assertTrue(np.all(gate > 0))

    def test_padding_does_not_change_valid_candidate_predictions(self) -> None:
        model, features, contexts, valid = self.fixture()
        base, _ = model.predict(features[:, :3], contexts, valid[:, :3])
        padded_features = np.concatenate(
            (features[:, :3], np.full((2, 2, 8), 100.0)),
            axis=1,
        )
        padded_valid = np.concatenate(
            (valid[:, :3], np.zeros((2, 2), dtype=bool)),
            axis=1,
        )
        padded, _ = model.predict(padded_features, contexts, padded_valid)
        np.testing.assert_allclose(base, padded[:, :3], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
