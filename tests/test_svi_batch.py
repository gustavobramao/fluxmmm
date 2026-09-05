import gzip
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "flux_svi_batch_tested",
    ROOT / "scripts" / "svi_batch.py",
)
if not SPEC or not SPEC.loader:
    raise RuntimeError("Unable to import SVI batch implementation.")
SVI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SVI
SPEC.loader.exec_module(SVI)

CLOUD_SPEC = importlib.util.spec_from_file_location(
    "flux_cloud_svi_task_tested",
    ROOT / "scripts" / "cloud_svi_task.py",
)
if not CLOUD_SPEC or not CLOUD_SPEC.loader:
    raise RuntimeError("Unable to import cloud SVI task implementation.")
CLOUD = importlib.util.module_from_spec(CLOUD_SPEC)
sys.modules[CLOUD_SPEC.name] = CLOUD
CLOUD_SPEC.loader.exec_module(CLOUD)


class SviBatchTest(unittest.TestCase):
    def test_backend_receipt_names_default_and_compilation_fallback(self):
        self.assertEqual(SVI.execution_backend_label(None), "numba-default")
        self.assertEqual(SVI.execution_backend_label("c"), "cvm-retry")

    def test_elbo_drift_detects_stability_and_nonfinite_history(self):
        stable = np.concatenate([np.linspace(120, 100, 500), np.full(500, 100.0)])
        self.assertLess(SVI.elbo_drift(stable, 200), 0.01)
        unstable = np.linspace(200, 100, 1_000)
        self.assertGreater(SVI.elbo_drift(unstable, 200), 0.1)
        self.assertTrue(np.isinf(SVI.elbo_drift(np.array([1.0, np.nan]), 2)))

    def test_seed_agreement_is_channel_aligned_and_scale_robust(self):
        close = [
            {"search": 1.2, "social": 2.5},
            {"search": 1.25, "social": 2.45},
        ]
        far = [
            {"search": 0.1, "social": 2.5},
            {"search": 4.0, "social": 2.5},
        ]
        self.assertLess(SVI.maximum_seed_difference(close), 0.05)
        self.assertGreater(SVI.maximum_seed_difference(far), 0.25)

    def test_cloud_manifest_payload_and_result_contracts_are_immutable(self):
        fingerprint = "a" * 64
        payload = gzip.compress(json.dumps({"fingerprint": fingerprint}).encode())
        entry = {
            "fingerprint": fingerprint,
            "payloadObject": "retry/payload.json.gz",
            "payloadSha256": hashlib.sha256(payload).hexdigest(),
            "outputObject": "retry/result.json.gz",
        }
        selected = CLOUD.manifest_entry(
            {"version": "flux-svi-cvm-retry-v1", "items": [entry]},
            0,
        )
        self.assertEqual(selected, entry)
        CLOUD.validate_payload(payload, entry)
        result = gzip.compress(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "executionBackend": "cvm-retry",
                    "status": "labelled",
                }
            ).encode()
        )
        self.assertEqual(CLOUD.validate_result(result, fingerprint)["status"], "labelled")

    def test_cloud_result_rejects_a_different_backend(self):
        fingerprint = "b" * 64
        result = gzip.compress(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "executionBackend": "numba-default",
                    "status": "labelled",
                }
            ).encode()
        )
        with self.assertRaisesRegex(ValueError, "C/VM backend"):
            CLOUD.validate_result(result, fingerprint)


if __name__ == "__main__":
    unittest.main()
