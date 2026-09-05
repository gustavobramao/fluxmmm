import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load("flux_cloud_shard_builder", ROOT / "scripts/build_cloud_svi_shards.py")
WORKER = load("flux_cloud_shard_worker", ROOT / "scripts/cloud_svi_shard_task.py")


class CloudSviShardTests(unittest.TestCase):
    def source(self):
        items = []
        for business in ("business-a", "business-b", "business-c"):
            for candidate in range(2):
                fingerprint = (str(candidate + 1) * 64) if business == "business-a" else (
                    (str(candidate + 3) * 64) if business == "business-b" else str(candidate + 5) * 64
                )
                items.append({
                    "fingerprint": fingerprint,
                    "businessId": business,
                    "candidateId": f"candidate-{candidate}",
                    "payloadObject": f"payloads/{fingerprint}",
                    "payloadSha256": "a" * 64,
                    "outputObject": f"results/{fingerprint}",
                })
        return {
            "version": "flux-svi-cvm-retry-v1",
            "protocolFreezeSha256": "f" * 64,
            "items": items,
        }

    def test_candidate_contracts_are_never_split_across_shards(self):
        result = BUILDER.build_shards(self.source(), 2, group_by="candidateId")
        self.assertEqual(result["businessCount"], 3)
        self.assertEqual(result["candidateCount"], 6)
        locations = {}
        for shard in result["shards"]:
            for group in shard["workGroups"]:
                self.assertNotIn(group["groupId"], locations)
                locations[group["groupId"]] = shard["shardIndex"]
                self.assertEqual(len(group["items"]), 3)

    def test_worker_validates_task_identity_and_entries(self):
        result = BUILDER.build_shards(self.source(), 2, group_by="candidateId")
        groups = WORKER.manifest_shard(result, 0)
        self.assertGreaterEqual(len(groups), 1)
        with self.assertRaises(IndexError):
            WORKER.manifest_shard(result, 2)

    def test_benchmark_prefix_changes_only_output_locations(self):
        result = BUILDER.build_shards(
            self.source(),
            1,
            group_by="businessId",
            group_limit=1,
            output_prefix="benchmark/results",
        )
        items = result["shards"][0]["workGroups"][0]["items"]
        self.assertEqual(len(items), 2)
        self.assertTrue(items[0]["outputObject"].startswith("benchmark/results/"))
        self.assertTrue(items[0]["payloadObject"].startswith("payloads/"))

    def test_worker_preserves_blas_flags_with_shared_compiler_cache(self):
        flags = WORKER.pytensor_flags(
            "/tmp/compiler-cache",
            "blas__ldflags=-lopenblas",
        )
        self.assertEqual(
            flags,
            "blas__ldflags=-lopenblas,base_compiledir=/tmp/compiler-cache",
        )

    def test_fingerprint_grouping_creates_one_candidate_per_task(self):
        result = BUILDER.build_shards(
            self.source(),
            6,
            group_by="fingerprint",
        )
        self.assertEqual(result["shardCount"], 6)
        for shard in result["shards"]:
            self.assertEqual(len(shard["workGroups"]), 1)
            self.assertEqual(len(shard["workGroups"][0]["items"]), 1)
            groups = WORKER.manifest_shard(result, shard["shardIndex"])
            self.assertEqual(len(groups[0]["items"]), 1)


if __name__ == "__main__":
    unittest.main()
