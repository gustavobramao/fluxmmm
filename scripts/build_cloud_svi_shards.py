#!/usr/bin/env python3
"""Group a candidate-level Flux SVI manifest into balanced execution shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


OUTPUT_VERSION = "flux-svi-cvm-structure-shards-v2"


def build_shards(
    source: dict[str, Any],
    shard_count: int,
    *,
    group_by: str = "candidateId",
    group_limit: int | None = None,
    items_per_group: int | None = None,
    output_prefix: str | None = None,
) -> dict[str, Any]:
    if source.get("version") != "flux-svi-cvm-retry-v1":
        raise ValueError("Unsupported candidate manifest version.")
    items = source.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Candidate manifest contains no items.")
    if group_by not in {"businessId", "candidateId", "fingerprint"}:
        raise ValueError("group_by must be businessId, candidateId, or fingerprint.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get(group_by), str):
            raise ValueError(f"Every candidate must declare its {group_by}.")
        normalized = dict(item)
        if output_prefix is not None:
            normalized["outputObject"] = (
                f"{output_prefix.rstrip('/')}/{item['fingerprint']}.result.json.gz"
            )
        grouped[item[group_by]].append(normalized)
    group_ids = sorted(grouped)
    if group_limit is not None:
        group_ids = group_ids[:group_limit]
    shard_count = min(max(shard_count, 1), len(group_ids))
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for index, group_id in enumerate(group_ids):
        candidates = sorted(
            grouped[group_id],
            key=lambda item: (item.get("businessId", ""), item["fingerprint"]),
        )
        if items_per_group is not None:
            candidates = candidates[:items_per_group]
        shards[index % shard_count].append({"groupId": group_id, "items": candidates})
    selected = [
        item
        for shard in shards
        for group in shard
        for item in group["items"]
    ]
    return {
        "version": OUTPUT_VERSION,
        "sourceManifestVersion": source["version"],
        "sourceManifestSha256": hashlib.sha256(
            json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "protocolFreezeSha256": source.get("protocolFreezeSha256"),
        "auditVersion": source.get("auditVersion"),
        "remotePrefix": source.get("remotePrefix"),
        "groupBy": group_by,
        "groupCount": len(group_ids),
        "businessCount": len({item["businessId"] for item in selected}),
        "candidateCount": len(selected),
        "shardCount": shard_count,
        "hiddenTruthDereferenced": False,
        "shards": [
            {"shardIndex": index, "workGroups": groups}
            for index, groups in enumerate(shards)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument(
        "--group-by",
        choices=("businessId", "candidateId", "fingerprint"),
        default="candidateId",
    )
    parser.add_argument("--group-limit", type=int)
    parser.add_argument("--items-per-group", type=int)
    parser.add_argument("--output-prefix")
    arguments = parser.parse_args()
    source = json.loads(Path(arguments.input).read_bytes())
    result = build_shards(
        source,
        arguments.shards,
        group_by=arguments.group_by,
        group_limit=arguments.group_limit,
        items_per_group=arguments.items_per_group,
        output_prefix=arguments.output_prefix,
    )
    target = Path(arguments.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(target),
        "groupBy": result["groupBy"],
        "groups": result["groupCount"],
        "businesses": result["businessCount"],
        "candidates": result["candidateCount"],
        "shards": result["shardCount"],
    }))


if __name__ == "__main__":
    main()
