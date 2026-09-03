"""How much untouched geography EuroSAT still has after a prepared benchmark.

Reads every patch in the official archive, compares it with an already prepared
manifest, and reports both the cell budget and the distance-tier fallback. This
reproduces the measurement behind ADR 0006. No download and no training occurs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eo_visual_retrieval.benchmarks.coverage import (
    cell_budget,
    distance_tiers,
    nearest_distance_percentiles,
)
from eo_visual_retrieval.benchmarks.eurosat import discover_candidates
from eo_visual_retrieval.datasets.eurosat import archive_members, verify_archive
from eo_visual_retrieval.hashing import file_md5, file_sha256
from eo_visual_retrieval.manifests import read_jsonl

DEFAULT_THRESHOLDS_KM = (5.0, 10.0, 20.0, 30.0, 50.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group-size-km", type=float, default=50.0)
    parser.add_argument(
        "--thresholds-km", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS_KM)
    )
    args = parser.parse_args()

    prepared = read_jsonl(args.manifest)
    used_members = set(archive_members(prepared))

    # discover_candidates() itself skips checksum verification so it can also run
    # against locally prepared archives, but this script's output becomes published
    # evidence (see docs/validation.md), so it must be pinned to the published
    # EuroSAT archive before any measurement happens.
    verify_archive(args.archive)
    candidates = discover_candidates(args.archive, group_size_m=args.group_size_km * 1000)
    budget = cell_budget(candidates, used_members)
    # Reference coordinates come from the verified source, just like unused
    # coordinates; edited or stale manifest metadata must not move the reference.
    used_lonlat = np.asarray(
        [(item.longitude, item.latitude) for item in candidates if item.member in used_members],
        dtype=np.float64,
    )
    tiers = distance_tiers(
        candidates,
        used_members=used_members,
        reference_lonlat=used_lonlat,
        thresholds_km=args.thresholds_km,
    )
    nearest_km_percentiles = nearest_distance_percentiles(
        candidates,
        used_members=used_members,
        reference_lonlat=used_lonlat,
    )
    unused_patches = budget.total_patches - budget.used_patches

    result: dict[str, Any] = {
        "measurement": "eurosat-cell-budget-and-distance-tiers",
        "group_size_km": args.group_size_km,
        "archive": str(args.archive),
        "archive_md5": file_md5(args.archive),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "prepared_patches": len(prepared),
        "unused_patches": unused_patches,
        "cell_budget": budget.to_dict(),
        "distance_from_prepared": tiers,
        "nearest_km_percentiles": nearest_km_percentiles,
        "notes": [
            "A cell counts as spent when any one of its patches was selected.",
            "Distance tiers are a weaker fallback than cell disjointness.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
