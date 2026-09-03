"""Selection is reproducible; the independent audit rejects protocol violations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from eo_visual_retrieval.benchmarks import bigearthnet_partitions as partitions
from eo_visual_retrieval.datasets.bigearthnet_footprints import add_centers
from eo_visual_retrieval.hashing import file_md5, file_sha256


def _case() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pytest.importorskip("rasterio")
    policy = copy.deepcopy(partitions.POLICY)
    policy.update(minimum_label_count=2, minimum_query_cells=2)
    rows = []
    for number, (part, spec) in enumerate(policy["partitions"].items()):
        spec["size"] = 10 if part == "index" else 6
        day = spec["start"].replace("-", "")
        for cell in range(3):
            for patch in range(8):
                left = 300_000 + number * 250_000 + cell * 60_000
                bottom = 5_000_000 + patch * 1200
                rows.append({
                    "patch_id": f"S2A_MSIL2A_{day}T101031_N9999_R022_T33UUP_"
                                f"{number * 3 + cell:02d}_{patch:02d}",
                    "labels": ["A", "B"] if patch < 2 else ["A"],
                    "split": spec["official_split"], "country": "synthetic",
                    "contains_seasonal_snow": False, "contains_cloud_or_shadow": False,
                    "epsg": 32633, "left": left, "bottom": bottom,
                    "right": left + 1200, "top": bottom + 1200,
                })
    add_centers(rows)
    return rows, policy


def test_deterministic_selection_and_source_geometry_based_audit() -> None:
    rows, policy = _case()
    selected, feasibility = partitions.select_partitions(rows, policy=policy)
    repeated = partitions.select_partitions(list(reversed(rows)), policy=policy)
    assert repeated == (selected, feasibility)
    selection = {"policy": policy, "partitions": selected}
    # Audit recomputes centers and cell keys from bounds, not saved derived coordinates.
    for row in rows:
        row.update(longitude=0, latitude=0, spatial_group="forged")
    result = partitions.audit_selection(selection, rows, policy=policy)
    assert result["partitions"]["index"]["patches"] == 10
    for value in result["pairwise_separation"].values():
        assert value["shared_cells"] == 0
        assert value["footprint_separation_lower_bound_m"] >= 5000
        assert value["temporal_gap_days"] >= 30


@pytest.mark.parametrize("case, message", [
    ("duplicate", "uniqueness"), ("unknown", "absent"), ("policy", "protocol"),
    ("split", "official split"), ("excluded", "excluded patch"),
    ("labels", "label coverage"), ("cells", "cell overlap"),
    ("distance", "guard-band"), ("time", "temporal guard"),
])
def test_audit_rejects_changed_inputs(case: str, message: str) -> None:
    rows, policy = _case()
    selected, _ = partitions.select_partitions(rows, policy=policy)
    selection = {"policy": copy.deepcopy(policy), "partitions": selected}
    by_id = {row["patch_id"]: row for row in rows}
    first = by_id[selected["index"][0]]
    if case == "duplicate":
        selected["index"][1] = selected["index"][0]
    elif case == "unknown":
        selected["index"][0] = "unknown"
    elif case == "policy":
        selection["policy"]["seed"] = 7
    elif case == "split":
        first["split"] = "test"
    elif case == "excluded":
        first["contains_cloud_or_shadow"] = True
    elif case == "labels":
        for identity in selected["index"]:
            by_id[identity]["labels"] = ["A"]
    elif case == "cells":
        other = by_id[selected["final"][0]]
        first.update({key: other[key] for key in ("left", "right", "top", "bottom")})
    elif case == "distance":
        policy["minimum_center_distance_m"] = 1_000_000
        selection["policy"] = copy.deepcopy(policy)
    elif case == "time":
        policy["minimum_temporal_gap_days"] = 1000
        selection["policy"] = copy.deepcopy(policy)
    with pytest.raises(ValueError, match=message):
        partitions.audit_selection(selection, rows, policy=policy)


def test_infeasible_selection_fails_without_relaxing_rules() -> None:
    rows, policy = _case()
    for row in rows:
        if row["split"] == "test":
            row["labels"] = ["A"]
    with pytest.raises(ValueError, match="insufficient spatially eligible label capacity: B"):
        partitions.select_partitions(rows, policy=policy)
    rows, policy = _case()
    policy["minimum_query_cells"] = 100
    with pytest.raises(ValueError, match="insufficient spatial cells"):
        partitions.select_partitions(rows, policy=policy)


def test_same_grid_on_multiple_dates_is_not_sampled_twice() -> None:
    rows, policy = _case()
    repeated = copy.deepcopy(rows[0])
    repeated["patch_id"] = repeated["patch_id"].replace("20170601", "20170701")
    rows.append(repeated)
    selected, _ = partitions.select_partitions(rows, policy=policy)
    grid_keys = [partitions._grid(identity) for identity in selected["index"]]
    assert len(set(grid_keys)) == len(grid_keys)


def test_candidate_loader_binds_inventory_metadata_and_reference_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eo_visual_retrieval.datasets import bigearthnet_footprints as footprints
    from eo_visual_retrieval.datasets.bigearthnet import MetadataAsset

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows, _ = _case()
    source = tmp_path / "source"
    source.mkdir()
    metadata = source / "metadata.parquet"
    inventory = tmp_path / "inventory.parquet"
    report = tmp_path / "report.json"
    reference = source / footprints.REFERENCE_ARCHIVE_FILENAME
    reference.write_bytes(b"synthetic reference archive")
    pq.write_table(pa.Table.from_pylist(rows), metadata)
    pq.write_table(pa.Table.from_pylist(rows), inventory)
    assets = (MetadataAsset(metadata.name, file_md5(metadata), "synthetic"),)
    monkeypatch.setattr(footprints, "METADATA_ASSETS", assets)
    monkeypatch.setattr(partitions, "METADATA_ASSETS", assets)
    evidence: dict[str, Any] = {
        "inventory_sha256": file_sha256(inventory),
        "reference_archive_sha256": file_sha256(reference),
        "metadata_sha256": {metadata.name: file_sha256(metadata)},
    }
    report.write_text(json.dumps(evidence), encoding="utf-8")
    assert len(partitions.load_candidates(source, inventory, report)) == len(rows)
    evidence["metadata_sha256"][metadata.name] = "wrong"
    report.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata differs"):
        partitions.load_candidates(source, inventory, report)
    evidence["metadata_sha256"][metadata.name] = file_sha256(metadata)
    report.write_text(json.dumps(evidence), encoding="utf-8")
    reference.write_bytes(b"changed reference archive")
    with pytest.raises(ValueError, match="reference archive differs"):
        partitions.load_candidates(source, inventory, report)
    reference.write_bytes(b"synthetic reference archive")
    inventory.write_bytes(b"damaged inventory")
    with pytest.raises(ValueError, match="inventory checksum mismatch"):
        partitions.load_candidates(source, inventory, report)


def test_source_audit_requires_independent_geometry_agreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, policy = _case()
    selected, _ = partitions.select_partitions(rows, policy=policy)
    selection = {"policy": policy, "partitions": selected}
    ids = {identity for values in selected.values() for identity in values}
    by_id = {row["patch_id"]: row for row in rows}
    fresh = [{key: row[key] for key in ("patch_id", "epsg", "left", "bottom", "right", "top")}
             for identity, row in by_id.items() if identity in ids]
    monkeypatch.setattr(partitions, "metadata_identities", lambda path: set(by_id))
    monkeypatch.setattr(partitions, "reference_footprints", lambda *args, **kwargs: iter(fresh))
    result = partitions.audit_selection_from_source(
        selection, rows, tmp_path, policy=policy,
    )
    assert result["selected_reference_footprints_reloaded"] == sum(
        len(values) for values in selected.values()
    )
    assert result["independent_geometry_readers_agree"] is True
    fresh[0]["left"] += 1
    with pytest.raises(ValueError, match="source geometry differs"):
        partitions.audit_selection_from_source(selection, rows, tmp_path, policy=policy)
