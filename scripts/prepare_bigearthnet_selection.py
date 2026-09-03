"""Freeze deterministic acquisition IDs and independently audit source footprints."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from collections.abc import Mapping
from pathlib import Path

from eo_visual_retrieval.benchmarks.bigearthnet_partitions import (
    POLICY,
    audit_selection_from_source,
    load_candidates,
    select_partitions,
)
from eo_visual_retrieval.hashing import file_sha256


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    """Promote complete JSON only; a failed write must not freeze a partial selection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    import PIL
    import pyarrow
    import rasterio
    import zstandard

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/downloads/bigearthnet-v2"))
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-report", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = [path.resolve() for path in
             (args.inventory, args.inventory_report, args.selection, args.report)]
    if len(set(paths)) != len(paths) or any(
        path.is_relative_to(args.source_dir.resolve()) for path in paths[2:]
    ):
        parser.error("outputs must be separate from inputs and each other")
    if args.selection.exists():
        parser.error("selection already exists; use a separate path for a reproducibility check")
    candidates = load_candidates(args.source_dir, args.inventory, args.inventory_report)
    print(f"Loaded {len(candidates):,} recommended footprints", flush=True)
    partitions, feasibility = select_partitions(candidates)
    selection = {
        "schema": "bigearthnet-acquisition-selection-v1", "policy": POLICY,
        "footprint_inventory_sha256": file_sha256(args.inventory),
        "footprint_report_sha256": file_sha256(args.inventory_report),
        "partitions": partitions,
    }
    print("Selection complete; independently reloading selected source footprints", flush=True)
    audit = audit_selection_from_source(selection, candidates, args.source_dir)
    _write_new_json(args.selection, selection)
    report = {
        "schema": "bigearthnet-selection-audit-v1", "policy": POLICY,
        "selection_sha256": file_sha256(args.selection),
        "footprint_inventory_sha256": file_sha256(args.inventory),
        "footprint_report_sha256": file_sha256(args.inventory_report),
        "feasibility": feasibility, **audit,
        "runtime": {"python": platform.python_version(), "pyarrow": pyarrow.__version__,
                    "rasterio": rasterio.__version__, "pillow": PIL.__version__,
                    "zstandard": zstandard.__version__},
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _write_new_json(args.report, report)
    print(payload, end="")


if __name__ == "__main__":
    main()
