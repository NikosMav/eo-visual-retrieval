"""Build the local footprint inventory from the small reference-map archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eo_visual_retrieval.datasets.bigearthnet_footprints import (
    build_inventory,
    download_reference_archive,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/downloads/bigearthnet-v2"))
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--download-reference", action="store_true")
    args = parser.parse_args()
    if args.report.resolve().is_relative_to(args.source_dir.resolve()) or (
        args.report.resolve() == args.inventory.resolve()
    ):
        parser.error("report must be separate from source files and inventory")
    if args.download_reference:
        download_reference_archive(args.source_dir)
    result = build_inventory(
        args.source_dir, args.inventory,
        progress=lambda n: print(f"Checked {n:,} maps", flush=True),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.report.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")


if __name__ == "__main__":
    main()
