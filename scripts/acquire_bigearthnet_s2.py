"""Plan, sample, or run one checksum-gated stream of the frozen S2 selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eo_visual_retrieval.datasets.acquisition_io import STORAGE_LIMIT, StorageBudget
from eo_visual_retrieval.datasets.bigearthnet import (
    REFERENCE_ARCHIVE_MD5,
    S2_ARCHIVE_BYTES,
    S2_ARCHIVE_MD5,
    S2_BANDS,
)
from eo_visual_retrieval.datasets.bigearthnet_s2 import (
    SAMPLE_NETWORK_BYTES,
    acquire,
    frozen_inputs,
)
from eo_visual_retrieval.hashing import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sample", "acquire"), default="acquire")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument(
        "--audit", type=Path, default=Path("docs/results/bigearthnet-selection-audit.json")
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing retained metadata/reference archive",
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="Dedicated staging directory under ignored data/"
    )
    parser.add_argument(
        "--network-budget",
        type=int,
        help="Cumulative response-body reservations including retries/replays",
    )
    parser.add_argument(
        "--max-seconds", type=float, help="Attempt deadline, checked between members and HTTP reads"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--download", action="store_true", help="Execute; default only prints a plan"
    )
    args = parser.parse_args()
    if args.resume and args.network_budget is None:
        parser.error("--resume requires an explicit cumulative --network-budget")
    footprints = frozen_inputs(args.selection, args.audit, args.inventory)
    binding = {
        "selection_sha256": file_sha256(args.selection),
        "audit_sha256": file_sha256(args.audit),
        "footprint_inventory_sha256": file_sha256(args.inventory),
        "reference_archive_md5": REFERENCE_ARCHIVE_MD5,
        "source_md5": S2_ARCHIVE_MD5,
        "source_bytes": S2_ARCHIVE_BYTES,
        "selected_ids": sorted(footprints),
    }
    network_budget = (
        args.network_budget
        if args.network_budget is not None
        else (SAMPLE_NETWORK_BYTES if args.mode == "sample" else S2_ARCHIVE_BYTES)
    )
    max_seconds = (
        args.max_seconds
        if args.max_seconds is not None
        else (120 if args.mode == "sample" else 24 * 60 * 60)
    )
    # Count sibling acquisition directories too, so another root cannot evade the ceiling.
    ancillary = [args.source_dir, args.selection.parent, args.inventory.parent]
    plan = {
        "mode": args.mode,
        "band_order": list(S2_BANDS),
        "frozen_patch_count": len(footprints),
        "storage_ceiling_bytes": STORAGE_LIMIT,
        "network_budget": network_budget,
        "max_seconds": max_seconds,
        "single_pass": True,
        "verify_every_selected_patch_inline": True,
        "abort_on_first_geometry_mismatch": True,
        "published_checksum_scope": "entire compressed archive",
        "source_bytes_required_for_checksum": S2_ARCHIVE_BYTES,
        "promotion_requires_complete_archive_md5": args.mode == "acquire",
        "process_restart": "replay from byte zero to preserve decompressor and whole-source digest",
    }
    if not args.download:
        # Read-only: use an existing ancestor for counting without creating the requested root.
        plan["existing_acquisition_bytes"] = StorageBudget(args.selection.parent, ancillary).used
        print(json.dumps(plan, indent=2))
        return
    print(json.dumps(plan, indent=2), flush=True)
    try:
        report = acquire(
            footprints,
            binding=binding,
            root=args.root,
            ancillary=ancillary,
            mode=args.mode,
            network_budget=network_budget,
            resume=args.resume,
            max_seconds=max_seconds,
        )
    except Exception as error:
        state_path = args.root / "state.json"
        stop = ""
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") == "geometry_mismatch":
                stop = (
                    f" Patch {state['geometry_mismatch_patch_id']} band "
                    f"{state['geometry_mismatch_band']} after compressed offset "
                    f"{state['failure_compressed_offset']}."
                )
        parser.exit(
            2,
            f"Acquisition stopped: {error}.{stop}\nNo new completion is implied. "
            f"Inspect {state_path} for executed evidence.\n",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
