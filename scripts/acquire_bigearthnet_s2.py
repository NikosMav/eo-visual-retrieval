"""Plan or run checksum-gated streaming acquisition of the frozen S2 selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eo_visual_retrieval.datasets.acquisition_io import STORAGE_LIMIT, StorageBudget
from eo_visual_retrieval.datasets.bigearthnet import S2_ARCHIVE_BYTES, S2_ARCHIVE_MD5, S2_BANDS
from eo_visual_retrieval.datasets.bigearthnet_s2 import (
    PILOT_NETWORK_BYTES,
    acquire,
    frozen_inputs,
    pilot_ids,
)
from eo_visual_retrieval.hashing import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "full"), default="pilot")
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
        "--pilot-root",
        type=Path,
        help="Verified pilot directory; mandatory before any full-phase request",
    )
    parser.add_argument(
        "--network-budget",
        type=int,
        default=PILOT_NETWORK_BYTES,
        help="Cumulative response-body byte reservations, including retries/replays",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=300,
        help="Attempt deadline, checked between members and HTTP reads",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--download", action="store_true", help="Execute; default only prints a plan"
    )
    args = parser.parse_args()
    footprints = frozen_inputs(args.selection, args.audit, args.inventory)
    binding = {
        "selection_sha256": file_sha256(args.selection),
        "audit_sha256": file_sha256(args.audit),
        "footprint_inventory_sha256": file_sha256(args.inventory),
        "source_md5": S2_ARCHIVE_MD5,
        "source_bytes": S2_ARCHIVE_BYTES,
        "selected_ids": sorted(footprints),
        "pilot_ids": pilot_ids(footprints),
    }
    # Count sibling acquisition directories too, so a second root cannot evade the ceiling.
    ancillary = [args.source_dir, args.selection.parent, args.inventory.parent]
    plan = {
        "phase": args.phase,
        "band_order": list(S2_BANDS),
        "pilot_ids": binding["pilot_ids"],
        "pilot_partition_counts": {"index": 10, "development": 10, "final": 10},
        "frozen_patch_count": len(footprints),
        "storage_ceiling_bytes": STORAGE_LIMIT,
        "network_budget": args.network_budget,
        "max_seconds": args.max_seconds,
        "published_checksum_scope": "entire compressed archive",
        "source_bytes_required_for_checksum": S2_ARCHIVE_BYTES,
        "bounded_prefix_can_authenticate_pilot": False,
        "resume_strategy": "replay from zero with cumulative byte reservations",
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
            phase=args.phase,
            network_budget=args.network_budget,
            resume=args.resume,
            max_seconds=args.max_seconds,
            pilot_root=args.pilot_root,
        )
    except (ValueError, OSError) as error:
        parser.exit(
            2,
            f"Acquisition stopped: {error}\nNo new completion is implied. "
            f"Inspect {args.root / 'state.json'} for executed evidence.\n",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
