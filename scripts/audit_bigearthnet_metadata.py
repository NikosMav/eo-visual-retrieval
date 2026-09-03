"""Verify and inspect the local BigEarthNet metadata without downloading imagery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eo_visual_retrieval.datasets.bigearthnet_audit import audit_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/downloads/bigearthnet-v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.metadata_dir.resolve()):
        parser.error("audit output must be outside the source metadata directory")
    report = audit_metadata(args.metadata_dir)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(args.output)
    print(payload, end="")


if __name__ == "__main__":
    main()
