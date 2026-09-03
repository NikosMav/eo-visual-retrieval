"""Print the BigEarthNet acquisition plan; optionally fetch checksum-verified metadata.

This command never downloads the 63.3 GB S2 image archive. The optional metadata
download is bounded to two allowlisted files of at most 8 MiB each.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eo_visual_retrieval.datasets.bigearthnet import acquisition_plan, download_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/downloads/bigearthnet-v2"))
    args = parser.parse_args()
    result = acquisition_plan()
    if args.download:
        try:
            result["verified_local_metadata"] = download_metadata(args.output_dir)
        except (OSError, ValueError) as error:
            parser.exit(1, f"Metadata acquisition failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
