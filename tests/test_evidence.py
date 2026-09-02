"""Guard public evidence against malformed provenance hashes during report promotion."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def test_published_result_hashes_are_well_formed() -> None:
    root = Path(__file__).resolve().parents[1] / "docs" / "results"

    def check(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("_sha256"):
                    assert isinstance(item, str) and re.fullmatch(r"[0-9a-fA-F]{64}", item), (
                        f"invalid SHA-256 at {location}.{key}"
                    )
                check(item, f"{location}.{key}")
        elif isinstance(value, list):
            for position, item in enumerate(value):
                check(item, f"{location}[{position}]")

    reports = list(root.glob("*.json"))
    assert reports
    for report in reports:
        check(json.loads(report.read_text(encoding="utf-8")), report.name)
