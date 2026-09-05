"""Serve an explicit, immutable snapshot of public evidence, never arbitrary files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eo_visual_retrieval.app.findings import DISPLAY_NAMES, reduce_reports
from eo_visual_retrieval.hashing import bytes_sha256

EUROSAT = {
    **{f"{key}-k10.json": name for key, name in DISPLAY_NAMES.items()
       if key != "eurosat-v1-terramind-tiny"},
    "terramind-v1.json": "TerraMind",
}
TEMPORAL = {
    "temporal-v1g-pca-32-k1.json": "PCA-32",
    "temporal-v1g-dinov2-vits14-k1.json": "DINOv2",
    "temporal-v1g-remoteclip-vit-b32-k1.json": "RemoteCLIP",
}
EXTRA = (
    "eurosat-v1-analysis.json", "temporal-availability-2024.json",
    "temporal-v1-guarded-split.json", "multimodal-temporal-provenance.json",
    "multimodal-v1-smoke.json", "temporal-v1-pca-32-k1.json",
    "temporal-v1-dinov2-vits14-k1.json",
)


class Evidence:
    def __init__(self, directory: Path | None) -> None:
        self.files: dict[str, bytes] = {}
        self.reports: dict[str, Any] = {}
        self.analysis = None
        if directory is not None:
            if not directory.is_dir():
                raise ValueError("evidence directory does not exist")
            for name in (*EUROSAT, *TEMPORAL, *EXTRA):
                path = directory / name
                if path.exists():
                    if not path.resolve().is_relative_to(directory.resolve()):
                        raise ValueError("evidence resolves outside its directory")
                    raw = path.read_bytes()
                    report = json.loads(raw)
                    if not isinstance(report, dict):
                        raise ValueError(f"evidence must be a JSON object: {name}")
                    self.files[name], self.reports[name] = raw, report
            if all(name in self.files for name in EXTRA[:2]):
                self.analysis = reduce_reports(self.reports[EXTRA[0]], self.reports[EXTRA[1]])

    def payload(self) -> dict[str, Any]:
        def rows(names: dict[str, str]) -> list[dict[str, Any]]:
            output = []
            for name, model in names.items():
                if name in self.reports:
                    raw = self.reports[name]
                    metrics = raw.get("metrics", raw)
                    output.append({"model": model, "source": name, **metrics})
            return output

        temporal = rows(TEMPORAL)
        places = sorted({place for row in temporal for place in row["per_class"]})
        return {
            "eurosat": rows(EUROSAT), "temporal": temporal, "places": places,
            "analysis": self.analysis,
            "sources": [{"name": name, "sha256": bytes_sha256(raw)}
                        for name, raw in sorted(self.files.items())],
        }
