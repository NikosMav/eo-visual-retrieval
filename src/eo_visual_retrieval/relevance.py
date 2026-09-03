"""Multi-label judgments and evaluation partitions, separate from embedding storage."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

EvaluationPartition = Literal["index", "development", "final"]
RELEVANCE_SCHEMA = "eo-multilabel-relevance-v1"


@dataclass(frozen=True)
class RelevanceRecord:
    item_id: str
    labels: tuple[str, ...]
    partition: EvaluationPartition

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id.strip():
            raise ValueError("relevance item_id must be a non-empty string")
        if not isinstance(self.partition, str) or self.partition not in {
            "index", "development", "final"
        }:
            raise ValueError("partition must be index, development, or final")
        if not isinstance(self.labels, tuple) or any(
            not isinstance(label, str) or not label.strip() for label in self.labels
        ):
            raise ValueError("labels must be a tuple of non-empty strings")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must not contain duplicates")
        if self.partition == "index" and not self.labels:
            raise ValueError("index items must have relevance labels")
        object.__setattr__(self, "labels", tuple(sorted(self.labels)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "labels": sorted(self.labels),
            "partition": self.partition,
        }


@dataclass(frozen=True)
class RelevanceManifest:
    dataset: str
    image_manifest_sha256: str
    records: tuple[RelevanceRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ValueError("dataset must be a non-empty string")
        if not isinstance(self.image_manifest_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.image_manifest_sha256
        ):
            raise ValueError("image_manifest_sha256 must be a lowercase SHA-256")
        if not self.records:
            raise ValueError("relevance manifest must contain records")
        if len({record.item_id for record in self.records}) != len(self.records):
            raise ValueError("relevance item IDs must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RELEVANCE_SCHEMA,
            "dataset": self.dataset,
            "image_manifest_sha256": self.image_manifest_sha256,
            "records": [record.to_dict() for record in self.records],
        }

    def save(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output)

    @classmethod
    def load(cls, path: Path) -> RelevanceManifest:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != RELEVANCE_SCHEMA:
            raise ValueError(f"unsupported relevance schema; expected {RELEVANCE_SCHEMA}")
        if not isinstance(value.get("records"), list):
            raise ValueError("relevance records must be a list")
        records: list[RelevanceRecord] = []
        for item in value["records"]:
            if not isinstance(item, dict) or not isinstance(item.get("labels"), list):
                raise ValueError("each relevance record must contain a labels list")
            if not {"item_id", "partition"}.issubset(item):
                raise ValueError("relevance record lacks item_id or partition")
            records.append(
                RelevanceRecord(
                    item_id=item["item_id"],
                    labels=tuple(item["labels"]),
                    partition=item["partition"],
                )
            )
        if not {"dataset", "image_manifest_sha256"}.issubset(value):
            raise ValueError("relevance manifest lacks dataset or image-manifest hash")
        return cls(
            dataset=value["dataset"],
            image_manifest_sha256=value["image_manifest_sha256"],
            records=tuple(records),
        )
