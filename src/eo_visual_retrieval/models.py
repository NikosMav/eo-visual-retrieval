"""Serializable records used at the data and embedding boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Split = Literal["index", "query"]


@dataclass(frozen=True)
class ImageRecord:
    """One local image that can be embedded and retrieved."""

    item_id: str
    path: str
    split: Split
    label: str | None = None
    source: str = "local"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("item_id must not be empty")
        if not self.path:
            raise ValueError("path must not be empty")
        if self.split not in {"index", "query"}:
            raise ValueError("split must be 'index' or 'query'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "path": self.path,
            "split": self.split,
            "label": self.label,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ImageRecord:
        return cls(
            item_id=str(value["item_id"]),
            path=str(value["path"]),
            split=value["split"],
            label=None if value.get("label") is None else str(value["label"]),
            source=str(value.get("source", "local")),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class StacItemRecord:
    """Stable, sanitized identity and metadata for one STAC item.

    Asset HREFs are intentionally excluded because providers may sign them with
    temporary credentials.
    """

    api_url: str
    collection: str
    item_id: str
    bbox: tuple[float, float, float, float] | None
    datetime: str | None
    asset_keys: tuple[str, ...]
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_url": self.api_url,
            "collection": self.collection,
            "item_id": self.item_id,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "datetime": self.datetime,
            "asset_keys": list(self.asset_keys),
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StacItemRecord:
        raw_bbox = value.get("bbox")
        bbox = tuple(float(item) for item in raw_bbox) if raw_bbox is not None else None
        if bbox is not None and len(bbox) != 4:
            raise ValueError("STAC item bbox must contain four coordinates")
        return cls(
            api_url=str(value["api_url"]),
            collection=str(value["collection"]),
            item_id=str(value["item_id"]),
            bbox=bbox,
            datetime=None if value.get("datetime") is None else str(value["datetime"]),
            asset_keys=tuple(str(item) for item in value.get("asset_keys", [])),
            properties=dict(value.get("properties", {})),
        )
