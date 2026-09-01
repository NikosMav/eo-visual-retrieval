"""Bounded, provider-neutral STAC discovery."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from eo_visual_retrieval.manifests import write_jsonl
from eo_visual_retrieval.models import ImageRecord, StacItemRecord

SAFE_PROPERTY_KEYS = (
    "constellation",
    "eo:cloud_cover",
    "gsd",
    "instruments",
    "platform",
    "proj:epsg",
    "s2:mgrs_tile",
    "s2:processing_baseline",
)


@dataclass(frozen=True)
class StacSearchConfig:
    api_url: str
    collection: str
    bbox: tuple[float, float, float, float]
    datetime: str
    limit: int = 20
    max_cloud_cover: float | None = None

    def __post_init__(self) -> None:
        west, south, east, north = self.bbox
        if not (-180 <= west < east <= 180):
            raise ValueError("bbox longitudes must satisfy -180 <= west < east <= 180")
        if not (-90 <= south < north <= 90):
            raise ValueError("bbox latitudes must satisfy -90 <= south < north <= 90")
        if not self.api_url.startswith("https://"):
            raise ValueError("api_url must use HTTPS")
        if not self.collection:
            raise ValueError("collection must not be empty")
        if not self.datetime:
            raise ValueError("datetime must not be empty")
        if not 1 <= self.limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if self.max_cloud_cover is not None and not 0 <= self.max_cloud_cover <= 100:
            raise ValueError("max_cloud_cover must be between 0 and 100")


def search_stac(config: StacSearchConfig) -> list[StacItemRecord]:
    """Search a STAC API without signing or persisting asset HREFs."""

    try:
        from pystac_client import Client
    except ImportError as error:
        message = 'STAC support is optional; install with pip install -e ".[stac]"'
        raise RuntimeError(message) from error

    query: dict[str, Any] | None = None
    if config.max_cloud_cover is not None:
        query = {"eo:cloud_cover": {"lt": config.max_cloud_cover}}

    catalog = Client.open(config.api_url)
    search = catalog.search(
        collections=[config.collection],
        bbox=list(config.bbox),
        datetime=config.datetime,
        query=query,
        max_items=config.limit,
    )
    items = search.item_collection()

    records: list[StacItemRecord] = []
    for item in items:
        item_collection = item.collection_id or config.collection
        item_datetime = item.datetime.isoformat() if item.datetime is not None else None
        item_bbox = tuple(float(value) for value in item.bbox) if item.bbox else None
        if item_bbox is not None and len(item_bbox) != 4:
            item_bbox = None
        properties = {
            key: item.properties[key]
            for key in SAFE_PROPERTY_KEYS
            if key in item.properties
        }
        records.append(
            StacItemRecord(
                api_url=config.api_url,
                collection=item_collection,
                item_id=item.id,
                bbox=item_bbox,
                datetime=item_datetime,
                asset_keys=tuple(sorted(item.assets)),
                properties=properties,
            )
        )

    return sorted(records, key=lambda record: (record.datetime or "", record.item_id))


def write_stac_jsonl(records: list[StacItemRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            serialized = record.to_dict()
            if "href" in json.dumps(serialized).lower():
                raise ValueError("sanitized STAC records must not contain asset HREFs")
            stream.write(json.dumps(serialized, sort_keys=True) + "\n")
    temporary.replace(output)


def read_stac_jsonl(path: Path) -> list[StacItemRecord]:
    records: list[StacItemRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(StacItemRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid STAC manifest record at line {line_number}") from error
    if not records:
        raise ValueError(f"STAC manifest is empty: {path}")
    return records


def _safe_filename(item_id: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", item_id).strip(".-") or "item"
    return f"{stem}{suffix}"


def _image_suffix(content_type: str) -> str:
    normalized = content_type.split(";", maxsplit=1)[0].strip().lower()
    suffixes = {
        "application/geotiff": ".tif",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/tiff": ".tif",
        "image/webp": ".webp",
    }
    if normalized not in suffixes:
        raise ValueError(f"asset response is not a supported image: {normalized or 'unknown'}")
    return suffixes[normalized]


def materialize_previews(
    records: list[StacItemRecord],
    *,
    output_dir: Path,
    image_manifest: Path,
    asset_key: str = "rendered_preview",
    signer: str = "none",
    max_bytes: int = 20 * 1024 * 1024,
) -> list[ImageRecord]:
    """Download bounded preview images while keeping asset URLs in memory only."""

    try:
        import requests
        from pystac_client import Client
    except ImportError as error:
        message = 'STAC support is optional; install with pip install -e ".[stac]"'
        raise RuntimeError(message) from error

    if signer not in {"none", "planetary-computer"}:
        raise ValueError("signer must be 'none' or 'planetary-computer'")
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=3))
    clients: dict[str, Any] = {}
    collections: dict[tuple[str, str], Any] = {}
    images: list[ImageRecord] = []

    for record in records:
        if record.api_url not in clients:
            clients[record.api_url] = Client.open(record.api_url)
        client = clients[record.api_url]
        collection_key = (record.api_url, record.collection)
        if collection_key not in collections:
            collections[collection_key] = client.get_collection(record.collection)
        collection = collections[collection_key]
        item = collection.get_item(record.item_id)
        if item is None:
            raise ValueError(f"STAC item no longer exists: {record.collection}/{record.item_id}")
        if signer == "planetary-computer":
            try:
                import planetary_computer
            except ImportError as error:
                message = "Planetary Computer signing requires the stac dependency group"
                raise RuntimeError(message) from error
            item = planetary_computer.sign(item)
        if asset_key not in item.assets:
            raise ValueError(f"STAC item does not expose asset '{asset_key}': {record.item_id}")

        asset = item.assets[asset_key]
        href = asset.href
        parsed = urlparse(href)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError(f"asset must use HTTPS without URL userinfo: {record.item_id}")

        try:
            with session.get(href, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                content_length = int(response.headers.get("content-length", 0))
                if content_length > max_bytes:
                    raise ValueError(
                        f"asset exceeds the {max_bytes}-byte limit: {record.item_id}"
                    )
                response_type = response.headers.get("content-type", "")
                if response_type.split(";", maxsplit=1)[0] == "application/octet-stream":
                    content_type = asset.media_type or response_type
                else:
                    content_type = response_type or asset.media_type or ""
                suffix = _image_suffix(content_type)
                destination = output_dir / _safe_filename(record.item_id, suffix)
                temporary = destination.with_suffix(destination.suffix + ".part")
                downloaded = 0
                try:
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            downloaded += len(chunk)
                            if downloaded > max_bytes:
                                message = (
                                    f"asset exceeds the {max_bytes}-byte limit: {record.item_id}"
                                )
                                raise ValueError(message)
                            stream.write(chunk)
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise
                temporary.replace(destination)
        except requests.RequestException as error:
            message = f"failed to download STAC preview after retries: {record.item_id}"
            raise RuntimeError(message) from error

        images.append(
            ImageRecord(
                item_id=record.item_id,
                path=destination.name,
                split="index",
                source="stac-preview",
                metadata={
                    "api_url": record.api_url,
                    "collection": record.collection,
                    "datetime": record.datetime,
                    "bbox": list(record.bbox) if record.bbox else None,
                    "asset_key": asset_key,
                    "signer": signer,
                },
            )
        )

    write_jsonl(images, image_manifest)
    return images
