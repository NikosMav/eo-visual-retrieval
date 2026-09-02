"""Windowed, georeferenced Sentinel-2 RGB chip materialization."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from numpy.typing import NDArray

from eo_visual_retrieval.hashing import file_sha256
from eo_visual_retrieval.manifests import write_jsonl
from eo_visual_retrieval.models import ImageRecord, StacItemRecord

RGB_ASSET_KEYS = ("B04", "B03", "B02")
DEFAULT_MASKED_SCL_CLASSES = (0, 1, 3, 7, 8, 9, 10, 11)
REFLECTANCE_NODATA = -9999.0


@dataclass(frozen=True)
class ChipArtifacts:
    """Files and manifest record produced for one STAC item."""

    reflectance_path: Path
    rgb_path: Path
    image_record: ImageRecord


def sentinel2_reflectance_parameters(processing_baseline: str) -> tuple[float, float]:
    """Return scale and offset for Sentinel-2 L2A BOA reflectance."""

    try:
        baseline = Decimal(processing_baseline)
    except InvalidOperation as error:
        raise ValueError("invalid Sentinel-2 processing baseline") from error
    if baseline <= 0:
        raise ValueError("Sentinel-2 processing baseline must be positive")
    offset = -0.1 if baseline >= Decimal("4.00") else 0.0
    return 0.0001, offset


def _validate_bounds(bounds: tuple[float, float, float, float]) -> None:
    west, south, east, north = bounds
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("chip bounds must contain finite coordinates")
    if west >= east or south >= north:
        raise ValueError("chip bounds must satisfy west < east and south < north")


def _safe_stem(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", item_id).strip(".-") or "item"


def _aligned_window(
    reference: Any,
    bounds: tuple[float, float, float, float],
    bounds_crs: str,
) -> tuple[Any, Any, int, int]:
    from rasterio.errors import WindowError
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    if reference.crs is None:
        raise ValueError("reference band does not declare a CRS")
    transformed = transform_bounds(bounds_crs, reference.crs, *bounds, densify_pts=21)
    requested = from_bounds(*transformed, transform=reference.transform)
    col_start = max(0, math.floor(requested.col_off))
    row_start = max(0, math.floor(requested.row_off))
    col_stop = min(reference.width, math.ceil(requested.col_off + requested.width))
    row_stop = min(reference.height, math.ceil(requested.row_off + requested.height))
    if col_start >= col_stop or row_start >= row_stop:
        raise ValueError("chip bounds do not overlap the reference band")
    window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
    try:
        window = window.intersection(Window(0, 0, reference.width, reference.height))
    except WindowError as error:
        raise ValueError("chip bounds do not overlap the reference band") from error
    width = int(window.width)
    height = int(window.height)
    return window, reference.window_transform(window), width, height


def _read_on_reference_grid(
    source: str | Path,
    *,
    reference_crs: Any,
    reference_transform: Any,
    reference_width: int,
    reference_height: int,
    window: Any,
    categorical: bool,
) -> Any:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    with rasterio.open(source) as dataset:
        if dataset.count < 1:
            raise ValueError("source raster contains no bands")
        if dataset.crs is None:
            raise ValueError("source raster does not declare a CRS")
        options: dict[str, Any] = {
            "crs": reference_crs,
            "transform": reference_transform,
            "width": reference_width,
            "height": reference_height,
            "resampling": Resampling.nearest if categorical else Resampling.bilinear,
        }
        if dataset.nodata is not None:
            options.update(src_nodata=dataset.nodata, nodata=dataset.nodata)
        with WarpedVRT(dataset, **options) as aligned:
            return aligned.read(1, window=window, masked=True)


def _write_chip_files(
    *,
    reflectance: NDArray[np.float32],
    rgb: NDArray[np.uint8],
    valid_mask: NDArray[np.bool_],
    crs: Any,
    transform: Any,
    reflectance_path: Path,
    rgb_path: Path,
    tags: dict[str, str],
) -> None:
    import rasterio
    from rasterio.enums import ColorInterp

    reflectance_temporary = reflectance_path.with_suffix(".tmp.tif")
    rgb_temporary = rgb_path.with_suffix(".tmp.tif")
    for temporary in (reflectance_temporary, rgb_temporary):
        temporary.unlink(missing_ok=True)

    height, width = valid_mask.shape
    common_profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 3,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "interleave": "pixel",
    }
    try:
        with rasterio.open(
            reflectance_temporary,
            "w",
            dtype="float32",
            nodata=REFLECTANCE_NODATA,
            **common_profile,
        ) as dataset:
            dataset.write(np.where(valid_mask, reflectance, REFLECTANCE_NODATA))
            dataset.write_mask(valid_mask.astype(np.uint8) * 255)
            dataset.descriptions = (
                "red BOA reflectance",
                "green BOA reflectance",
                "blue BOA reflectance",
            )
            dataset.update_tags(**tags, artifact="boa-reflectance")

        with rasterio.open(rgb_temporary, "w", dtype="uint8", **common_profile) as dataset:
            dataset.write(rgb)
            dataset.write_mask(valid_mask.astype(np.uint8) * 255)
            dataset.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            dataset.descriptions = ("red", "green", "blue")
            dataset.update_tags(**tags, artifact="model-ready-rgb")

        reflectance_temporary.replace(reflectance_path)
        rgb_temporary.replace(rgb_path)
    except Exception:
        reflectance_temporary.unlink(missing_ok=True)
        rgb_temporary.unlink(missing_ok=True)
        raise


def build_sentinel2_chip(
    sources: Mapping[str, str | Path],
    *,
    output_dir: Path,
    item_id: str,
    api_url: str,
    collection: str,
    datetime: str | None,
    bounds: tuple[float, float, float, float],
    processing_baseline: str,
    bounds_crs: str = "EPSG:4326",
    reflectance_min: float = 0.0,
    reflectance_max: float = 0.3,
    mask_scl: bool = True,
    max_pixels: int = 1024 * 1024,
) -> ChipArtifacts:
    """Build reflectance and model-ready RGB artifacts from local or remote bands."""

    try:
        import rasterio
        from rasterio.transform import array_bounds
    except ImportError as error:
        message = 'geospatial chip support is optional; install with pip install -e ".[geo]"'
        raise RuntimeError(message) from error

    _validate_bounds(bounds)
    if reflectance_min >= reflectance_max:
        raise ValueError("reflectance_min must be less than reflectance_max")
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")
    missing = [key for key in RGB_ASSET_KEYS if key not in sources]
    if mask_scl and "SCL" not in sources:
        missing.append("SCL")
    if missing:
        raise ValueError(f"missing Sentinel-2 assets: {', '.join(missing)}")

    scale, offset = sentinel2_reflectance_parameters(processing_baseline)
    with rasterio.open(sources["B04"]) as reference:
        window, transform, width, height = _aligned_window(reference, bounds, bounds_crs)
        if width * height > max_pixels:
            raise ValueError(f"chip contains {width * height} pixels; limit is {max_pixels}")
        reference_crs = reference.crs
        reference_transform = reference.transform
        reference_width = reference.width
        reference_height = reference.height

    band_values: list[NDArray[np.float32]] = []
    invalid = np.zeros((height, width), dtype=np.bool_)
    for asset_key in RGB_ASSET_KEYS:
        band = _read_on_reference_grid(
            sources[asset_key],
            reference_crs=reference_crs,
            reference_transform=reference_transform,
            reference_width=reference_width,
            reference_height=reference_height,
            window=window,
            categorical=False,
        )
        values = np.asarray(band.filled(0), dtype=np.float32)
        invalid |= np.ma.getmaskarray(band) | (values == 0)
        band_values.append(values)

    if mask_scl:
        scl = _read_on_reference_grid(
            sources["SCL"],
            reference_crs=reference_crs,
            reference_transform=reference_transform,
            reference_width=reference_width,
            reference_height=reference_height,
            window=window,
            categorical=True,
        )
        scl_values = np.asarray(scl.filled(0), dtype=np.uint8)
        invalid |= np.ma.getmaskarray(scl)
        invalid |= np.isin(scl_values, DEFAULT_MASKED_SCL_CLASSES)

    reflectance = np.asarray(np.stack(band_values) * scale + offset, dtype=np.float32)
    valid_mask = ~invalid
    normalized = np.clip(
        (reflectance - reflectance_min) / (reflectance_max - reflectance_min),
        0.0,
        1.0,
    )
    rgb = np.asarray(np.rint(normalized * 255.0), dtype=np.uint8)
    rgb[:, ~valid_mask] = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(item_id)
    reflectance_path = output_dir / f"{stem}-reflectance.tif"
    rgb_path = output_dir / f"{stem}-rgb.tif"
    output_bounds = array_bounds(height, width, transform)
    tags = {
        "source_api_url": api_url,
        "source_collection": collection,
        "source_item_id": item_id,
        "source_datetime": datetime or "",
        "source_assets": json.dumps(RGB_ASSET_KEYS),
        "bounds_crs": bounds_crs,
        "requested_bounds": json.dumps(bounds),
        "processing_baseline": processing_baseline,
        "reflectance_scale": str(scale),
        "reflectance_offset": str(offset),
        "rgb_reflectance_min": str(reflectance_min),
        "rgb_reflectance_max": str(reflectance_max),
        "scl_mask_enabled": str(mask_scl).lower(),
        "scl_masked_classes": json.dumps(DEFAULT_MASKED_SCL_CLASSES if mask_scl else ()),
    }
    _write_chip_files(
        reflectance=reflectance,
        rgb=rgb,
        valid_mask=valid_mask,
        crs=reference_crs,
        transform=transform,
        reflectance_path=reflectance_path,
        rgb_path=rgb_path,
        tags=tags,
    )

    metadata = {
        "api_url": api_url,
        "collection": collection,
        "datetime": datetime,
        "bbox": list(bounds),
        "bbox_crs": bounds_crs,
        "crs": str(reference_crs),
        "transform": list(transform)[:6],
        "bounds": list(output_bounds),
        "width": width,
        "height": height,
        "gsd": [abs(transform.a), abs(transform.e)],
        "assets": list(RGB_ASSET_KEYS),
        "processing_baseline": processing_baseline,
        "reflectance_scale": scale,
        "reflectance_offset": offset,
        "reflectance_min": reflectance_min,
        "reflectance_max": reflectance_max,
        "scl_masked_classes": list(DEFAULT_MASKED_SCL_CLASSES if mask_scl else ()),
        "reflectance_path": reflectance_path.name,
        "reflectance_sha256": file_sha256(reflectance_path),
        "sha256": file_sha256(rgb_path),
    }
    image_record = ImageRecord(
        item_id=item_id,
        path=rgb_path.name,
        split="index",
        source="stac-sentinel2-chip",
        metadata=metadata,
    )
    serialized = json.dumps(image_record.to_dict()).lower()
    if "href" in serialized or "token" in serialized or "signature" in serialized:
        raise ValueError("chip manifest metadata contains a forbidden access field")
    return ChipArtifacts(
        reflectance_path=reflectance_path,
        rgb_path=rgb_path,
        image_record=image_record,
    )


def materialize_sentinel2_chip(
    record: StacItemRecord,
    *,
    output_dir: Path,
    image_manifest: Path,
    bounds: tuple[float, float, float, float],
    signer: str = "none",
    reflectance_min: float = 0.0,
    reflectance_max: float = 0.3,
    mask_scl: bool = True,
    max_pixels: int = 1024 * 1024,
) -> ChipArtifacts:
    """Resolve one STAC item and materialize a bounded Sentinel-2 chip."""

    try:
        from pystac_client import Client
    except ImportError as error:
        message = 'STAC support is optional; install with pip install -e ".[stac,geo]"'
        raise RuntimeError(message) from error

    if signer not in {"none", "planetary-computer"}:
        raise ValueError("signer must be 'none' or 'planetary-computer'")
    if record.collection != "sentinel-2-l2a":
        raise ValueError("stac-chip currently supports only the sentinel-2-l2a collection")

    client = Client.open(record.api_url)
    collection = client.get_collection(record.collection)
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

    required = (*RGB_ASSET_KEYS, "SCL") if mask_scl else RGB_ASSET_KEYS
    missing = [key for key in required if key not in item.assets]
    if missing:
        raise ValueError(f"STAC item is missing required assets: {', '.join(missing)}")
    sources: dict[str, str] = {}
    for key in required:
        href = item.assets[key].href
        parsed = urlparse(href)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError(f"STAC asset must use HTTPS without URL userinfo: {key}")
        sources[key] = href

    processing_baseline = item.properties.get("s2:processing_baseline")
    if processing_baseline is None:
        raise ValueError("STAC item does not declare s2:processing_baseline")
    try:
        artifacts = build_sentinel2_chip(
            sources,
            output_dir=output_dir,
            item_id=record.item_id,
            api_url=record.api_url,
            collection=record.collection,
            datetime=record.datetime,
            bounds=bounds,
            processing_baseline=str(processing_baseline),
            reflectance_min=reflectance_min,
            reflectance_max=reflectance_max,
            mask_scl=mask_scl,
            max_pixels=max_pixels,
        )
    except (RuntimeError, ValueError):
        raise
    except Exception as error:
        raise RuntimeError(f"failed to materialize Sentinel-2 chip: {record.item_id}") from error
    write_jsonl([artifacts.image_record], image_manifest)
    return artifacts
