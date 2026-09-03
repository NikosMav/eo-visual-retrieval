"""Bounded reference-map acquisition and source-derived footprint inventory.

No maps are extracted to disk. TIFF headers, rather than reference-map pixels,
provide the geometry. S2 imagery still needs its own source-integrity audit.
"""

from __future__ import annotations

import io
import math
import tarfile
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import urlopen

from eo_visual_retrieval.datasets.bigearthnet import (
    BIGEARTHNET_RECORD_URL,
    METADATA_ASSETS,
    REFERENCE_ARCHIVE_BYTES,
    REFERENCE_ARCHIVE_FILENAME,
    REFERENCE_ARCHIVE_MD5,
    verify_metadata_file,
)
from eo_visual_retrieval.hashing import file_md5, file_sha256

MAX_MAP_BYTES = 64 * 1024
MAX_WINDOW_BYTES = 16 * 1024 * 1024
MAX_INVENTORY_BYTES = 128 * 1024 * 1024
CELL_SIZE_M = 50_000


def verify_reference_archive(path: Path) -> None:
    if path.stat().st_size != REFERENCE_ARCHIVE_BYTES:
        raise ValueError("reference archive byte count mismatch")
    if file_md5(path) != REFERENCE_ARCHIVE_MD5:
        raise ValueError("reference archive checksum mismatch")


def download_reference_archive(directory: Path) -> Path:
    """One bounded attempt; verify a cached file or atomically promote a download."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / REFERENCE_ARCHIVE_FILENAME
    if destination.exists():
        verify_reference_archive(destination)
        return destination
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as file:
        temporary = Path(file.name)
    try:
        url = f"{BIGEARTHNET_RECORD_URL}/files/{REFERENCE_ARCHIVE_FILENAME}?download=1"
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            if response.status != 200:
                raise ValueError("reference archive requires a complete HTTP 200 response")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != REFERENCE_ARCHIVE_BYTES:
                raise ValueError("reference archive Content-Length mismatch")
            total = 0
            while chunk := response.read(min(1024 * 1024, REFERENCE_ARCHIVE_BYTES - total + 1)):
                total += len(chunk)
                if total > REFERENCE_ARCHIVE_BYTES:
                    raise ValueError("reference archive exceeds byte limit")
                output.write(chunk)
        verify_reference_archive(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def metadata_identities(directory: Path) -> set[str]:
    import pyarrow.parquet as pq

    identities: set[str] = set()
    for asset in METADATA_ASSETS:
        path = directory / asset.filename
        verify_metadata_file(path, asset)
        values = pq.read_table(path, columns=["patch_id"])["patch_id"].to_pylist()
        ids = set(values)
        if len(ids) != len(values) or ids & identities:
            raise ValueError("duplicate reference metadata identity")
        identities.update(ids)
    return identities


def _map_geometry(payload: bytes, patch_id: str) -> dict[str, Any]:
    from rasterio.io import MemoryFile

    with MemoryFile(payload) as memory, memory.open() as source:
        if (source.width, source.height, source.count, source.dtypes) != (
            120, 120, 1, ("uint16",)
        ):
            raise ValueError("unexpected reference-map dimensions or dtype")
        epsg = source.crs.to_epsg() if source.crs else None
        tile = patch_id.split("_T")[-1].split("_")[0]
        zone = int(tile[:2])
        expected_epsg = (32600 if tile[2] >= "N" else 32700) + zone
        if epsg != expected_epsg:
            raise ValueError("reference CRS disagrees with its MGRS zone")
        affine = source.transform
        if (affine.a, affine.b, affine.d, affine.e) != (10, 0, 0, -10):
            raise ValueError("unexpected reference-map pixel transform")
        bounds = list(source.bounds)
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("non-finite reference-map footprint")
        return dict(zip(
            ("patch_id", "epsg", "left", "bottom", "right", "top"),
            (patch_id, epsg, *bounds), strict=True,
        ))


def _tag_geometry(payload: bytes, patch_id: str) -> dict[str, Any]:
    """Read the source's strict GeoTIFF tag profile without a GDAL dataset per row.

    OGC GeoTIFF 19-008r4: pixel scale + tiepoint, PixelIsArea and projected EPSG.
    Unsupported profiles fail; selected maps are independently reopened with Rasterio.
    """
    from PIL import Image, TiffImagePlugin

    with Image.open(io.BytesIO(payload)) as source:
        if not isinstance(source, TiffImagePlugin.TiffImageFile):
            raise ValueError("reference map must be a TIFF")
        tags = source.tag_v2
        if (source.size, tags.get(258), tags.get(277), tags.get(339, (1,))) != (
            (120, 120), (16,), 1, (1,)
        ):
            raise ValueError("unexpected reference-map dimensions or dtype")
        if tags.get(33550) != (10.0, 10.0, 0.0) or 34264 in tags:
            raise ValueError("unsupported reference-map pixel transform")
        tie = tags.get(33922, ())
        if len(tie) != 6 or tie[:3] != (0.0, 0.0, 0.0) or tie[5] != 0:
            raise ValueError("unsupported reference-map tiepoint")
        keys = tags.get(34735, ())
        if len(keys) < 4 or keys[:3] not in ((1, 1, 0), (1, 1, 1)) or (
            len(keys) != 4 + 4 * keys[3]
        ):
            raise ValueError("invalid reference-map GeoKey directory")
        values = {keys[i]: tuple(keys[i + 1:i + 4]) for i in range(4, len(keys), 4)}
        tile = patch_id.split("_T")[-1].split("_")[0]
        epsg = (32600 if tile[2] >= "N" else 32700) + int(tile[:2])
        expected = {1024: (0, 1, 1), 1025: (0, 1, 1),
                    3072: (0, 1, epsg), 3076: (0, 1, 9001)}
        if len(values) != keys[3] or any(values.get(key) != value
                                        for key, value in expected.items()):
            raise ValueError("reference CRS or raster type disagrees with the required profile")
        left, top = float(tie[3]), float(tie[4])
        if not math.isfinite(left) or not math.isfinite(top):
            raise ValueError("non-finite reference-map footprint")
        return {"patch_id": patch_id, "epsg": epsg, "left": left, "bottom": top - 1200,
                "right": left + 1200, "top": top}


def reference_footprints(
    archive: Path, expected_ids: set[str], *, selected_ids: set[str] | None = None,
    progress: Callable[[int], None] | None = None,
    geometry_reader: Callable[[bytes, str], dict[str, Any]] = _map_geometry,
) -> Iterator[dict[str, Any]]:
    """Validate archive membership completely, parsing only requested TIFF headers.

    The checksum is checked before tar parsing. The iterator must be exhausted to
    establish complete metadata coverage, including when selecting a few maps.
    """
    import zstandard as zstd
    from rasterio import Env

    verify_reference_archive(archive)
    if selected_ids is not None and not selected_ids <= expected_ids:
        raise ValueError("selected reference ID is absent from metadata")
    remaining = set(expected_ids)
    with Env(), archive.open("rb") as compressed:
        header = compressed.read(18)
        if zstd.get_frame_parameters(header).window_size > MAX_WINDOW_BYTES:
            raise ValueError("reference decoder window exceeds limit")
        compressed.seek(0)
        decoder = zstd.ZstdDecompressor(max_window_size=MAX_WINDOW_BYTES)
        with decoder.stream_reader(compressed) as decoded, tarfile.open(
            fileobj=decoded, mode="r|"
        ) as bundle:
            for member in bundle:
                # Python 3.11 otherwise caches every TarInfo, even in forward-only mode.
                bundle.members.clear()  # type: ignore[attr-defined]
                parts = PurePosixPath(member.name).parts
                if "\\" in member.name or not parts or parts[0] != "Reference_Maps":
                    raise ValueError("unexpected reference-map member path")
                if ".." in parts or "." in member.name.split("/"):
                    raise ValueError("unsafe reference-map member path")
                if member.isdir() and len(parts) <= 3:
                    continue
                if not member.isfile() or len(parts) != 4:
                    raise ValueError("unexpected reference-map member type or path")
                patch_id = parts[2]
                if parts[1] != patch_id.rsplit("_", 2)[0] or parts[3] != (
                    patch_id + "_reference_map.tif"
                ):
                    raise ValueError("reference-map path does not match patch ID")
                if patch_id not in remaining:
                    raise ValueError("duplicate or unknown reference-map ID")
                remaining.remove(patch_id)
                if not 0 < member.size <= MAX_MAP_BYTES:
                    raise ValueError("reference-map member exceeds byte limit")
                if selected_ids is None or patch_id in selected_ids:
                    stream = bundle.extractfile(member)
                    if stream is None:
                        raise ValueError("reference-map payload missing")
                    with stream:
                        payload = stream.read(MAX_MAP_BYTES + 1)
                    if len(payload) != member.size:
                        raise ValueError("reference-map payload size mismatch")
                    yield geometry_reader(payload, patch_id)
                count = len(expected_ids) - len(remaining)
                if progress and count % 25_000 == 0:
                    progress(count)
    if remaining:
        raise ValueError(f"reference archive is missing {len(remaining)} metadata IDs")


def add_centers(rows: list[dict[str, Any]]) -> None:
    """Batch CRS transforms; retain native bounds so audits can recompute positions."""
    from rasterio.warp import transform

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["epsg"]].append(row)
    for epsg, group in groups.items():
        x = [(row["left"] + row["right"]) / 2 for row in group]
        y = [(row["bottom"] + row["top"]) / 2 for row in group]
        longitude, latitude = transform(epsg, 4326, x, y)
        cell_x, cell_y = transform(epsg, 6933, x, y)
        for row, lon, lat, cx, cy in zip(group, longitude, latitude, cell_x, cell_y, strict=True):
            if not all(math.isfinite(value) for value in (lon, lat, cx, cy)):
                raise ValueError("non-finite transformed reference footprint")
            row.update(longitude=lon, latitude=lat,
                       spatial_group=f"epsg6933:{math.floor(cx / CELL_SIZE_M)}:"
                                     f"{math.floor(cy / CELL_SIZE_M)}")


def build_inventory(
    directory: Path, output: Path, *, progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.resolve().is_relative_to(directory.resolve()):
        raise ValueError("inventory output must be outside the source directory")
    if output.exists():
        raise ValueError("inventory output already exists")
    archive = directory / REFERENCE_ARCHIVE_FILENAME
    ids = metadata_identities(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    crs_counts: Counter[str] = Counter()
    writer = None
    batch: list[dict[str, Any]] = []
    source = {
        "reference_archive_sha256": file_sha256(archive),
        "metadata_files": {a.filename: {"file_sha256": file_sha256(directory / a.filename)}
                           for a in METADATA_ASSETS},
    }
    try:
        for row in reference_footprints(
            archive, ids, progress=progress, geometry_reader=_tag_geometry,
        ):
            batch.append(row)
            crs_counts[str(row["epsg"])] += 1
            if len(batch) == 8192 or sum(crs_counts.values()) == len(ids):
                add_centers(batch)
                table = pa.Table.from_pylist(batch)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                batch.clear()
                if temporary.stat().st_size > MAX_INVENTORY_BYTES:
                    raise ValueError("footprint inventory exceeds byte limit")
        if writer is None:
            raise ValueError("empty footprint inventory")
        writer.close()
        writer = None
        if temporary.stat().st_size > MAX_INVENTORY_BYTES:
            raise ValueError("footprint inventory exceeds byte limit")
        report = {
            "schema": "bigearthnet-footprints-v1", **source,
            "reference_archive_bytes": archive.stat().st_size,
            "reference_archive_md5": REFERENCE_ARCHIVE_MD5,
            "inventory_sha256": file_sha256(temporary),
            "inventory_bytes": temporary.stat().st_size,
            "patches": len(ids), "crs_counts": dict(sorted(crs_counts.items())),
            "geometry": "120 x 120 pixels, 10 m, north-up UTM, 1200 m square",
            "inventory_reader": "Pillow GeoTIFF tags; selected maps independently read by Rasterio",
            "reference_pixels_read": False, "s2_subset_footprints_verified": False,
        }
        temporary.replace(output)
        return report
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
