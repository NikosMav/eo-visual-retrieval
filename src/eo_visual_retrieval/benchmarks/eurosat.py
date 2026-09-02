"""Prepare a bounded, spatially separated EuroSAT retrieval benchmark."""

from __future__ import annotations

import hashlib
import math
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eo_visual_retrieval.datasets.eurosat import (
    EUROSAT_ARCHIVE,
    EUROSAT_ARCHIVE_MD5,
    EUROSAT_CLASSES,
    EUROSAT_CLASSES_SET,
    EUROSAT_DOI,
    EUROSAT_SOURCE,
    verify_archive,
)
from eo_visual_retrieval.hashing import file_md5
from eo_visual_retrieval.manifests import write_jsonl
from eo_visual_retrieval.models import ImageRecord, Split

__all__ = [
    "EUROSAT_ARCHIVE",
    "EUROSAT_ARCHIVE_MD5",
    "EUROSAT_CLASSES",
    "EUROSAT_DOI",
    "EUROSAT_SOURCE",
    "EuroSatAudit",
    "EuroSatCandidate",
    "EuroSatPreparation",
    "EuroSatSplit",
    "audit_eurosat_manifest",
    "discover_candidates",
    "file_md5",
    "prepare_eurosat_benchmark",
    "select_spatial_split",
]

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class EuroSatCandidate:
    """Geographic identity discovered from one archive member."""

    member: str
    label: str
    source_crs: str
    source_bounds: tuple[float, float, float, float]
    longitude: float
    latitude: float
    equal_area_x: float
    equal_area_y: float
    spatial_group: str


@dataclass(frozen=True)
class EuroSatSplit:
    """Selected candidates and audited spatial separation."""

    index: tuple[EuroSatCandidate, ...]
    query: tuple[EuroSatCandidate, ...]
    minimum_separation_m: float
    excluded_near_query: int


@dataclass(frozen=True)
class EuroSatPreparation:
    """Artifacts and counts produced by benchmark preparation."""

    records: tuple[ImageRecord, ...]
    discovered: int
    minimum_separation_m: float
    excluded_near_query: int


@dataclass(frozen=True)
class EuroSatAudit:
    """Independent checks over a prepared manifest and its optional files."""

    items: int
    index: int
    query: int
    per_class: dict[str, dict[str, int]]
    spatial_groups: int
    minimum_separation_m: float
    manifest_sha256: str
    verified_files: int

    def to_dict(self) -> dict[str, object]:
        return {
            "items": self.items,
            "index": self.index,
            "query": self.query,
            "per_class": self.per_class,
            "spatial_groups": self.spatial_groups,
            "minimum_separation_km": self.minimum_separation_m / 1000,
            "manifest_sha256": self.manifest_sha256,
            "verified_files": self.verified_files,
        }


def _bounds4(values: Iterable[float]) -> tuple[float, float, float, float]:
    """Narrow an iterable of coordinates to the fixed-width bounds contract."""

    west, south, east, north = (float(value) for value in values)
    return west, south, east, north


def _rank(seed: int, namespace: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).digest()


def _spatial_group(x: float, y: float, group_size_m: float) -> str:
    return f"epsg6933:{math.floor(x / group_size_m)}:{math.floor(y / group_size_m)}"


def discover_candidates(
    archive: Path,
    *,
    group_size_m: float = 50_000.0,
) -> list[EuroSatCandidate]:
    """Read georeferencing from every official EuroSAT multispectral patch."""

    if group_size_m <= 0:
        raise ValueError("group_size_m must be positive")
    verify_archive(archive, expected_md5=None)

    try:
        from rasterio.io import MemoryFile
        from rasterio.warp import transform
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("EuroSAT preparation requires the 'geo' dependency group") from error

    candidates: list[EuroSatCandidate] = []
    with zipfile.ZipFile(archive) as bundle:
        members = sorted(
            member
            for member in bundle.namelist()
            if member.lower().endswith((".tif", ".tiff"))
        )
        for member in members:
            parts = Path(member).parts
            if len(parts) < 3 or parts[-2] not in EUROSAT_CLASSES_SET:
                continue
            label = parts[-2]
            with MemoryFile(bundle.read(member)) as memory_file, memory_file.open() as dataset:
                if dataset.crs is None:
                    raise ValueError(f"archive member has no CRS: {member}")
                if dataset.count < 4:
                    raise ValueError(f"archive member has fewer than four bands: {member}")
                bounds = _bounds4(dataset.bounds)
                center_x = (bounds[0] + bounds[2]) / 2
                center_y = (bounds[1] + bounds[3]) / 2
                longitude, latitude = transform(dataset.crs, "EPSG:4326", [center_x], [center_y])
                equal_x, equal_y = transform(dataset.crs, "EPSG:6933", [center_x], [center_y])
                source_crs = dataset.crs.to_string()
            candidates.append(
                EuroSatCandidate(
                    member=member,
                    label=label,
                    source_crs=source_crs,
                    source_bounds=bounds,
                    longitude=float(longitude[0]),
                    latitude=float(latitude[0]),
                    equal_area_x=float(equal_x[0]),
                    equal_area_y=float(equal_y[0]),
                    spatial_group=_spatial_group(equal_x[0], equal_y[0], group_size_m),
                )
            )

    if not candidates:
        raise ValueError(f"no EuroSAT multispectral images found in {archive}")
    return candidates


def _spread_sample(
    candidates: list[EuroSatCandidate],
    count: int,
    *,
    seed: int,
    namespace: str,
) -> list[EuroSatCandidate]:
    """Sample deterministically while using every available group before reuse."""

    by_group: dict[str, list[EuroSatCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_group[candidate.spatial_group].append(candidate)
    for group, items in by_group.items():
        items.sort(key=lambda item: _rank(seed, f"{namespace}:{group}", item.member))

    group_order = sorted(
        by_group,
        key=lambda group: _rank(seed, namespace, group),
    )
    selected: list[EuroSatCandidate] = []
    depth = 0
    while len(selected) < count:
        added = False
        for group in group_order:
            items = by_group[group]
            if depth < len(items):
                selected.append(items[depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError(f"only {len(selected)} candidates available; requested {count}")
        depth += 1
    return selected


def _distances_m(candidate: EuroSatCandidate, right_lonlat: np.ndarray) -> np.ndarray:
    candidate_lon = math.radians(candidate.longitude)
    candidate_lat = math.radians(candidate.latitude)
    right_lon = np.radians(right_lonlat[:, 0])
    right_lat = np.radians(right_lonlat[:, 1])
    delta_lon = right_lon - candidate_lon
    delta_lat = right_lat - candidate_lat
    haversine = np.sin(delta_lat / 2) ** 2 + (
        math.cos(candidate_lat) * np.cos(right_lat) * np.sin(delta_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))


def _minimum_distance(
    left: list[EuroSatCandidate], right: list[EuroSatCandidate]
) -> float:
    if not left or not right:
        return math.inf
    right_lonlat = np.asarray([(item.longitude, item.latitude) for item in right])
    minimum = math.inf
    for item in left:
        minimum = min(minimum, float(np.min(_distances_m(item, right_lonlat))))
    return minimum


def select_spatial_split(
    candidates: list[EuroSatCandidate],
    *,
    queries_per_class: int,
    index_per_class: int,
    minimum_separation_m: float = 5_000.0,
    seed: int = 42,
    labels: tuple[str, ...] = EUROSAT_CLASSES,
) -> EuroSatSplit:
    """Build a class-balanced split with disjoint spatial cells and a guard band."""

    if queries_per_class <= 0 or index_per_class <= 0:
        raise ValueError("per-class query and index counts must be positive")
    if minimum_separation_m < 0:
        raise ValueError("minimum_separation_m must not be negative")

    candidates_by_label: dict[str, list[EuroSatCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.label in labels:
            candidates_by_label[candidate.label].append(candidate)
    missing = [label for label in labels if len(candidates_by_label[label]) < queries_per_class]
    if missing:
        raise ValueError(f"insufficient query candidates for class: {missing[0]}")

    minimum_query_groups = min(10, queries_per_class)
    grouped_candidates: dict[str, list[EuroSatCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.label in labels:
            grouped_candidates[candidate.spatial_group].append(candidate)
    query_groups: set[str] = set()
    query_candidate_counts = {label: 0 for label in labels}
    query_group_counts = {label: 0 for label in labels}
    ordered_groups = sorted(
        grouped_candidates,
        key=lambda group: _rank(seed, "query-groups", group),
    )
    for group in ordered_groups:
        group_items = grouped_candidates[group]
        group_labels = {item.label for item in group_items}
        helps_balance = any(
            query_candidate_counts[label] < queries_per_class
            or query_group_counts[label] < minimum_query_groups
            for label in group_labels
        )
        if not helps_balance:
            continue
        query_groups.add(group)
        for label in labels:
            label_count = sum(item.label == label for item in group_items)
            if label_count:
                query_candidate_counts[label] += label_count
                query_group_counts[label] += 1
        if all(
            query_candidate_counts[label] >= queries_per_class
            and query_group_counts[label] >= minimum_query_groups
            for label in labels
        ):
            break
    if not all(
        query_candidate_counts[label] >= queries_per_class
        and query_group_counts[label] >= minimum_query_groups
        for label in labels
    ):
        raise ValueError("insufficient spatially diverse query groups")

    queries: list[EuroSatCandidate] = []
    for label in labels:
        queries.extend(
            _spread_sample(
                [
                    candidate
                    for candidate in candidates_by_label[label]
                    if candidate.spatial_group in query_groups
                ],
                queries_per_class,
                seed=seed,
                namespace=f"query:{label}",
            )
        )

    query_lonlat = np.asarray([(item.longitude, item.latitude) for item in queries])
    eligible_by_label: dict[str, list[EuroSatCandidate]] = defaultdict(list)
    excluded_near_query = 0
    for candidate in candidates:
        if candidate.label not in labels or candidate.spatial_group in query_groups:
            continue
        if float(np.min(_distances_m(candidate, query_lonlat))) < minimum_separation_m:
            excluded_near_query += 1
            continue
        eligible_by_label[candidate.label].append(candidate)

    indexes: list[EuroSatCandidate] = []
    for label in labels:
        if len(eligible_by_label[label]) < index_per_class:
            raise ValueError(
                f"only {len(eligible_by_label[label])} leakage-safe index candidates "
                f"available for class {label}; requested {index_per_class}"
            )
        indexes.extend(
            _spread_sample(
                eligible_by_label[label],
                index_per_class,
                seed=seed,
                namespace=f"index:{label}",
            )
        )

    overlap = query_groups.intersection(item.spatial_group for item in indexes)
    if overlap:
        raise AssertionError("spatial groups overlap across index and query")
    observed_minimum = _minimum_distance(queries, indexes)
    if observed_minimum + 1e-6 < minimum_separation_m:
        raise AssertionError("spatial guard band was not preserved")

    return EuroSatSplit(
        index=tuple(sorted(indexes, key=lambda item: item.member)),
        query=tuple(sorted(queries, key=lambda item: item.member)),
        minimum_separation_m=observed_minimum,
        excluded_near_query=excluded_near_query,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rgb_from_multispectral(source: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Apply the published EuroSAT RGB band choice and fixed 0-2750 stretch."""

    rgb = np.clip(source.astype(np.float32), 0, 2750) * (255.0 / 2750.0)
    result = np.rint(rgb).astype(np.uint8)
    result[:, ~valid] = 0
    return result


def _materialize_candidate(
    bundle: zipfile.ZipFile,
    candidate: EuroSatCandidate,
    split: Split,
    output_dir: Path,
    *,
    group_size_m: float,
    minimum_separation_m: float,
    seed: int,
) -> ImageRecord:
    try:
        from rasterio.enums import ColorInterp
        from rasterio.io import MemoryFile
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("EuroSAT preparation requires the 'geo' dependency group") from error

    payload = bundle.read(candidate.member)
    relative = Path(candidate.label) / Path(candidate.member).name
    destination = output_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with MemoryFile(payload) as memory_file, memory_file.open() as source:
        bands = source.read((4, 3, 2))
        valid = np.all(source.read_masks((4, 3, 2)) > 0, axis=0)
        rgb = _rgb_from_multispectral(bands, valid)
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            count=3,
            dtype="uint8",
            compress="deflate",
            predictor=2,
            nodata=None,
        )
        with MemoryFile() as output_memory:
            with output_memory.open(**profile) as target:
                target.write(rgb)
                target.write_mask(valid.astype(np.uint8) * 255)
                target.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                target.update_tags(
                    source_dataset=EUROSAT_SOURCE,
                    source_bands="B04,B03,B02",
                    stretch="fixed-linear-0-2750",
                )
            output_payload = output_memory.read()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(output_payload)
    temporary.replace(destination)

    return ImageRecord(
        item_id=relative.as_posix(),
        path=relative.as_posix(),
        split=split,
        label=candidate.label,
        source=EUROSAT_SOURCE,
        metadata={
            "sha256": _sha256_bytes(output_payload),
            "dataset_doi": EUROSAT_DOI,
            "archive": EUROSAT_ARCHIVE,
            "archive_member": candidate.member,
            "source_crs": candidate.source_crs,
            "source_bounds": list(candidate.source_bounds),
            "centroid_lonlat": [candidate.longitude, candidate.latitude],
            "centroid_epsg6933": [candidate.equal_area_x, candidate.equal_area_y],
            "spatial_group": candidate.spatial_group,
            "spatial_group_size_m": group_size_m,
            "minimum_index_query_separation_m": minimum_separation_m,
            "split_seed": seed,
            "rgb_bands": [4, 3, 2],
            "rgb_stretch_dn": [0, 2750],
        },
    )


def prepare_eurosat_benchmark(
    archive: Path,
    *,
    output_dir: Path,
    manifest: Path,
    queries_per_class: int = 40,
    index_per_class: int = 160,
    group_size_m: float = 50_000.0,
    minimum_separation_m: float = 5_000.0,
    seed: int = 42,
    expected_md5: str | None = EUROSAT_ARCHIVE_MD5,
    labels: tuple[str, ...] = EUROSAT_CLASSES,
) -> EuroSatPreparation:
    """Validate, split, convert, and record one benchmark version."""

    verify_archive(archive, expected_md5=expected_md5)
    candidates = discover_candidates(archive, group_size_m=group_size_m)
    split = select_spatial_split(
        candidates,
        queries_per_class=queries_per_class,
        index_per_class=index_per_class,
        minimum_separation_m=minimum_separation_m,
        seed=seed,
        labels=labels,
    )
    records: list[ImageRecord] = []
    partitions: tuple[tuple[Split, tuple[EuroSatCandidate, ...]], ...] = (
        ("index", split.index),
        ("query", split.query),
    )
    with zipfile.ZipFile(archive) as bundle:
        for split_name, selected in partitions:
            for candidate in selected:
                records.append(
                    _materialize_candidate(
                        bundle,
                        candidate,
                        split_name,
                        output_dir,
                        group_size_m=group_size_m,
                        minimum_separation_m=minimum_separation_m,
                        seed=seed,
                    )
                )
    write_jsonl(records, manifest)
    return EuroSatPreparation(
        records=tuple(sorted(records, key=lambda record: record.item_id)),
        discovered=len(candidates),
        minimum_separation_m=split.minimum_separation_m,
        excluded_near_query=split.excluded_near_query,
    )


def audit_eurosat_manifest(
    manifest: Path,
    *,
    image_root: Path | None = None,
    expected_labels: tuple[str, ...] = EUROSAT_CLASSES,
    expected_index_per_class: int = 160,
    expected_queries_per_class: int = 40,
) -> EuroSatAudit:
    """Verify class balance, spatial separation, hashes, and manifest identity."""

    from eo_visual_retrieval.manifests import read_jsonl

    records = read_jsonl(manifest)
    per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"index": 0, "query": 0})
    groups_by_split: dict[str, set[str]] = {"index": set(), "query": set()}
    candidates_by_split: dict[str, list[EuroSatCandidate]] = {"index": [], "query": []}
    declared_separations: set[float] = set()
    verified_files = 0

    for record in records:
        if record.source != EUROSAT_SOURCE or record.label is None:
            raise ValueError(f"record is not a labeled {EUROSAT_SOURCE} item: {record.item_id}")
        try:
            group = str(record.metadata["spatial_group"])
            x, y = (float(value) for value in record.metadata["centroid_epsg6933"])
            minimum = float(record.metadata["minimum_index_query_separation_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"record lacks spatial audit metadata: {record.item_id}") from error
        per_class[record.label][record.split] += 1
        groups_by_split[record.split].add(group)
        declared_separations.add(minimum)
        candidates_by_split[record.split].append(
            EuroSatCandidate(
                member=str(record.metadata.get("archive_member", record.item_id)),
                label=record.label,
                source_crs=str(record.metadata.get("source_crs", "unknown")),
                source_bounds=_bounds4(record.metadata["source_bounds"]),
                longitude=float(record.metadata["centroid_lonlat"][0]),
                latitude=float(record.metadata["centroid_lonlat"][1]),
                equal_area_x=x,
                equal_area_y=y,
                spatial_group=group,
            )
        )
        if image_root is not None:
            path = image_root / record.path
            if not path.is_file():
                raise ValueError(f"manifest references missing image: {path}")
            observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed_sha256 != record.metadata.get("sha256"):
                raise ValueError(f"image checksum mismatch: {path}")
            verified_files += 1

    overlap = groups_by_split["index"].intersection(groups_by_split["query"])
    if overlap:
        raise ValueError(f"spatial group crosses index/query: {sorted(overlap)[0]}")
    if len(declared_separations) != 1:
        raise ValueError("manifest contains inconsistent separation policies")
    if set(per_class) != set(expected_labels):
        missing = sorted(set(expected_labels) - set(per_class))
        unexpected = sorted(set(per_class) - set(expected_labels))
        raise ValueError(f"class set mismatch; missing={missing}, unexpected={unexpected}")
    for label in expected_labels:
        counts = per_class[label]
        if counts != {
            "index": expected_index_per_class,
            "query": expected_queries_per_class,
        }:
            raise ValueError(
                f"class count mismatch for {label}: observed {counts}, expected "
                f"index={expected_index_per_class}, query={expected_queries_per_class}"
            )
    observed_minimum = _minimum_distance(
        candidates_by_split["query"], candidates_by_split["index"]
    )
    declared_minimum = next(iter(declared_separations))
    if observed_minimum + 1e-6 < declared_minimum:
        raise ValueError(
            f"minimum separation is {observed_minimum:.3f} m; expected {declared_minimum:.3f} m"
        )

    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return EuroSatAudit(
        items=len(records),
        index=sum(record.split == "index" for record in records),
        query=sum(record.split == "query" for record in records),
        per_class={label: dict(counts) for label, counts in sorted(per_class.items())},
        spatial_groups=len(groups_by_split["index"] | groups_by_split["query"]),
        minimum_separation_m=observed_minimum,
        manifest_sha256=manifest_digest,
        verified_files=verified_files,
    )
