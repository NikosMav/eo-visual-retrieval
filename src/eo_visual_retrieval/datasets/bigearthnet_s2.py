"""Single-pass staged acquisition of the frozen S2 selection."""

from __future__ import annotations

import io
import json
import re
import tarfile
import time
from collections import Counter
from collections.abc import Iterator
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from typing import Any

from eo_visual_retrieval.datasets.acquisition_io import (
    READ_BYTES,
    AcquisitionLimit,
    RangeStream,
    StorageBudget,
    acquisition_lock,
)
from eo_visual_retrieval.datasets.bigearthnet import (
    BIGEARTHNET_RECORD_URL,
    S2_ARCHIVE_BYTES,
    S2_ARCHIVE_FILENAME,
    S2_ARCHIVE_MD5,
    S2_BANDS,
    S2_GSD,
)
from eo_visual_retrieval.hashing import bytes_sha256, file_sha256

MAX_MEMBER_BYTES = 256 * 1024
MAX_NAME_BYTES = 4096
MAX_WINDOW_BYTES = 16 * 1024 * 1024
SAMPLE_NETWORK_BYTES = 512 * 1024 * 1024
_PATCH = re.compile(r"S2[AB]_MSIL2A_\d{8}T\d{6}_N\d{4}_R\d{3}_T\d{2}[A-Z]{3}_\d{2}_\d{2}")


class GeometryMismatch(ValueError):
    """Terminal stop: changing the reference footprint requires a separate decision."""


def frozen_inputs(selection: Path, audit: Path, inventory: Path) -> dict[str, dict[str, Any]]:
    """Read only frozen IDs/geometry, not final labels or any representation outputs."""
    import pyarrow.parquet as pq

    evidence = json.loads(audit.read_text(encoding="utf-8"))
    if file_sha256(selection) != evidence["selection_sha256"] or (
        file_sha256(inventory) != evidence["footprint_inventory_sha256"]
    ):
        raise ValueError("frozen selection or inventory hash mismatch")
    frozen = json.loads(selection.read_text(encoding="utf-8"))
    if frozen["schema"] != "bigearthnet-acquisition-selection-v1" or (
        frozen["footprint_inventory_sha256"] != evidence["footprint_inventory_sha256"]
        or frozen["footprint_report_sha256"] != evidence["footprint_report_sha256"]
        or frozen["policy"] != evidence["policy"]
    ):
        raise ValueError("frozen selection provenance mismatch")
    partitions: dict[str, str] = {}
    expected = {"index": 4000, "development": 500, "final": 500}
    if set(frozen["partitions"]) != set(expected):
        raise ValueError("unknown frozen partition")
    for partition, count in expected.items():
        ids = frozen["partitions"][partition]
        if len(ids) != count or len(set(ids)) != count or set(ids) & partitions.keys():
            raise ValueError("frozen partition size or ID uniqueness mismatch")
        if any(_PATCH.fullmatch(identity) is None for identity in ids):
            raise ValueError("invalid frozen patch ID")
        partitions.update(dict.fromkeys(ids, partition))
    columns = ["patch_id", "epsg", "left", "bottom", "right", "top", "spatial_group"]
    rows = pq.read_table(
        inventory, columns=columns, filters=[("patch_id", "in", list(partitions))]
    ).to_pylist()
    result = {row["patch_id"]: {**row, "partition": partitions[row["patch_id"]]} for row in rows}
    if set(result) != set(partitions) or len(rows) != len(result):
        raise ValueError("frozen footprints do not exactly cover the selection")
    return result


def _exact(stream: Any, count: int) -> bytes:
    if count < 0 or count > MAX_MEMBER_BYTES:
        raise ValueError("tar read exceeds bounded member size")
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(min(READ_BYTES, remaining))
        if not chunk:
            raise ValueError("truncated tar stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def bounded_tar(stream: Any, *, deadline: float) -> Iterator[tuple[tarfile.TarInfo, bytes]]:
    """Strict GNU-tar profile, with bounded long names before any allocation.

    TarFile's general parser reads GNU/PAX extension sizes before yielding a
    member. This reader bounds those headers too and never extracts filesystem paths.
    Unsupported extensions and links fail closed.
    """
    pending_name: str | None = None
    while True:
        if time.monotonic() >= deadline:
            raise AcquisitionLimit("acquisition wall-clock limit reached")
        header = _exact(stream, 512)
        if header == bytes(512):
            if pending_name is not None or _exact(stream, 512) != bytes(512):
                raise ValueError("invalid tar end marker")
            return
        member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
        if member.type == tarfile.GNUTYPE_LONGNAME:
            if pending_name is not None or not 0 < member.size <= MAX_NAME_BYTES:
                raise ValueError("invalid or oversized tar long name")
            pending_name = _exact(stream, member.size).rstrip(b"\0").decode("ascii")
            _exact(stream, -member.size % 512)
            continue
        if pending_name is not None:
            member.name, pending_name = pending_name, None
        if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
            raise ValueError("unsupported tar entry type or extension")
        if member.size < 0 or member.size > MAX_MEMBER_BYTES or (member.isdir() and member.size):
            raise ValueError("tar member exceeds its size limit")
        payload = _exact(stream, member.size)
        _exact(stream, -member.size % 512)
        yield member, payload


def member_identity(member: tarfile.TarInfo) -> tuple[str, str] | None:
    name = member.name.rstrip("/")
    parts = PurePosixPath(name).parts
    if (
        not name
        or "\\" in name
        or any(x in ("", ".", "..") for x in name.split("/"))
        or (not parts or parts[0] != "BigEarthNet-S2")
    ):
        raise ValueError("unsafe or unexpected S2 member path")
    if member.isdir() and len(parts) <= 3:
        return None
    if not member.isfile() or len(parts) != 4:
        raise ValueError("unexpected S2 member type or layout")
    patch = parts[2]
    if _PATCH.fullmatch(patch) is None or parts[1] != patch.rsplit("_", 2)[0]:
        raise ValueError("S2 member disagrees with its patch identity")
    band = parts[3].removeprefix(patch + "_").removesuffix(".tif")
    if band not in S2_BANDS or parts[3] != f"{patch}_{band}.tif":
        raise ValueError("unknown S2 band or filename; expected explicit 12-band L2A contract")
    return patch, band


def check_raster(payload: bytes, band: str, footprint: dict[str, Any]) -> dict[str, Any]:
    """Decode the actual band, compare native geometry, and retain no pixel statistics."""
    from rasterio.io import MemoryFile

    gsd = S2_GSD[band]
    shape = 1200 // gsd
    expected_bounds = [footprint[x] for x in ("left", "bottom", "right", "top")]
    expected_transform = [gsd, 0, footprint["left"], 0, -gsd, footprint["top"]]
    with MemoryFile(payload) as memory, memory.open() as raster:
        observed = {
            "bounds": list(raster.bounds),
            "transform": list(raster.transform)[:6],
            "epsg": raster.crs.to_epsg() if raster.crs else None,
            "shape": [raster.height, raster.width],
            "dtypes": list(raster.dtypes),
        }
        if (
            observed["bounds"] != expected_bounds
            or observed["transform"] != expected_transform
            or (observed["epsg"] != footprint["epsg"])
        ):
            raise GeometryMismatch(
                json.dumps(
                    {
                        "patch_id": footprint["patch_id"],
                        "band": band,
                        "expected": {
                            "bounds": expected_bounds,
                            "transform": expected_transform,
                            "epsg": footprint["epsg"],
                        },
                        "observed": observed,
                    },
                    sort_keys=True,
                )
            )
        if (
            raster.count != 1
            or raster.dtypes != ("uint16",)
            or (raster.height, raster.width) != (shape, shape)
        ):
            raise ValueError("S2 raster shape/dtype disagrees with native band resolution")
        pixels = raster.read(1)
        if pixels.shape != (shape, shape):
            raise ValueError("decoded S2 pixel dimensions disagree")
    return {
        "band": band,
        "file_bytes": len(payload),
        "file_sha256": bytes_sha256(payload),
        "geometry_agrees": True,
        "pixels_decoded": True,
        **observed,
    }


def require_complete(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    """Only this gate makes a download usable; a directory of TIFFs is insufficient."""
    marker = root / "COMPLETE.json"
    if not marker.is_file():
        raise ValueError("verified acquisition completion marker is missing")
    report = json.loads(marker.read_text(encoding="utf-8"))
    if (
        report["binding"] != binding
        or report["mode"] != "acquire"
        or (
            report["status"] != "verified"
            or report["complete_archive_checksum_verified"] is not True
            or report["source_md5"] != S2_ARCHIVE_MD5
            or report["source_bytes"] != S2_ARCHIVE_BYTES
            or report["band_order"] != list(S2_BANDS)
        )
    ):
        raise ValueError("completion marker provenance mismatch")
    expected = set(binding["selected_ids"])
    receipts = report["patch_receipts"]
    if set(receipts) != expected:
        raise ValueError("completion marker has partial or foreign IDs")
    for identity, digest in receipts.items():
        path = root / "files" / identity / "patch.json"
        if file_sha256(path) != digest:
            raise ValueError("completed patch receipt checksum mismatch")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["patch_id"] != identity or [b["band"] for b in record["bands"]] != list(S2_BANDS):
            raise ValueError("completed patch band identity/order mismatch")
        for band in record["bands"]:
            raster = path.parent / f"{band['band']}.tif"
            if file_sha256(raster) != band["file_sha256"]:
                raise ValueError("completed band checksum mismatch")
    return report


def acquire(
    footprints: dict[str, dict[str, Any]],
    *,
    binding: dict[str, Any],
    root: Path,
    ancillary: list[Path],
    mode: str,
    network_budget: int,
    resume: bool = False,
    max_seconds: float = 300,
) -> dict[str, Any]:
    """Run one staged acquisition or a bounded sample using the same verification path."""
    if mode not in ("sample", "acquire") or network_budget <= 0 or max_seconds <= 0:
        raise ValueError("invalid mode, network budget, or time limit")
    if sorted(footprints) != binding["selected_ids"]:
        raise ValueError("acquisition binding disagrees with the frozen IDs")
    wanted = set(binding["selected_ids"])
    root.mkdir(parents=True, exist_ok=True)
    StorageBudget(root, ancillary).reserve(1)  # Account for initial lock-file creation too.
    with acquisition_lock(root, resume=resume):
        budget = StorageBudget(root, ancillary)
        state_path = root / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["binding"] != binding or state["mode"] != mode:
                raise ValueError("resume provenance/mode mismatch")
            if state["status"] in ("geometry_mismatch", "integrity_failure"):
                raise ValueError("terminal acquisition failure requires a separate decision")
            if (root / "COMPLETE.json").exists():
                return require_complete(root, binding)
            if not resume:
                raise ValueError("interrupted acquisition exists; use --resume")
        else:
            if list(root.glob("files/*")):
                raise ValueError("untracked staging exists without a resumable state")
            state = {
                "binding": binding,
                "mode": mode,
                "attempts": 0,
                "network_reserved_bytes": 0,
                "network_received_bytes": 0,
                "http_requests": 0,
                "transport_retries": 0,
                "wall_seconds": 0.0,
            }
        state.update(
            status="in_progress",
            attempts=state["attempts"] + 1,
            attempt_compressed_bytes=0,
            resume_strategy="replay compressed stream from zero",
            network_budget=network_budget,
        )
        state.update(scanned_members=0, scanned_raster_members=0)
        for field in (
            "error",
            "attempt_prefix_digest",
            "diagnostic_complete_patches",
            "diagnostic_bands",
            "diagnostic_partition_counts",
            "last_patch_seen",
        ):
            state.pop(field, None)
        state["complete_archive_checksum_verified"] = False
        started = time.monotonic()
        previous_seconds = state["wall_seconds"]

        def persist() -> None:
            state["wall_seconds"] = previous_seconds + time.monotonic() - started
            budget.checkpoint(state_path, state)

        persist()
        observed: dict[str, dict[str, dict[str, Any]]] = {}
        receipts: dict[str, str] = {}
        source = RangeStream(
            f"{BIGEARTHNET_RECORD_URL}/files/{S2_ARCHIVE_FILENAME}?download=1",
            S2_ARCHIVE_BYTES,
            network_budget=network_budget,
            state=state,
            persist=persist,
            deadline=started + max_seconds,
        )
        try:
            import zstandard as zstd
            from rasterio import Env

            with source, io.BufferedReader(source, buffer_size=READ_BYTES) as compressed:
                frame = zstd.get_frame_parameters(compressed.peek(18))
                if frame.window_size > MAX_WINDOW_BYTES:
                    raise ValueError("S2 decompressor window exceeds limit")
                state["first_frame_window_bytes"] = frame.window_size
                state["first_frame_content_bytes"] = frame.content_size
                decoder = zstd.ZstdDecompressor(max_window_size=MAX_WINDOW_BYTES)
                with (
                    Env(),
                    decoder.stream_reader(
                        compressed, closefd=False, read_across_frames=False
                    ) as decoded,
                ):
                    for member, payload in bounded_tar(decoded, deadline=started + max_seconds):
                        state["scanned_members"] += 1
                        identity = member_identity(member)
                        if identity is None:
                            continue
                        state["scanned_raster_members"] += 1
                        state["last_patch_seen"] = identity[0]
                        if identity[0] not in wanted:
                            continue
                        patch, band = identity
                        bands = observed.setdefault(patch, {})
                        if band in bands:
                            raise ValueError("duplicate selected band in source stream")
                        record = check_raster(payload, band, footprints[patch])
                        path = root / "files" / patch / f"{band}.tif"
                        if path.exists():
                            if file_sha256(path) != record["file_sha256"]:
                                raise ValueError("staged band differs from replayed source")
                        else:
                            budget.write(path, payload)
                        bands[band] = record
                        if len(bands) == len(S2_BANDS):
                            receipt = path.parent / "patch.json"
                            budget.json(
                                receipt,
                                {
                                    "patch_id": patch,
                                    "partition": footprints[patch]["partition"],
                                    "bands": [bands[value] for value in S2_BANDS],
                                },
                            )
                            receipts[patch] = file_sha256(receipt)
                            persist()
                    tail_bytes = 0
                    while tail := decoded.read(READ_BYTES):
                        tail_bytes += len(tail)
                        if tail_bytes > 1024 * 1024 or any(tail):
                            raise ValueError("unexpected data after tar end markers")
                # Hash every compressed byte, including bytes beyond tar's logical EOF.
                while compressed.read(READ_BYTES):
                    if time.monotonic() >= started + max_seconds:
                        raise AcquisitionLimit("acquisition wall-clock limit reached")
            digest = source.digest.values()
            if digest["bytes"] != S2_ARCHIVE_BYTES or digest["md5"] != S2_ARCHIVE_MD5:
                raise ValueError("complete S2 source checksum/byte count mismatch")
            state["complete_archive_checksum_verified"] = True
            if set(receipts) != wanted:
                raise ValueError("source lacks selected patches or complete 12-band groups")
            if mode == "sample":
                raise AcquisitionLimit("throughput sample reached source end; staging not promoted")
            state["status"] = "verified"
            persist()
            result = {
                "schema": "bigearthnet-s2-acquisition-v1",
                "status": "verified",
                "binding": binding,
                "mode": mode,
                "source_md5": digest["md5"],
                "source_sha256": digest["sha256"],
                "source_bytes": digest["bytes"],
                "complete_archive_checksum_verified": True,
                "all_selected_patch_geometry_verified": True,
                "band_order": list(S2_BANDS),
                "patch_receipts": receipts,
                "partition_counts": dict(Counter(footprints[x]["partition"] for x in wanted)),
                "wall_seconds": state["wall_seconds"],
                "network_received_bytes": state["network_received_bytes"],
                "network_reserved_bytes": state["network_reserved_bytes"],
                "storage_peak_bytes": state["storage_peak_bytes"],
                "single_pass": True,
            }
            # Written last, after every integrity gate, including its own atomic storage cost.
            budget.checkpoint(root / "COMPLETE.json", result)
            return result
        except BaseException as error:
            source.close()
            state["status"] = (
                "geometry_mismatch"
                if isinstance(error, GeometryMismatch)
                else "incomplete"
                if isinstance(error, (AcquisitionLimit, OSError, HTTPException, KeyboardInterrupt))
                else "integrity_failure"
            )
            state["error"] = str(error)
            state["diagnostic_complete_patches"] = len(receipts)
            state["diagnostic_bands"] = sum(len(bands) for bands in observed.values())
            state["diagnostic_partition_counts"] = dict(
                Counter(footprints[x]["partition"] for x in receipts)
            )
            state["attempt_prefix_digest"] = source.digest.values()
            state["failure_compressed_offset"] = source.offset
            state["failure_offset_definition"] = (
                "compressed bytes delivered to the application; decoder buffering may read ahead"
            )
            if isinstance(error, GeometryMismatch):
                mismatch = json.loads(str(error))
                state["geometry_mismatch_patch_id"] = mismatch["patch_id"]
                state["geometry_mismatch_band"] = mismatch["band"]
            # A storage guard may leave no room even for an expanded error report.
            # The earlier durable in_progress state and absence of COMPLETE remain safe.
            try:
                persist()
            except AcquisitionLimit:
                pass
            raise
