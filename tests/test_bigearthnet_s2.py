"""Synthetic source integrity, native geometry, bounded storage and restart gates."""

from __future__ import annotations

import io
import json
import tarfile
import time
from pathlib import Path
from typing import Any
from urllib.request import Request

import numpy as np
import pytest

from eo_visual_retrieval.datasets import acquisition_io as aio
from eo_visual_retrieval.datasets import bigearthnet_s2 as s2
from eo_visual_retrieval.datasets.bigearthnet import S2_BANDS, S2_GSD
from eo_visual_retrieval.hashing import StreamDigests, file_sha256


def identity(column: int = 0) -> str:
    return f"S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_{column:02d}_57"


def geometry(column: int = 0, partition: str = "index") -> dict[str, Any]:
    return dict(
        patch_id=identity(column),
        epsg=32633,
        left=331200,
        bottom=5330400,
        right=332400,
        top=5331600,
        spatial_group=f"cell{column}",
        partition=partition,
    )


def raster(band: str, **changes: Any) -> bytes:
    rio = pytest.importorskip("rasterio")
    gsd = S2_GSD[band]
    options = dict(
        driver="GTiff",
        width=1200 // gsd,
        height=1200 // gsd,
        count=1,
        dtype="uint16",
        crs="EPSG:32633",
        transform=rio.Affine(gsd, 0, 331200, 0, -gsd, 5331600),
    )
    options.update(changes)
    with rio.MemoryFile() as memory:
        with memory.open(**options) as out:
            out.write(np.zeros((options["height"], options["width"]), dtype=options["dtype"]), 1)
        return bytes(memory.read())


def name(patch: str, band: str) -> str:
    return f"BigEarthNet-S2/{patch.rsplit('_', 2)[0]}/{patch}/{patch}_{band}.tif"


def archive(entries: list[tuple[str, bytes]], *, extra: bytes = b"") -> bytes:
    zstd = pytest.importorskip("zstandard")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for path, payload in entries:
            member = tarfile.TarInfo(path)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
    return bytes(zstd.ZstdCompressor().compress(buffer.getvalue() + extra))


class Response(io.BytesIO):
    def __init__(self, payload: bytes, start: int, end: int) -> None:
        super().__init__(payload[start : end + 1])
        self.status = 206
        self.headers = {
            "Content-Range": f"bytes {start}-{end}/{len(payload)}",
            "Content-Length": str(end - start + 1),
        }


def serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> list[tuple[int, int]]:
    requests: list[tuple[int, int]] = []

    def open_range(request: Request, *, timeout: float) -> Response:
        assert timeout > 0
        assert request.headers["Accept-encoding"] == "identity"
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        requests.append((start, end))
        return Response(payload, start, end)

    monkeypatch.setattr(aio, "urlopen", open_range)
    return requests


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    rows = {
        identity(i): geometry(i, part) for i, part in enumerate(("index", "development", "final"))
    }
    # Small source fixture; production source constants are tested elsewhere.
    binding = {"selected_ids": sorted(rows)}
    # Reverse archive order to ensure receipt order is independent of tar order.
    entries = [(name(patch, band), raster(band)) for patch in rows for band in reversed(S2_BANDS)]
    # This non-selected member must never be decoded or written.
    entries.append((name(identity(99), "B01"), b"unselected bytes"))
    payload = archive(entries)
    pin_source(monkeypatch, payload)
    return rows, binding, payload


def pin_source(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    digest = StreamDigests()
    digest.update(payload)
    monkeypatch.setattr(s2, "S2_ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(s2, "S2_ARCHIVE_MD5", digest.values()["md5"])


def test_single_acquisition_requires_entire_checksum_and_ordered_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: tuple[dict[str, Any], dict[str, Any], bytes],
) -> None:
    rows, binding, payload = source
    requests = serve(monkeypatch, payload)
    pilot = tmp_path / "pilot"
    report = s2.acquire(
        rows, binding=binding, root=pilot, ancillary=[], mode="acquire", network_budget=len(payload)
    )
    assert report["source_bytes"] == len(payload) and len(report["patch_receipts"]) == 3
    assert report["partition_counts"] == {"index": 1, "development": 1, "final": 1}
    assert len(list(pilot.rglob("*.tif"))) == 36
    assert not list(pilot.rglob("*.zst"))
    receipt = json.loads((pilot / "files" / identity() / "patch.json").read_text())
    assert [x["band"] for x in receipt["bands"]] == list(S2_BANDS)
    assert all(x["geometry_agrees"] and x["pixels_decoded"] for x in receipt["bands"])
    assert s2.require_complete(pilot, binding) == report
    assert (
        s2.acquire(
            rows,
            binding=binding,
            root=pilot,
            ancillary=[],
            mode="acquire",
            network_budget=len(payload),
        )
        == report
    )
    assert len(requests) == 1  # Existing completed data is checked locally, not fetched again.
    assert report["mode"] == "acquire" and report["single_pass"] is True


def test_sample_uses_same_full_selection_path_but_never_promotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: tuple[dict[str, Any], dict[str, Any], bytes],
) -> None:
    rows, binding, payload = source
    serve(monkeypatch, payload)
    with pytest.raises(aio.AcquisitionLimit, match="staging not promoted"):
        s2.acquire(
            rows,
            binding=binding,
            root=tmp_path,
            ancillary=[],
            mode="sample",
            network_budget=len(payload),
        )
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["diagnostic_complete_patches"] == len(rows)
    assert state["complete_archive_checksum_verified"] is True
    assert state["status"] == "incomplete" and not (tmp_path / "COMPLETE.json").exists()


def test_restart_replays_checksums_preserves_files_and_charges_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: tuple[dict[str, Any], dict[str, Any], bytes],
) -> None:
    rows, binding, payload = source
    requests = serve(monkeypatch, payload)
    original_write = aio.StorageBudget.write

    def interrupt(self: aio.StorageBudget, path: Path, data: bytes) -> None:
        if path.parent.name == identity(1) and path.suffix == ".tif":
            raise KeyboardInterrupt("simulated interruption after first complete patch")
        original_write(self, path, data)

    monkeypatch.setattr(aio.StorageBudget, "write", interrupt)
    options: dict[str, Any] = dict(binding=binding, root=tmp_path, ancillary=[], mode="acquire")
    with pytest.raises(KeyboardInterrupt):
        s2.acquire(rows, **options, network_budget=len(payload))
    assert not (tmp_path / "COMPLETE.json").exists()
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "incomplete" and state["diagnostic_complete_patches"] == 1
    retained = tmp_path / "files" / identity() / "B01.tif"
    old_hash, old_mtime = file_sha256(retained), retained.stat().st_mtime_ns
    with pytest.raises(ValueError, match="use --resume"):
        s2.acquire(rows, **options, network_budget=len(payload))
    monkeypatch.setattr(aio.StorageBudget, "write", original_write)
    with pytest.raises(aio.AcquisitionLimit, match="cumulative network"):
        s2.acquire(rows, **options, network_budget=len(payload), resume=True)
    assert len(requests) == 1
    report = s2.acquire(rows, **options, network_budget=2 * len(payload), resume=True)
    assert report["network_reserved_bytes"] == 2 * len(payload)
    assert report["network_received_bytes"] == 2 * len(payload)
    assert requests[0][0] == requests[1][0] == 0
    assert file_sha256(retained) == old_hash and retained.stat().st_mtime_ns == old_mtime


@pytest.mark.parametrize(
    "case, message",
    [
        ("checksum", "checksum/byte count"),
        ("geometry", "expected"),
        ("missing", "lacks selected"),
        ("duplicate", "duplicate selected"),
        ("tail", "after tar end"),
        ("band", "unknown S2 band"),
    ],
)
def test_bad_source_never_promotes_and_terminal_failures_cannot_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: tuple[dict[str, Any], dict[str, Any], bytes],
    case: str,
    message: str,
) -> None:
    rows, binding, payload = source
    if case == "checksum":
        monkeypatch.setattr(s2, "S2_ARCHIVE_MD5", "0" * 32)
    elif case == "geometry":
        rows[identity()]["epsg"] = 32632
    else:
        entries = [(name(patch, band), raster(band)) for patch in rows for band in S2_BANDS]
        if case == "missing":
            entries.pop()
        elif case == "duplicate":
            entries.append(entries[0])
        elif case == "band":
            entries[0] = (name(identity(), "B10"), raster("B01"))
        payload = archive(entries, extra=b"bad" if case == "tail" else b"")
        pin_source(monkeypatch, payload)
    requests = serve(monkeypatch, payload)
    options: dict[str, Any] = dict(
        binding=binding, root=tmp_path, ancillary=[], mode="acquire", network_budget=len(payload)
    )
    with pytest.raises(ValueError, match=message):
        s2.acquire(rows, **options)
    assert not (tmp_path / "COMPLETE.json").exists()
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == ("geometry_mismatch" if case == "geometry" else "integrity_failure")
    if case == "geometry":
        assert state["geometry_mismatch_patch_id"] == identity()
        assert state["geometry_mismatch_band"] in S2_BANDS
        assert state["failure_compressed_offset"] > 0
    assert state["complete_archive_checksum_verified"] is (case == "missing")
    with pytest.raises(ValueError, match="terminal"):
        s2.acquire(rows, **options, resume=True)
    assert len(requests) == 1


@pytest.mark.parametrize("band", S2_BANDS)
def test_native_band_geometry_and_resolution(band: str) -> None:
    result = s2.check_raster(raster(band), band, geometry())
    assert result["shape"] == [1200 // S2_GSD[band]] * 2
    assert result["bounds"] == [331200, 5330400, 332400, 5331600]


@pytest.mark.parametrize(
    "changes",
    [
        {"crs": None},
        {"crs": "EPSG:32632"},
        {"width": 119},
        {"transform": (10, 0, 331201, 0, -10, 5331600)},
    ],
)
def test_native_geometry_disagreement_stops(changes: dict[str, Any]) -> None:
    rio = pytest.importorskip("rasterio")
    if "transform" in changes:
        changes["transform"] = rio.Affine(*changes["transform"])
    with pytest.raises(s2.GeometryMismatch):
        s2.check_raster(raster("B02", **changes), "B02", geometry())


def test_native_dtype_disagreement_stops() -> None:
    with pytest.raises(ValueError, match="dtype"):
        s2.check_raster(raster("B02", dtype="uint8"), "B02", geometry())


@pytest.mark.parametrize(
    "kind, size, message",
    [
        (tarfile.GNUTYPE_LONGNAME, 4097, "long name"),
        (tarfile.REGTYPE, s2.MAX_MEMBER_BYTES + 1, "size limit"),
        (tarfile.DIRTYPE, 1, "size limit"),
        (tarfile.SYMTYPE, 0, "entry type"),
        (tarfile.XHDTYPE, 0, "entry type"),
    ],
)
def test_tar_extensions_and_members_are_bounded_before_body_read(
    kind: bytes,
    size: int,
    message: str,
) -> None:
    member = tarfile.TarInfo("anything")
    member.type, member.size = kind, size
    stream = io.BytesIO(member.tobuf(format=tarfile.GNU_FORMAT))
    with pytest.raises(ValueError, match=message):
        list(s2.bounded_tar(stream, deadline=float("inf")))
    assert stream.tell() == 512


@pytest.mark.parametrize(
    "path",
    [
        "/BigEarthNet-S2/a",
        "BigEarthNet-S2/../a",
        "BigEarthNet-S2//a",
        "BigEarthNet-S2/./a",
        "wrong",
        "a\\b",
    ],
)
def test_tar_paths_cannot_escape(path: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        s2.member_identity(tarfile.TarInfo(path))


def test_tar_deadline_and_truncation() -> None:
    with pytest.raises(aio.AcquisitionLimit, match="wall-clock"):
        list(s2.bounded_tar(io.BytesIO(), deadline=0))
    with pytest.raises(ValueError, match="truncated"):
        list(s2.bounded_tar(io.BytesIO(b"short"), deadline=float("inf")))


def test_storage_guard_accounts_ancillary_atomic_replacement_and_stale_files(
    tmp_path: Path,
) -> None:
    root, reference = tmp_path / "root", tmp_path / "reference"
    root.mkdir()
    reference.write_bytes(bytes(30))
    target = root / "band.tif"
    budget = aio.StorageBudget(root, [reference, root], limit=100)
    budget.write(target, bytes(40))
    assert budget.used == 70
    with pytest.raises(aio.AcquisitionLimit):
        budget.write(target, bytes(40))  # Old and atomic replacement would coexist: 110 bytes.
    assert target.stat().st_size == 40 and not list(root.glob("*.tmp"))
    budget.write(target, bytes(20))
    assert budget.used == 50 and budget.peak == 90
    with pytest.raises(ValueError, match="outside"):
        budget.write(tmp_path / "escape", b"x")
    staging = root / "band.tif.tmp"
    staging.write_bytes(b"interrupted")
    with pytest.raises(ValueError, match="stale"):
        budget.write(target, b"x")
    with pytest.raises(ValueError, match="resume"):
        with aio.acquisition_lock(root, resume=False):
            pass
    with aio.acquisition_lock(root, resume=True):
        assert not staging.exists()
    assert aio.StorageBudget(root, [reference]).used == 51  # Includes durable 1-byte lock.
    with pytest.raises(aio.AcquisitionLimit):
        aio.StorageBudget(root, [reference], limit=50)


def ledger() -> dict[str, Any]:
    return dict(
        network_reserved_bytes=0, network_received_bytes=0, http_requests=0, transport_retries=0
    )


def test_range_stream_preserves_hash_across_ranges_and_bounds_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"0123456789"
    requests = serve(monkeypatch, payload)
    state = ledger()
    with aio.RangeStream(
        "https://example.test",
        len(payload),
        network_budget=10,
        state=state,
        persist=lambda: None,
        range_bytes=4,
    ) as stream:
        assert stream.read(0) == b""
        with pytest.raises(ValueError, match="unbounded"):
            stream.read()
        assert b"".join(iter(lambda: stream.read(3), b"")) == payload
        assert stream.digest.values()["bytes"] == 10
    assert requests == [(0, 3), (4, 7), (8, 9)]
    assert state["network_reserved_bytes"] == state["network_received_bytes"] == 10
    with pytest.raises(ValueError, match="closed"):
        stream.read(1)


def test_truncated_range_retries_from_live_offset_and_charges_failed_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, offsets = b"0123456789", []

    def interrupted(request: Request, *, timeout: float) -> Response:
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        offsets.append(start)
        response = Response(payload, start, end)
        if len(offsets) == 1:
            response.truncate(3)
        return response

    monkeypatch.setattr(aio, "urlopen", interrupted)
    state = ledger()
    with aio.RangeStream(
        "https://example.test", 10, network_budget=17, state=state, persist=lambda: None
    ) as stream:
        assert b"".join(iter(lambda: stream.read(4), b"")) == payload
    assert offsets == [0, 3] and state["transport_retries"] == 1
    assert state["network_reserved_bytes"] == 17 and state["network_received_bytes"] == 10


@pytest.mark.parametrize(
    "change",
    [
        {"status": 200},
        {"Content-Range": "bytes 1-9/10"},
        {"Content-Encoding": "gzip"},
        {"Content-Length": "999"},
    ],
)
def test_server_must_honor_exact_range_before_any_body_read(
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, Any],
) -> None:
    response = Response(b"0123456789", 0, 9)
    if "status" in change:
        response.status = change["status"]
    else:
        response.headers.update(change)
    monkeypatch.setattr(aio, "urlopen", lambda *a, **kw: response)
    state = ledger()
    with aio.RangeStream(
        "https://example.test", 10, network_budget=10, state=state, persist=lambda: None
    ) as stream:
        with pytest.raises(ValueError, match="exact identity"):
            stream.read(1)
    assert state["network_received_bytes"] == 0 and state["network_reserved_bytes"] == 10
    assert response.closed


def test_network_budget_and_deadline_stop_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = serve(monkeypatch, b"0123456789")
    with aio.RangeStream(
        "https://example.test", 10, network_budget=3, state=ledger(), persist=lambda: None
    ) as stream:
        assert stream.read(10) == b"012"
        with pytest.raises(aio.AcquisitionLimit, match="network"):
            stream.read(10)
    with aio.RangeStream(
        "https://example.test",
        10,
        network_budget=3,
        state=ledger(),
        persist=lambda: None,
        deadline=time.monotonic() - 1,
    ) as stream:
        with pytest.raises(aio.AcquisitionLimit, match="wall-clock"):
            stream.read(10)
    assert requests == [(0, 2)]


def test_frozen_inputs_require_exact_hashes_partitions_and_inventory(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = []
    partitions: dict[str, list[str]] = {"index": [], "development": [], "final": []}
    for i in range(5000):
        part = "index" if i < 4000 else "development" if i < 4500 else "final"
        patch = identity(i // 100).removesuffix("57") + f"{i % 100:02d}"
        rows.append({**geometry(), "patch_id": patch})
        partitions[part].append(patch)
    inventory, selection, audit = (
        tmp_path / name for name in ("inv.parquet", "ids.json", "audit.json")
    )
    pq.write_table(pa.Table.from_pylist(rows), inventory)
    frozen: dict[str, Any] = {
        "schema": "bigearthnet-acquisition-selection-v1",
        "partitions": partitions,
        "policy": {"fixed": True},
        "footprint_inventory_sha256": file_sha256(inventory),
        "footprint_report_sha256": "a" * 64,
    }

    def write_inputs() -> None:
        selection.write_text(json.dumps(frozen))
        audit.write_text(
            json.dumps(
                {
                    "selection_sha256": file_sha256(selection),
                    "policy": {"fixed": True},
                    "footprint_report_sha256": "a" * 64,
                    "footprint_inventory_sha256": file_sha256(inventory),
                }
            )
        )

    write_inputs()
    result = s2.frozen_inputs(selection, audit, inventory)
    assert len(result) == 5000 and result[partitions["final"][0]]["partition"] == "final"
    # Extra labels/partition columns in the inventory must not leak through the column projection.
    assert "policy" not in next(iter(result.values()))
    selection.write_text("corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        s2.frozen_inputs(selection, audit, inventory)
    frozen["policy"] = {"fixed": False}
    write_inputs()
    with pytest.raises(ValueError, match="provenance"):
        s2.frozen_inputs(selection, audit, inventory)
    frozen["policy"] = {"fixed": True}
    partitions["final"][0] = partitions["index"][0]
    write_inputs()
    with pytest.raises(ValueError, match="uniqueness"):
        s2.frozen_inputs(selection, audit, inventory)


def test_checkpoint_storage_measurements_include_atomic_checkpoint(tmp_path: Path) -> None:
    budget = aio.StorageBudget(tmp_path, [], limit=4096)
    state: dict[str, Any] = {"status": "incomplete"}
    path = tmp_path / "state.json"
    budget.checkpoint(path, state)
    assert state["storage_bytes"] == path.stat().st_size == budget.used
    old_size = path.stat().st_size
    state["message"] = "interrupted" * 20
    budget.checkpoint(path, state)
    assert state["storage_bytes"] == path.stat().st_size == budget.used
    assert state["storage_peak_bytes"] == old_size + path.stat().st_size == budget.peak


def test_corrupt_staging_and_changed_bindings_fail_without_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: tuple[dict[str, Any], dict[str, Any], bytes],
) -> None:
    rows, binding, payload = source
    requests = serve(monkeypatch, payload)
    root = tmp_path / "pilot"
    s2.acquire(
        rows, binding=binding, root=root, ancillary=[], mode="acquire", network_budget=len(payload)
    )
    # Simulate a crash after files/checksum but before writing the completion marker.
    (root / "COMPLETE.json").unlink()
    (root / "files" / identity() / "B01.tif").write_bytes(b"disk corruption")
    with pytest.raises(ValueError, match="provenance"):
        s2.acquire(
            rows,
            binding={**binding, "different_audit": True},
            root=root,
            ancillary=[],
            mode="acquire",
            network_budget=2 * len(payload),
            resume=True,
        )
    assert len(requests) == 1
    with pytest.raises(ValueError, match="staged band differs"):
        s2.acquire(
            rows,
            binding=binding,
            root=root,
            ancillary=[],
            mode="acquire",
            network_budget=2 * len(payload),
            resume=True,
        )
    assert not (root / "COMPLETE.json").exists()
