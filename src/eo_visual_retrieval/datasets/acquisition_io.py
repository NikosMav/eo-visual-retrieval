"""Bounded files and HTTP range streaming for acquisition, with explicit retry costs."""

from __future__ import annotations

import io
import json
import os
import stat
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from eo_visual_retrieval.hashing import StreamDigests

STORAGE_LIMIT = 2_147_483_648
RANGE_BYTES = 16 * 1024 * 1024
READ_BYTES = 64 * 1024


class AcquisitionLimit(ValueError):
    """A hard byte guard stopped the acquisition before the disallowed read/write."""


def _plain(path: Path) -> None:
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"acquisition paths cannot contain links: {path}")


class StorageBudget:
    """Account logical file bytes, including atomic-write staging and retained sources.

    One owner holds the acquisition lock. Ancillary roots must not change during
    the run; they contain the already pinned metadata and footprint inputs.
    """

    def __init__(self, root: Path, ancillary: list[Path], *, limit: int = STORAGE_LIMIT) -> None:
        if not 0 < limit <= STORAGE_LIMIT:
            raise ValueError("storage ceiling must be between one byte and 2 GiB")
        self.root = root.resolve()
        self.limit = limit
        paths: set[Path] = set()
        for base in [root, *ancillary]:
            _plain(base)
            if base.is_dir():
                for path in base.rglob("*"):
                    _plain(path)
                    if path.is_file():
                        paths.add(path.resolve())
            elif base.is_file():
                paths.add(base.resolve())
            else:
                raise ValueError(f"missing budget input: {base}")
        self.used = sum(path.stat().st_size for path in paths)
        self.peak = self.used
        self.reserve(0)

    def reserve(self, size: int) -> None:
        if size < 0 or self.used + size > self.limit:
            raise AcquisitionLimit("2 GiB acquisition storage ceiling exceeded")

    def write(self, path: Path, payload: bytes) -> None:
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("write outside acquisition directory")
        for parent in (path, *path.parents):
            _plain(parent)
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            raise ValueError("stale staging file: resume recovery is required")
        self.reserve(len(payload))
        path.parent.mkdir(parents=True, exist_ok=True)
        old_size = path.stat().st_size if path.exists() else 0
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            self.peak = max(self.peak, self.used + len(payload))
            temporary.replace(path)
            self.used += len(payload) - old_size
        finally:
            temporary.unlink(missing_ok=True)

    def json(self, path: Path, value: dict[str, Any]) -> None:
        self.write(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())

    def checkpoint(self, path: Path, value: dict[str, Any]) -> None:
        """Include this atomic checkpoint itself in its measured storage accounting."""
        old_size = path.stat().st_size if path.exists() else 0
        previous_peak = max(value.get("storage_peak_bytes", 0), self.peak)
        while True:
            payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
            used = self.used + len(payload) - old_size
            peak = max(previous_peak, self.used + len(payload))
            if value.get("storage_bytes") == used and value.get("storage_peak_bytes") == peak:
                break
            value.update(storage_bytes=used, storage_peak_bytes=peak)
        self.write(path, payload)


@contextmanager
def acquisition_lock(root: Path, *, resume: bool) -> Iterator[None]:
    for path in (root, *root.parents):
        _plain(path)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".lock"
    _plain(lock_path)
    with lock_path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            # Only this dedicated workspace owns these exact staging suffixes.
            for path in root.rglob("*.tmp"):
                _plain(path)
                if not resume:
                    raise ValueError("interrupted staging exists; use --resume")
                if not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError("staging recovery outside acquisition directory")
                path.unlink()
            yield
        finally:
            if sys.platform == "win32":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class RangeStream(io.RawIOBase):
    """Read bounded identity-encoded ranges; retries continue the live decoder.

    Reserve an entire HTTP response body before each request. On a crash/error
    its reservation remains charged, even if the received-byte count is unknown.
    This conservative durable ledger prevents retries silently multiplying costs.
    """

    def __init__(
        self,
        url: str,
        size: int,
        *,
        network_budget: int,
        state: dict[str, Any],
        persist: Callable[[], None],
        range_bytes: int = RANGE_BYTES,
        retries: int = 3,
        timeout: float = 30,
        deadline: float = float("inf"),
    ) -> None:
        super().__init__()
        self.url, self.size = url, size
        self.network_budget, self.state, self.persist = network_budget, state, persist
        self.range_bytes, self.retries, self.timeout = range_bytes, retries, timeout
        self.deadline = deadline
        self.offset = 0
        self.digest = StreamDigests()
        self.response: Any = None
        self.end = 0

    def readable(self) -> bool:
        return True

    def _disconnect(self) -> None:
        if self.response is not None:
            self.response.close()
            self.response = None

    def _connect(self) -> None:
        if time.monotonic() >= self.deadline:
            raise AcquisitionLimit("acquisition wall-clock limit reached")
        available = self.network_budget - self.state["network_reserved_bytes"]
        if available <= 0:
            raise AcquisitionLimit("cumulative network byte budget exhausted; source unverified")
        count = min(self.range_bytes, self.size - self.offset, available)
        self.end = self.offset + count
        self.state["network_reserved_bytes"] += count
        self.state["http_requests"] += 1
        self.persist()  # Charge before opening the connection, including failed requests.
        request = Request(
            self.url,
            headers={
                "Range": f"bytes={self.offset}-{self.end - 1}",
                "Accept-Encoding": "identity",
            },
        )
        response = urlopen(request, timeout=min(self.timeout, self.deadline - time.monotonic()))
        expected = f"bytes {self.offset}-{self.end - 1}/{self.size}"
        if (
            response.status != 206
            or response.headers.get("Content-Range") != expected
            or (response.headers.get("Content-Encoding", "identity") != "identity")
            or int(response.headers.get("Content-Length", count)) != count
        ):
            response.close()
            raise ValueError("server did not return the exact identity-encoded byte range")
        self.response = response

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("read on closed range stream")
        if size < 0:
            raise ValueError("unbounded source reads are forbidden")
        if size == 0 or self.offset == self.size:
            return b""
        failures = 0
        while True:
            if time.monotonic() >= self.deadline:
                raise AcquisitionLimit("acquisition wall-clock limit reached")
            try:
                if self.response is None:
                    self._connect()
                data = self.response.read(min(size, READ_BYTES, self.end - self.offset))
                if not data:
                    raise OSError("truncated HTTP range response")
                if len(data) > min(size, READ_BYTES, self.end - self.offset):
                    raise ValueError("HTTP response exceeded requested read size")
                self.offset += len(data)
                self.digest.update(data)
                self.state["network_received_bytes"] += len(data)
                self.state["attempt_compressed_bytes"] = self.offset
                if self.offset == self.end:
                    self._disconnect()
                    self.persist()
                return bytes(data)
            except (OSError, URLError, HTTPException):
                self._disconnect()
                failures += 1
                self.state["transport_retries"] += 1
                self.persist()
                if failures > self.retries:
                    raise

    def readinto(self, buffer: Any) -> int:
        payload = self.read(len(buffer))
        buffer[: len(payload)] = payload
        return len(payload)

    def close(self) -> None:
        self._disconnect()
        super().close()
