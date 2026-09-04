"""Streaming content digests shared by every provenance-recording component.

Manifests, chips, embedding stores, benchmark reports, and model checkpoints all
record content identity. Keeping one implementation here means the chunk size,
the hexadecimal case, and the verification error text cannot drift apart.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_BYTES = 1024 * 1024


class StreamDigests:
    """Incremental identities for a non-retained source stream; never pickle hash state."""

    def __init__(self) -> None:
        self._md5 = hashlib.md5(usedforsecurity=False)
        self._sha256 = hashlib.sha256()
        self.bytes = 0

    def update(self, payload: bytes) -> None:
        self._md5.update(payload)
        self._sha256.update(payload)
        self.bytes += len(payload)

    def values(self) -> dict[str, str | int]:
        return {"bytes": self.bytes, "md5": self._md5.hexdigest(),
                "sha256": self._sha256.hexdigest()}


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stream_digest(path: Path, digest: hashlib._Hash) -> str:
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one local file."""

    return _stream_digest(path, hashlib.sha256())


def file_md5(path: Path) -> str:
    """Return a streaming MD5 digest used only for published archive identity.

    MD5 is not a security control here. It is the digest that the EuroSAT
    publisher distributes, so it is the digest that can be compared.
    """

    return _stream_digest(path, hashlib.md5(usedforsecurity=False))


def verify_sha256(path: Path, expected: str) -> str:
    """Validate a file's identity before deserializing or trusting it."""

    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")
    observed = file_sha256(path)
    if observed.lower() != expected.lower():
        raise ValueError(f"checkpoint checksum mismatch: expected {expected}, observed {observed}")
    return observed
