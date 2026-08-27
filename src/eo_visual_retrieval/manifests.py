"""Build and read deterministic, portable image manifests."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from eo_visual_retrieval.models import ImageRecord

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_rank(key: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{key}".encode()).digest()


def build_image_manifest(
    image_root: Path,
    *,
    query_fraction: float = 0.2,
    seed: int = 42,
) -> list[ImageRecord]:
    """Scan class folders and assign deterministic index/query splits.

    The first path component below ``image_root`` is used as a label. Images
    directly inside the root are left unlabeled. Identical files receive the
    same split because the content hash is the split key.
    """

    if not 0 < query_fraction < 1:
        raise ValueError("query_fraction must be between 0 and 1")
    if not image_root.is_dir():
        raise ValueError(f"image root does not exist: {image_root}")

    paths = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"no supported images found under {image_root}")

    entries: list[tuple[Path, str, str | None]] = []
    label_by_hash: dict[str, str | None] = {}
    for path in paths:
        relative = path.relative_to(image_root)
        content_hash = _sha256(path)
        label = relative.parts[0] if len(relative.parts) > 1 else None
        if content_hash in label_by_hash and label_by_hash[content_hash] != label:
            raise ValueError("identical image content has conflicting labels")
        label_by_hash[content_hash] = label
        entries.append((relative, content_hash, label))

    hashes_by_label: dict[str | None, set[str]] = defaultdict(set)
    for _, content_hash, label in entries:
        hashes_by_label[label].add(content_hash)

    query_hashes: set[str] = set()
    for content_hashes in hashes_by_label.values():
        ordered = sorted(content_hashes, key=lambda value: _split_rank(value, seed))
        if len(ordered) < 2:
            continue
        query_count = round(len(ordered) * query_fraction)
        query_count = max(1, min(len(ordered) - 1, query_count))
        query_hashes.update(ordered[:query_count])

    records: list[ImageRecord] = []
    for relative, content_hash, label in entries:
        split = "query" if content_hash in query_hashes else "index"
        records.append(
            ImageRecord(
                item_id=relative.as_posix(),
                path=relative.as_posix(),
                split=split,
                label=label,
                metadata={"sha256": content_hash},
            )
        )

    _validate_manifest(records)
    return records


def _validate_manifest(records: Iterable[ImageRecord]) -> None:
    seen_ids: set[str] = set()
    split_by_hash: dict[str, str] = {}
    for record in records:
        if record.item_id in seen_ids:
            raise ValueError(f"duplicate item_id: {record.item_id}")
        seen_ids.add(record.item_id)
        content_hash = record.metadata.get("sha256")
        if content_hash:
            previous = split_by_hash.setdefault(str(content_hash), record.split)
            if previous != record.split:
                raise ValueError("identical image content appears across splits")


def write_jsonl(records: Iterable[ImageRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    ordered = sorted(records, key=lambda record: record.item_id)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in ordered:
            stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    temporary.replace(output)


def read_jsonl(path: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(ImageRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid manifest record at line {line_number}") from error
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    _validate_manifest(records)
    return sorted(records, key=lambda record: record.item_id)
