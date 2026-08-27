from pathlib import Path

import pytest
from PIL import Image

from eo_visual_retrieval.manifests import build_image_manifest, read_jsonl, write_jsonl


def _make_images(root: Path) -> None:
    for label, channel in (("forest", 1), ("harbor", 2)):
        folder = root / label
        folder.mkdir(parents=True)
        for index in range(8):
            color = [0, 0, 0]
            color[channel] = 20 + index
            Image.new("RGB", (16, 16), tuple(color)).save(folder / f"{index}.png")


def test_manifest_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    _make_images(image_root)

    first = build_image_manifest(image_root, query_fraction=0.25, seed=7)
    second = build_image_manifest(image_root, query_fraction=0.25, seed=7)
    assert first == second
    assert {record.label for record in first} == {"forest", "harbor"}
    assert {record.split for record in first} == {"index", "query"}
    for label in ("forest", "harbor"):
        assert {record.split for record in first if record.label == label} == {"index", "query"}

    output = tmp_path / "manifest.jsonl"
    write_jsonl(first, output)
    assert read_jsonl(output) == first


def test_identical_content_with_conflicting_labels_is_rejected(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    for label in ("a", "b"):
        folder = image_root / label
        folder.mkdir(parents=True)
        Image.new("RGB", (8, 8), (1, 2, 3)).save(folder / "duplicate.png")

    with pytest.raises(ValueError, match="conflicting labels"):
        build_image_manifest(image_root, query_fraction=0.5, seed=3)
