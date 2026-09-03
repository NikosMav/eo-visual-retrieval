from pathlib import Path

import numpy as np
from PIL import Image

from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.models import ImageRecord, Split
from eo_visual_retrieval.visualization import write_result_grid


def test_result_grid_writes_one_query_row_per_class(tmp_path: Path) -> None:
    ids = ("a-i1", "a-i2", "b-i1", "b-i2", "a-q", "b-q")
    labels = ("a", "a", "b", "b", "a", "b")
    splits = ("index", "index", "index", "index", "query", "query")
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.95, 0.05],
            [0.05, 0.95],
        ],
        dtype=np.float32,
    )
    records: list[ImageRecord] = []
    for index, (item_id, label, split) in enumerate(zip(ids, labels, splits, strict=True)):
        path = tmp_path / f"{item_id}.png"
        Image.new("RGB", (16, 16), (index * 20, 30, 40)).save(path)
        records.append(
            ImageRecord(
                item_id=item_id,
                path=path.name,
                label=label,
                split=split,  # type: ignore[arg-type]
            )
        )
    store = EmbeddingStore(
        ids=ids,
        labels=labels,
        splits=splits,
        vectors=vectors,
        metadata={"backend": "test"},
    )
    output = tmp_path / "grid.png"

    selected = write_result_grid(
        store,
        records,
        image_root=tmp_path,
        output=output,
        k=2,
        mode="worst",
    )

    assert [evaluation.label for evaluation in selected] == ["a", "b"]
    with Image.open(output) as grid:
        assert grid.format == "PNG"
        assert grid.width == 336
        assert grid.height > 200


def test_grid_title_distinguishes_two_variants_of_one_backend(tmp_path: Path) -> None:
    """A 13-band and an RGB SSL4EO grid would otherwise carry the same title."""
    from PIL import Image

    root = tmp_path / "images"
    root.mkdir()
    records = []
    splits: tuple[Split, ...] = ("index", "index", "query")
    for index, split in enumerate(splits):
        name = f"forest/img-{index}.png"
        (root / "forest").mkdir(exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), 40 + index * 10, dtype=np.uint8)).save(root / name)
        records.append(
            ImageRecord(item_id=name, path=name, split=split, label="forest")
        )
    store = EmbeddingStore(
        ids=tuple(r.item_id for r in records),
        vectors=np.asarray([[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]], dtype=np.float32),
        labels=tuple(r.label for r in records),
        splits=tuple(r.split for r in records),
        metadata={"backend": "ssl4eo-s12", "model": "ssl4eo-s12-rgb-moco-resnet50"},
    )

    output = tmp_path / "grid.png"
    write_result_grid(store, records, image_root=root, output=output, k=1, mode="best")

    assert output.is_file()
    # The specific model wins over the shared backend name.
    assert store.metadata["model"] != store.metadata["backend"]
