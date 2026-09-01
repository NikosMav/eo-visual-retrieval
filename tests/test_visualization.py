from pathlib import Path

import numpy as np
from PIL import Image

from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.models import ImageRecord
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
