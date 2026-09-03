"""Qualitative result grids for inspecting retrieval successes and failures."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.evaluation import QueryEvaluation, evaluate_queries
from eo_visual_retrieval.models import ImageRecord


def _thumbnail(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        return ImageOps.fit(image, (size, size), method=Image.Resampling.BICUBIC)


def _selected_queries(
    evaluations: list[QueryEvaluation], mode: str
) -> list[QueryEvaluation]:
    if mode not in {"best", "worst"}:
        raise ValueError("mode must be 'best' or 'worst'")
    by_label: dict[str, list[QueryEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        by_label[evaluation.label].append(evaluation)
    selected: list[QueryEvaluation] = []
    for _label, values in sorted(by_label.items()):
        ordered = sorted(
            values,
            key=lambda value: (value.average_precision_at_k, value.query_id),
            reverse=mode == "best",
        )
        selected.append(ordered[0])
    return selected


def write_result_grid(
    store: EmbeddingStore,
    records: list[ImageRecord],
    *,
    image_root: Path,
    output: Path,
    k: int = 5,
    mode: str = "worst",
) -> list[QueryEvaluation]:
    """Render one best or worst query per class with its exact top-k results."""

    evaluations, _ = evaluate_queries(store, k=k)
    if not evaluations:
        raise ValueError("no eligible queries are available for a result grid")
    selected = _selected_queries(evaluations, mode)
    path_by_id = {record.item_id: image_root / record.path for record in records}
    needed_ids = {
        item_id
        for evaluation in selected
        for item_id in (evaluation.query_id, *evaluation.ranked_ids)
    }
    missing_records = sorted(needed_ids - path_by_id.keys())
    if missing_records:
        raise ValueError(f"embedding ID is missing from manifest: {missing_records[0]}")
    missing_files = sorted(
        str(path_by_id[item_id])
        for item_id in needed_ids
        if not path_by_id[item_id].is_file()
    )
    if missing_files:
        raise ValueError(f"manifest references missing image: {missing_files[0]}")

    tile_size = 96
    cell_width = 112
    cell_height = 130
    top_margin = 42
    width = (k + 1) * cell_width
    height = top_margin + len(selected) * cell_height + 8
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    # Two variants of one backend produce visually similar grids, so the title
    # carries the specific model when the store records one.
    label = str(store.metadata.get("model") or store.metadata.get("backend") or "unknown")
    draw.text((8, 8), f"{label} | {mode} query per class | exact cosine top-{k}", fill="black")
    draw.text((8, 24), "blue=query  green=relevant  red=not relevant", fill=(70, 70, 70))

    for row, evaluation in enumerate(selected):
        y = top_margin + row * cell_height
        ids = (evaluation.query_id, *evaluation.ranked_ids)
        borders = ((40, 100, 220),) + tuple(
            (35, 150, 70) if relevant else (205, 55, 55)
            for relevant in evaluation.relevance
        )
        for column, (item_id, border) in enumerate(zip(ids, borders, strict=True)):
            x = column * cell_width + 8
            image = _thumbnail(path_by_id[item_id], tile_size)
            canvas.paste(image, (x, y))
            draw.rectangle(
                (x - 2, y - 2, x + tile_size + 1, y + tile_size + 1),
                outline=border,
                width=3,
            )
            caption = "query" if column == 0 else f"#{column}"
            draw.text((x, y + tile_size + 4), caption, fill="black")
        score = evaluation.average_precision_at_k
        label = evaluation.label[:18]
        draw.text((8, y + tile_size + 18), f"{label}  AP@{k}={score:.3f}", fill="black")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    canvas.save(temporary, format="PNG", optimize=True)
    temporary.replace(output)
    return selected
