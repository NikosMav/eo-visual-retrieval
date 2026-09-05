"""Real-model execution checks, explicitly not a judged semantic benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

from eo_visual_retrieval.app.search import decode_search_image
from eo_visual_retrieval.embeddings.remoteclip import RemoteClipEncoder
from eo_visual_retrieval.hashing import file_sha256
from eo_visual_retrieval.multimodal import MultimodalSearch
from eo_visual_retrieval.search_plan import SearchFilters, SearchPlan, plan_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    encoder = RemoteClipEncoder()
    engine = MultimodalSearch.load(
        manifest=args.manifest,
        image_root=args.image_root,
        embeddings=args.embeddings,
        encoder=encoder,
    )
    example = next(
        item
        for item, split in zip(engine.store.ids, engine.store.splits, strict=True)
        if split == "query"
    )
    path = engine.image_path(example)
    with Image.open(path) as source:
        image = source.convert("RGB")
    uploaded = decode_search_image(path.read_bytes())
    vector = encoder.encode_image(image)
    uploaded_vector = encoder.encode_image(uploaded)
    position = engine.store.ids.index(example)
    error = float(np.max(np.abs(vector - engine.store.vectors[position])))
    upload_error = float(np.max(np.abs(vector - uploaded_vector)))
    assert error < 1e-5 and upload_error < 1e-6
    text = "Industrial buildings, roads and agricultural fields"
    text_result = engine.search(SearchPlan(text), k=5)
    image_result = engine.search(SearchPlan(""), image=uploaded, k=5)
    hybrid = engine.search(SearchPlan(text), image=uploaded, text_weight=0.65, k=5)
    assert all(len(result["results"]) == 5 for result in (text_result, image_result, hybrid))
    assert [r["item_id"] for r in image_result["results"]] == [
        r["item_id"] for r in engine.search(SearchPlan(""), item_id=example, k=5)["results"]
    ]
    for result in hybrid["results"]:
        assert (
            abs(result["score"] - (0.65 * result["text_score"] + 0.35 * result["image_score"]))
            < 1e-6
        )
    prompt = "Sentinel imagery showing recent urban expansion near Athens with low cloud coverage."
    recent_plan = plan_query(prompt, today=date(2026, 9, 5))
    recent = engine.search(recent_plan, k=5)
    assert recent["results"] == []  # This smoke uses historical imagery.
    historical = engine.search(
        plan_query(
            "Sentinel imagery of agricultural landscapes with low cloud coverage",
            overrides=SearchFilters(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                bbox=(23.4, 38.4, 23.8, 38.7),
            ),
        ),
        k=5,
    )
    assert historical["candidate_count"] > 0
    for result in historical["results"]:
        metadata = result["metadata"]
        assert metadata["date"].startswith("2024") and metadata["cloud_cover"] <= 10
        lon, lat = metadata["centroid_lonlat"]
        assert 23.4 <= lon <= 23.8 and 38.4 <= lat <= 38.7
    try:
        encoder.encode_text("industrial " * 100)
    except ValueError:
        long_text_rejected = True
    else:
        raise AssertionError("overlength CLIP text was not rejected")
    image.close()
    uploaded.close()
    corpus = engine.describe()
    report = {
        "evidence_role": "execution-smoke-not-semantic-quality",
        "date": "2026-09-05",
        "model": encoder.metadata,
        "manifest_sha256": file_sha256(args.manifest),
        "embedding_store_sha256": file_sha256(args.embeddings),
        "items": len(engine.store.ids),
        "index_items": corpus["index_items"],
        "date_range": corpus["date_range"],
        "metadata_coverage": corpus["metadata_coverage"],
        "image_query_max_abs_difference_from_store": error,
        "uploaded_image_max_abs_difference": upload_error,
        "modes_returned_five_results": [r["mode"] for r in (text_result, image_result, hybrid)],
        "hybrid_weight": 0.65,
        "hybrid_score_formula_verified": True,
        "recent_athens_matches": recent["candidate_count"],
        "historical_explicit_bbox": [23.4, 38.4, 23.8, 38.7],
        "historical_filtered_matches": historical["candidate_count"],
        "long_text_rejected": long_text_rejected,
        "semantic_quality_measured": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
