"""Command-line interface for the reproducible retrieval workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.evaluation import evaluate_store
from eo_visual_retrieval.manifests import build_image_manifest, read_jsonl, write_jsonl
from eo_visual_retrieval.retrieval import ExactCosineIndex
from eo_visual_retrieval.stac import (
    StacSearchConfig,
    materialize_previews,
    read_stac_jsonl,
    search_stac,
    write_stac_jsonl,
)


def _paths(records: list[Any], image_root: Path) -> list[Path]:
    paths = [image_root / record.path for record in records]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"manifest references missing image: {missing[0]}")
    return paths


def _store(records: list[Any], vectors: Any, metadata: dict[str, Any]) -> EmbeddingStore:
    return EmbeddingStore(
        ids=tuple(record.item_id for record in records),
        vectors=vectors,
        labels=tuple(record.label for record in records),
        splits=tuple(record.split for record in records),
        metadata=metadata,
    )


def _manifest_build(args: argparse.Namespace) -> None:
    records = build_image_manifest(
        args.images,
        query_fraction=args.query_fraction,
        seed=args.seed,
    )
    write_jsonl(records, args.output)
    counts = {
        split: sum(record.split == split for record in records)
        for split in ("index", "query")
    }
    print(json.dumps({"output": str(args.output), "items": len(records), **counts}, indent=2))


def _stac_search(args: argparse.Namespace) -> None:
    config = StacSearchConfig(
        api_url=args.api_url,
        collection=args.collection,
        bbox=tuple(args.bbox),
        datetime=args.datetime,
        limit=args.limit,
        max_cloud_cover=args.max_cloud_cover,
    )
    records = search_stac(config)
    write_stac_jsonl(records, args.output)
    print(json.dumps({"output": str(args.output), "items": len(records)}, indent=2))


def _stac_materialize(args: argparse.Namespace) -> None:
    records = read_stac_jsonl(args.manifest)
    if args.limit is not None:
        records = records[: args.limit]
    images = materialize_previews(
        records,
        output_dir=args.output_dir,
        image_manifest=args.image_manifest,
        asset_key=args.asset,
        signer=args.signer,
        max_bytes=args.max_bytes,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "image_manifest": str(args.image_manifest),
                "items": len(images),
            },
            indent=2,
        )
    )


def _stac_chip(args: argparse.Namespace) -> None:
    from eo_visual_retrieval.chips import materialize_sentinel2_chip

    records = read_stac_jsonl(args.manifest)
    matches = [record for record in records if record.item_id == args.item_id]
    if not matches:
        raise ValueError(f"item ID not found in STAC manifest: {args.item_id}")
    artifacts = materialize_sentinel2_chip(
        matches[0],
        output_dir=args.output_dir,
        image_manifest=args.image_manifest,
        bounds=tuple(args.bbox),
        signer=args.signer,
        reflectance_min=args.reflectance_min,
        reflectance_max=args.reflectance_max,
        mask_scl=args.mask_scl,
        max_pixels=args.max_pixels,
    )
    print(
        json.dumps(
            {
                "reflectance": str(artifacts.reflectance_path),
                "rgb": str(artifacts.rgb_path),
                "image_manifest": str(args.image_manifest),
                "item_id": artifacts.image_record.item_id,
                "shape": [
                    artifacts.image_record.metadata["height"],
                    artifacts.image_record.metadata["width"],
                ],
            },
            indent=2,
        )
    )


def _embed_dinov2(args: argparse.Namespace) -> None:
    from eo_visual_retrieval.embeddings.dinov2 import dinov2_embeddings

    records = read_jsonl(args.manifest)
    vectors = dinov2_embeddings(
        _paths(records, args.image_root),
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    store = _store(records, vectors, {"backend": "dinov2", "model": args.model})
    store.save(args.output)
    print(json.dumps({"output": str(args.output), "shape": list(vectors.shape)}, indent=2))


def _embed_pca(args: argparse.Namespace) -> None:
    from eo_visual_retrieval.embeddings.pca import pca_embeddings

    records = read_jsonl(args.manifest)
    vectors = pca_embeddings(
        _paths(records, args.image_root),
        [record.split for record in records],
        components=args.components,
        image_size=args.image_size,
        seed=args.seed,
    )
    store = _store(
        records,
        vectors,
        {
            "backend": "pca",
            "components": args.components,
            "image_size": args.image_size,
            "seed": args.seed,
        },
    )
    store.save(args.output)
    print(json.dumps({"output": str(args.output), "shape": list(vectors.shape)}, indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    summary = evaluate_store(EmbeddingStore.load(args.embeddings), k=args.k)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


def _query(args: argparse.Namespace) -> None:
    store = EmbeddingStore.load(args.embeddings)
    try:
        query_position = store.ids.index(args.item_id)
    except ValueError as error:
        raise ValueError(f"item ID not found: {args.item_id}") from error
    index_positions = [i for i, split in enumerate(store.splits) if split == "index"]
    index = ExactCosineIndex(
        [store.ids[i] for i in index_positions],
        store.vectors[index_positions],
    )
    results = index.search(store.vectors[query_position], k=args.k, exclude_id=args.item_id)
    payload = [{"item_id": result.item_id, "score": result.score} for result in results]
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eovr", description="EO visual retrieval workflow")
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("manifest-build", help="build a deterministic image manifest")
    manifest.add_argument("--images", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--query-fraction", type=float, default=0.2)
    manifest.add_argument("--seed", type=int, default=42)
    manifest.set_defaults(handler=_manifest_build)

    stac = commands.add_parser("stac-search", help="write a sanitized STAC item manifest")
    stac.add_argument("--api-url", required=True)
    stac.add_argument("--collection", required=True)
    stac.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
    stac.add_argument("--datetime", required=True)
    stac.add_argument("--max-cloud-cover", type=float)
    stac.add_argument("--limit", type=int, default=20)
    stac.add_argument("--output", type=Path, required=True)
    stac.set_defaults(handler=_stac_search)

    materialize = commands.add_parser(
        "stac-materialize",
        help="download bounded preview images from a sanitized STAC manifest",
    )
    materialize.add_argument("--manifest", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--image-manifest", type=Path, required=True)
    materialize.add_argument("--asset", default="rendered_preview")
    materialize.add_argument(
        "--signer",
        choices=("none", "planetary-computer"),
        default="none",
    )
    materialize.add_argument("--limit", type=int)
    materialize.add_argument("--max-bytes", type=int, default=20 * 1024 * 1024)
    materialize.set_defaults(handler=_stac_materialize)

    chip = commands.add_parser(
        "stac-chip",
        help="materialize one aligned Sentinel-2 RGB chip",
    )
    chip.add_argument("--manifest", type=Path, required=True)
    chip.add_argument("--item-id", required=True)
    chip.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
    chip.add_argument("--output-dir", type=Path, required=True)
    chip.add_argument("--image-manifest", type=Path, required=True)
    chip.add_argument(
        "--signer",
        choices=("none", "planetary-computer"),
        default="none",
    )
    chip.add_argument("--reflectance-min", type=float, default=0.0)
    chip.add_argument("--reflectance-max", type=float, default=0.3)
    chip.add_argument(
        "--mask-scl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mask nodata, defective, cloud, cirrus, shadow, and snow SCL classes",
    )
    chip.add_argument("--max-pixels", type=int, default=1024 * 1024)
    chip.set_defaults(handler=_stac_chip)

    dinov2 = commands.add_parser("embed-dinov2", help="embed manifest images with DINOv2")
    dinov2.add_argument("--manifest", type=Path, required=True)
    dinov2.add_argument("--image-root", type=Path, required=True)
    dinov2.add_argument("--output", type=Path, required=True)
    dinov2.add_argument("--model", default="dinov2_vits14")
    dinov2.add_argument("--batch-size", type=int, default=8)
    dinov2.add_argument("--device", default="auto")
    dinov2.set_defaults(handler=_embed_dinov2)

    pca = commands.add_parser("embed-pca", help="embed images with index-fitted PCA")
    pca.add_argument("--manifest", type=Path, required=True)
    pca.add_argument("--image-root", type=Path, required=True)
    pca.add_argument("--output", type=Path, required=True)
    pca.add_argument("--components", type=int, default=64)
    pca.add_argument("--image-size", type=int, default=64)
    pca.add_argument("--seed", type=int, default=42)
    pca.set_defaults(handler=_embed_pca)

    evaluate = commands.add_parser("evaluate", help="evaluate label-proxy retrieval metrics")
    evaluate.add_argument("--embeddings", type=Path, required=True)
    evaluate.add_argument("--k", type=int, default=10)
    evaluate.set_defaults(handler=_evaluate)

    query = commands.add_parser("query", help="retrieve nearest index images for one item")
    query.add_argument("--embeddings", type=Path, required=True)
    query.add_argument("--item-id", required=True)
    query.add_argument("--k", type=int, default=5)
    query.set_defaults(handler=_query)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
