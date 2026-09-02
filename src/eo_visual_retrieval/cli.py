"""Command-line interface for the reproducible retrieval workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
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


def _run_metadata(
    records: list[Any], manifest: Path, packages: tuple[str, ...]
) -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "items": len(records),
        "index_items": sum(record.split == "index" for record in records),
        "query_items": sum(record.split == "query" for record in records),
        "python": platform.python_version(),
        "packages": versions,
    }


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


def _benchmark_eurosat_prepare(args: argparse.Namespace) -> None:
    from eo_visual_retrieval.benchmarks.eurosat import prepare_eurosat_benchmark

    result = prepare_eurosat_benchmark(
        args.archive,
        output_dir=args.output_dir,
        manifest=args.manifest,
        queries_per_class=args.queries_per_class,
        index_per_class=args.index_per_class,
        group_size_m=args.group_size_km * 1000,
        minimum_separation_m=args.minimum_separation_km * 1000,
        seed=args.seed,
    )
    counts = {
        split: sum(record.split == split for record in result.records)
        for split in ("index", "query")
    }
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "manifest": str(args.manifest),
                "discovered": result.discovered,
                **counts,
                "minimum_separation_km": result.minimum_separation_m / 1000,
                "excluded_near_query": result.excluded_near_query,
            },
            indent=2,
        )
    )


def _benchmark_eurosat_audit(args: argparse.Namespace) -> None:
    from eo_visual_retrieval.benchmarks.eurosat import audit_eurosat_manifest

    audit = audit_eurosat_manifest(
        args.manifest,
        image_root=args.image_root,
        expected_index_per_class=args.expected_index_per_class,
        expected_queries_per_class=args.expected_queries_per_class,
    )
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))


def _embed_dinov2(args: argparse.Namespace) -> None:
    from eo_visual_retrieval.embeddings.dinov2 import dinov2_embeddings

    records = read_jsonl(args.manifest)
    vectors = dinov2_embeddings(
        _paths(records, args.image_root),
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    store = _store(
        records,
        vectors,
        {
            "backend": "dinov2",
            "model": args.model,
            "device_requested": args.device,
            "preprocessing": "RGB, 224x224 bicubic, ImageNet normalization",
            **_run_metadata(records, args.manifest, ("numpy", "Pillow", "torch", "torchvision")),
        },
    )
    store.save(args.output)
    print(json.dumps({"output": str(args.output), "shape": list(vectors.shape)}, indent=2))


def _embed_ssl4eo(args: argparse.Namespace) -> None:
    from eo_visual_retrieval.benchmarks.eurosat import EUROSAT_ARCHIVE_MD5
    from eo_visual_retrieval.embeddings.ssl4eo import (
        CHECKPOINT_FILENAME,
        CHECKPOINT_REPOSITORY,
        CHECKPOINT_REVISION,
        CHECKPOINT_SHA256,
        MODEL_NAME,
        SSL4EO_BAND_ORDER,
        ssl4eo_embeddings,
    )

    records = read_jsonl(args.manifest)
    vectors = ssl4eo_embeddings(
        records,
        archive=args.archive,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
    )
    store = _store(
        records,
        vectors,
        {
            "backend": "ssl4eo-s12",
            "model": MODEL_NAME,
            "checkpoint_repository": CHECKPOINT_REPOSITORY,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_filename": CHECKPOINT_FILENAME,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "archive_md5": EUROSAT_ARCHIVE_MD5,
            "bands": list(SSL4EO_BAND_ORDER),
            "device_requested": args.device,
            "frozen": True,
            "preprocessing": (
                "13-band L1C DN clipped to 0-10000 and divided by 10000; "
                "resize 256, center crop 224"
            ),
            **_run_metadata(
                records,
                args.manifest,
                ("numpy", "torch", "torchvision", "rasterio"),
            ),
        },
    )
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
            "fit_partition": "index",
            "preprocessing": "RGB, square resize, 0-1 scaling, flattened pixels",
            **_run_metadata(records, args.manifest, ("numpy", "Pillow", "scikit-learn")),
        },
    )
    store.save(args.output)
    print(json.dumps({"output": str(args.output), "shape": list(vectors.shape)}, indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    summary = evaluate_store(EmbeddingStore.load(args.embeddings), k=args.k)
    payload = json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        temporary.replace(args.output)
    print(payload, end="")


def _result_grid(args: argparse.Namespace) -> None:
    from eo_visual_retrieval.visualization import write_result_grid

    selected = write_result_grid(
        EmbeddingStore.load(args.embeddings),
        read_jsonl(args.manifest),
        image_root=args.image_root,
        output=args.output,
        k=args.k,
        mode=args.mode,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "mode": args.mode,
                "k": args.k,
                "queries": [
                    {
                        "item_id": evaluation.query_id,
                        "label": evaluation.label,
                        "average_precision_at_k": evaluation.average_precision_at_k,
                    }
                    for evaluation in selected
                ],
            },
            indent=2,
        )
    )


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

    eurosat = commands.add_parser(
        "benchmark-eurosat-prepare",
        help="prepare a class-balanced, spatially separated EuroSAT benchmark",
    )
    eurosat.add_argument("--archive", type=Path, required=True)
    eurosat.add_argument("--output-dir", type=Path, required=True)
    eurosat.add_argument("--manifest", type=Path, required=True)
    eurosat.add_argument("--queries-per-class", type=int, default=40)
    eurosat.add_argument("--index-per-class", type=int, default=160)
    eurosat.add_argument("--group-size-km", type=float, default=50.0)
    eurosat.add_argument("--minimum-separation-km", type=float, default=5.0)
    eurosat.add_argument("--seed", type=int, default=42)
    eurosat.set_defaults(handler=_benchmark_eurosat_prepare)

    eurosat_audit = commands.add_parser(
        "benchmark-eurosat-audit",
        help="audit a prepared EuroSAT manifest and optionally verify image hashes",
    )
    eurosat_audit.add_argument("--manifest", type=Path, required=True)
    eurosat_audit.add_argument("--image-root", type=Path)
    eurosat_audit.add_argument("--expected-index-per-class", type=int, default=160)
    eurosat_audit.add_argument("--expected-queries-per-class", type=int, default=40)
    eurosat_audit.set_defaults(handler=_benchmark_eurosat_audit)

    dinov2 = commands.add_parser("embed-dinov2", help="embed manifest images with DINOv2")
    dinov2.add_argument("--manifest", type=Path, required=True)
    dinov2.add_argument("--image-root", type=Path, required=True)
    dinov2.add_argument("--output", type=Path, required=True)
    dinov2.add_argument("--model", default="dinov2_vits14")
    dinov2.add_argument("--batch-size", type=int, default=8)
    dinov2.add_argument("--device", default="auto")
    dinov2.set_defaults(handler=_embed_dinov2)

    ssl4eo = commands.add_parser(
        "embed-ssl4eo",
        help="embed selected EuroSAT members with frozen 13-band SSL4EO-S12",
    )
    ssl4eo.add_argument("--manifest", type=Path, required=True)
    ssl4eo.add_argument("--archive", type=Path, required=True)
    ssl4eo.add_argument("--checkpoint", type=Path, required=True)
    ssl4eo.add_argument("--output", type=Path, required=True)
    ssl4eo.add_argument("--batch-size", type=int, default=16)
    ssl4eo.add_argument("--device", default="auto")
    ssl4eo.set_defaults(handler=_embed_ssl4eo)

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
    evaluate.add_argument("--output", type=Path, help="optionally write the JSON result atomically")
    evaluate.set_defaults(handler=_evaluate)

    grid = commands.add_parser(
        "result-grid",
        help="render one best or worst exact-retrieval query per class",
    )
    grid.add_argument("--embeddings", type=Path, required=True)
    grid.add_argument("--manifest", type=Path, required=True)
    grid.add_argument("--image-root", type=Path, required=True)
    grid.add_argument("--output", type=Path, required=True)
    grid.add_argument("--k", type=int, default=5)
    grid.add_argument("--mode", choices=("best", "worst"), default="worst")
    grid.set_defaults(handler=_result_grid)

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
