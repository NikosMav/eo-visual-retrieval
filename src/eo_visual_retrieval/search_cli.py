"""Optional multimodal command registration; heavy imports stay in handlers."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from eo_visual_retrieval.embeddings.remoteclip import RemoteClipEncoder
from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.hashing import file_sha256
from eo_visual_retrieval.manifests import read_jsonl
from eo_visual_retrieval.multimodal import MultimodalSearch
from eo_visual_retrieval.search_plan import SearchFilters, plan_query


def _encoder(args: argparse.Namespace, *, download: bool = False) -> RemoteClipEncoder:
    return RemoteClipEncoder(
        device=args.device,
        checkpoint=args.checkpoint,
        allow_download=download,
    )


def _embed(args: argparse.Namespace) -> None:
    from eo_visual_retrieval.cli import _paths, _run_metadata

    records = read_jsonl(args.manifest)
    paths = _paths(records, args.image_root)
    if not paths:
        raise ValueError("manifest must contain images")
    encoder = _encoder(args, download=True)
    vectors = encoder.encode_paths(paths, batch_size=args.batch_size)
    metadata = _run_metadata(records, args.manifest, ("torch", "torchvision", "open-clip-torch"))
    EmbeddingStore(
        ids=tuple(r.item_id for r in records),
        vectors=vectors,
        splits=tuple(r.split for r in records),
        labels=tuple(r.label for r in records),
        metadata={**metadata, **encoder.metadata},
    ).save(args.output)
    print(
        json.dumps(
            {"items": len(records), "output": str(args.output), **encoder.metadata}, indent=2
        )
    )


def _load(args: argparse.Namespace) -> MultimodalSearch:
    # Check the inexpensive corpus binding before loading a large model.
    store = EmbeddingStore.load(args.embeddings)
    if store.metadata.get("manifest_sha256") != file_sha256(args.manifest):
        raise ValueError("manifest SHA-256 does not match the embedding store")
    return MultimodalSearch(
        read_jsonl(args.manifest),
        store,
        _encoder(args),
        image_root=args.image_root,
    )


def _search(args: argparse.Namespace) -> None:
    from PIL import Image

    filters = SearchFilters(
        bbox=tuple(args.bbox) if args.bbox else None,
        start_date=args.start_date,
        end_date=args.end_date,
        max_cloud_cover=args.max_cloud_cover,
        collection=args.collection,
    )
    plan = plan_query(args.text or "", overrides=filters, interpret=not args.no_prompt_defaults)
    if args.plan_only:
        print(json.dumps(plan.to_dict(), indent=2))
        return
    engine = _load(args)
    image = None
    if args.image:
        with Image.open(args.image) as source:
            image = source.convert("RGB")
    try:
        result = engine.search(
            plan,
            image=image,
            item_id=args.item_id,
            text_weight=args.text_weight,
            k=args.k,
        )
    finally:
        if image is not None:
            image.close()
    print(json.dumps(result, indent=2, allow_nan=False))


def _serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn

        from eo_visual_retrieval.app.search import create_search_app
    except ImportError as error:
        raise RuntimeError('install the app extra: pip install -e ".[app]"') from error
    engine = _load(args)
    comparison = None
    if args.comparison_manifest or args.comparison_image_root or args.comparison_store:
        from eo_visual_retrieval.app.catalog import Catalog

        if not (args.comparison_manifest and args.comparison_image_root and args.comparison_store):
            raise ValueError("comparison needs a manifest, image root and at least one store")
        comparison = Catalog.load(manifest=args.comparison_manifest,
                                  image_root=args.comparison_image_root,
                                  stores=args.comparison_store,
                                  projection=args.comparison_projection)
    print(
        json.dumps(
            {
                "url": f"http://{args.host}:{args.port}",
                "index_items": engine.describe()["index_items"],
            }
        )
    )
    uvicorn.run(create_search_app(engine, results_dir=args.results_dir, comparison=comparison),
                host=args.host, port=args.port)


def register_search_commands(commands: Any) -> None:
    embed = commands.add_parser(
        "embed-remoteclip", help="build aligned RGB/text RemoteCLIP vectors"
    )
    search = commands.add_parser("search", help="text, example-image or hybrid exact search")
    serve = commands.add_parser("serve-search", help="serve the text/image/hybrid search interface")
    for command in (embed, search, serve):
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--image-root", type=Path, required=True)
        command.add_argument(
            "--checkpoint", type=Path, help="optional hash-verified local checkpoint"
        )
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    for command in (search, serve):
        command.add_argument("--embeddings", type=Path, required=True)
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--batch-size", type=int, default=16)
    embed.set_defaults(handler=_embed)
    search.add_argument("--text")
    example = search.add_mutually_exclusive_group()
    example.add_argument("--image", type=Path)
    example.add_argument("--item-id")
    search.add_argument("--text-weight", type=float, default=0.5)
    search.add_argument("--bbox", nargs=4, type=float)
    search.add_argument("--start-date", type=date.fromisoformat)
    search.add_argument("--end-date", type=date.fromisoformat)
    search.add_argument("--max-cloud-cover", type=float)
    search.add_argument("--collection")
    search.add_argument("--no-prompt-defaults", action="store_true")
    search.add_argument(
        "--plan-only", action="store_true", help="inspect filters without loading a model"
    )
    search.add_argument("--k", type=int, default=12)
    search.set_defaults(handler=_search)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8002)
    serve.add_argument("--results-dir", type=Path)
    serve.add_argument("--comparison-manifest", type=Path)
    serve.add_argument("--comparison-image-root", type=Path)
    serve.add_argument("--comparison-store", type=Path, action="append")
    serve.add_argument("--comparison-projection", type=Path)
    serve.set_defaults(handler=_serve)
