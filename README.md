# EO Visual Retrieval

An educational, retrieval-first project for searching Earth-observation imagery with classical PCA features and DINOv2 embeddings.

The repository is being evolved from a 2022 image-analysis notebook into a reproducible system that covers STAC discovery, embedding generation, exact vector search, and retrieval evaluation. The original notebook remains available under [`legacy/`](legacy/README.md).

## What this project demonstrates

- Provider-neutral discovery of public imagery through STAC APIs.
- Privacy-safe STAC manifests that never persist signed asset URLs.
- Frozen DINOv2 features for content-based image retrieval.
- PCA as a transparent classical baseline.
- Exact cosine search with Precision@k, Recall@k, mAP@k, and nDCG@k.
- Reproducible splits, cached embeddings, tests, and CI.

## Architecture

```text
STAC API                  Local labeled image folders
   |                                 |
   v                                 v
safe item manifest          deterministic image manifest
                                      |
                           +----------+----------+
                           |                     |
                         PCA                 DINOv2
                           |                     |
                           +----------+----------+
                                      |
                              normalized vectors
                                      |
                           exact cosine retrieval
                                      |
                               ranked evaluation
```

STAC discovery and image retrieval are deliberately separated. STAC manifests describe where imagery comes from; embedding commands operate on materialized local RGB chips. This keeps catalog credentials and temporary signed URLs out of Git while allowing the retrieval layer to work with any provider.

## Setup

Python 3.11 is recommended for the full ML stack. On Windows, create the environment at a short path because PyTorch distributions contain deeply nested files.

```powershell
py -3.11 -m venv C:\Users\<you>\.venvs\eovr
C:\Users\<you>\.venvs\eovr\Scripts\python -m pip install --upgrade pip
C:\Users\<you>\.venvs\eovr\Scripts\python -m pip install -e ".[dev,stac,ml]"
```

For lightweight development and unit tests:

```powershell
py -3.11 -m venv C:\Users\<you>\.venvs\eovr
C:\Users\<you>\.venvs\eovr\Scripts\python -m pip install -e ".[dev]"
C:\Users\<you>\.venvs\eovr\Scripts\python -m pytest
```

## Learn the workflow

### 1. Discover public EO items with STAC

This bounded example searches Sentinel-2 Level-2A items around a public demonstration area. It stores stable metadata, not signed download URLs.

```powershell
eovr stac-search `
  --api-url https://planetarycomputer.microsoft.com/api/stac/v1 `
  --collection sentinel-2-l2a `
  --bbox -122.2751 47.5469 -121.9613 47.7458 `
  --datetime 2024-06-01/2024-06-30 `
  --max-cloud-cover 20 `
  --limit 20 `
  --output data/manifests/stac-items.jsonl
```

See [`docs/learning-stac.md`](docs/learning-stac.md) for the concepts behind this command.

Materialize bounded public preview images by resolving each stable item ID at runtime:

```powershell
eovr stac-materialize `
  --manifest data/manifests/stac-items.jsonl `
  --output-dir data/stac-previews `
  --image-manifest data/manifests/stac-images.jsonl `
  --asset preview `
  --signer planetary-computer `
  --limit 20
```

The access token produced by the signer exists only in memory. Preview images are useful for learning and qualitative DINOv2 retrieval. They are overview products, not analysis-ready chips, and should not be used for the final quantitative benchmark.

### 2. Build a reproducible image manifest

Place RGB chips in class folders:

```text
data/images/
  forest/
  harbor/
  residential/
```

Then create deterministic index/query assignments:

```powershell
eovr manifest-build --images data/images --output data/manifests/images.jsonl
```

### 3. Generate embeddings

```powershell
eovr embed-dinov2 `
  --manifest data/manifests/images.jsonl `
  --image-root data/images `
  --output artifacts/dinov2-vits14.npz
```

The first run downloads the official DINOv2 model through PyTorch Hub. Start with the default `dinov2_vits14` model and a small batch size. Verify `torch.cuda.is_available()` before selecting CUDA; the default PyPI wheel may be CPU-only. Use the [official PyTorch installer](https://pytorch.org/get-started/locally/) when GPU acceleration is required.

Create the classical baseline with:

```powershell
eovr embed-pca `
  --manifest data/manifests/images.jsonl `
  --image-root data/images `
  --components 64 `
  --output artifacts/pca-64.npz
```

### 4. Evaluate retrieval

```powershell
eovr evaluate --embeddings artifacts/dinov2-vits14.npz --k 10
eovr query --embeddings artifacts/dinov2-vits14.npz --item-id forest/example.jpg --k 5
```

## Current scope

Version 0.1 focuses on exact, offline retrieval. Approximate search with Faiss, STAC chip materialization, geographic leakage controls, EO-specific encoders, and a service/demo layer are planned after the baseline is measured.

DINOv2 consumes RGB imagery, so it does not directly use all bands in multispectral products such as Sentinel-2. A later experiment will compare RGB DINOv2 against an EO-specific multispectral encoder.

See [`docs/validation.md`](docs/validation.md) for the current verified gates and the difference between smoke evidence and benchmark evidence.

For the migration decisions, prioritized milestones, and release gate, see
[`docs/project-context.md`](docs/project-context.md).

## Privacy and data policy

- Do not commit EO imagery, credentials, signed URLs, proprietary areas of interest, or job-specific configuration.
- Keep local data under `data/` and generated vectors under `artifacts/`; both are ignored by Git.
- Commit only public, generic examples and sanitized aggregate results.

## License

[MIT](LICENSE.md)
