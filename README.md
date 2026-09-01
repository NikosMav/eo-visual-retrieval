# EO Visual Retrieval

An educational Earth-observation image-retrieval system built to demonstrate the complete
retrieval workflow: public-data discovery, reproducible manifests, image embeddings, exact
ranking, and honest offline evaluation.

The project compares a transparent PCA baseline with frozen DINOv2 features. It currently
provides a tested offline pipeline and command-line interface. Retrieval quality on a
representative, leakage-safe EO benchmark has **not** yet been established.

## What the project does

- Searches public Earth-observation catalogs through STAC.
- Stores stable item identities and safe metadata without signed asset URLs.
- Materializes bounded preview imagery for learning and qualitative inspection.
- Builds aligned, georeferenced Sentinel-2 reflectance and model-ready RGB chips.
- Prepares a bounded, class-balanced EuroSAT benchmark with spatial leakage controls.
- Builds deterministic index/query manifests from labeled local images.
- Generates PCA or DINOv2 image embeddings.
- Ranks images with exact cosine similarity.
- Reports Precision@k, Recall@k, mAP@k, and nDCG@k.
- Records the difference between executed evidence and planned work.

## System at a glance

```text
STAC API                                      Labeled local RGB images
   |                                                    |
   v                                                    v
sanitized item manifest                     deterministic image manifest
   |                                                    |
   +--> bounded previews                               |
   |                                         +----------+----------+
   +--> reflectance + RGB chip ------------->|                     |
                                             PCA                 DINOv2
                                              |                     |
                                              +----------+----------+
                                                         |
                                                 normalized vectors
                                                         |
                                                 exact cosine search
                                                         |
                                                 ranked evaluation
```

STAC discovery and benchmark evaluation are separate on purpose. Preview assets help teach the
data-access workflow, but they are not analysis-ready chips and must not be used to support
quantitative retrieval claims.

See [Architecture](docs/architecture.md) for the components, boundaries, and design decisions.

## Current benchmark phase

The first real evaluation uses a 2,000-image subset of the official georeferenced EuroSAT
multispectral archive: 1,600 index images and 400 queries, balanced across 10 classes. Index and
query use disjoint 50 km spatial cells and a 5 km guard band. See the
[EuroSAT benchmark guide](docs/benchmark-eurosat.md) for provenance, preparation, and limitations.

## Quick start

Python 3.11 is recommended for the complete ML stack. On Windows, use a short environment path
because PyTorch packages contain deeply nested files.

```powershell
py -3.11 -m venv C:\Users\<you>\.venvs\eovr
C:\Users\<you>\.venvs\eovr\Scripts\python -m pip install --upgrade pip
C:\Users\<you>\.venvs\eovr\Scripts\python -m pip install -e ".[dev,stac,geo,ml]"
```

Confirm the local code is healthy:

```powershell
C:\Users\<you>\.venvs\eovr\Scripts\python -m ruff check .
C:\Users\<you>\.venvs\eovr\Scripts\python -m pytest
```

## EO chip workflow

Search for one or more public Sentinel-2 items:

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

Select one item ID from that sanitized manifest and materialize a bounded spatial window:

```powershell
eovr stac-chip `
  --manifest data/manifests/stac-items.jsonl `
  --item-id <stable-stac-item-id> `
  --bbox -122.15 47.60 -122.13 47.62 `
  --output-dir data/stac-chips `
  --image-manifest data/manifests/stac-chip.jsonl `
  --signer planetary-computer
```

The command produces a float32 BOA-reflectance GeoTIFF and a fixed-stretch uint8 RGB GeoTIFF. The
image manifest points to the RGB artifact used by the embedding pipeline and records the
reflectance artifact, grid, mask policy, processing baseline, and hashes.

## Retrieval workflow

Organize labeled RGB images by class:

```text
data/images/
  forest/
  harbor/
  residential/
```

Build a deterministic manifest:

```powershell
eovr manifest-build `
  --images data/images `
  --output data/manifests/images.jsonl
```

Generate both embedding baselines:

```powershell
eovr embed-pca `
  --manifest data/manifests/images.jsonl `
  --image-root data/images `
  --components 64 `
  --output artifacts/pca-64.npz

eovr embed-dinov2 `
  --manifest data/manifests/images.jsonl `
  --image-root data/images `
  --output artifacts/dinov2-vits14.npz
```

Evaluate and inspect results:

```powershell
eovr evaluate --embeddings artifacts/dinov2-vits14.npz --k 10
eovr query --embeddings artifacts/dinov2-vits14.npz --item-id forest/example.jpg --k 5
```

The full data-discovery and retrieval procedure is explained in
[Pipeline and CLI](docs/pipeline-and-cli.md).

## Current evidence

The verified code-health gates pass on Python 3.11, and CI tests Python 3.11 and 3.12. Bounded
STAC, analysis-ready chip, PCA, and DINOv2 smoke runs have executed successfully. These runs prove
that the workflow operates; they do not measure representative EO retrieval quality.

The missing benchmark must compare PCA and DINOv2 on the same labeled images using spatially and
temporally safe splits. Until then, metric values from artificial smoke data must not be used as
portfolio performance claims.

See [Validation](docs/validation.md) for exact evidence and limitations.

## Documentation

Read the guides in this order:

1. [Project context and roadmap](docs/project-context.md) — goal, current state, and next work.
2. [Architecture](docs/architecture.md) — components, data boundaries, and design choices.
3. [Sentinel-2 chip decision](docs/decisions/0001-windowed-sentinel2-chip-materialization.md) — selected design and trade-offs.
4. [EuroSAT benchmark](docs/benchmark-eurosat.md) — dataset, split, commands, and limits.
5. [Benchmark decision](docs/decisions/0002-georeferenced-eurosat-benchmark.md) — alternatives and rationale.
6. [Pipeline and CLI](docs/pipeline-and-cli.md) — each action, input, and output.
7. [Models and metrics](docs/models-and-metrics.md) — PCA, DINOv2, cosine search, and evaluation.
8. [Learning STAC](docs/learning-stac.md) — EO catalog concepts and retrieval pitfalls.
9. [Development](docs/development.md) — environment, tools, tests, and contribution workflow.
10. [Validation](docs/validation.md) — what has and has not been verified.

## Data and privacy policy

- Do not commit EO imagery, generated embeddings, credentials, or signed URLs.
- Do not commit private areas of interest or proprietary data.
- Keep local data under `data/` and generated vectors under `artifacts/`; both are ignored.
- Persist stable STAC identities and allowlisted metadata only.
- Use public, generic examples and sanitized aggregate results in documentation.

## Roadmap

The next milestone is a labeled, leakage-aware EO benchmark comparing PCA and DINOv2 on identical
inputs and splits. Retrieval analysis, approximate search, and a small interactive product surface
follow that benchmark.

## License

[MIT](LICENSE.md)
