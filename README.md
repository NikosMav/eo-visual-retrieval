# EO Visual Retrieval

An educational Earth-observation image-retrieval system built to demonstrate the complete
retrieval workflow: public-data discovery, reproducible manifests, image embeddings, exact
ranking, and honest offline evaluation.

The project compares transparent PCA, frozen RGB DINOv2, and frozen 13-band SSL4EO-S12 features.
It provides a tested offline pipeline, command-line interface, and a first spatially separated
EuroSAT benchmark. Broader temporal and cross-dataset generalization has **not** yet been
established.

## What the project does

- Searches public Earth-observation catalogs through STAC.
- Stores stable item identities and safe metadata without signed asset URLs.
- Materializes bounded preview imagery for learning and qualitative inspection.
- Builds aligned, georeferenced Sentinel-2 reflectance and model-ready RGB chips.
- Prepares a bounded, class-balanced EuroSAT benchmark with spatial leakage controls.
- Builds deterministic index/query manifests from labeled local images.
- Generates PCA, DINOv2, or EuroSAT-specific SSL4EO-S12 image embeddings.
- Ranks images with exact cosine similarity.
- Reports Precision@k, Recall@k, mAP@k, and nDCG@k.
- Records the difference between executed evidence and planned work.

## System at a glance

```text
STAC API                                      Selected EO patches + manifest
   |                                                    |
   v                                         +----------+-----------+
sanitized item manifest                      | RGB derivatives      | 13 bands
   |                                         v             v        v
   +--> bounded previews                    PCA         DINOv2   SSL4EO-S12
   |                                         |             |        |
   +--> reflectance + RGB chip               +-------------+--------+
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
STAC and analysis-ready chip smoke runs have executed successfully. The EuroSAT v1 benchmark uses
1,600 index images and 400 queries with disjoint 50 km spatial cells and a 5 km guard band.

At `k=10`, PCA-64 achieved mAP 0.19698, DINOv2 ViT-S/14 achieved 0.60763, and 13-band
SSL4EO-S12 achieved 0.81360 on the same selected patches, split, relevance definition, and exact
ranker. See [EuroSAT v1 results](docs/results/eurosat-v1.md) for all metrics, per-class slices,
qualitative examples, reproducibility evidence, and limitations.

See [Validation](docs/validation.md) for exact evidence and limitations.

## Documentation

Read the guides in this order:

1. [Project context and roadmap](docs/project-context.md) — goal, current state, and next work.
2. [Architecture](docs/architecture.md) — components, data boundaries, and design choices.
3. [Sentinel-2 chip decision](docs/decisions/0001-windowed-sentinel2-chip-materialization.md) — selected design and trade-offs.
4. [EuroSAT benchmark](docs/benchmark-eurosat.md) — dataset, split, commands, and limits.
5. [Benchmark decision](docs/decisions/0002-georeferenced-eurosat-benchmark.md) — alternatives and rationale.
6. [Multispectral encoder decision](docs/decisions/0003-ssl4eo-s12-multispectral-encoder.md) — model choice, alternatives, preprocessing, and risks.
7. [EuroSAT v1 results](docs/results/eurosat-v1.md) — metrics, examples, evidence, and interpretation.
8. [Pipeline and CLI](docs/pipeline-and-cli.md) — each action, input, and output.
9. [Models and metrics](docs/models-and-metrics.md) — representations, cosine search, and evaluation.
10. [Learning STAC](docs/learning-stac.md) — EO catalog concepts and retrieval pitfalls.
11. [Development](docs/development.md) — environment, tools, tests, and contribution workflow.
12. [Validation](docs/validation.md) — what has and has not been verified.

## Data and privacy policy

- Do not commit EO imagery, generated embeddings, credentials, or signed URLs.
- Do not commit private areas of interest or proprietary data.
- Keep local data under `data/` and generated vectors under `artifacts/`; both are ignored.
- Persist stable STAC identities and allowlisted metadata only.
- Use public, generic examples and sanitized aggregate results in documentation.

## Roadmap

The next milestone is approximate search: add Faiss while retaining exact cosine search as the
quality reference, then measure recall, query latency, build time, index size, and memory. A small
interactive product surface follows the scaling analysis.

## License

[MIT](LICENSE.md)
