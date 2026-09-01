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
- Builds deterministic index/query manifests from labeled local images.
- Generates PCA or DINOv2 image embeddings.
- Ranks images with exact cosine similarity.
- Reports Precision@k, Recall@k, mAP@k, and nDCG@k.
- Records the difference between executed evidence and planned work.

## System at a glance

```text
STAC API                                     Labeled local RGB images
   |                                                   |
   v                                                   v
sanitized item manifest                    deterministic image manifest
   |                                                   |
   v                                        +----------+----------+
bounded preview materialization             |                     |
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

## Quick start

Python 3.11 is recommended for the complete ML stack. On Windows, use a short environment path
because PyTorch packages contain deeply nested files.

```powershell
py -3.11 -m venv C:\Users\<you>\.venvs\eovr
C:\Users\<you>\.venvs\eovr\Scripts\python -m pip install --upgrade pip
C:\Users\<you>\.venvs\eovr\Scripts\python -m pip install -e ".[dev,stac,ml]"
```

Confirm the local code is healthy:

```powershell
C:\Users\<you>\.venvs\eovr\Scripts\python -m ruff check .
C:\Users\<you>\.venvs\eovr\Scripts\python -m pytest
```

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
STAC, PCA, and DINOv2 smoke runs have executed successfully. These runs prove that the workflow
operates; they do not measure representative EO retrieval quality.

The missing benchmark must compare PCA and DINOv2 on the same labeled images using spatially and
temporally safe splits. Until then, metric values from artificial smoke data must not be used as
portfolio performance claims.

See [Validation](docs/validation.md) for exact evidence and limitations.

## Documentation

Read the guides in this order:

1. [Project context and roadmap](docs/project-context.md) — goal, current state, and next work.
2. [Architecture](docs/architecture.md) — components, data boundaries, and design choices.
3. [Pipeline and CLI](docs/pipeline-and-cli.md) — each action, input, and output.
4. [Models and metrics](docs/models-and-metrics.md) — PCA, DINOv2, cosine search, and evaluation.
5. [Learning STAC](docs/learning-stac.md) — EO catalog concepts and retrieval pitfalls.
6. [Development](docs/development.md) — environment, tools, tests, and contribution workflow.
7. [Validation](docs/validation.md) — what has and has not been verified.

## Data and privacy policy

- Do not commit EO imagery, generated embeddings, credentials, or signed URLs.
- Do not commit private areas of interest or proprietary data.
- Keep local data under `data/` and generated vectors under `artifacts/`; both are ignored.
- Persist stable STAC identities and allowlisted metadata only.
- Use public, generic examples and sanitized aggregate results in documentation.

## Roadmap

The next milestone is one reproducible, analysis-ready Sentinel-2 RGB chip with tested spatial
windows, band alignment, reflectance scaling, nodata/cloud handling, and sanitized geospatial
metadata. A labeled leakage-aware benchmark follows, then retrieval analysis, approximate search,
and a small interactive product surface.

## License

[MIT](LICENSE.md)
