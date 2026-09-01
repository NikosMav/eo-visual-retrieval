# Project context and roadmap

## Goal

Build a public, educational Earth-observation visual-retrieval project that demonstrates sound
retrieval engineering from data discovery through evaluation. The project should make each step
understandable, reproducible, and safe to publish.

The intended learning outcomes are:

- understand STAC catalogs and safe EO data access;
- create deterministic datasets and leakage-aware evaluation splits;
- compare classical and modern image representations fairly;
- implement exact ranking before approximate vector search;
- interpret retrieval metrics without overstating what they prove;
- evolve an evaluated offline pipeline into a small usable application.

## Current state

The repository currently contains:

- a typed Python package and `eovr` command-line interface;
- provider-neutral STAC discovery with sanitized JSONL manifests;
- bounded public preview materialization with optional in-memory signing;
- deterministic image manifests with index/query partitions;
- PCA and frozen DINOv2 embedding backends;
- portable compressed embedding stores;
- exact cosine retrieval and four ranked-retrieval metrics;
- unit tests, linting, CI, and executed smoke validation.

The current evidence proves that the workflow runs. It does **not** establish retrieval quality on
a representative EO task. No portfolio claim should exceed the evidence in `docs/validation.md`.

## Project boundaries

- The project is generic and public; it contains no employer or role-specific context.
- STAC manifests contain stable identity and allowlisted metadata, never signed asset URLs.
- Preview images are learning inputs, not analysis-ready benchmark data.
- DINOv2 is an RGB baseline, not a multispectral EO model.
- PCA and any other learned preprocessing must be fitted on the index/training partition only.
- Benchmark splits must prevent spatial and temporal leakage.
- Exact cosine search remains the quality reference when approximate search is introduced.
- Raw imagery, generated vectors, credentials, and private areas of interest remain outside Git.

## Prioritized roadmap

### Milestone 1: analysis-ready Sentinel-2 RGB chip

Implement one narrow, reproducible materialization path for Sentinel-2 `B04`, `B03`, and `B02`
bands that:

- reads only a requested spatial window;
- aligns the bands on a documented pixel grid;
- records CRS, affine transform, ground sampling distance, and source item identity;
- applies documented reflectance scaling and deterministic RGB normalization;
- handles nodata and cloud/SCL information explicitly;
- writes a sanitized image manifest without source URLs;
- includes local raster fixtures and tests for pixel, metadata, and failure behavior.

Bulk downloading is out of scope until this vertical slice is reproducible.

### Milestone 2: labeled EO retrieval benchmark

Create a small benchmark with defensible relevance labels. A public scene dataset such as EuroSAT
or PatternNet can support the initial representation comparison. A STAC-derived benchmark becomes
appropriate once geographic groups and temporal holdouts are available.

Record:

- dataset name, version, and provenance;
- label semantics and known limitations;
- split algorithm, seed, and exclusions;
- class balance and query/index counts;
- geographic and temporal grouping rules;
- model and preprocessing configuration.

**Benchmark gate:** report PCA and DINOv2 Precision@k, Recall@k, mAP@k, and nDCG@k on the same
leakage-aware split, accompanied by qualitative success and failure examples.

### Milestone 3: retrieval analysis

- Compare PCA and DINOv2 fairly on identical inputs and splits.
- Add query-result grids and per-class error slices.
- Add geographic, seasonal, and cloud-condition slices when metadata supports them.
- Explain what class-label relevance captures and what it misses about user intent.
- Evaluate one EO-specific or multispectral encoder as a separate experiment.

### Milestone 4: approximate search

Add Faiss while retaining exact cosine search as the reference. Measure:

- recall relative to exact search;
- query latency;
- index build time;
- serialized index size;
- runtime memory at multiple corpus sizes.

### Milestone 5: usable project surface

Expose the evaluated workflow through a small API or interactive demo. A user should be able to
select or upload a public query chip, inspect ranked results and metadata, and see which model and
index generated the ranking.

Containers and release automation should be added when they support this workflow, not as isolated
portfolio decoration.

## Portfolio-ready definition

The project is portfolio-ready when:

- a clean clone can reproduce the documented benchmark;
- tests and CI pass on supported Python versions;
- data provenance, relevance assumptions, and leakage controls are explicit;
- performance claims link to recorded evidence and configuration;
- PCA and DINOv2 results are directly comparable;
- representative successes and failures are visible;
- no private data, sensitive geography, signed URLs, or generated datasets are committed;
- the repository name and release metadata match the EO visual-retrieval identity.

## Next task

Implement Milestone 1 as a single tested vertical slice: materialize one small public Sentinel-2
RGB window into an analysis-ready chip and sanitized manifest. Do not expand to batch processing
until the spatial metadata and pixel-processing behavior are validated.
