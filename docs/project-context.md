# Project context and roadmap

This is the durable, sanitized context for continuing the project. It captures the useful
decisions from the planning conversation without storing a transcript or any private job details.

## Goal

Transform the original PCA image-analysis assignment into a complete Earth-observation visual
retrieval project. The project should build practical experience with STAC, DINOv2, retrieval
evaluation, vector search, and recommendation-style ranking while remaining honest, reproducible,
and safe to publish.

The existing repository is being evolved instead of replaced because the classical-to-modern
progression tells a coherent engineering story. The original state is preserved by the
`legacy-pca-v1` tag and under `legacy/`.

## What exists now

- A tested Python package and CLI rather than a notebook-only workflow.
- Provider-neutral STAC search with sanitized manifests.
- Bounded public preview materialization with signing kept in memory.
- Deterministic image manifests with separate index and query partitions.
- PCA and frozen DINOv2 ViT-S/14 embedding pipelines.
- Exact cosine retrieval and Precision@k, Recall@k, mAP@k, and nDCG@k.
- Unit tests, linting, CI, learning notes, an architecture decision, and a validation record.

The current validation proves that the workflow executes. It does **not** yet prove retrieval
quality on a representative EO task. See `docs/validation.md` for the exact evidence.

## Key decisions

1. Keep one repository and preserve the old project as legacy history.
2. Store stable STAC item identity, never expiring or signed asset URLs.
3. Separate catalog discovery from local, reproducible image materialization.
4. Use exact search as the reference implementation before introducing Faiss.
5. Compare PCA and DINOv2 first; add an EO-specific encoder only after the benchmark is sound.
6. Keep all examples public and generic; local or role-specific context stays outside Git.
7. Delay the repository rename and main-branch migration until measured baseline results exist.

## Prioritized roadmap

### Milestone 1: analysis-ready RGB chips

Add a bounded materializer for Sentinel-2 RGB bands (`B04`, `B03`, `B02`) that:

- reads only the requested spatial window;
- aligns bands and records CRS, transform, ground sampling distance, and source item identity;
- applies documented reflectance scaling and deterministic RGB normalization;
- handles nodata and cloud/SCL filtering explicitly;
- produces a reproducible image manifest without source URLs;
- includes tests for windows, metadata, scaling, and failure cases.

Preview assets remain useful for learning and qualitative checks, but not for benchmark claims.

### Milestone 2: labeled EO retrieval benchmark

Create a small, reproducible benchmark with defensible relevance labels. A public scene dataset
such as EuroSAT or PatternNet is suitable for the first comparison; a STAC-derived benchmark is
valuable once geographic groups and temporal holdouts are available.

The split must prevent overlapping or nearby observations of the same place from appearing in
both index and query partitions. Record dataset version, label semantics, split seed, exclusions,
and class balance.

**Release gate:** report PCA and DINOv2 Precision@k, Recall@k, mAP@k, and nDCG@k with leakage-aware
splits, plus qualitative success and failure examples. Only then merge the modernization branch,
publish v0.1, and rename the repository to `eo-visual-retrieval`.

### Milestone 3: retrieval analysis

- Compare PCA and DINOv2 fairly on the same images and splits.
- Add query-result grids and error slices by class, geography, season, and cloud condition where
  the data supports them.
- Document what class-label relevance captures and what it misses about user intent.
- Evaluate one EO-specific or multispectral encoder as a separate experiment.

### Milestone 4: approximate search

Add a Faiss index while retaining exact cosine search as the reference. Measure recall relative to
exact search, latency, build time, index size, and memory use at more than one corpus size.

### Milestone 5: usable project surface

Expose the evaluated workflow through a small API or interactive demo. A user should be able to
select a public query chip, inspect ranked results and metadata, and understand which model and
index produced them. Add containers and release automation only when they support this workflow.

## Definition of a portfolio-ready project

- A clean clone can reproduce the documented benchmark.
- Tests and CI pass on supported Python versions.
- Dataset provenance, relevance assumptions, and leakage controls are explicit.
- Performance claims link to recorded evidence and configuration.
- The README stays concise; deeper reasoning lives in `docs/`.
- No private data, sensitive geography, signed URLs, or generated datasets are committed.

## Next task

Implement Milestone 1 as a narrow vertical slice: materialize one small public Sentinel-2 RGB
window into an analysis-ready chip and a sanitized manifest, then test the geospatial metadata and
pixel-processing path. Do not expand to bulk downloads until that slice is reproducible.

