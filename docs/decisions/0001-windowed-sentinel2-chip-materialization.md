# ADR-0001: Use windowed Rasterio reads for Sentinel-2 chips

**Status:** Accepted

**Date:** 2026-09-01

**Decider:** Repository owner

## Context

The project needs a reproducible bridge between stable STAC item identities and local RGB inputs.
Provider-rendered previews are convenient for inspection but hide radiometric processing,
resolution, projection, resampling, and cloud handling. They cannot support the labeled retrieval
benchmark.

The first implementation must remain bounded, understandable, locally testable, and compatible
with remote Cloud-Optimized GeoTIFF assets. It must preserve physical reflectance for analysis
while also creating the byte RGB representation expected by the current embedding backends.

## Decision

Use Rasterio to read a requested spatial window from Sentinel-2 L2A `B04`, `B03`, and `B02`
assets. Use the red 10 m band as the reference grid and align every other input to that grid with
an in-memory `WarpedVRT`. Resample spectral bands bilinearly and the categorical `SCL` layer with
nearest-neighbour resampling.

Produce two georeferenced artifacts:

1. a three-band float32 BOA-reflectance GeoTIFF for analysis and auditability;
2. a three-band uint8 RGB GeoTIFF using a fixed recorded reflectance stretch for embeddings.

Write a dataset mask for nodata and selected SCL classes. The image manifest points to the RGB
artifact and records the reflectance artifact, grid, processing parameters, source identity, and
content hashes without persisting source URLs.

The first CLI command processes exactly one item selected from a sanitized STAC manifest and
enforces a maximum output pixel count.

## Options considered

### Provider-rendered previews

| Dimension | Assessment |
|---|---|
| Implementation complexity | Low |
| Radiometric control | Low |
| Reproducibility | Low to medium |
| Educational visibility | Low |

**Pros:** Already supported and inexpensive to download.

**Cons:** Rendering choices are provider-controlled and the pixels are not suitable benchmark
inputs.

### Direct Rasterio window reads

| Dimension | Assessment |
|---|---|
| Implementation complexity | Medium |
| Radiometric control | High |
| Reproducibility | High |
| Educational visibility | High |

**Pros:** Makes CRS, transforms, resampling, scaling, masks, and output metadata explicit. Reads
only the required COG window and supports small local raster fixtures.

**Cons:** Requires careful grid alignment and explicit Sentinel-2 radiometric rules.

### Datacube loading with odc-stac or stackstac

| Dimension | Assessment |
|---|---|
| Implementation complexity | Medium to high |
| Radiometric control | High |
| Reproducibility | High |
| Scaling to many items | High |

**Pros:** Strong multi-item and lazy-array capabilities.

**Cons:** Adds a larger abstraction and dependency surface before the single-chip contract is
understood and tested.

## Trade-off analysis

Direct Rasterio reads expose the geospatial operations the project is intended to teach and keep
the first slice small. A datacube library may become useful for multi-item benchmark construction,
but adopting one now would hide important grid and mask decisions behind a higher-level loader.

Keeping reflectance and RGB artifacts avoids choosing between scientific auditability and current
model compatibility. The extra local file is acceptable because generated raster artifacts are
ignored by Git.

## Consequences

- Rasterio becomes an optional `geo` dependency and is installed in CI for local-fixture tests.
- Sentinel-2 processing-baseline metadata determines the BOA reflectance offset.
- The output grid is explicit and testable.
- SCL masking is reproducible but remains a policy choice recorded in metadata.
- The uint8 RGB artifact is a model input, not a replacement for the reflectance artifact.
- Batch processing, mosaicking, and datacube abstractions remain future work.

## Action items

1. [x] Implement pure local-raster chip construction.
2. [x] Add STAC item resolution with in-memory signing.
3. [x] Add the bounded `stac-chip` CLI command.
4. [x] Test alignment, scaling, masking, metadata, and failure cases.
5. [x] Execute one bounded public Sentinel-2 smoke validation.
