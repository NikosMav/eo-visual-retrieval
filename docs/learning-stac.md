# Learning STAC through this project

STAC—the SpatioTemporal Asset Catalog specification—is a common way to describe and search geospatial assets. It standardizes metadata and discovery; it does not prescribe one storage provider or one image format.

## The four concepts to learn first

1. **Catalog:** a linked entry point for browsing STAC resources.
2. **Collection:** a group of related items, such as a satellite product.
3. **Item:** a GeoJSON feature representing one observation at a place and time.
4. **Asset:** a file or service associated with an item, such as a spectral band, thumbnail, or metadata document.

A STAC API adds searchable HTTP endpoints. Typical filters include:

- `collections`: which product to search;
- `bbox` or `intersects`: where on Earth;
- `datetime`: when;
- `query` or CQL2 filters: properties such as cloud cover.

## What `eovr stac-search` does

The command opens a conforming API with `pystac-client`, submits a bounded search, and writes one JSON object per item. Each record contains:

- API and collection identity;
- item ID;
- bounding box and timestamp;
- available asset keys;
- a small allowlist of useful EO properties.

It intentionally excludes asset HREFs. Some providers attach temporary access tokens to HREF query strings. Persisting those URLs creates noisy, expiring manifests and can disclose credentials. The `stac-materialize` command retrieves an item by stable identity and keeps the resolved, optionally signed asset URL in memory only.

Preview materialization remains available for quick inspection. The `stac-chip` command is the
controlled analysis path: it resolves raw `B04`, `B03`, `B02`, and `SCL` assets in memory, reads a
bounded window, aligns grids, applies documented reflectance conversion and cloud policy, and
writes georeferenced local artifacts.

## What an analysis-ready chip adds

A rendered preview answers “what does this scene roughly look like?” A controlled chip also makes
the following choices explicit:

- **CRS:** the coordinate system used by the output grid;
- **affine transform:** the mapping between pixel positions and projected coordinates;
- **ground sampling distance:** the physical size represented by one pixel;
- **resampling:** bilinear for continuous spectral values and nearest-neighbour for SCL classes;
- **radiometry:** conversion from stored digital numbers to BOA reflectance;
- **normalization:** the fixed reflectance range mapped to model-ready byte RGB;
- **validity mask:** which nodata, cloud, shadow, cirrus, and snow/ice pixels are excluded.

Sentinel-2 processing baseline 04.00 introduced a -1000 digital-count BOA offset. With the 10,000
quantification value, current products use `reflectance = DN * 0.0001 - 0.1`. DN zero remains
nodata. The physical reflectance file preserves negative values; only the derived RGB view clips
to its recorded display range.

The Scene Classification Layer is categorical. The default policy masks classes 0, 1, 3, 7, 8,
9, 10, and 11: nodata, defective pixels, cloud shadows, low-probability cloud/unclassified,
medium/high cloud, cirrus, and snow/ice. This is a reproducible project policy, not a universal
definition of valid imagery.

## Safe experiments

- Use small bounding boxes, short date windows, low result limits, and public collections.
- Keep private areas of interest in `configs/local/`, which Git ignores.
- Inspect collection metadata before assuming band names or units.
- Record the collection ID, date range, spatial bounds, filters, and item IDs for reproducibility.
- Treat cloud cover metadata as a scene-level hint, not a guarantee that a specific chip is cloud-free.

## Retrieval-specific pitfalls

- **Spatial leakage:** nearby or overlapping chips can make evaluation unrealistically easy.
- **Temporal leakage:** repeated observations of the same place can cross query/index splits.
- **Resolution mismatch:** sensors and products may use different ground sampling distances.
- **Spectral mismatch:** DINOv2 expects RGB while many EO products are multispectral.
- **Proxy relevance:** sharing a scene class does not prove two images are useful to the same user.

The current image manifest uses deterministic splits. Geographic grouping and temporal holdouts
must be added before STAC-derived imagery becomes part of the benchmark.

## Public references

- [STAC specification](https://stacspec.org/en/about/stac-spec/)
- [PySTAC Client usage](https://github.com/stac-utils/pystac-client/blob/main/docs/usage.rst)
- [Planetary Computer STAC quickstart](https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/)
