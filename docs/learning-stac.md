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

For the learning workflow, materialization is restricted to bounded preview images over HTTPS, with optional provider signing performed in memory. Analysis-ready RGB chip generation will be a separate step because it must handle projections, pixel windows, spectral scaling, cloud masks, and spatial leakage deliberately.

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
