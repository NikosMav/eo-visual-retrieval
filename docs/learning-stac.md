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

## Providers: Planetary Computer and Copernicus Data Space

`--api-url` takes any conforming STAC API, and two are exercised. Microsoft Planetary Computer
mirrors Sentinel-2 and signs asset URLs on request. The **Copernicus Data Space Ecosystem** is
ESA's own distribution of the same archive:

```powershell
eovr stac-search `
  --api-url https://stac.dataspace.copernicus.eu/v1 `
  --collection sentinel-2-l2a `
  --bbox 23.4 38.4 23.8 38.7 `
  --datetime 2025-06-01/2025-08-31 `
  --max-cloud-cover 10 --limit 4 `
  --output data/manifests/copernicus-items.jsonl
```

That runs with **no account and no credentials**. Copernicus answers searches anonymously.

Its access model then splits in a way worth understanding: catalogue queries are open, but assets
are addressed as `s3://eodata/...` on Copernicus object storage, which needs a free account and S3
keys. So discovery works today and pixel materialization does not — `stac-materialize` refuses an
`s3://` asset with a message that says why, rather than reporting a generic scheme error.

### The same fact under two names

Providers publish identical information under two generations of STAC extension names, so an
allowlist written for one silently drops fields on the other:

| Fact | Planetary Computer | Copernicus |
|---|---|---|
| Tile identity | `s2:mgrs_tile` | `grid:code` |
| Processing baseline | `s2:processing_baseline` | `processing:version` |

Both spellings are allowlisted, and a key a provider does not publish is simply not recorded.

Copernicus also supplies viewing geometry the older keys omit — `view:sun_elevation`,
`view:sun_azimuth`, `sat:relative_orbit`, `sat:orbit_state`, `eo:snow_cover`. These matter for
retrieval rather than being trivia. Sun elevation explains why one place photographs differently in
March and September, and relative orbit is how repeat passes over a single location are found: a
search over Attica returned four scenes all on relative orbit 93, the track that revisits that area.

Nothing credential-shaped enters a manifest. Copernicus publishes `auth:schemes` and
`storage:schemes` properties describing how to authenticate; both are excluded.

### Why this matters for evaluation

EuroSAT publishes no acquisition timestamps, which is why the
[structure analysis](results/eurosat-v1-analysis.md) records seasonal slicing as unachievable
rather than outstanding. A Copernicus-derived corpus carries per-scene dates, so it can support
questions this project currently cannot ask — including retrieval relevance defined as *the same
place on a different date*, a ground truth that needs no class labels at all.

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
- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) and its
  [STAC API](https://stac.dataspace.copernicus.eu/v1)
- [Copernicus Browser](https://browser.dataspace.copernicus.eu/) for inspecting a scene visually
  before searching for it programmatically
