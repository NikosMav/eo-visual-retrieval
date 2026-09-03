# BigEarthNet v2: acquisition and development evaluation

This phase supplies the dataset acquisition plan and multi-label evaluation machinery specified
by [ADR 0006](decisions/0006-confirmatory-evaluation-data.md). It has produced no BigEarthNet
retrieval score. Acquisition selection and usable image partitions are distinct stages.

## Source and acquisition budget — checked 2026-09-03

The authoritative source is [Zenodo record 10891137](https://zenodo.org/records/10891137), version
2.0.0, DOI `10.5281/zenodo.10891137`. The dataset is distributed under
[CDLA-Permissive-1.0](https://cdla.dev/permissive-1-0/), as stated in the
[official dataset description](https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf).

| Asset | Advertised download size | Published MD5 |
|---|---:|---|
| `BigEarthNet-S2.tar.zst` | 63.3 GB, approximately 59 GiB | `2245ed2d1a93f6ce637d839bc856396e` |
| `metadata.parquet` | 3.6 MB | `55687065e77b6d0b0f1ff604a6e7b49c` |
| `metadata_for_patches_with_snow_cloud_or_shadow.parquet` | 710.2 kB | `fe31856f4986d446c9468b59d6387c91` |

Sizes above are the source listing's rounded values, not measured local byte counts. The source
lists one compressed S2 tar archive, with no S2 shards or member index. The API subsequently
reported its exact compressed size as **63,251,710,377 bytes**. Two 32-byte range probes returned
HTTP 206, but the archive lacks the standard Zstandard seek-table footer. Selective patch download
has therefore not been established. HTTP range support alone does not provide a member index.

The first acquisition stage is metadata only: two allowlisted files, each limited to 8 MiB,
under ignored `data/downloads/bigearthnet-v2/`. It requires local CPU and disk and no paid service.
The supplied command has no image-archive download option. Cached metadata is checksum-verified
before reuse, and new downloads are checksum-verified before atomic promotion.

Both metadata files were acquired and checksum-verified on 2026-09-03 after earlier transient
access failures. Their exact local sizes are 3,616,349 and 710,162 bytes, totaling 4,326,511 bytes.
The image archive has not been downloaded. See the [executed recovery](validation.md) for evidence.

```powershell
# Inspect the pinned inventory without network access.
python scripts/download_bigearthnet_metadata.py

# Fetch only the two small metadata files.
python scripts/download_bigearthnet_metadata.py --download
```

The subsequent imagery stage must specify an exact byte ceiling, a storage layout that avoids
extracting the full dataset, and a reproducible member-access strategy before starting the bulk
transfer. The 4,000 index / 500 development / 500 final patch target remains unchanged.
The [bounded acquisition proposal](decisions/0007-bounded-bigearthnet-acquisition.md) recommends
streaming with a 2 GiB retained-data ceiling. A 1 MiB prefix probe successfully decoded one complete
12-band patch in memory; it did not verify the full archive or prepare the benchmark.

## Executed metadata audit — 2026-09-03

Install the optional reader and audit the two verified local files without any download:

```powershell
uv sync --locked --extra dev --extra geo --extra search --extra pca --extra bigearthnet
uv run --locked --no-sync python scripts/audit_bigearthnet_metadata.py `
  --output outputs/bigearthnet-metadata-audit.json
```

The [committed aggregate report](results/bigearthnet-metadata-audit.json) records the exact input
hashes, schema, per-label/country/month counts, and repeated grid identities. It contains no imagery
or selected benchmark partition. Both source files passed validation for unique IDs, valid calendar
timestamps, non-empty labels, recognized splits, and consistent exclusion flags. Their ID sets
are disjoint.

| Official split | Recommended patches | Distinct acquisition dates | Rarest label count |
|---|---:|---:|---:|
| train | 237,871 | 79 | 670, Coastal wetlands |
| validation | 122,342 | 77 | 426, Beaches, dunes, sands |
| test | 119,825 | 72 | 117, Coastal wetlands |

The recommended file contains **480,038 patches**, all 19 labels, and 54 MGRS tiles, dated
2017-06-13 through 2018-05-29. The other file excludes 69,450 patches: 60,773 flagged for seasonal
snow and 8,677 for cloud/shadow, with no patch flagged for both. Together they cover 549,488 IDs.

Usable dates are present in every patch ID. However, train/validation share 77 dates and both
train/test and validation/test share 72. The official splits do not establish temporal holdout.

Among recommended patches, 233,966 distinct tile/row/column keys include 139,499 keys observed on
multiple dates. No identical key crosses the official splits. These keys do not reveal metric
distances or footprints shared by neighboring tiles. Full source-georeferencing checks, independent
cell/guard-band audits, and a frozen temporal rule are supplied by the following stage.

## Footprints and acquisition selection

[ADR 0008](decisions/0008-bigearthnet-selection-protocol.md) fixes the temporal windows, spatial
guards, label minimums, seed, and allocation algorithm. It preserves 4,000 index / 500 development /
500 final IDs inside the respective official splits.

Use the small reference-map archive for geometry. Its exact compressed size is 282,391,301 bytes,
with MD5 `95d85a222fa983faddcac51a19f28917`. The downloader makes one bounded attempt, validates the
complete file, and reuses a verified cache. The reader streams TIFF headers into a compact Parquet
inventory without extracting maps or reading their pixel values. It requires exact coverage of both
metadata files and rejects unexpected members, duplicate IDs, oversized maps, and invalid geometry.

```powershell
python scripts/prepare_bigearthnet_footprints.py --download-reference `
  --inventory data/bigearthnet-v2/footprints.parquet `
  --report outputs/bigearthnet-footprints.json

python scripts/prepare_bigearthnet_selection.py `
  --inventory data/bigearthnet-v2/footprints.parquet `
  --inventory-report outputs/bigearthnet-footprints.json `
  --selection data/bigearthnet-v2/acquisition-selection.json `
  --report outputs/bigearthnet-selection-audit.json
```

Omit `--download-reference` to operate entirely offline. Inventory/selection outputs must be new
paths; rerun to separate paths for reproducibility checks. Both source archives and generated
geometry/ID files remain local. Commit aggregate reports only. The optional `bigearthnet` and `geo`
groups provide PyArrow, Zstandard, and Rasterio; use the locked installation above.

The second command verifies source/inventory identities, selects the IDs, then reloads their TIFF
headers from the original reference archive. Inventory reading uses Pillow's TIFF tags; the audit
uses Rasterio and requires identical native geometry, then recomputes cells and distances from
bounds rather than trusting stored centres. Final allocation occurs before development/index;
neither retrieval outputs nor reference-map pixel values enter selection. The later comparison
will measure combined geography and season shifts under this protocol.

The acquisition-selection JSON is **not** an image or relevance manifest. S2 imagery has not yet
been acquired, and its 12 bands must be checked against these footprints before use. Source hashes
and aggregate outcomes are recorded in the [footprint report](results/bigearthnet-footprints.json)
and [selection audit](results/bigearthnet-selection-audit.json). Pretraining overlap and overlap
with historical EuroSAT geography remain separate, unresolved audits.

The executed selection contains 4,000 / 500 / 500 patches across 162 / 20 / 20 disjoint cells,
with all 19 labels in every partition. The smallest achieved centre separation is 7.176 km;
adjacent temporal gaps are 32 and 44 days. The five retained dataset files occupy 284.79 MiB.
See [validation](validation.md) for the full scope and limitations of these checks.

## SSL4EO L2A gate — absent in the agreed sources

The bounded search checked:

- the [SSL4EO-S12 model list at revision 2156913](https://github.com/zhu-xlab/SSL4EO-S12/blob/2156913c5d8e5a2c572a5b000f0d5eaed6fc3192/README.md);
- the [TorchGeo ResNet registry at revision 39711ba](https://github.com/microsoft/torchgeo/blob/39711baadcd4a02b88dc7e83cffc29f841123d3e/torchgeo/models/resnet.py);
- all 36 public models returned by the [Hugging Face torchgeo organization](https://huggingface.co/torchgeo) catalog on 2026-09-03, including the Sentinel-2 model cards.

The registered SSL4EO-S12 ResNet-50 Sentinel-2 multispectral weights use 13 channels. The 12-channel
SeCo-Eco registry entry belongs to the separate SSL4Eco dataset, so it does not satisfy ADR 0006's
SSL4EO-S12 checkpoint gate. No qualifying 12-band L2A ResNet-50 checkpoint was found in the
agreed sources, and none was downloaded or silently substituted.
The [inspection snapshot](decisions/evidence/ssl4eo-l2a-gate-2026-09-03.json) records the
registry entries, source hashes, and model-catalog revisions used for this decision.

SSL4EO is absent from the planned BigEarthNet comparison under this gate. Its existing EuroSAT
results remain valid within their recorded scope. This is a dated, source-bounded conclusion,
not a claim that such a checkpoint could never be published. Reopening the gate requires an
explicit protocol update before final scoring.

## Multi-label relevance contract

An embedding store continues to carry vectors, IDs, `index`/`query` splits, and its original image
manifest hash. Its single-label entries must all be `None`. A separate `RelevanceManifest` binds
multi-label judgments and `index`/`development`/`final` partitions to that same image-manifest
SHA-256. Its IDs must exactly cover the store and its partitions must agree with the store's
index/query assignments. Existing EuroSAT stores and the `evaluate` command are unchanged.

Example shape, with the full SHA-256 of the corresponding image manifest in place of the placeholder:

```json
{
  "schema": "eo-multilabel-relevance-v1",
  "dataset": "BigEarthNet-v2.0.0",
  "image_manifest_sha256": "<64 lowercase hexadecimal characters>",
  "records": [
    {"item_id": "index-patch", "labels": ["Arable land", "Pastures"], "partition": "index"},
    {"item_id": "dev-patch", "labels": ["Pastures"], "partition": "development"},
    {"item_id": "final-patch", "labels": ["Arable land"], "partition": "final"}
  ]
}
```

For label sets A and B, Jaccard is `|A intersect B| / |A union B|`. The development evaluator uses:

- binary relevance at `Jaccard >= 0.5` for Precision@k, Recall@k, and mAP@k;
- raw Jaccard as nDCG gain, with the ideal ranking computed over the entire index;
- the number of all binary-relevant index items as the Recall denominator, and
  `min(k, number of binary-relevant index items)` as the AP denominator;
- zero binary metrics for labeled queries with no binary positives, while retaining their
  independently graded nDCG and reporting their count;
- an explicit skip count for unlabeled development queries, and rejection of unlabeled index
  items. If no labeled development queries remain, evaluation fails instead of publishing a score.

This eligibility policy is fixed before any BigEarthNet scoring: a stricter threshold must not
improve a mean by removing difficult queries. Per-label slices include a query in every label it
carries, so their populations overlap; the overall result is the mean over queries, not over
these slices.

```powershell
eovr evaluate-multilabel `
  --embeddings artifacts/bigearthnet-development.npz `
  --relevance data/bigearthnet-v2/relevance.json `
  --k 10 --threshold 0.5 `
  --output outputs/bigearthnet-development-k10.json
```

This command requires prepared inputs; none are bundled. It scores development queries only.
Threshold sensitivity is limited to the pre-registered 0.3, 0.5, and 0.7 values. Final records
cannot enter the index, query set, normalization, or metric aggregation. A future final-scoring
entry point must bind an audited split and frozen configuration and enforce the one-shot gate.
This development command exposes no final-scoring option.

Every output records the evaluated partition, threshold, dataset, image-manifest SHA-256,
relevance-manifest SHA-256, and embedding-store SHA-256. These bind the result to its inputs;
they do not replace the still-required spatial, temporal, and pretraining-overlap audits.
