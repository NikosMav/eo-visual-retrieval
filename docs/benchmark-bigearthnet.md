# BigEarthNet v2: acquisition and development evaluation

This phase supplies the dataset acquisition plan and multi-label evaluation machinery specified
by [ADR 0006](decisions/0006-confirmatory-evaluation-data.md). It has produced no BigEarthNet
retrieval score. Preparing and auditing real partitions remains a separate gate.

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
lists one compressed S2 tar archive, with no S2 shards or member index. Selective patch download
has therefore not been established. Reading compressed tar members is a sequential operation
unless a compatible seek index is provided; HTTP range support alone would not establish one.

The first acquisition stage is metadata only: two allowlisted files, each limited to 8 MiB,
under ignored `data/downloads/bigearthnet-v2/`. It requires local CPU and disk and no paid service.
The supplied command has no image-archive download option. Cached metadata is checksum-verified
before reuse, and new downloads are checksum-verified before atomic promotion.

```powershell
# Inspect the pinned inventory without network access.
python scripts/download_bigearthnet_metadata.py

# Fetch only the two small metadata files.
python scripts/download_bigearthnet_metadata.py --download
```

The subsequent imagery stage must specify an exact byte ceiling, a storage layout that avoids
extracting the full dataset, and a reproducible member-access strategy before starting the bulk
transfer. The 4,000 index / 500 development / 500 final patch target remains unchanged.

The documented metadata contains patch IDs, multi-label lists, official train/validation/test
assignments, country, and snow/cloud exclusions. Acquisition time is encoded in each S2 patch ID
as `YYYYMMDDTHHMMSS`. The official format establishes how dates can be read; it does not establish
that usable temporal and geographic separation can be achieved for the selected partitions.
The metadata listing does not supply the patch footprints needed for the independent spatial
audit; those must be obtained and verified from the source georeferencing.

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
