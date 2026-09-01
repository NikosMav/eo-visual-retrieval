# Architecture

## Purpose

EO Visual Retrieval is an offline reference system for learning how image-retrieval systems are
built and evaluated. It separates data acquisition, representation, ranking, and evaluation so
that each stage can be inspected and changed independently.

The current system is deliberately small:

- files and JSONL manifests instead of a database;
- compressed NumPy arrays instead of a vector service;
- exact cosine search instead of an approximate index;
- a command-line interface instead of an API or web application.

Those choices keep the quality baseline understandable before scale-oriented components are added.

## High-level design

```text
                          DATA ACQUISITION

STAC API --search--> StacItemRecord JSONL --resolve/sign--> local preview
                           |                                  |
                    stable metadata only               qualitative use


                         RETRIEVAL PIPELINE

labeled image folders
         |
         v
ImageRecord JSONL --load--> image pixels --embed--> EmbeddingStore NPZ
                                                     |
                                      +--------------+--------------+
                                      |                             |
                                      v                             v
                             ExactCosineIndex                 offline evaluator
                                      |                             |
                                      v                             v
                               ranked item IDs        P@k / R@k / mAP@k / nDCG@k
```

The acquisition and retrieval paths meet at local image files, not at provider URLs. This is the
main security and reproducibility boundary.

## Components

| Component | Responsibility | Main module |
|---|---|---|
| STAC search | Validate a bounded query and collect safe item metadata | `stac.py` |
| Preview materializer | Resolve an item, optionally sign in memory, and download a bounded image | `stac.py` |
| Image manifest builder | Hash images, infer labels, and assign deterministic splits | `manifests.py` |
| PCA backend | Learn a classical index-fitted projection and produce normalized vectors | `embeddings/pca.py` |
| DINOv2 backend | Produce normalized frozen vision-transformer features | `embeddings/dinov2.py` |
| Embedding store | Save vectors and retrieval metadata in a portable NPZ file | `embeddings/store.py` |
| Exact index | Rank every index vector by cosine similarity | `retrieval.py` |
| Evaluator | Calculate label-proxy ranked-retrieval metrics | `evaluation.py` |
| CLI | Validate arguments and connect all stages | `cli.py` |

## Data contracts

### STAC item manifest

`StacItemRecord` describes where public imagery came from without storing an access URL. It
contains:

- STAC API and collection identity;
- stable item ID;
- bounding box and acquisition time;
- available asset keys;
- a small allowlist of EO properties.

Asset HREFs are excluded because providers may add expiring signatures or credentials. At
materialization time, the item is fetched again by stable identity and any signed URL remains in
memory.

### Image manifest

`ImageRecord` describes a local file that can be embedded:

- `item_id`: stable identifier used in results;
- `path`: path relative to the image root;
- `split`: `index` or `query`;
- `label`: optional relevance proxy;
- `source`: origin category such as `local` or `stac-preview`;
- `metadata`: content hash or sanitized source metadata.

The first folder below the image root becomes the class label. Exact duplicate content is kept in
one split, and conflicting labels for identical content are rejected.

### Embedding store

`EmbeddingStore` packages:

- ordered item IDs;
- a two-dimensional float32 vector matrix;
- labels and splits aligned with the vectors;
- backend configuration metadata.

The compressed NPZ format is portable and sufficient for offline experiments. It is not intended
as a scalable online index format.

## Required invariants

The system depends on the following rules:

1. Signed or credential-bearing URLs are never persisted.
2. STAC queries and downloads are bounded.
3. Learned preprocessing is fitted on index/training data only.
4. Every embedding row stays aligned with its ID, label, and split.
5. Stored IDs are unique.
6. Zero-length vectors are rejected before cosine search.
7. The query image is excluded when it also exists in the index.
8. Benchmark claims use leakage-aware splits and recorded configuration.
9. Exact search remains available as the reference for approximate-search experiments.

## Why two manifests exist

A STAC item and a retrieval image are not the same concept:

- A STAC item may expose many assets, bands, resolutions, and projections.
- A retrieval image is one concrete local tensor-ready input with a label and split.

Combining them too early would make provider access details part of model and evaluation code. The
two-manifest design allows a later analysis-ready chip generator to turn stable STAC identities
into controlled image inputs without changing the retrieval layer.

## Current limitations

- The STAC materializer downloads previews, not analysis-ready band windows.
- Preview records are unlabeled and assigned to the index, so they cannot form a benchmark alone.
- Deterministic content-hash splitting prevents exact duplicate leakage but not geographic or
  temporal leakage.
- The PCA transformer is not persisted for embedding unseen images later.
- The query command accepts an ID already in an embedding store, not a new uploaded image.
- Exact search is intentionally linear in corpus size.
- There is no API, database, job runner, or interactive result viewer.

These constraints define the roadmap rather than hidden production claims.
