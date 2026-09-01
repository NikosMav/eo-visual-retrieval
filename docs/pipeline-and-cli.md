# Pipeline and CLI

## Mental model

The command-line workflow moves through four durable artifacts:

```text
STAC manifest -> local images -> image manifest -> embedding store -> rankings/metrics
```

Each command performs one bounded transformation. This makes failures easier to diagnose and lets
PCA and DINOv2 reuse the same data and split.

## Install the command

From the repository root:

```powershell
C:\Users\<you>\.venvs\eovr\Scripts\python -m pip install -e ".[dev,stac,ml]"
eovr --help
```

Optional dependency groups are separated by responsibility:

- `dev`: Ruff, Pytest, and coverage;
- `stac`: PySTAC Client, Requests, and Planetary Computer support;
- `ml`: scikit-learn, PyTorch, and torchvision.

## Path A: discover public EO imagery

### 1. Search a STAC catalog

```powershell
eovr stac-search `
  --api-url https://planetarycomputer.microsoft.com/api/stac/v1 `
  --collection sentinel-2-l2a `
  --bbox -122.2751 47.5469 -121.9613 47.7458 `
  --datetime 2024-06-01/2024-06-30 `
  --max-cloud-cover 20 `
  --limit 20 `
  --output data/manifests/stac-items.jsonl
```

Inputs:

- `--bbox W S E N`: longitude/latitude search rectangle;
- `--datetime`: one timestamp or an interval;
- `--max-cloud-cover`: optional scene-level metadata filter;
- `--limit`: maximum returned items, constrained to 1–1000.

Output: one sanitized `StacItemRecord` per line. The file contains item identity and safe metadata,
but no asset URLs.

Cloud cover is a scene-level hint. It does not guarantee that a future chip is cloud-free.

### 2. Materialize bounded previews

```powershell
eovr stac-materialize `
  --manifest data/manifests/stac-items.jsonl `
  --output-dir data/stac-previews `
  --image-manifest data/manifests/stac-images.jsonl `
  --asset rendered_preview `
  --signer planetary-computer `
  --limit 20
```

The command resolves each item at runtime, optionally signs it in memory, validates HTTPS and
image media type, streams the response with retries, enforces a per-file byte limit, and replaces
the destination atomically.

Output records are unlabeled `stac-preview` images assigned to the index. They support embedding
smoke tests and visual exploration, not quantitative evaluation.

## Path B: build a labeled retrieval dataset

### 1. Organize images

```text
data/images/
  class-a/
    image-001.png
    image-002.png
  class-b/
    image-003.png
```

The first directory component is treated as the label. Files directly under the root are
unlabeled and will be skipped by evaluation.

Supported extensions are JPEG, PNG, WebP, TIFF, and GeoTIFF-style TIFF names.

### 2. Build the manifest

```powershell
eovr manifest-build `
  --images data/images `
  --output data/manifests/images.jsonl `
  --query-fraction 0.2 `
  --seed 42
```

The builder hashes file contents, groups unique hashes by label, ranks them deterministically from
the seed, and assigns query/index membership. Where a label has at least two unique images, it
keeps at least one item on each side.

The hash protects against exact duplicates crossing the split. It does not identify nearby crops,
overlapping scenes, or repeated observations of the same location.

## Generate embeddings

### PCA

```powershell
eovr embed-pca `
  --manifest data/manifests/images.jsonl `
  --image-root data/images `
  --components 64 `
  --image-size 64 `
  --seed 42 `
  --output artifacts/pca-64.npz
```

PCA is fitted only on index items and then transforms every item. The requested component count
cannot exceed the number of index samples or flattened input dimensions.

### DINOv2

```powershell
eovr embed-dinov2 `
  --manifest data/manifests/images.jsonl `
  --image-root data/images `
  --model dinov2_vits14 `
  --batch-size 8 `
  --device auto `
  --output artifacts/dinov2-vits14.npz
```

`--device auto` selects CUDA only when the installed PyTorch build reports it as available. The
first run may download model code and weights through the official PyTorch Hub entrypoint.

Both commands preserve manifest order and write IDs, vectors, labels, splits, and backend metadata
to a compressed NPZ store.

## Query the index

```powershell
eovr query `
  --embeddings artifacts/dinov2-vits14.npz `
  --item-id forest/example.jpg `
  --k 5
```

The query ID must already exist in the store. The command creates an index from records whose
split is `index`, excludes the query ID if necessary, and prints ranked item IDs with cosine
scores as JSON.

## Evaluate rankings

```powershell
eovr evaluate --embeddings artifacts/dinov2-vits14.npz --k 10
```

Evaluation requires at least one index item and one query item. `k` cannot exceed the index size.
A result is relevant when its label equals the query label. Unlabeled queries and labels with no
matching index item are skipped and reported separately.

See [Models and metrics](models-and-metrics.md) before interpreting the output.

## Reproducibility checklist

For a benchmark run, record:

- dataset and version;
- image/chip-generation configuration;
- manifest hash or immutable copy;
- split seed and grouping policy;
- index/query counts and class balance;
- embedding backend, model, dimensions, and preprocessing;
- `k` values;
- package versions, Python version, and device;
- exclusions, skipped queries, and known leakage risks.

Generated data and embeddings stay outside Git. Commit sanitized configuration, aggregate metrics,
and enough provenance for another person to reproduce the run.

## Common failure conditions

| Symptom | Meaning |
|---|---|
| `manifest references missing image` | Image root and manifest path do not match |
| `identical image content has conflicting labels` | A duplicate file appears under different classes |
| `components must be between...` | PCA dimension exceeds available index samples/input dimensions |
| `no labeled queries have relevant index items` | The split cannot support label-proxy evaluation |
| `query dimension does not match the index` | Vectors from incompatible stores/models were mixed |
| `asset exceeds ... byte limit` | The materializer stopped an unexpectedly large download |
| CUDA requested but unavailable | The installed PyTorch build or machine does not expose CUDA |
