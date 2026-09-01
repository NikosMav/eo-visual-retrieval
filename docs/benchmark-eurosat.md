# EuroSAT benchmark

## What this experiment asks

Given a held-out Sentinel-2 RGB patch, can an embedding rank patches from the same broad EuroSAT
land-use/land-cover class above patches from other classes when index and query geography are
separated?

The experiment compares representation quality, not supervised classification. PCA and DINOv2 do
not train on EuroSAT labels. Labels are used only to define relevance during evaluation.

## Dataset and provenance

[EuroSAT](https://github.com/phelber/EuroSAT) contains 27,000 georeferenced Sentinel-2 patches in
10 classes. This project uses the official 13-band multispectral archive from
[Zenodo record 7711810](https://zenodo.org/records/7711810):

| Field | Value |
|---|---|
| File | `EuroSAT_MS.zip` |
| Size | 2,065,402,329 bytes |
| DOI | `10.5281/zenodo.7711810` |
| Published MD5 | `091174add3c8e680a49244acf185b9f0` |
| License | MIT, according to the official project repository |

The archive is used instead of the smaller RGB package because its GeoTIFFs preserve the CRS and
affine transform needed for leakage control. Only the selected RGB derivatives are embedded.

## Download and prepare

Install the geospatial and model dependencies:

```powershell
C:\Users\<you>\.venvs\eovr\Scripts\python -m pip install -e ".[dev,geo,ml]"
```

Download the official archive into an ignored directory. `--continue-at -` resumes an interrupted
transfer:

```powershell
New-Item -ItemType Directory -Force data/downloads | Out-Null
curl.exe -L --continue-at - `
  --output data/downloads/EuroSAT_MS.zip `
  https://zenodo.org/api/records/7711810/files/EuroSAT_MS.zip/content
```

Prepare benchmark version 1:

```powershell
eovr benchmark-eurosat-prepare `
  --archive data/downloads/EuroSAT_MS.zip `
  --output-dir data/eurosat-v1/images `
  --manifest data/eurosat-v1/manifest.jsonl `
  --queries-per-class 40 `
  --index-per-class 160 `
  --group-size-km 50 `
  --minimum-separation-km 5 `
  --seed 42
```

Preparation performs five explicit actions:

1. verifies the complete archive against the published MD5;
2. reads CRS and transforms each patch centroid into EPSG:6933 and WGS84;
3. deterministically selects class-balanced query and index samples;
4. enforces disjoint 50 km spatial cells plus a 5 km great-circle centroid guard band;
5. writes georeferenced uint8 RGB GeoTIFFs and one auditable JSONL manifest.

The query pool spans at least 10 spatial cells per class. Sampling uses every available selected
cell before taking a second image from one cell, avoiding a query set dominated by one locality.

Run the independent manifest and file audit after preparation:

```powershell
eovr benchmark-eurosat-audit `
  --manifest data/eurosat-v1/manifest.jsonl `
  --image-root data/eurosat-v1/images
```

The audit recomputes class counts, group disjointness, minimum geodesic centroid distance,
manifest SHA-256, and every selected image hash.

RGB uses source bands 4, 3, and 2 and a fixed 0–2750 digital-number stretch. It is deterministic
and does not calculate per-image contrast, which would make model inputs depend on each image's
histogram.

## Run both baselines

Both commands consume the exact same manifest and image root:

```powershell
eovr embed-pca `
  --manifest data/eurosat-v1/manifest.jsonl `
  --image-root data/eurosat-v1/images `
  --components 64 `
  --image-size 64 `
  --seed 42 `
  --output artifacts/eurosat-v1-pca-64.npz

eovr embed-dinov2 `
  --manifest data/eurosat-v1/manifest.jsonl `
  --image-root data/eurosat-v1/images `
  --model dinov2_vits14 `
  --batch-size 8 `
  --device auto `
  --output artifacts/eurosat-v1-dinov2-vits14.npz
```

Evaluate at the same cutoffs, starting with `k=10`:

```powershell
eovr evaluate `
  --embeddings artifacts/eurosat-v1-pca-64.npz `
  --k 10 `
  --output artifacts/eurosat-v1-pca-64-k10.json

eovr evaluate `
  --embeddings artifacts/eurosat-v1-dinov2-vits14.npz `
  --k 10 `
  --output artifacts/eurosat-v1-dinov2-vits14-k10.json
```

Evaluation returns macro means over eligible queries plus the same metrics per class. The class
slices show whether an aggregate improvement is broad or concentrated in a few easy classes.

Render both ends of each model's query distribution for qualitative inspection:

```powershell
eovr result-grid `
  --embeddings artifacts/eurosat-v1-dinov2-vits14.npz `
  --manifest data/eurosat-v1/manifest.jsonl `
  --image-root data/eurosat-v1/images `
  --output artifacts/eurosat-v1-dinov2-best.png `
  --k 5 `
  --mode best

eovr result-grid `
  --embeddings artifacts/eurosat-v1-dinov2-vits14.npz `
  --manifest data/eurosat-v1/manifest.jsonl `
  --image-root data/eurosat-v1/images `
  --output artifacts/eurosat-v1-dinov2-worst.png `
  --k 5 `
  --mode worst
```

Each row contains one query and its exact top five results. Blue marks the query, green a
class-relevant result, and red a non-relevant result. “Best” and “worst” are selected by AP@5
within each query class; they are diagnostic examples, not additional aggregate evidence.

## Split audit checklist

Before reporting model scores, verify and record:

- exactly 160 index and 40 query items for every class;
- no `spatial_group` value appears in both splits;
- observed minimum index/query centroid separation is at least 5 km;
- all 400 queries have relevant index items and none are skipped;
- the manifest hash is unchanged between model runs;
- PCA was fitted only on the 1,600 index images;
- model, package, Python, and device versions are recorded.

## What a score means

Class agreement is a convenient binary proxy, not ground truth for every retrieval need. For
example, two `Residential` scenes can differ greatly in density and structure, while a `River`
scene can visually overlap `SeaLake` or vegetated classes. Aggregate metrics therefore need
per-class results and result grids showing successes and failures.

This benchmark does not establish seasonal, temporal, sensor, cloud, or multispectral
generalization. Those are later experiments with metadata that can support the corresponding
holdouts.
