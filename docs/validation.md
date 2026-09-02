# Validation record

## Evidence policy

This file records executed checks. A passing smoke run proves that a path operates under the
recorded conditions; it does not prove representative retrieval quality, generalization, or
production readiness.

Update this record only after executing the relevant validation. Keep planned work in
`docs/project-context.md`.

```mermaid
flowchart LR
    Tests[Tests and CI<br/>code behavior] --> Smoke[Smoke runs<br/>small real paths]
    Smoke --> Benchmark[EuroSAT v1 benchmark<br/>bounded model comparison]
    Benchmark -. does not establish .-> Missing[Temporal · cross-dataset · product · scale]
```

The first three boxes record different kinds of evidence; passing one does not imply the next.
See [Understanding the benchmarks](learning-benchmarks.md) for the evidence ladder and the exact
training boundary.

## Code health — 2026-09-02

Executed locally against the current checkout with Python 3.11.5:

| Gate | Command or evidence | Status |
|---|---|---|
| Static quality | `python -m ruff check .` | Passed |
| Unit tests | `python -m pytest` | 23 passed |
| Dependency consistency | `python -m pip check` | Passed |
| Coverage report | `pytest --cov=eo_visual_retrieval --cov-report=term-missing` | 58% total |
| Current-source import | `eo_visual_retrieval.__file__` resolved under this checkout's `src/` | Passed |
| GitHub CI | Ruff and tests on Python 3.11 and 3.12 for commit `1b851ca` | Passed |

Coverage is strongest in EuroSAT preparation/audit, visualization, manifests, storage, exact
retrieval, records, evaluation, SSL4EO input preparation, and the pure chip-processing path. The
CLI, PCA, and DINOv2 modules do not yet have direct unit coverage; STAC network resolution is only
partially covered. CI reports coverage but does not enforce a minimum percentage.

## Spatially separated EuroSAT v1 benchmark — updated 2026-09-02

The official EuroSAT multispectral archive was downloaded from DOI `10.5281/zenodo.7711810` and
verified before preparation.

### Dataset and split audit

| Gate | Executed evidence | Status |
|---|---|---|
| Archive identity | 2,065,402,329 bytes; MD5 `091174add3c8e680a49244acf185b9f0` | Passed |
| Source discovery | 27,000 georeferenced multispectral patches | Passed |
| Selected benchmark | 1,600 index + 400 query; 160/40 for each of 10 classes | Passed |
| Spatial groups | 725 represented groups; no 50 km EPSG:6933 cell crossed the split | Passed |
| Guard band | Observed minimum great-circle index/query centroid distance 5.06623 km | Passed |
| File integrity | All 2,000 selected RGB SHA-256 values recomputed successfully | Passed |
| Manifest identity | SHA-256 `bc0b10bf3e3cf29d7f7732529ce5f419b514e2ded3a5e2a5e6e88ebcdea45338` | Passed |
| Model alignment | Ordered IDs, labels, splits, and manifest SHA matched across all three stores | Passed |
| Vector normalization | PCA, DINOv2, and SSL4EO-S12 vector norms were approximately 1.0 | Passed |
| SSL4EO checkpoint | 94,487,109 bytes; SHA-256 `df8b932e2a23a0773febedf3f650aa7d342b805f7876ca5ed6b139d7245d7c09` | Passed |

### Exact-retrieval results at k=10

All 400 queries were evaluated and none were skipped.

| Model | P@10 | R@10 | mAP@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| PCA-64 | 0.3015 | 0.01884 | 0.19698 | 0.31013 |
| DINOv2 ViT-S/14 | 0.69475 | 0.04342 | 0.60763 | 0.70545 |
| SSL4EO-S12 MoCo ResNet-50 | 0.8530 | 0.05331 | 0.81360 | 0.86472 |

PCA used 64 components fitted only on the 1,600 index items. DINOv2 used frozen
`dinov2_vits14` features and produced 384-dimensional vectors. Both ranked the same index with
exact cosine similarity. Execution used Python 3.11.5 and PyTorch 2.13.0 CPU; xFormers was not
available, so no acceleration or throughput claim is made.

SSL4EO-S12 read the same 2,000 selected patches from the 13-band source archive, reordered `B8A`
to the checkpoint's expected position, applied the registered clipping/scaling and 224-pixel crop,
and produced frozen 2,048-dimensional vectors. Generation took 126.87 seconds on CPU. Its norm
range was 0.99999976–1.00000012. The aggregate and per-class scores exceeded DINOv2, but this run
changes both pretraining domain and model input bands; it is not an extra-band ablation.

Per-class metrics and inspected best/worst AP@5 grids are recorded in
`docs/results/eurosat-v1.md`, with machine-readable k=10 results under `docs/results/`.

## Analysis-ready Sentinel-2 chip smoke validation — 2026-09-01

Executed with Rasterio 1.4.4 against one public Planetary Computer Sentinel-2 L2A item:

```text
S2A_MSIL2A_20240625T185941_R013_T10TET_20240626T030520
```

Requested WGS84 bounds:

```text
[-122.15, 47.60, -122.13, 47.62]
```

| Gate | Evidence | Status |
|---|---|---|
| Windowed COG access | Read `B04`, `B03`, `B02`, and `SCL` through signed in-memory URLs | Passed |
| Grid alignment | 153 × 225 pixels in EPSG:32610 at 10 m | Passed |
| Reflectance output | 3-band float32 GeoTIFF with scale 0.0001 and offset -0.1 | Passed |
| Model RGB output | 3-band uint8 GeoTIFF with fixed 0.0–0.3 reflectance stretch | Passed |
| SCL/nodata mask | 34,293 valid pixels and 132 masked pixels | Passed |
| Manifest sanitization | No HREF, token, or signature text persisted | Passed |
| Reproducibility metadata | CRS, transform, bounds, GSD, policy, baseline, and hashes recorded | Passed |
| DINOv2 bridge | Model-ready GeoTIFF produced one normalized 384-dimensional ViT-S/14 vector | Passed |

Observed valid BOA reflectance ranged from approximately -0.0079 to 0.76. Negative values were
preserved in the reflectance artifact. Values outside the configured 0.0–0.3 display range were
clipped only in the model-ready RGB artifact.

Artifacts remained under ignored local `data/` paths and were not committed.

The DINOv2 bridge ran on CPU without xFormers. The missing xFormers optimization emitted warnings
but did not affect correctness; no acceleration or throughput claim is made.

## External-service and model smoke validation — 2026-08-27

### Environment

- Python 3.11
- PyTorch 2.13.0 CPU build
- torchvision 0.28.0 CPU build
- scikit-learn 1.9.0
- pystac-client 0.9.0
- DINOv2 `dinov2_vits14`

The machine had an NVIDIA RTX 3050 Laptop GPU, but the executed environment used a CPU-only
PyTorch wheel. CUDA acceleration and throughput were not validated.

### Executed gates

| Gate | Evidence | Status |
|---|---|---|
| Public STAC search | Bounded Sentinel-2 L2A query returned 2 items | Passed |
| Manifest sanitization | No HREF, signature, or token fields persisted | Passed |
| Signed materialization | 2 public 343 × 343 RGB preview GeoTIFFs | Passed |
| PCA smoke retrieval | 16 controlled synthetic images, 4-dimensional vectors | Passed |
| DINOv2 smoke retrieval | 16 controlled synthetic images, 384-dimensional vectors | Passed |
| DINOv2 EO smoke | 2 Sentinel-2 previews, 384-dimensional vectors | Passed |

The official ViT-S/14 checkpoint was cached at 88,283,115 bytes with local SHA-256:

```text
B938BF1BC15CD2EC0FEACFE3A1BB553FE8EA9CA46A7E1D8D00217F29AEF60CD9
```

### Smoke observations

- Synthetic PCA and DINOv2 runs produced Precision@3, mAP@3, and nDCG@3 of 1.0. The images were
  deliberately simple color patterns, so these values validate execution only.
- Two same-area Sentinel-2 previews captured five days apart had DINOv2 cosine similarity 0.9416.
  Two images without relevance judgments cannot support a retrieval-quality conclusion.

## Evidence not yet available

- Temporal or seasonal leakage controls; EuroSAT does not expose acquisition timestamps.
- Exact versus approximate search recall, latency, build-time, size, and memory measurements.
- CUDA correctness or throughput measurements.
- API, interactive-demo, or deployment validation.

## Allowed claims

The repository may claim that it contains a tested offline retrieval pipeline; that bounded STAC
and analysis-ready Sentinel-2 chip paths have executed; that DINOv2 ViT-S/14 outperformed PCA-64;
and that the frozen 13-band SSL4EO-S12 representation outperformed both RGB baselines on the
recorded spatially separated EuroSAT v1 class-retrieval benchmark.

It must not generalize that result to temporal or seasonal transfer, the causal benefit of
non-visible bands, other datasets, analyst utility, production readiness, approximate-search
performance, or GPU acceleration until those claims are supported by recorded validation.
