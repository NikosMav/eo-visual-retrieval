# Validation record

## Evidence policy

This file records executed checks. A passing smoke run proves that a path operates under the
recorded conditions; it does not prove representative retrieval quality, generalization, or
production readiness.

Update this record only after executing the relevant validation. Keep planned work in
`docs/project-context.md`.

## Code health — 2026-09-01

Executed locally against the current checkout with Python 3.11.5:

| Gate | Command or evidence | Status |
|---|---|---|
| Static quality | `python -m ruff check .` | Passed |
| Unit tests | `python -m pytest` | 13 passed |
| Dependency consistency | `python -m pip check` | Passed |
| Coverage report | `pytest --cov=eo_visual_retrieval --cov-report=term-missing` | 50% total |
| Current-source import | `eo_visual_retrieval.__file__` resolved under this checkout's `src/` | Passed |
| GitHub CI | Ruff and tests on Python 3.11 and 3.12 for merge commit `d1c76a5` | Passed |

Coverage is strongest in manifests, storage, exact retrieval, records, evaluation, and the pure
chip-processing path. The CLI, PCA, and DINOv2 modules do not yet have direct unit coverage; STAC
network resolution is only partially covered. CI reports coverage but does not enforce a minimum
percentage.

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

- Retrieval metrics on a representative labeled EO benchmark.
- Geographic and temporal leakage controls.
- A fair PCA-versus-DINOv2 benchmark table.
- Per-class and qualitative error analysis.
- DINOv2 versus an EO-specific multispectral encoder.
- Exact versus approximate search recall, latency, build-time, size, and memory measurements.
- CUDA correctness or throughput measurements.
- API, interactive-demo, or deployment validation.

## Allowed claims

The repository may currently claim that it contains a tested offline retrieval pipeline and that
bounded STAC, analysis-ready Sentinel-2 chip, PCA, and DINOv2 smoke paths have executed
successfully.

It must not claim a representative EO retrieval score, production readiness, geographic
generalization, multispectral capability, approximate-search performance, or GPU acceleration
until those claims are supported by recorded validation.
