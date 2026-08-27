# Validation record

Validated locally on 2026-08-27 with Python 3.11.

## Environment

- PyTorch 2.13.0 CPU build
- torchvision 0.28.0 CPU build
- scikit-learn 1.9.0
- pystac-client 0.9.0
- DINOv2 `dinov2_vits14`

The machine has an NVIDIA RTX 3050 Laptop GPU, but the verified environment uses a CPU-only PyTorch wheel. CUDA acceleration is therefore not claimed.

## Verified gates

| Gate | Evidence | Status |
|---|---|---|
| Static quality | `ruff check .` | Passed |
| Unit tests | 9 tests on the lightweight core | Passed |
| Public STAC search | Bounded Sentinel-2 L2A query returned 2 items | Passed |
| Manifest sanitization | No HREF, signature, or token fields persisted | Passed |
| Signed materialization | 2 public 343×343 RGB preview GeoTIFFs | Passed |
| PCA smoke retrieval | 16 synthetic images, 4-dimensional vectors | Passed |
| DINOv2 smoke retrieval | 16 synthetic images, 384-dimensional vectors | Passed |
| DINOv2 EO smoke | 2 Sentinel-2 previews, 384-dimensional vectors | Passed |

The official ViT-S/14 checkpoint was cached at 88,283,115 bytes with local SHA-256 `B938BF1BC15CD2EC0FEACFE3A1BB553FE8EA9CA46A7E1D8D00217F29AEF60CD9`.

## Smoke observations

- The synthetic PCA and DINOv2 runs achieved Precision@3, mAP@3, and nDCG@3 of 1.0. The images were deliberately simple color patterns, so these values validate execution only.
- Two same-area Sentinel-2 previews from five days apart had DINOv2 cosine similarity `0.9416`. With only two images and no relevance labels, this is qualitative evidence only.

## Evidence not yet available

- Retrieval metrics on a labeled EO benchmark.
- Geographic or temporal leakage controls.
- DINOv2 versus an EO-specific multispectral encoder.
- Exact versus approximate search latency, recall, and memory.
- CUDA throughput measurements.

No portfolio claim should exceed the verified gates above.
