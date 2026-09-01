# EuroSAT v1 benchmark results

## Outcome

Frozen DINOv2 ViT-S/14 substantially outperformed the 64-component flattened-pixel PCA baseline
on the same spatially separated EuroSAT RGB benchmark. DINOv2 improved every evaluated class,
with the largest mAP@10 gains on `Residential` and `Industrial`.

| Model | P@10 | R@10 | mAP@10 | nDCG@10 | Queries | Skipped |
|---|---:|---:|---:|---:|---:|---:|
| PCA-64 | 0.3015 | 0.01884 | 0.19698 | 0.31013 | 400 | 0 |
| DINOv2 ViT-S/14 | **0.69475** | **0.04342** | **0.60763** | **0.70545** | 400 | 0 |
| Absolute change | +0.39325 | +0.02458 | +0.41066 | +0.39532 | — | — |

Recall looks numerically small because every query has 160 relevant index images while only 10
results can be returned. The maximum possible R@10 is therefore `10 / 160 = 0.0625`. DINOv2's
P@10 corresponds to 6.9475 relevant results in the average top 10, compared with 3.015 for PCA.

Machine-readable results:

- [PCA-64 metrics](eurosat-v1-pca-64-k10.json)
- [DINOv2 ViT-S/14 metrics](eurosat-v1-dinov2-vits14-k10.json)

## Per-class comparison

| Class | PCA P@10 | DINOv2 P@10 | PCA mAP@10 | DINOv2 mAP@10 | mAP change |
|---|---:|---:|---:|---:|---:|
| AnnualCrop | 0.2750 | 0.6650 | 0.1551 | 0.5616 | +0.4065 |
| Forest | 0.6250 | 0.9125 | 0.5040 | 0.8685 | +0.3645 |
| HerbaceousVegetation | 0.2800 | 0.6950 | 0.1729 | 0.5822 | +0.4093 |
| Highway | 0.1275 | 0.5075 | 0.0675 | 0.4037 | +0.3361 |
| Industrial | 0.3025 | 0.8875 | 0.1542 | 0.8542 | +0.7001 |
| Pasture | 0.3700 | 0.6675 | 0.2420 | 0.5771 | +0.3351 |
| PermanentCrop | 0.1800 | 0.5575 | 0.0958 | 0.4116 | +0.3158 |
| Residential | 0.1175 | 0.8350 | 0.0509 | 0.7715 | +0.7206 |
| River | 0.2200 | 0.4000 | 0.1238 | 0.2744 | +0.1507 |
| SeaLake | 0.5175 | 0.8200 | 0.4036 | 0.7715 | +0.3679 |

DINOv2's strongest mAP@10 classes were `Forest` (0.8685) and `Industrial` (0.8542). Its weakest
were `River` (0.2744), `Highway` (0.4037), and `PermanentCrop` (0.4116). This makes River-like
linear features and heterogeneous agricultural patterns priorities for deeper error analysis.

## Qualitative inspection

Green borders mark same-class results, red borders mark different-class results, and blue marks
the query. Each grid selects one query per class by AP@5; these are distribution endpoints, not
additional aggregate evidence.

### DINOv2 best cases

![DINOv2 best query per class](../assets/eurosat-v1-dinov2-best.png)

All selected best cases retrieved five same-class results. The rows show that DINOv2 can group
recognizable field geometry, forest texture, transport structure, dense built form, waterways,
and open water across separated locations.

### DINOv2 failure cases

![DINOv2 worst query per class](../assets/eurosat-v1-dinov2-worst.png)

Failure rows remain important despite the strong average. Visually mixed or atypical patches can
be ranked with a plausible neighboring class: agricultural mosaics resemble other crop classes,
thin rivers compete with vegetated or road-like structure, and built classes overlap where a patch
contains both settlement and industry. Some failures may also expose the limits of one broad class
label as the relevance definition rather than purely model error.

### PCA best and worst cases

![PCA best query per class](../assets/eurosat-v1-pca-best.png)

![PCA worst query per class](../assets/eurosat-v1-pca-worst.png)

PCA succeeds when dominant color and coarse texture are distinctive, especially for forest and
open water. Its worst rows show why pixel variance is a limited semantic representation: similar
color palettes can connect unrelated land-use classes, while the same class can vary greatly in
layout and appearance.

## Reproducibility evidence

| Property | Executed value |
|---|---|
| Source | Official EuroSAT multispectral archive, DOI `10.5281/zenodo.7711810` |
| Archive integrity | 2,065,402,329 bytes; MD5 `091174add3c8e680a49244acf185b9f0` |
| Discovered patches | 27,000 |
| Selected set | 1,600 index + 400 query; 160/40 per class |
| Spatial policy | Disjoint 50 km EPSG:6933 cells plus 5 km geodesic guard band |
| Observed minimum separation | 5.06623 km |
| Represented spatial groups | 725 |
| Verified RGB file hashes | 2,000 |
| Manifest SHA-256 | `bc0b10bf3e3cf29d7f7732529ce5f419b514e2ded3a5e2a5e6e88ebcdea45338` |
| Python | 3.11.5 |
| PCA | 64 components, 64×64 RGB, fitted on index only; scikit-learn 1.9.0 |
| DINOv2 | `dinov2_vits14`, 224×224 RGB, frozen 384-d vectors; PyTorch 2.13.0 CPU |
| Ranking | Exact cosine similarity |
| Relevance | Binary EuroSAT class agreement |

Both stores contain identical ordered IDs, labels, splits, and manifest hashes. All vector norms
were within floating-point tolerance of 1.0.

## Evidence boundary

This result supports a narrow claim: on EuroSAT v1 under the recorded spatial split and binary
class relevance, frozen RGB DINOv2 ranks same-class images substantially better than PCA-64.

It does **not** establish temporal or seasonal generalization because EuroSAT does not expose
acquisition timestamps. It also does not establish multispectral performance, transfer to another
dataset, analyst usefulness, online latency, or production readiness. Best/worst examples were
chosen after evaluation and must not be treated as representative averages.

