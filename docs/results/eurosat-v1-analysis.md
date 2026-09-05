# EuroSAT v1 retrieval structure — 2026-09-04

What the published aggregates hide. Every number here is a regrouping of the same per-query
results behind [`eurosat-v1.md`](eurosat-v1.md), sliced four ways. Pooling any slice reproduces
the published aggregate exactly, for all five representations, which is the check that the
regrouping neither invented nor lost a query.

Evidence role: **development analysis of already-published stores**. No new embedding was
computed, no ranking rule changed, and no claim here extends beyond EuroSAT v1.

Machine-readable output: [`eurosat-v1-analysis.json`](eurosat-v1-analysis.json), produced by
`eovr analyze-retrieval`.

## What one chip actually is

Worth stating before interpreting anything. Each EuroSAT patch is 64 × 64 pixels of Sentinel-2
imagery at 10 m ground sampling, so one chip covers roughly **640 m × 640 m** — the manifest's
`source_bounds` confirm it. A "Highway" chip is therefore not a picture of a road. It is a
640-metre square of landscape that happens to contain a road, and the road may occupy a few
percent of the pixels. That single fact explains most of the failure structure below.

The 13-band variant sees all Sentinel-2 MSI bands, including red-edge and short-wave infrared.
Those bands respond to vegetation structure and moisture, which the human-visible red, green and
blue do not separate well.

## 1. Geography barely moves quality, except at the edges

mAP@10 by 5° latitude band. Queries span 34.7° N to 63.3° N — Mediterranean to northern
Scandinavia.

| Band | n | PCA-64 | DINOv2 | SSL4EO-RGB | SSL4EO-13band | TerraMind |
|---|---|---|---|---|---|---|
| 30–35° | 8 | 0.174 | 0.557 | 0.616 | 0.829 | 0.735 |
| 35–40° | 59 | 0.129 | 0.537 | 0.630 | 0.764 | 0.541 |
| 40–45° | 31 | 0.138 | 0.661 | 0.769 | 0.896 | 0.735 |
| 45–50° | 125 | 0.242 | 0.609 | 0.758 | 0.820 | 0.693 |
| 50–55° | 125 | 0.197 | 0.638 | 0.764 | 0.802 | 0.693 |
| 55–60° | 44 | 0.154 | 0.562 | 0.788 | 0.826 | 0.752 |
| 60–65° | 8 | 0.485 | 0.719 | 0.878 | 0.860 | 0.974 |

The two bands carrying most of the corpus (45–55°, 250 of 400 queries) are stable and close to
each other. The 8-query bands at both extremes are too thin to read and are flagged as such in
the JSON. The one substantive signal is the **35–40° band**, with 59 queries: every
representation scores below its own average there. Southern Europe is the hardest region in this
corpus for all five, and nothing in the analysis says why — a genuine open question rather than a
result.

### Consistency across places, not just average quality

Spread of per-cell mean mAP@10 across the 52 spatial cells that carry queries. Reported as a
distribution, not per-cell scores: the median cell holds 7 queries, far too few to score
individually.

| | min | p10 | median | p90 | max | sd |
|---|---|---|---|---|---|---|
| PCA-64 | 0.024 | 0.046 | 0.177 | 0.463 | 1.000 | 0.181 |
| DINOv2 | 0.025 | 0.319 | 0.633 | 0.833 | 1.000 | 0.194 |
| SSL4EO-RGB | 0.166 | 0.591 | 0.777 | 0.995 | 1.000 | 0.186 |
| **SSL4EO-13band** | **0.362** | **0.690** | 0.831 | 0.973 | 1.000 | **0.131** |
| TerraMind | 0.168 | 0.438 | 0.723 | 0.997 | 1.000 | 0.202 |

**This is a finding the aggregate cannot show.** SSL4EO-13band is not merely better on average —
it is the most *consistent* across locations, with both the highest floor (0.362 versus DINOv2's
0.025) and the lowest spread (0.131). TerraMind has the widest spread of all five despite a
middling average: it is excellent in some places and poor in others. A user who cares about
predictable behaviour across regions would rank these representations differently from a user
reading only mAP.

## 2. The representations retrieve almost entirely different images

Mean overlap@10 — the share of each query's ten results that two representations both return.

| Pair | Overlap |
|---|---|
| SSL4EO-13band vs SSL4EO-RGB | 0.300 |
| SSL4EO-13band vs TerraMind | 0.285 |
| SSL4EO-RGB vs TerraMind | 0.184 |
| DINOv2 vs SSL4EO-RGB | 0.147 |
| DINOv2 vs SSL4EO-13band | 0.135 |
| DINOv2 vs TerraMind | 0.106 |
| PCA-64 vs TerraMind | 0.058 |
| PCA-64 vs SSL4EO-RGB | 0.054 |
| PCA-64 vs SSL4EO-13band | 0.046 |
| DINOv2 vs PCA-64 | 0.035 |

The highest agreement anywhere is **3 shared images out of 10**, between the two SSL4EO variants —
same architecture, same pretraining corpus, differing only in which bands they read. DINOv2 and
PCA share **one image in thirty**.

Top-1 correctness across all five, over 400 queries:

- all five correct: **99 (24.8%)**
- some but not all correct: **295 (73.8%)**
- exactly one correct: 21 (5.2%)
- none correct: **6 (1.5%)**

27 of the 32 possible correctness patterns actually occur. The most common single pattern is
"everything except PCA" (142 queries, 35.5%).

**What this means for the metric.** With 160 index images per class, a great many different result
sets all score perfectly under class-label relevance. So representations can post very different
mAP values while agreeing on almost nothing, and two representations with the *same* score need
not be interchangeable in use. Class-label relevance measures whether the retrieved thing is the
right *kind*; it is blind to whether it is the right *one*. That is the answer to the roadmap's
open question about what label relevance misses, and it argues that any future user-facing
evaluation needs a relevance notion finer than the class.

## 3. Failure structure separates semantic from photometric representations

What comes back when a result is wrong, for the three most distinct representations:

**SSL4EO-13band** — Highway → Residential 35, River 32, Industrial 24. River → Highway 41,
SeaLake 21. PermanentCrop → Highway 31, AnnualCrop 30.

**DINOv2** — River → Highway 87, Pasture 51. Highway → River 59, Industrial 38.
AnnualCrop → PermanentCrop 83.

**PCA-64** — Residential → SeaLake 57, HerbaceousVegetation 53. Highway → Forest 63, SeaLake 59.

The two learned representations fail *coherently*: Highway confuses with River (both are narrow
linear features crossing a 640 m square), and the crop and vegetation classes confuse with each
other (they are genuinely similar surfaces). PCA fails *incoherently*: Residential retrieves
SeaLake, Highway retrieves Forest. Those pairs share brightness statistics, not content.

A single mAP number ranks PCA below DINOv2. The confusion structure shows they are not the same
kind of system at all — one models content, the other models colour.

### The six queries nothing ranks correctly

| Class | Query |
|---|---|
| Highway | `Highway/Highway_1780.tif`, `Highway/Highway_373.tif`, `Highway/Highway_436.tif` |
| Pasture | `Pasture/Pasture_143.tif` |
| River | `River/River_180.tif`, `River/River_2319.tif` |

Five of six are linear features. In a 640 m square at 10 m resolution, a road or a river is a thin
ribbon surrounded by whatever it passes through — so the chip's dominant appearance is the
surroundings, and the label describes a minority of the pixels. These are labelling-granularity
failures more than representation failures.

## 4. Proximity is not what makes the benchmark work

The concern behind a spatially separated split is that retrieval could succeed by finding
geographically adjacent images rather than by learning what a class looks like. Two distances test
that, both measured great-circle from the query.

**Distance to the nearest index image of any class** (quartiles at 5.1 / 13.4 / 19.1 / 27.3 /
142.8 km):

| | q1 | q2 | q3 | q4 |
|---|---|---|---|---|
| PCA-64 | 0.150 | 0.231 | 0.227 | 0.179 |
| DINOv2 | 0.591 | 0.612 | 0.599 | 0.629 |
| SSL4EO-RGB | 0.682 | 0.794 | 0.743 | 0.760 |
| SSL4EO-13band | 0.774 | 0.844 | 0.810 | 0.827 |
| TerraMind | 0.591 | 0.764 | 0.705 | 0.687 |

Flat, and if anything slightly *better* far away. Corpus density does not drive the result. The
minimum distance observed is 5.1 km, which independently confirms the 5 km guard band held.

**Distance to the nearest index image of the same class** (quartiles at 5.1 / 21.7 / 36.6 / 71.5 /
1016.8 km):

| | q1 | q2 | q3 | q4 |
|---|---|---|---|---|
| PCA-64 | 0.223 | 0.236 | 0.196 | **0.133** |
| DINOv2 | 0.599 | 0.646 | 0.601 | 0.585 |
| SSL4EO-RGB | 0.746 | 0.758 | 0.783 | 0.691 |
| SSL4EO-13band | 0.801 | 0.842 | 0.846 | **0.765** |
| TerraMind | 0.699 | 0.731 | 0.730 | **0.588** |

Here every representation is at or near its worst in the far quartile, where the nearest same-class
image sits 72 to 1017 km away. The size of the drop differs sharply: PCA loses 40% of its q2 value
and TerraMind 20%, while SSL4EO-13band and DINOv2 lose under 10%.

The relationship is not monotonic — q2 or q3 is often the peak — so this is "the far quartile is
hardest", not "quality decays with distance". Read carefully, it says the strong representations
generalize across geography while the weaker ones lean on finding a nearby example of the same
thing. That is a distinction between memorizing a place and learning a class, and it is invisible
in the aggregate.

## What this analysis cannot do

- **No seasonal or cloud slices.** EuroSAT publishes no acquisition timestamps and no cloud
  metadata. Those two roadmap items are not blocked on effort; they are unachievable on this
  dataset and need a source such as a STAC-derived corpus that carries acquisition dates.
- **No causal claim.** That the 13-band variant is more consistent across regions is a measurement,
  not a demonstration that non-visible bands caused it. The two SSL4EO variants differ in bands,
  but a controlled attribution needs more than one comparison.
- **No claim beyond EuroSAT v1.** Nothing here says the same structure appears in another corpus.
  That remains the project's largest open question.
- **Thin slices are flagged, not hidden.** Bands with fewer than five queries carry
  `below_minimum_queries: true` in the JSON and should not be read as results.

## Reproducing

```powershell
eovr analyze-retrieval `
  --manifest data/eurosat-v1/manifest.jsonl `
  --store artifacts/eurosat-v1-pca-64.npz `
  --store artifacts/eurosat-v1-dinov2-vits14.npz `
  --store artifacts/eurosat-v1-ssl4eo-s12-rgb-moco-resnet50.npz `
  --store artifacts/eurosat-v1-ssl4eo-s12-moco-resnet50.npz `
  --store artifacts/eurosat-v1-terramind-tiny.npz `
  --k 10 --output docs/results/eurosat-v1-analysis.json
```

The command refuses any store built from a different manifest, because slicing another corpus's
queries by this geography would mislabel every result. Census and confusion objects in the JSON are
written alphabetically for stable diffs; sort them by value to read them by frequency.
