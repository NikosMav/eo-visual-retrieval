# Retrieval across a change of season — 2026-09-05

The [first temporal run](temporal-v1.md) could not answer the question it was built for: its
queries sat a median of 13 days from their nearest answer, so it measured retrieval between images
taken a fortnight apart. This re-splits the same 193 chips behind a temporal guard band and asks
the intended question — **given a place in July, retrieve that same place in another season.**

No imagery was downloaded. `eovr temporal-resplit` re-partitions the existing corpus.

## The guard

Spreading queries through a timeline measures the gap to their nearest answer; it does not create
one. The guard instead does what the benchmark's 5 km spatial guard band does — it *excludes*.

Queries are the acquisitions nearest 15 July, so the excluded zone is one contiguous block of
summer rather than scattered holes. Every index acquisition within 90 days of any query is dropped.

| | Unguarded | Guarded |
|---|---|---|
| Query to nearest answer, median | 13 days | **110 days** |
| Query to nearest answer, minimum | 1 day | **90 days** |
| Images | 193 | 97 (36 query, 61 index) |
| Places | 12 | 12, none dropped |

## Result

Top-1 accuracy — does the single nearest neighbour come from the same place. Reported because it
has **no ceiling artifact** in either split: every place has at least one index acquisition, so a
perfect ranking scores 1.0 regardless of how many images the place has. The first run showed how
badly precision@5 misleads when that is not true.

| Representation | Gap 13 days | Gap 110 days | Change |
|---|---|---|---|
| DINOv2 ViT-S/14 | 0.938 | **0.806** | −14.1% |
| PCA-32 | 0.896 | **0.667** | −25.6% |

**Both degrade across seasons, and PCA degrades about 1.8 times as much.** The learned
representation survives a change of season markedly better than one built from raw pixels — which
is what you would hope, and what the unguarded split could not show, since it reported 0.938
against 0.917 and made the two look nearly equivalent.

PCA is reported at 32 components in both splits, because the guarded index holds only 61 images
and cannot support 64. At matched dimensionality the unguarded figure is 0.896 against PCA-64's
0.917, so dimensionality accounts for about two points and the guard accounts for the rest.

## Where it fails, and it is not random

DINOv2 top-1 per place, across seasons:

| Place | Latitude | Top-1 |
|---|---|---|
| andalusia-guadalquivir | 37.6 | 1.00 |
| attica-thriasio | 38.5 | 1.00 |
| crete-messara | 35.1 | 1.00 |
| po-valley-cremona | 45.1 | 1.00 |
| brandenburg-spreewald | 51.9 | 1.00 |
| flevoland-polder | 52.5 | 1.00 |
| ostrobothnia-vaasa | 63.1 | 1.00 |
| beauce-orleans | 48.1 | 0.67 |
| jutland-viborg | 56.5 | 0.67 |
| smaland-vaxjo | 56.9 | 0.67 |
| lapland-rovaniemi | 66.5 | 0.67 |
| **uppland-uppsala** | **59.9** | **0.00** |

Every Mediterranean place is perfect. Three of the four places scoring 0.67 sit above 56° N, and
Uppland — forest and farmland mosaic at 59.9° N — fails **completely**: not one July query
retrieves its own place from another season.

The pattern is what the physical geography predicts. Southern Europe changes modestly between July
and the shoulder seasons; northern Europe changes drastically, with snow cover and a short
growing season turning the same ground into a visually different scene. Ostrobothnia at 63.1° N
breaking the pattern with a perfect score is the interesting exception, and this corpus is too
small to explain it.

## What this establishes, and what it does not

**Establishes**, on this corpus: a temporal guard changes the answer substantially, so any
same-place retrieval result reported without one should be treated as measuring near-in-time
similarity. Frozen DINOv2 features are markedly more robust to seasonal change than PCA on raw
pixels. Seasonal robustness degrades with latitude, with one complete failure.

**Does not establish**: anything with statistical weight — 36 queries over 12 places, three
queries per place, so a single place's score moves in steps of 0.33. Nothing about *why* Uppland
fails; snow is a hypothesis this corpus cannot test, because the chips carry no snow annotation.
Nothing about the multispectral representations, which read 13-band archive members and cannot
consume these three-band chips. And nothing about other years, other regions, or a 90-day guard
being the right threshold rather than one that was feasible.

The obvious next step is more places, so a per-place score is not three queries wide.
