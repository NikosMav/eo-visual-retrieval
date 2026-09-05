# Same-place retrieval, first corpus — 2026-09-05

193 Sentinel-2 chips over 12 European places, each place seen on up to 20 dates in 2024, with
retrieval judged by identity rather than category: given one view of a place, retrieve that same
place on another date. No class labels are involved, and "correct" means literally the right place.

Evidence role: **development evidence for a pipeline, not a benchmark result.** The corpus does
not yet test what it was built to test — see the limitation below, which is the most useful thing
on this page.

Build report: [`temporal-corpus-v1.json`](temporal-corpus-v1.json). Imagery stays out of Git.

## The corpus

| | |
|---|---|
| Places | 12, spanning 35.1° N to 66.5° N |
| Images | 193 — 145 index, 48 query |
| Dates per place | 6 to 20, capped at 20 |
| Window | 2,560 m, 261 × 260 px at 10 m |
| Size on disk | 113 MB |
| Build | 12 places, 0 failures, about 25 minutes on Planetary Computer with no account |

## Results at k=5

| Representation | P@5 | mAP@5 | nDCG@5 | Recall@5 |
|---|---|---|---|---|
| DINOv2 ViT-S/14 | 0.854 | 0.916 | 0.928 | 0.435 |
| PCA-64 | 0.725 | 0.752 | 0.797 | 0.327 |

Both score far above their EuroSAT class-retrieval figures (DINOv2 0.712, PCA 0.325 at k=5). That
is expected and is not an improvement: telling twelve visually distinct European landscapes apart
is an easier task than telling ten land-cover classes apart.

## A trap: per-place precision is uninterpretable on its own

Precision@5 cannot exceed the number of relevant images available. A place with only 3 index
acquisitions caps at 3/5 = 0.600 no matter how perfect the ranking.

| Place | Index images | Ceiling | DINOv2 P@5 | Share of achievable |
|---|---|---|---|---|
| andalusia-guadalquivir | 16 | 1.00 | 1.000 | 1.000 |
| attica-thriasio | 16 | 1.00 | 1.000 | 1.000 |
| brandenburg-spreewald | 9 | 1.00 | 1.000 | 1.000 |
| crete-messara | 16 | 1.00 | 1.000 | 1.000 |
| po-valley-cremona | 16 | 1.00 | 1.000 | 1.000 |
| uppland-uppsala | 14 | 1.00 | 1.000 | 1.000 |
| **flevoland-polder** | **3** | **0.60** | **0.600** | **1.000** |
| ostrobothnia-vaasa | 16 | 1.00 | 0.950 | 0.950 |
| smaland-vaxjo | 13 | 1.00 | 0.950 | 0.950 |
| **beauce-orleans** | **2** | **0.40** | **0.300** | **0.750** |
| jutland-viborg | 9 | 1.00 | 0.750 | 0.750 |
| lapland-rovaniemi | 15 | 1.00 | 0.700 | 0.700 |

Read raw, Flevoland's 0.600 and Beauce's 0.300 look like the two worst failures in the corpus.
Flevoland is in fact **perfect** — it retrieved every image it was possible to retrieve — and
Beauce reached three quarters of its ceiling. The two places that look worst are the two the cloud
survey already showed have the fewest clear days in Europe; their low scores measure the weather,
not the representation.

Against the achievable ceiling, DINOv2 averages 0.925 and is perfect at 7 of 12 places. The
genuinely harder places are **Lapland (0.700)** and **Jutland (0.750)** — both high-latitude, both
with strong seasonal snow and vegetation change.

This is a concrete instance of the point [ADR 0011](../decisions/0011-requirements-and-thresholds.md)
makes about metric definitions. A "top-5 accuracy above 85%" target would be met here by DINOv2's
raw 0.854, on a number that is partly an artifact of how many images each place happens to have.

## The limitation, which matters more than the results

**This corpus does not test seasonal invariance, which is why it was built.**

Queries were held out spread through each place's timeline, on the reasoning that consecutive
dates would leave every query beside a near-identical neighbour. With 20 dates spread across a
year, that reasoning fails: the dates are roughly 18 days apart, so spreading the queries still
leaves each one close to an index acquisition.

The recorded gaps say so plainly. Median gap from a query to its nearest same-place index
acquisition: **13 days**. 44 of 48 queries are within 30 days.

| Gap | Queries | DINOv2, share of ceiling |
|---|---|---|
| 0–30 days | 44 | 0.918 |
| 31–90 days | 3 | 1.000 |
| 91–180 days | 1 | 1.000 |

So the headline numbers largely measure retrieval between images of the same place taken a
fortnight apart, under similar sun and similar vegetation. That is a much weaker question than the
one intended, and the three long-gap queries are far too few to say anything.

**The fix is a temporal guard band, by direct analogy with the spatial one.** EuroSAT v1 excludes
any index image within 5 km of a query; the temporal corpus needs to exclude any index acquisition
within N days of a query. Spreading the queries measures the gap; only excluding neighbours
enforces it. Building the measurement first is what revealed that the mechanism did not deliver
the guard.

## What this establishes

- The pipeline works end to end: discovery, windowed chips over an identical footprint, a
  place-labelled manifest, embedding, and evaluation, with zero failures.
- Cost is settled: about 6.3 s and 700 KB per chip, so corpus size is not a practical constraint.
- Per-place precision must be read against its ceiling, or it misleads in exactly the cases
  scarce data makes most interesting.
- Twelve distinct places is too easy a task, as expected. Six places score perfectly.

## What it does not establish

- Nothing about seasonal or illumination invariance, for the reason above.
- Nothing comparable to the EuroSAT results: a different corpus, a different relevance notion, and
  a different task difficulty.
- Only RGB representations were run. SSL4EO and TerraMind read 13-band archive members and cannot
  consume these three-band chips.
