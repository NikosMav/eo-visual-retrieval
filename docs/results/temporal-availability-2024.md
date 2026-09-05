# Clear-sky availability across Europe, 2024 — 2026-09-04

Before building a corpus of places observed on many dates, this measures whether the dates exist.
Executed against Microsoft Planetary Computer's `sentinel-2-l2a` over the frozen
`europe-latitude-spread-v1` selection, for calendar 2024, with a 2,560 m window per place and a
10% scene cloud ceiling.

No imagery was downloaded. Machine-readable output:
[`temporal-availability-2024.json`](temporal-availability-2024.json).

## Result

All 12 places are usable. 303 distinct acquisition days in total, median 19 per place.

| Place | Latitude | Clear days | Months covered |
|---|---|---|---|
| crete-messara | 35.05 | **71** | 12 |
| andalusia-guadalquivir | 37.55 | 28 | 12 |
| attica-thriasio | 38.53 | 43 | 8 |
| po-valley-cremona | 45.14 | 33 | 11 |
| beauce-orleans | 48.05 | **6** | 3 |
| brandenburg-spreewald | 51.88 | 13 | 8 |
| flevoland-polder | 52.52 | 7 | 6 |
| jutland-viborg | 56.45 | 13 | 6 |
| smaland-vaxjo | 56.88 | 17 | 8 |
| uppland-uppsala | 59.86 | 18 | 8 |
| ostrobothnia-vaasa | 63.10 | 35 | 11 |
| lapland-rovaniemi | 66.50 | 19 | 9 |

## What this shows

**Clear-sky availability is not monotonic with latitude.** The intuitive expectation — the further
north, the cloudier — is wrong. The worst coverage in Europe is the temperate maritime band around
48–53° N: Beauce got **6 usable days in an entire year**, Flevoland 7. Both the Mediterranean
(Crete, 71 days) and the Nordic interior (Ostrobothnia, 35 days) do far better. Atlantic frontal
systems dominate north-west Europe year-round, while the Mediterranean has a reliable dry season
and the Nordic interior is drier than its reputation.

The practical consequence for anyone building an EO corpus: **the cloud budget, not the satellite
revisit interval, decides how much data a place yields.** Sentinel-2's revisit is roughly uniform
over this area. The 12-fold spread between Crete and Beauce is entirely weather.

**A balanced corpus cannot have equal dates per place.** Levelling down to the scarcest place would
cap every location at 6 dates and discard 65 of Crete's 71. Either the corpus is imbalanced, or it
is small, and that trade-off has to be chosen deliberately rather than discovered later.

**Attica has no clear scene before 2024-05-22** under this cloud ceiling, despite 43 days
afterwards. Seasonal availability is uneven within a place as well as between places, so a corpus
sampled uniformly across the calendar is not achievable everywhere.

## What this does not show

- **Nothing about retrieval quality.** No imagery was downloaded and no representation was run.
- **Nothing about a usable benchmark yet.** Twelve places spread from Crete to Lapland are
  visually distinct, so "find the same place" would likely be easy and therefore uninformative. A
  real temporal benchmark needs many more places, including neighbouring ones, so that the task is
  to distinguish a place from its neighbours rather than a continent from another.
- **One provider, one year, one cloud ceiling.** A 20% ceiling would change every count, and
  scene-level cloud cover does not mean the specific 2,560 m window is clear — it describes the
  whole tile.

## Reproducing

```powershell
eovr temporal-survey `
  --datetime 2024-01-01/2024-12-31 `
  --max-cloud-cover 10 --window-m 2560 --limit 80 `
  --output docs/results/temporal-availability-2024.json
```

Defaults to Planetary Computer and the frozen selection, needs no account, and downloads no
pixels. Tiles from one overpass are collapsed to a single acquisition day, keeping the least
cloudy, because two tiles of one moment are one observation.
