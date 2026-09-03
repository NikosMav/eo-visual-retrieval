# ADR 0008: BigEarthNet acquisition selection before model evaluation

- Status: accepted selection policy; S2 acquisition and scoring remain separate gates
- Date: 2026-09-03
- Scope: freeze the IDs to retain during the future bounded S2 stream

## Decision

Preserve ADR 0006's sizes and official split assignments. Add chronological windows, independent
spatial cells, and a minimum label count before seeing imagery or retrieval scores.

| Partition | Official split | Inclusive acquisition window | Patches |
|---|---|---|---:|
| Index | train | 2017-06-01 through 2017-09-30 | 4,000 |
| Development | validation | 2017-11-01 through 2018-02-28 | 500 |
| Final | test | 2018-04-01 through 2018-05-31 | 500 |

The windows leave at least 30 days between adjacent partitions. Dates come from the source patch
identities. These windows intentionally introduce season and geography shifts; a later score
cannot isolate either effect by itself.

Require at least five examples of every one of the 19 labels in each partition. This is a
coverage floor, not a balanced sample or a guarantee of precise per-label estimates. Inspecting
metadata and label availability to construct the sample is permitted; model performance must
not influence selection. Reference-map pixel values are not read during footprint preparation.

Each partition contains at most one observation of a given MGRS tile/row/column. Across partitions,
50 km EPSG:6933 cells are disjoint and source-derived centres are at least 7 km apart. Each query
partition uses at least 20 cells. The independent audit also checks a conservative footprint
separation estimate above 5 km: minimum spherical centre distance minus each partition's largest
centre-to-corner radius, with radii increased by 1%. Every source map is a north-up, 1,200 m UTM
square. This distance model is recorded explicitly; it is not an exact ellipsoidal polygon-distance
calculation. S2 bands must later match these native bounds, transforms, and CRS before use.

## Deterministic selection

1. Use seed 42 for SHA-256 ordering of IDs and cells. Filter to recommended metadata, the official
   split, and the stated dates. Exclude the previously inspected S2 probe ID from final selection.
2. Allocate final first, then development, then index. Later partitions exclude cells already used
   by earlier partitions and patches closer than the 7 km centre guard.
3. Deduplicate observations by tile/row/column using the seeded order before counting capacity.
4. For query partitions, visit rare labels first. Allocate cells that satisfy the label minimum,
   using seeded cell order to break ties. Add cells in seeded order until both the 20-cell minimum
   and the requested patch capacity are available. This avoids spending every cell on the first
   query partition. The index can use all remaining eligible cells.
5. Select one patch per allocated cell, fill missing label quotas, then fill the exact requested
   size by round-robin sampling across those cells. Fail if any rule is infeasible; do not silently
   shrink the sample, change the seed, or relax the constraints.
6. Build the full inventory with Pillow's TIFF-tag reader using the source's strict
   [GeoTIFF scale/tiepoint and projected-CRS profile](https://docs.ogc.org/is/19-008r4/19-008r4.html).
   Independently reload all selected TIFF headers with Rasterio from the checksum-verified archive,
   require agreement with the inventory, and recompute centres, cells, distances, dates,
   exclusions, and label counts. Publish aggregate
   evidence and content hashes; retain full selection IDs only under ignored `data/`.

The resulting acquisition-selection JSON is not an image manifest or a relevance manifest. It
records which patches to obtain. Future acquisition must bind its content hash, verify the complete
S2 stream, check every selected band against the reference geometry, and only then prepare model
inputs and their manifests. The 2 GiB acquisition ceiling in ADR 0007 remains unchanged.

## Interpretation and remaining limits

The committed [footprint inventory report](../results/bigearthnet-footprints.json) and
[selection audit](../results/bigearthnet-selection-audit.json) are the execution evidence. The
selection must be reproducible from verified sources before it is used for acquisition.

Final imagery and model scores remain untouched. The protected final set still needs a frozen
model/search configuration and a one-shot scoring gate. Geographic checks here concern the three
BigEarthNet partitions; they do not establish separation from all historical EuroSAT observations
or encoder pretraining corpora. Those overlaps require a separate audit before generalization
claims. No BigEarthNet retrieval quality is established by this decision.
