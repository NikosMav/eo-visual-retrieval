# ADR 0002: use georeferenced EuroSAT for the first retrieval benchmark

- Status: accepted
- Date: 2026-09-01

## Context

The pipeline needs a real labeled dataset before its PCA and DINOv2 scores mean anything. The
first benchmark must be small enough to reproduce locally, use one unchanged manifest for both
models, and make geographic leakage measurable.

Three public datasets were considered:

| Dataset | Advantage | Reason not selected for this phase |
|---|---|---|
| PatternNet | Designed for remote-sensing image retrieval; 30,400 images in 38 classes | The published image package does not provide a sufficiently clear geographic identity for a defensible spatial split |
| BigEarthNet v2 | Large, georeferenced, multi-label Sentinel-1/2 corpus with official splits | The Sentinel-2 archive is about 59 GiB and its multi-label task adds scale and evaluation complexity before the baseline is established |
| EuroSAT | 27,000 labeled, georeferenced Sentinel-2 patches in 10 classes; manageable official archive | Labels are broad land-use/land-cover proxies and acquisition timestamps are not provided |

## Decision

Use the official EuroSAT multispectral archive from DOI
[`10.5281/zenodo.7711810`](https://doi.org/10.5281/zenodo.7711810). Read its georeferencing to
construct the split, then materialize RGB from Sentinel-2 bands 4, 3, and 2 with the published
fixed digital-number range 0–2750.

Version 1 of the benchmark contains 2,000 images:

- 10 official classes;
- 160 index images per class, or 1,600 total;
- 40 query images per class, or 400 total;
- seed 42;
- 50 km cells in global equal-area CRS EPSG:6933;
- no spatial cell shared between index and query;
- at least 5 km between every query centroid and every index centroid.
- at least 10 query spatial cells represented per class.

The manifest records the source archive member, DOI, source CRS and bounds, longitude/latitude
centroid, spatial group, split parameters, RGB bands/stretch, and output hash. Raw imagery and the
generated manifest remain ignored local artifacts.

Relevance is binary class agreement: an index image is relevant when its EuroSAT class equals the
query class. Both embedding backends must use the same immutable manifest and RGB files.

## Why this split

A random image split can put adjacent patches or repeat observations of nearby ground on both
sides. Models can then succeed by recognizing location-specific appearance instead of transferring
to separated geography. Grouping in EPSG:6933 provides a stable Europe-wide metre-based grid,
while a great-circle guard band also protects boundaries between adjacent cells.

The algorithm samples across spatial groups before selecting a second image from a group. It then
removes all index candidates from query groups and all candidates inside the guard band. It fails
rather than silently reducing a per-class quota.

## Consequences and limits

- The benchmark measures retrieval under a bounded spatial holdout, not random-split memorization.
- EuroSAT class labels provide reproducible relevance, but they do not capture every visual or
  analyst intent.
- DINOv2 still receives RGB only; this decision does not make it a multispectral model.
- The multispectral archive provides coordinates but not acquisition timestamps. Spatial
  separation prevents the same or nearby locations crossing the split, including repeat
  observations there, but seasonal and temporal generalization are **not evaluated**.
- A later BigEarthNet or STAC-derived experiment should add timestamps, temporal holdouts,
  multilabel relevance, and an EO-specific multispectral encoder.

## Integrity boundary

Preparation accepts only a local archive whose MD5 equals the official published value
`091174add3c8e680a49244acf185b9f0`. MD5 is used here to match the publisher's file integrity
record, not as a security primitive.
