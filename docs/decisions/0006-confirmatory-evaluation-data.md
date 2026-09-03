# ADR 0006: BigEarthNet v2 as the confirmatory evaluation set

- Status: accepted
- Date: 2026-09-03
- Decision owners: project maintainers

## Context

ADR 0005 required genuinely new, geographically separated data before confirmatory model
selection, and named BigEarthNet v2 as the likely source. It did not establish where an untouched
partition would come from. The cheaper plan was a holdout drawn from the 25,000 EuroSAT patches
v1 never selected.

Measurement rejected that plan. EuroSAT's 27,000 patches occupy only 845 distinct 50 km
equal-area cells, and preparing v1 consumed 725 of them.

| Quantity | Value |
|---|---:|
| Total patches | 27,000 |
| Total 50 km cells | 845 |
| Cells used by v1 | 725 |
| Cells untouched | 120 |
| Patches in untouched cells | 778 |

In those untouched cells, `HerbaceousVegetation` has zero patches, `PermanentCrop` has one, and
`AnnualCrop` has four. A class-balanced holdout is impossible, let alone the three mutually
disjoint partitions the protocol requires.

The cause is v1's own sampling: `_spread_sample` uses every available spatial group before reusing
one. That maximised v1's geographic diversity and spent the cell budget doing it. It was the right
choice for v1 and is not a defect.

Relaxing cell disjointness to a distance rule does not rescue the plan.

| Minimum distance from every v1 patch | Patches | Classes present | Smallest class (of all 10) |
|---|---:|---:|---:|
| 5 km | 16,024 | 10 | 1,094 |
| 10 km | 8,445 | 10 | 250 |
| 20 km | 1,473 | 10 | 12 |
| 30 km | 355 | 10 | 1 |
| 50 km | 65 | 5 | 0 |

A 10 km set is constructible before internal partition separation is subtracted, but its guarantee
is far weaker than v1's disjoint 50 km cells plus 5 km guard band, and it still cannot answer
whether SSL4EO's advantage is specific to EuroSAT.

Reproduce with `scripts/eurosat_cell_budget.py`.

## Decision

1. EuroSAT v1 is permanently a regression and development benchmark. No confirmatory claim rests
   on EuroSAT data.
2. BigEarthNet v2 (reBEN) is the single confirmatory evaluation set.
3. Relevance over its 19 CORINE labels is set similarity, not equality. Precision@k, Recall@k, and
   mAP@k are binary at `Jaccard >= 0.5`; nDCG@k uses the raw Jaccard as graded gain. The threshold
   is pre-registered. Sensitivity at 0.3, 0.5, and 0.7 is reported on the development partition
   only; the final partition is scored once at 0.5.
4. Three pre-registered partitions — 4,000 index, 500 development queries, 500 final queries — are
   drawn inside reBEN's official geographically separated splits and then independently audited
   with this project's own cell and guard-band machinery.
5. Acquisition is bounded. The distribution size, licence, and DOI are recorded before download;
   the archive is checksum-verified under `data/downloads/`; selected members are read directly
   from it rather than materialised as a duplicate dataset.
6. SSL4EO-S12 enters the confirmatory comparison only through a pre-registered gate, below.
7. The existing single-label evaluator path is not modified, so published EuroSAT results stay
   reproducible.

### The SSL4EO gate

BigEarthNet is 12-band Level-2A; sen2cor drops B10. The selected SSL4EO-S12 reference consumes
13-band Level-1C. The mismatch is both band count and radiometric quantity.

TerraMind registers `untok_sen2l2a@224` at 12 bands alongside `untok_sen2l1c@224` at 13, verified
against the installed TerraTorch, so it transfers to L2A natively. PCA and DINOv2 consume RGB and
are unaffected.

- If a published SSL4EO-S12 L2A ResNet-50 checkpoint exists in the SSL4EO-S12 repository, the
  TorchGeo weight registry, or the Hugging Face `torchgeo` organisation, and can be pinned by
  SHA-256, SSL4EO enters the comparison on L2A.
- Otherwise SSL4EO is recorded as absent from it, and EuroSAT v1 remains its only evidence.

Slicing the 13-channel `conv1` to 12 bands is rejected: it would silently alter a frozen
pretrained model, violate the frozen-encoder boundary, and produce scores not comparable with the
published EuroSAT numbers.

## Options considered

| Option | Evidence strength | Cost | Assessment |
|---|---|---|---|
| EuroSAT holdout from untouched cells | Would test new geography | None | **Impossible.** One class has zero patches available. |
| EuroSAT holdout at 10 km separation | Weak; same dataset and taxonomy | None | Rejected. Invites overclaiming from a guarantee weaker than v1's own. |
| BigEarthNet v2 | Cross-dataset, multi-label, timestamped | Download plus evaluator work | **Selected.** |
| Re-derive v1 with smaller cells | Would free cells | Invalidates published v1 results | Rejected. Re-opens ADR 0002 and discards executed evidence. |
| Another dataset, e.g. So2Sat LCZ42 | Single-label, city-separated | Research plus download | Deferred. Reconsider only if BigEarthNet acquisition proves unviable. |

## Consequences

- The project may gain its first control for season and acquisition date, once reBEN's
  acquisition metadata is confirmed to be present and usable. EuroSAT exposes no timestamps, a
  limitation recorded in `docs/validation.md`.
- Multi-label relevance is new machinery with a judgement in it. `tau = 0.5` is not derived from
  anything; it is pre-registered so it cannot be chosen after seeing scores.
- If the SSL4EO gate resolves to absent, the confirmatory comparison weakens to TerraMind against
  the RGB baselines, and every report of it must say so.
- Overlap between BigEarthNet and the pretraining corpora of the frozen encoders is a separate
  audit, unresolved here.
- Nothing in this ADR produces a score. It specifies data and protocol only.

## Action items

- [ ] Measure the reBEN distribution size and whether shard-level download is possible; record
      size, licence, and DOI before downloading.
- [ ] Confirm reBEN exposes usable acquisition dates before relying on temporal separation.
- [ ] Resolve the SSL4EO L2A checkpoint gate to present-with-SHA or absent.
- [ ] Implement multi-label relevance as a separate path beside the single-label evaluator.
- [ ] Prepare and audit the three partitions; publish the achieved label distribution.
- [ ] Only then run the development comparison, freeze configuration, and score the final set once.

## Primary references

- [BigEarthNet](https://bigearth.net/)
- [reBEN: Refined BigEarthNet dataset](https://arxiv.org/pdf/2407.03653)
- [SSL4EO-S12](https://arxiv.org/pdf/2211.07044)
- [ADR 0002](0002-georeferenced-eurosat-benchmark.md), [ADR 0005](0005-evaluation-foundations-before-product.md)
