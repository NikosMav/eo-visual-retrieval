# Confirmatory evaluation data: design

- Date: 2026-09-03
- Status: implemented and validated, 2026-09-03
- Deliverable: ADR 0006 plus the reproducible evidence behind it

All deliverables below are complete. The final decision is recorded in
[ADR 0006](../../decisions/0006-confirmatory-evaluation-data.md), with executed checks in
[Validation](../../validation.md). BigEarthNet acquisition and evaluation remain subsequent work.

## Why this work exists

Every remaining roadmap item is blocked on the same thing. Optuna tuning, the Qdrant adapter
experiment, and the product-backend choice all need a partition that has never influenced a
decision. ADR 0005 recorded that requirement and named BigEarthNet v2 as the likely source, but it
did not establish where an untouched partition would actually come from.

The plan going into this design was two-tiered: a cheap EuroSAT holdout drawn from the ~25,000
patches v1 never selected, followed by BigEarthNet for the cross-dataset question. Measurement
killed the first tier.

## Finding: EuroSAT is exhausted

EuroSAT's 27,000 patches occupy only **845 distinct 50 km equal-area cells**. Preparing v1
consumed **725 of them, 86%**.

| Quantity | Value |
|---|---:|
| Total patches | 27,000 |
| Total 50 km cells | 845 |
| Cells used by v1 | 725 |
| Cells untouched | 120 |
| Patches in untouched cells | 778 |

Per class, in untouched cells: HerbaceousVegetation **0**, PermanentCrop 1, AnnualCrop 4,
Forest 55, SeaLake 70 (across 4 cells). A class-balanced holdout is impossible, let alone the
three mutually disjoint partitions the protocol requires.

The cause is v1's own sampling. `_spread_sample` in `benchmarks/eurosat.py` uses every available
spatial group before reusing one, which maximised v1's geographic diversity and spent the cell
budget doing it. This was the right choice for v1 and it is not a defect. It is a consequence
nobody could have seen without counting.

Relaxing cell-disjointness to a distance rule does not rescue the tier:

| Minimum distance from every v1 patch | Patches | Classes present | Smallest class |
|---|---:|---:|---:|
| 5 km | 16,024 | 10 | 1,094 |
| 10 km | 8,445 | 10 | 250 |
| 20 km | 1,473 | 10 | 12 |
| 30 km | 355 | 10 | 1 |
| 50 km | 65 | 5 | 0 |

The median unused patch is 7.0 km from a v1 patch. A 10 km set is technically constructible before
internal index/development/final separation is subtracted, but its guarantee is far weaker than
v1's own disjoint 50 km cells plus 5 km guard band, and it still cannot answer whether SSL4EO's
advantage is specific to EuroSAT.

**Consequence:** EuroSAT v1 is permanently a regression and development benchmark. No confirmatory
claim may ever rest on EuroSAT data.

## Decision

BigEarthNet v2 (reBEN) becomes the single confirmatory evaluation set.

### Relevance

Relevance moves from label equality to set similarity over 19 CORINE labels.

- **Precision@k, Recall@k, mAP@k** — binary: relevant iff `Jaccard(query, result) >= tau`,
  with **tau = 0.5 pre-registered**.
- **nDCG@k** — graded: gain is the raw Jaccard value.

nDCG has been reported since v1 against binary relevance, which wastes it. Graded gain is what the
measure was designed for, and multi-label data is the first thing this project has that can supply
one.

Threshold sensitivity at 0.3, 0.5, and 0.7 is reported on the **development partition only**. The
final partition is scored once, at tau = 0.5.

The existing single-label path in `evaluate_store` is not modified. Multi-label relevance is a
separate function selected by manifest type, so every published EuroSAT result stays reproducible.

### Partitions

Three pre-registered partitions: index, development-query, final-query.

They are drawn inside reBEN's own geographically separated official splits, then **independently
audited** with this project's existing machinery — equal-area cell disjointness plus a guard band,
following the `audit_eurosat_manifest` pattern. Accepting an upstream split without re-verifying it
would be inconsistent with how v1 was treated, and the audit is cheap.

BigEarthNet carries acquisition dates. Temporal separation is recorded and enforced where the split
permits. This is the first time the project can control for season at all; EuroSAT exposes no
timestamps, a limitation `docs/validation.md` already records.

Patches the dataset flags as snow, cloud, or shadow are excluded, and the exclusion count is
reported alongside the split.

**Scale:** exactly 5,000 patches — 4,000 index, 500 development queries, 500 final queries —
keeping runtimes within the same order of magnitude as v1's 2,000.

Class balance in v1's sense does not transfer: a patch carrying several labels cannot be assigned
to one class quota. Selection therefore targets even *label frequency* across the 19 CORINE labels
as far as the multi-label structure permits, and the achieved label distribution is reported rather
than forced. A label that cannot reach its target is recorded, not silently dropped.

### The SSL4EO gate

BigEarthNet is 12-band Level-2A; sen2cor drops B10, the cirrus band. The selected SSL4EO-S12
reference consumes 13-band Level-1C. The two are not interchangeable, and the mismatch is both
band count and radiometric quantity.

Verified locally against the installed TerraTorch: TerraMind v1 registers
`untok_sen2l2a@224` at 12 bands alongside `untok_sen2l1c@224` at 13, so TerraMind transfers to
L2A natively. PCA and DINOv2 consume RGB and are unaffected.

Pre-registered gate:

- **If** a publicly published SSL4EO-S12 L2A ResNet-50 checkpoint exists in the SSL4EO-S12
  repository, the TorchGeo weight registry, or the Hugging Face `torchgeo` organisation, and can
  be pinned by SHA-256, SSL4EO enters the confirmatory comparison on L2A.
- **Otherwise** SSL4EO is recorded as absent from it, and EuroSAT v1 remains its only evidence.

Slicing the 13-channel `conv1` down to 12 bands is explicitly rejected. It would silently alter a
frozen pretrained model, violate the frozen-encoder boundary in `AGENTS.md`, and produce scores not
comparable with the published EuroSAT numbers.

### Acquisition

Bounded and recorded before anything is downloaded:

1. Measure the distribution size and check whether selective or shard-level download is possible.
2. Record the measured size, licence, and DOI in the ADR before committing to the download.
3. Keep the archive under `data/downloads/`, verified by published checksum.
4. Read selected members directly from the archive rather than materialising a duplicate dataset —
   the pattern `embeddings/ssl4eo.py` already uses for EuroSAT.

## Deliverables

| Artifact | Purpose |
|---|---|
| `docs/decisions/0006-confirmatory-evaluation-data.md` | The decision, its options, and consequences |
| `scripts/eurosat_cell_budget.py` | Reproduces both tables above: cell budget and distance tiers |
| `docs/validation.md` entry | Records the measurement as executed evidence |
| `docs/project-context.md` update | Roadmap reflects that Tier A is dead and why |

The script matters. The exhaustion finding is the load-bearing claim in this ADR, and this project
does not accept load-bearing claims that cannot be re-executed.

## Out of scope

This ADR specifies data and protocol. It produces no scores.

Explicitly excluded: downloading BigEarthNet, implementing the multi-label evaluator, preparing any
partition, Optuna tuning, the Qdrant adapter, and the product surface. Each is authorised by this
ADR but planned separately.

## Acceptance criteria

1. ADR 0006 follows the numbering, status, and Context/Decision/Options/Consequences shape of
   ADRs 0002 through 0005.
2. Every quantity in the ADR is reproducible by the committed script against the local archive.
3. `docs/validation.md` records the measurement under the executed-evidence policy, and claims
   nothing the measurement does not support.
4. No published EuroSAT v1 result changes.
5. `ruff`, `mypy`, and the test suite pass, and coverage stays at or above 75%.

## Risks and open questions

| Item | Status |
|---|---|
| reBEN download size and whether shard-level download is possible | **Unverified.** Action item before acquisition. |
| Existence of a pinnable SSL4EO-S12 L2A checkpoint | **Unverified.** Resolved by the gate above; may resolve to "absent". |
| tau = 0.5 is a judgement, not a derived value | Accepted. Justified in the ADR, sensitivity reported on development data only. |
| If SSL4EO is absent, the confirmatory comparison weakens to TerraMind against RGB baselines | Accepted and must be stated plainly wherever the result is reported. |
| Overlap between BigEarthNet and the pretraining corpora of the frozen encoders | Audited separately, per ADR 0005. Not resolved here. |
