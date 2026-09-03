# ADR 0009: freeze the confirmatory model roster at four representations

- Status: accepted
- Date: 2026-09-03
- Decision owners: project maintainers

## Context

[ADR 0006](0006-confirmatory-evaluation-data.md) pre-registered a gate: SSL4EO-S12 would enter the
confirmatory comparison if a published Level-2A ResNet-50 checkpoint could be pinned by SHA-256,
and would otherwise be recorded as absent.

The gate resolved **absent**. The recorded evidence in
[`evidence/ssl4eo-l2a-gate-2026-09-03.json`](evidence/ssl4eo-l2a-gate-2026-09-03.json) enumerates
the official SSL4EO-S12 model list, the TorchGeo ResNet-50 weight registry, and all 36 public
TorchGeo Hugging Face entries. Every SSL4EO-S12 Sentinel-2 ResNet-50 entry is 13-band or RGB. A
12-channel `SENTINEL2_ALL_SECO_ECO` entry exists and was excluded, correctly, because it is
pretrained on the SSL4Eco dataset rather than the pre-registered SSL4EO-S12 corpus.

Read literally, ADR 0006 then leaves the confirmatory comparison as PCA-64, DINOv2, and
TerraMind-Tiny. That outcome is worse than the gate's wording suggests. TerraMind already scored
below SSL4EO on EuroSAT v1 — mAP@10 `0.68688` against `0.81360` — so the confirmatory set would
contain no multispectral representation that has won anything in this project, and the headline
finding recorded in [validation](../validation.md), that frozen 13-band SSL4EO-S12 outperformed
both RGB baselines, would go untested on new data.

The BigEarthNet partitions are frozen by [ADR 0008](0008-bigearthnet-selection-protocol.md), but no
imagery has been transferred and no score exists. This is therefore the last moment at which
changing the roster is pre-registration rather than post-hoc model selection. Once a single
confirmatory score exists, it is too late permanently.

## Decision

### 1. Four representations enter the confirmatory comparison

| Model | Input | Role |
|---|---|---|
| PCA-64 | RGB | Transparency floor, fitted on the index partition only |
| DINOv2 ViT-S/14 | RGB | General-purpose visual pretraining |
| SSL4EO-S12 RGB MoCo ResNet-50 | RGB | **Added here.** EO-domain pretraining without extra bands |
| TerraMind-Tiny | 12-band L2A | Multispectral, native Level-2A |

### 2. The added checkpoint is pinned

| Field | Value |
|---|---|
| Repository | `torchgeo/resnet50_sentinel2_rgb_moco` |
| Revision | `e6704867d1bf7f77c403d8078f41ccf5b2ffaa6c` |
| Filename | `resnet50_sentinel2_rgb_moco-2b57ba8b.pth` |
| Size | 94,361,669 bytes |
| SHA-256 | `2b57ba8b9964dbe1c409aac1bb79b4d97c19c874ffe7934799b7c8ad94ff85f0` |

These are the provider's published identity, read from the Hugging Face API at the revision the
gate evidence already records. The filename prefix matches the digest, the same convention as the
existing `resnet50_sentinel2_all_moco-df8b932e.pth`. The local file is verified against this digest
at first use, exactly as the 13-band checkpoint is; nothing here asserts a local verification that
has not happened.

### 3. SSL4EO-RGB is also evaluated on EuroSAT v1, as a band ablation

Against the existing 13-band SSL4EO result this is a **controlled** comparison: identical ResNet-50
architecture, identical SSL4EO-S12 pretraining corpus, the same 2,000 selected patches, the same
split, the same class-label relevance, and the same exact-cosine ranker. Only the input bands
differ.

[ADR 0003](0003-ssl4eo-s12-multispectral-encoder.md) recorded that its result "does not by itself
prove that non-visible bands caused the difference", and `docs/models-and-metrics.md` still states
the comparison "is not a controlled band ablation". This supplies the missing control.

EuroSAT v1 is a regression and development benchmark, so adding a model to it is legitimate. The
result is **not** confirmatory evidence and must never be reported as such.

### 4. Which comparisons are controlled is stated, not implied

Three of the four pairings are confounded. Reporting them without saying so would repeat the
mistake ADR 0003 had to disclaim.

| Comparison | Dataset | Differs in | Controlled |
|---|---|---|---|
| SSL4EO-RGB vs SSL4EO 13-band | EuroSAT v1 | Input bands only | **Yes** |
| SSL4EO-RGB vs DINOv2 | Both | Pretraining domain *and* architecture | No |
| SSL4EO-RGB vs TerraMind-L2A | BigEarthNet | Bands, architecture, and pretraining | No |
| PCA-64 vs anything | Both | Everything | No — a floor, not a control |

### 5. This does not resurrect the 13-band reference

Adding an RGB variant of the same pretraining corpus does not test the 13-band representation. The
BigEarthNet result will not confirm or refute the project's existing multispectral finding, and
every report of it must say so. ADR 0006's gate outcome stands unchanged.

### 6. Published evidence is untouched

The single-label evaluator, the exact ranker, and every existing result under `docs/results/`
remain unchanged. A new model produces a new result file; it does not edit an old one.

## Options considered

| Option | Evidence value | Cost | Assessment |
|---|---|---|---|
| Proceed with three models | The incumbent never faces the confirmatory test | None | Rejected. Spends the full acquisition to compare one already-losing multispectral model against two RGB baselines. |
| Add SSL4EO-S12 RGB MoCo | Supplies the missing band ablation and an EO-pretrained RGB control | One adapter path, one embedding run | **Selected.** |
| Add a flexible-band foundation model (DOFA, Copernicus-FM, CROMA) | A second true multispectral representation | A new model contract to build and verify | Deferred. Stronger multispectral coverage, but does not resolve the pretraining-versus-bands confound, and is more new machinery before any score exists. |
| Defer until imagery arrives | None | None | Rejected. Choosing models after seeing data is the pre-registration violation ADR 0005 exists to prevent. |

## Consequences

- EuroSAT v1 finally supports a controlled band ablation. Its status as a regression benchmark is
  unchanged; the ablation is development evidence.
- `embeddings/ssl4eo.py` hardcodes a `(64, 13, 7, 7)` `conv1` check. A 3-band path is required and
  must not weaken the 13-band contract, which protects the published result.
- SSL4EO-RGB may lose to DINOv2. A pre-registered model that loses is evidence, not a failure; its
  result is published either way.
- The roster changed after the partitions were frozen. That is legitimate only because no imagery
  had been transferred and no score existed, and the sequence is recorded here so it stays
  auditable.
- Nothing in this ADR produces a score. It specifies the roster only.

## Action items

- [ ] Add a 3-band SSL4EO adapter path with the pinned checkpoint, leaving the 13-band contract intact.
- [ ] Verify the downloaded checkpoint against the recorded SHA-256 before first use.
- [ ] Run SSL4EO-RGB on EuroSAT v1 and publish it as a regression result with the ablation caveat.
- [ ] Include SSL4EO-RGB in the BigEarthNet confirmatory embedding run.
- [ ] Reconsider a flexible-band foundation model only after the first confirmatory result exists.

## Primary references

- [SSL4EO-S12](https://arxiv.org/abs/2211.07044)
- [TorchGeo pretrained weights](https://docs.torchgeo.org/en/stable/tutorials/pretrained_weights.html)
- [ADR 0003](0003-ssl4eo-s12-multispectral-encoder.md), [ADR 0006](0006-confirmatory-evaluation-data.md), [ADR 0008](0008-bigearthnet-selection-protocol.md)
