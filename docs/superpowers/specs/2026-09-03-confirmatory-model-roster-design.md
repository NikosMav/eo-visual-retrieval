# Confirmatory model roster: design

- Date: 2026-09-03
- Status: approved, not yet implemented
- Deliverable: ADR 0009 plus supporting doc updates

## Why this work exists, and why now

ADR 0006 pre-registered a gate: SSL4EO-S12 would enter the confirmatory comparison if a 12-band
Level-2A ResNet-50 checkpoint could be pinned by SHA-256, and would otherwise be recorded absent.

The gate resolved **absent**. `docs/decisions/evidence/ssl4eo-l2a-gate-2026-09-03.json` enumerates
every SSL4EO-S12 Sentinel-2 ResNet-50 entry: all are 13-band or RGB. A 12-channel
`SENTINEL2_ALL_SECO_ECO` exists and was correctly excluded, because it is pretrained on the
SSL4Eco dataset rather than the pre-registered SSL4EO-S12.

Taken literally, that leaves the confirmatory comparison as PCA-64, DINOv2, and TerraMind-Tiny.
The consequence is worse than it looks: TerraMind already scored **below** SSL4EO on EuroSAT v1
(mAP@10 0.68688 against 0.81360), so the confirmatory set would contain no multispectral model
that has ever won anything here, and the project's headline finding — that frozen 13-band
SSL4EO-S12 beat both RGB baselines — would go untested on new data.

The BigEarthNet partitions are frozen but no imagery has been transferred and no score exists.
This is therefore the last moment at which changing the roster is pre-registration rather than
post-hoc selection. After a single score exists it is too late, permanently.

## Decision

Four models enter the confirmatory comparison:

| Model | Input | Role |
|---|---|---|
| PCA-64 | RGB | Transparency floor, fitted on index only |
| DINOv2 ViT-S/14 | RGB | General-purpose visual pretraining |
| **SSL4EO-S12 RGB MoCo ResNet-50** | RGB | **New.** EO-domain pretraining, no extra bands |
| TerraMind-Tiny | 12-band L2A | Multispectral, native L2A |

### The checkpoint

Pinned from the published Hugging Face LFS identity at the revision already recorded in the
SSL4EO gate evidence:

| Field | Value |
|---|---|
| Repository | `torchgeo/resnet50_sentinel2_rgb_moco` |
| Revision | `e6704867d1bf7f77c403d8078f41ccf5b2ffaa6c` |
| Filename | `resnet50_sentinel2_rgb_moco-2b57ba8b.pth` |
| Size | 94,361,669 bytes |
| SHA-256 | `2b57ba8b9964dbe1c409aac1bb79b4d97c19c874ffe7934799b7c8ad94ff85f0` |

The filename prefix matches the digest, the same convention as the existing
`resnet50_sentinel2_all_moco-df8b932e.pth`. This is the *published* identity, recorded from the
provider API; the local file is verified against it at first use, exactly as the 13-band
checkpoint is.

### The band ablation on EuroSAT v1

SSL4EO-RGB is also evaluated on EuroSAT v1, as a regression-benchmark result.

Against the existing 13-band SSL4EO run this is a **controlled** comparison: same ResNet-50
architecture, same SSL4EO-S12 pretraining corpus, same 2,000 selected patches, same split, same
class-label relevance, same exact-cosine ranker. Only the input bands differ.

That is the ablation ADR 0003 explicitly recorded as missing, and `docs/models-and-metrics.md`
still states the multispectral comparison "is not a controlled band ablation". EuroSAT v1 is a
regression and development benchmark, so adding a model to it is legitimate; the result is
**not** confirmatory evidence and must never be reported as such.

### What is controlled and what is not

Stating this explicitly is the point of the ADR, because three of the four pairings are confounded.

| Comparison | Dataset | Differs in | Controlled |
|---|---|---|---|
| SSL4EO-RGB vs SSL4EO-13-band | EuroSAT v1 | Input bands only | **Yes** |
| SSL4EO-RGB vs DINOv2 | Both | Pretraining domain *and* architecture | No |
| SSL4EO-RGB vs TerraMind-L2A | BigEarthNet | Bands, architecture, pretraining | No |
| PCA-64 vs anything | Both | Everything | No — a floor, not a control |

### What this does not do

Adding an RGB variant of the SSL4EO-S12 corpus does **not** resurrect the 13-band reference. The
BigEarthNet result will still not test the 13-band representation on new data, and every report of
it must say so. ADR 0006's gate outcome stands.

## Deliverables

| Artifact | Purpose |
|---|---|
| `docs/decisions/0009-confirmatory-model-roster.md` | The decision, options, and consequences |
| `docs/validation.md` entry | Records the checkpoint identity lookup as executed evidence |
| `docs/models-and-metrics.md` update | The "not a controlled band ablation" caveat gains its planned resolution |
| `docs/project-context.md` update | Roadmap reflects the four-model roster |

## Out of scope

This ADR specifies the roster. It produces no scores and writes no model code.

Excluded and planned separately: the 3-band SSL4EO adapter, downloading the checkpoint, running
the EuroSAT ablation, BigEarthNet acquisition, and the confirmatory run itself.

## Acceptance criteria

1. ADR 0009 follows the shape of ADRs 0006-0008.
2. Every checkpoint field is traceable to the provider API response, and is described as a
   published identity rather than a locally verified one.
3. `docs/validation.md` claims only the lookup that was executed.
4. No published EuroSAT v1 result changes, and `evaluation.py` and `retrieval.py` are untouched.
5. Ruff, mypy, and the suite pass; coverage stays at or above 75%.

## Risks

| Item | Status |
|---|---|
| `embeddings/ssl4eo.py` hardcodes a `(64, 13, 7, 7)` `conv1` check | A 3-band path is required. It must not weaken the 13-band contract. Adapter work, planned separately. |
| SSL4EO-RGB may simply lose to DINOv2 on RGB | Acceptable. A pre-registered model that loses is evidence, not a failure. |
| The band ablation runs on the benchmark that already informed decisions | Accepted and stated: regression evidence only, never confirmatory. |
| Roster changed after partitions were frozen | Legitimate: no imagery transferred, no score exists. Recorded so the sequence is auditable. |
