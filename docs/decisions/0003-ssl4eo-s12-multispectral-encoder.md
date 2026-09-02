# ADR 0003: SSL4EO-S12 for the first multispectral retrieval experiment

- Status: accepted
- Date: 2026-09-02
- Decision owners: project maintainers

## Context

EuroSAT v1 was prepared from the official 13-band Sentinel-2 Level-1C archive, but the first two
retrieval baselines consume only deterministic RGB derivatives. Milestone 3 requires one frozen
EO-specific multispectral encoder on the unchanged 1,600-index/400-query split. The experiment
must preserve exact cosine ranking and class-agreement relevance so only the representation path
changes.

The selected model needs a public checkpoint, reproducible preprocessing, support for the source
bands, and an implementation small enough to study. It must not require invented metadata or
supervised fitting on the benchmark.

## Decision

Use the SSL4EO-S12 MoCo ResNet-50 Sentinel-2 checkpoint registered by TorchGeo. It was pretrained
on Sentinel-2 Level-1C imagery and accepts all 13 bands, matching the EuroSAT product and band set.
The encoder remains frozen and its 2,048-dimensional output is L2-normalized before exact cosine
search.

The implementation reads only the manifest-selected TIFF members from the checksum-verified
EuroSAT archive. EuroSAT stores bands as
`B01,B02,B03,B04,B05,B06,B07,B08,B09,B10,B11,B12,B8A`; the checkpoint expects
`B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12`. The loader therefore moves `B8A` into
position nine before inference.

Preprocessing follows the registered checkpoint transform:

1. clip Level-1C digital numbers to `[0, 10000]` and divide by `10000`;
2. resize all 13 channels to 256 × 256 with bilinear interpolation;
3. take the centered 224 × 224 crop;
4. run the frozen encoder and L2-normalize its feature vector.

The checkpoint is pinned by repository revision, filename, and SHA-256. Weights are downloaded to
an ignored local directory and are not redistributed by this repository.

## Options considered

| Option | Fit for this experiment | Main trade-off |
|---|---|---|
| SSL4EO-S12 MoCo ResNet-50 | Direct 13-band Sentinel-2 Level-1C match | Older CNN architecture and possible unknown pretraining-location overlap |
| Prithvi-EO-2.0 | Strong EO foundation model | HLS checkpoint uses six fixed bands, discarding seven EuroSAT bands |
| Clay v1.5 | Flexible multispectral encoder | Requires wavelength plus location/time metadata; EuroSAT lacks acquisition timestamps |
| DOFA | Wavelength-aware multisensor model | Less direct first comparison and a larger integration surface |

## Consequences

Positive consequences:

- all 13 source bands are used with documented ordering and scaling;
- no EuroSAT labels or query examples influence representation learning;
- the same IDs, splits, relevance judgments, metrics, and exact ranker remain in use;
- raw imagery and generated embeddings remain outside Git;
- a compact ResNet implementation makes the input path inspectable.

Limitations and risks:

- this comparison changes both pretraining domain and input channels, so it does not isolate the
  causal value of the ten non-RGB bands;
- EuroSAT has no acquisition timestamps, so temporal leakage cannot be audited;
- public checkpoint documentation does not let us rule out geographic overlap between its
  pretraining corpus and EuroSAT;
- fixed 64 × 64 patches are enlarged substantially for a 224 × 224 model input;
- class agreement remains only a proxy for retrieval intent.

## Follow-up actions

Completed in this phase:

- recorded aggregate and per-class results on the unchanged EuroSAT v1 manifest;
- inspected best and worst result grids using the existing RGB derivatives for readability;
- preserved DINOv2 and PCA as the fixed quality references.

Remaining:

- If attribution to extra spectral bands matters, add a controlled RGB-only SSL4EO ablation rather
  than inferring it from this three-model comparison.
- Move to approximate search only after this representation comparison is complete.

## References

- [SSL4EO-S12 project and checkpoint documentation](https://github.com/zhu-xlab/SSL4EO-S12)
- [SSL4EO-S12 paper](https://arxiv.org/abs/2211.07044)
- [TorchGeo pretrained weights registry](https://github.com/torchgeo/torchgeo)
