# ADR 0012: Local text, image, and hybrid scene search

- Status: accepted for the first implementation; model quality remains experimental
- Date: 2026-09-05
- Scope: a new search surface over prepared RGB corpora

## Context

An image-only encoder such as DINOv2 has no aligned text tower. Feeding a prompt to an unrelated
text encoder and comparing its vectors with DINOv2 vectors is invalid, even if their dimensions
happen to match. We need a shared image/text representation and a separate treatment of exact
metadata constraints. A query for recent imagery near Athens is not satisfied by an old image
that merely looks urban.

## Research and alternatives

Sources were checked on 2026-09-05. Published results on other datasets are selection context,
not evidence of superiority on this project's Sentinel-2 chips.

| Candidate | Evidence and practical trade-off | Decision |
|---|---|---|
| [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP), [paper](https://arxiv.org/abs/2306.11029) | EO image/text dual encoder, official OpenCLIP loading path and downloadable weights. RGB input; aerial-image pretraining does not establish performance on 10 m Sentinel-2 chips. | Start with the smaller ViT-B/32 checkpoint, frozen. |
| [GeoRSCLIP / RS5M](https://github.com/om-ai-lab/RS5M), [paper](https://arxiv.org/abs/2306.11300) | Remote-sensing vision/language model trained using a large image-caption dataset. A credible domain-specific challenger with its own preprocessing contract. | Compare on judged development queries before adding another runtime adapter. |
| [SigLIP 2](https://huggingface.co/docs/transformers/model_doc/siglip2), [paper](https://arxiv.org/abs/2502.14786) | Modern general vision/language baseline with multilingual capability and a maintained Transformers interface. Domain transfer remains a measurement question. | Preferred general-purpose challenger for a later controlled comparison. |
| Existing PCA / DINOv2 / SSL4EO / TerraMind | Existing image retrieval and regression evidence. No text alignment in the installed representations. | Preserve those stores and their comparison surface. |

The [STAC EO extension](https://github.com/stac-extensions/eo) supplies scene cloud metadata.
The [STAC query specification](https://github.com/stac-api-extensions/query) distinguishes property
filters from similarity; it recommends CQL2 Filter for new catalog integrations. This change
filters local manifests and does not change the existing provider discovery implementation.

## Decision

1. Add `embed-remoteclip`, `search`, and `serve-search` to the existing package. Use FastAPI and
   browser-native forms/JavaScript, matching the current deployment stack. Keep model dependencies
   in an optional `multimodal` extra; the lightweight comparison viewer still needs no neural model.
2. Pin the model repository revision, checkpoint digest, OpenCLIP version, RGB preprocessing, and
   output dimension. Strictly load weights with `weights_only=True`. The image index and query
   encoder must share this identity. Serving uses cached/local weights and never downloads them.
3. Keep normalized exact cosine as the reference. At this corpus size, introducing a remote vector
   database would add deployment and data-management work without an established benefit.
4. For hybrid requests, score all eligible scenes with
   `alpha * cosine(text, scene) + (1-alpha) * cosine(example, scene)`. Display both terms and alpha.
   This is an interpretable starting policy, not calibrated relevance. Never average vectors from
   different models. If we later combine DINOv2 and RemoteCLIP, evaluate rank fusion separately;
   their scores and coordinates are not interchangeable.
5. Apply location, acquisition date, collection, and cloud filters before top-k. Unknown metadata
   fails an active constraint. Dates are UTC calendar dates and inclusive. Bounding boxes match
   chip centers; dateline-crossing boxes are rejected explicitly.
6. Use a small, visible prompt helper, with explicit overrides and a disable switch. It recognizes
   the documented Athens/recent/low-cloud/Sentinel example vocabulary. It is not a general geocoder,
   intent parser, or LLM agent. No external language service is required. If broader interpretation
   is later added, it must emit the same validated schema and expose uncertainty and assumptions.
7. Treat expansion/change language as a warning that the request needs temporal evidence. A
   single-scene embedding can find plausible appearances; it cannot verify that expansion occurred.

## Consequences and evaluation gate

Prepared corpora need a new RemoteCLIP store. Existing representations are not converted. The
new service holds one model in memory, serializes inference, and defaults to localhost. This is
a local implementation, not a public service with a load, authentication, or GPU capacity claim.
Uploads are bounded and decoded before inference; filenames are not filesystem destinations.

Before claiming useful text/hybrid relevance, collect independent descriptions and graded
judgments, including negative and empty-result queries. Freeze a development/final partition
with geographic and temporal controls and audit overlap with model pretraining where possible.
Compare text-only, image-only, and hybrid at fixed weights using nDCG@k and Recall@k, plus constraint
satisfaction and missing-metadata slices. Choose weights on development data only. Existing
EuroSAT class metrics and same-place metrics do not evaluate prompt relevance or urban expansion.

Implementation, unit/API/browser validation, and real-model smoke evidence are recorded in
[the search guide](../multimodal-search.md) and [validation](../validation.md). No new model enters
the frozen BigEarthNet confirmatory roster through this product experiment.
