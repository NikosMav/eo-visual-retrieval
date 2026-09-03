# Product surface: design

- Date: 2026-09-03
- Status: approved, not yet implemented
- Deliverable: a served comparison interface over the existing EuroSAT v1 evidence

## Why this work exists

Milestone 5 asks for "a small API or interactive demo" where a user can select or upload a query
chip, inspect ranked results and metadata, and see which model and index generated the ranking.

The retrieval capability now exists. `eovr query --image` embeds an unseen image with the backend
recorded in a store, `embed-pca --projection-output` persists the one basis this project fits
itself, and five embedding stores sit on disk. What is missing is the interface.

This work is deliberately **independent of BigEarthNet acquisition**. It runs on EuroSAT v1, which
is already local, so it is not blocked by a 14-hour transfer.

## What makes this surface worth building

A generic single-model retrieval demo would waste what this repository uniquely holds. The
distinctive claim here is that **representation choice determines what "similar" means**, and the
band ablation now quantifies it: only 34% of SSL4EO's advantage over DINOv2 is spectral.

So the central view is a comparison. One query, several representations ranked side by side.
Placing SSL4EO 13-band directly above SSL4EO RGB makes the ablation visible rather than tabular:
the `River` query that RGB answers with buildings and 13-band answers correctly.

## Decision

### Shape

A new optional extra `app` — FastAPI, Uvicorn, Jinja2, and `python-multipart` — living in
`src/eo_visual_retrieval/app/` and launched by `eovr serve`. Optional, so CI, the benchmark path,
and the core install stay untouched, matching how `stac`, `geo`, `ml`, and `search` already work.

The command takes the inputs explicitly, in the style of the existing CLI, rather than discovering
them:

```powershell
eovr serve `
  --manifest data/eurosat-v1/manifest.jsonl `
  --image-root data/eurosat-v1/images `
  --store artifacts/eurosat-v1-pca-64.npz `
  --store artifacts/eurosat-v1-dinov2-vits14.npz `
  --store artifacts/eurosat-v1-ssl4eo-s12-rgb-moco-resnet50.npz `
  --store artifacts/eurosat-v1-ssl4eo-s12-moco-resnet50.npz `
  --projection artifacts/eurosat-v1-pca-64-projection.npz `
  --host 127.0.0.1 --port 8000
```

`--store` is repeatable and the page compares every store given, in the order given. `--projection`
is optional; without it the upload path is disabled and the page says so rather than failing.

**No PyTorch in the served process.** Ranking is a matrix-vector product over precomputed vectors,
and uploads go through the persisted PCA basis, which needs only NumPy and Pillow. This keeps the
deployable image small, the cold start immediate, and the untrusted-input attack surface narrow.

The measured payload supports this:

| Item | Size |
|---|---:|
| All 2,000 index images as JPEG thumbnails | ~3 MB |
| All five embedding stores | ~29 MB |
| PCA projection for the upload path | ~3 MB |

### Components

Four units. Only one knows about HTTP.

| Unit | Responsibility | Depends on |
|---|---|---|
| `app/catalog.py` | Load stores and manifest, build one `ExactCosineIndex` per model, answer queries, expose provenance | `embeddings.store`, `retrieval`, `manifests` |
| `app/thumbnails.py` | Decode a GeoTIFF and return bounded JPEG bytes, LRU-cached | Pillow |
| `app/uploads.py` | Validate and decode untrusted bytes, or refuse | Pillow |
| `app/main.py` + templates | Routing and rendering | FastAPI, Jinja2, the three above |

`catalog.py` imports nothing web-related, so the ranking and provenance logic is testable without
starting a server. `thumbnails.py` is not decoration: browsers cannot render GeoTIFF, so converting
is a functional requirement.

### The load-bearing invariant

The catalog **refuses to start** unless every loaded store shares the same `manifest_sha256` and an
identical ID ordering.

A side-by-side comparison of models that ranked different corpora would be meaningless, and it
would look completely correct on screen. This is the one defect in this design that could silently
invalidate everything the page claims, so it is a startup failure rather than a warning.

### What the page shows

Rows are representations, columns are ranks. The colour language deliberately matches the committed
result grids — blue query, green relevant, red not relevant — so the served view and the published
evidence read the same way.

Each ranking carries its provenance: model name, checkpoint SHA-256, manifest SHA-256, index size,
and ranker. That is milestone 5's "see which model and index generated the ranking", and the stores
already record all of it.

An honest asymmetry, stated on the page rather than left for the visitor to infer:

| Query source | Relevance colouring | Per-query metrics |
|---|---|---|
| Corpus item | Yes — the label is known | Yes |
| Upload | No — no label exists | No |

The absence of colour on an upload means "unknown", not "wrong".

### Untrusted input

The surface is intended to be publicly reachable eventually, so uploads are treated as hostile:

- Bytes are capped **before** decode, and never written to disk.
- `Image.MAX_IMAGE_PIXELS` bounds decompression bombs.
- Type is established by decoding, not by trusting a declared content type.
- No user-supplied string reaches a filesystem path.
- The whole request path is read-only.

### Failure behaviour

| Condition | Response |
|---|---|
| An embedding store is missing | Startup fails naming the exact command that generates it |
| Stores disagree on manifest or ID order | Startup fails; see the invariant above |
| Upload too large, undecodable, or a pixel bomb | 400 with a plain message |
| Unknown item ID | 404 |

### Deployment

A host-agnostic container definition and no host-specific code. It runs locally with one command.

Publishing stays a separate, reversible step taken whenever an account exists somewhere. Because
the served process needs no GPU, no PyTorch, and about 35 MB of data, it fits a free CPU tier, so
this does not reverse ADR 0005's deferral of paid infrastructure so much as avoid needing it. A
short ADR records the decision to expose the surface publicly and why it costs nothing.

EuroSAT is MIT-licensed and openly accessible, confirmed from Zenodo record 7711810, so serving a
1,600-image index subset with attribution is permitted.

## Out of scope

- PyTorch-backed uploads. DINOv2, SSL4EO, and TerraMind uploads would require a 2-3 GB image and
  put model loading behind a public endpoint.
- BigEarthNet. Not acquired, and this work is deliberately independent of it.
- Authentication, analytics, and rate limiting beyond the byte caps.
- Any change to the evaluator, the ranker, or a published result.

## Acceptance criteria

1. `eovr serve` starts against local EuroSAT v1 artifacts and renders a comparison for a corpus
   query across every store passed with `--store`, in the given order.
2. An uploaded RGB image is embedded through the persisted PCA basis and ranked, with no colouring
   and an explicit statement of why.
3. Mismatched stores cause a startup failure, covered by a test.
4. Upload limits are enforced and tested: oversize, undecodable, and pixel-bomb inputs.
5. `catalog.py` is tested without starting a server.
6. Ruff, Mypy, and the suite pass; coverage stays at or above 75%.
7. No published result changes, and `evaluation.py` and `retrieval.py` are untouched.

## Risks

| Item | Status |
|---|---|
| Upload works only for PCA | Accepted, and stated on the page itself. A reviewer will notice; better that the page says it first. |
| Thumbnail decoding cost | 2,000 GeoTIFFs at ~1.3 KB JPEG each is cheap, but must be LRU-cached or every page view re-decodes. |
| New web dependencies in a deliberately lean project | Confined to an optional extra; CI's install set is unchanged. |
| The demo needs prepared local data | Inherent. Startup failures name the commands that produce it. |
| A future public deployment inherits upload risk | The threat model above is designed for it now rather than retrofitted. |
