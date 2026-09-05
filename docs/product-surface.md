# Product surface

The local product brings four areas together under `eovr serve-search`:

| Page | Purpose | Current evidence boundary |
|---|---|---|
| Search (`/`) | Text, uploaded/selected image, and hybrid retrieval; visible constraints and explanations | Frozen RemoteCLIP; text/hybrid relevance not yet benchmarked |
| Compare models (`/models/`) | Inspect the same EuroSAT query across supplied representations | Saved vectors; same-class proxy; PCA-only upload in this viewer |
| Findings (`/findings`) | Metrics, definitions, sample counts, per-place slices, and downloadable evidence | Development results, not confirmatory generalization |
| Data & experiments (`/research`) | Search corpus coverage, model identity, and staged research plan | Acquisition and training stages are plans, not running jobs |

The detailed existing charts are available at `/findings/analysis`. Navigation connects all four
areas while retaining the lightweight standalone `eovr serve` comparison viewer below.

## Launch the combined product

Prepare the optional [RemoteCLIP runtime and store](multimodal-search.md) and the EuroSAT stores.
From the repository root, using that Python environment:

```powershell
python -m eo_visual_retrieval.cli serve-search `
  --manifest data/temporal-v1/manifest-guarded.jsonl `
  --image-root data/temporal-v1/images `
  --embeddings artifacts/temporal-v1g-remoteclip-vit-b32.npz `
  --results-dir docs/results `
  --comparison-manifest data/eurosat-v1/manifest.jsonl `
  --comparison-image-root data/eurosat-v1/images `
  --comparison-store artifacts/eurosat-v1-pca-64.npz `
  --comparison-store artifacts/eurosat-v1-dinov2-vits14.npz `
  --comparison-store artifacts/eurosat-v1-ssl4eo-s12-rgb-moco-resnet50.npz `
  --comparison-store artifacts/eurosat-v1-ssl4eo-s12-moco-resnet50.npz `
  --comparison-store artifacts/eurosat-v1-terramind-tiny.npz `
  --comparison-projection artifacts/eurosat-v1-pca-64-projection.npz
```

Open `http://127.0.0.1:8002/`. Comparison stores may be omitted; the page then explains that they
are not configured. Missing evidence is shown as unavailable, never replaced by example scores.
An explicit results directory must exist. Public report names are allowlisted; downloads use the
same startup byte snapshot and SHA-256 identity as the displayed metrics. No arbitrary directory
is exposed through HTTP. Changes to reports require a restart.

## Transparency contract

Each search shows the original prompt and interpreted filters; independent pass/fail/missing
counts for every constraint; total exclusions and candidate counts; and engine wall time.
Independent filter counts overlap and are not a sequential funnel. Metadata checks drive both
eligibility and explanations through one function.

Each result shows the text and image cosines, weighted contributions, combined rank, component
ranks over the same eligible pool, date/cloud/collection/center metadata, and filter outcomes.
Stable corpus order breaks ties. These explanations describe arithmetic and metadata; no generated
object explanation, confidence estimate, or causal interpretation is implied. Model revision,
checkpoint/preprocessing identity, manifest digest, query inputs and results can be exported in a
JSON search record. Uploaded pixels are not included in the export or added to the corpus.

Timing includes inference queueing but excludes HTTP upload/decoding, model startup and rendering.
The 50% weight is an untuned numerical blend; it does not guarantee equal ranking influence because
text and image score distributions can differ. A single scene cannot verify urban expansion.

## Research direction

See [the multimodal temporal finding](results/multimodal-temporal.md) and
[the research plan](research-roadmap.md). Priority is judged semantic relevance, independent data,
and frozen-model/ranking comparisons before fine-tuning. Retrieval quality must determine the
model choice separately for each task.

## Standalone representation explorer

The explorer makes the EuroSAT comparison interactive: choose a scene class and query, then
inspect the nearest images returned by every supplied representation. Every result names its
scene label and cosine score. Expand image identity or model provenance for the underlying IDs,
input bands, manifest hash, checkpoint hash when recorded, and exact ranker.

Same-class labels are a broad relevance proxy. Cosine scores are neither probabilities nor a
shared scale across representations. Rows follow the operator's store order, not measured quality.

## Start locally

Use Python 3.11 or 3.12. With the existing EuroSAT artifacts:

```powershell
python -m pip install -e ".[app]"
eovr serve `
  --manifest data/eurosat-v1/manifest.jsonl `
  --image-root data/eurosat-v1/images `
  --store artifacts/eurosat-v1-pca-64.npz `
  --store artifacts/eurosat-v1-dinov2-vits14.npz `
  --store artifacts/eurosat-v1-ssl4eo-s12-rgb-moco-resnet50.npz `
  --store artifacts/eurosat-v1-ssl4eo-s12-moco-resnet50.npz `
  --store artifacts/eurosat-v1-terramind-tiny.npz `
  --projection artifacts/eurosat-v1-pca-64-projection.npz
```

Open `http://127.0.0.1:8000`. `--port` changes the port; `--k` sets the number of results.
Supply only the stores you have; at least one is required. No model framework is imported by the
server, and no inference checkpoint or source archive needs to be mounted.

A fresh clone does not contain imagery or embeddings. Follow the [EuroSAT preparation guide](
benchmark-eurosat.md) and [embedding commands](pipeline-and-cli.md) to prepare them separately.
PCA upload support needs the saved projection generated alongside its store by
`embed-pca --projection-output`; do not mix projections from different fits. Omit `--projection`
for a comparison-only viewer. Startup rejects mismatched manifest hashes, ordered IDs, labels,
or partitions across stores and against the manifest. It also rejects unsafe image paths,
missing files, declared image checksum mismatches, and non-finite or degenerate vector rows.
When a projection is supplied, startup checks it against every stored PCA row in batches of 32
with `atol=1e-5`, `rtol=1e-4`. A projection from an unrelated fit is rejected even when its shape
matches. These read-only checks add startup work; they do not refit PCA or rewrite the stores.

Older generic manifests may omit image checksums; no pixel-identity guarantee is inferred for
those files. Keep the corpus read-only while serving because startup verification does not monitor
later filesystem edits.

## Query and upload behavior

- Class filtering narrows the 400-query picker. Without JavaScript, the grouped native picker
  still works. Query URLs can be bookmarked.
- Corpus queries use their precomputed vectors across all supplied representations. Images show
  both textual relevance and colored borders, so color is never the only indicator.
- Uploads use the saved PCA projection only. They carry no label, so no relevance coloring or
  per-query metric is produced. RGB satellite chips are the intended input.
- The complete upload request, including multipart overhead, is capped at 8 MiB before form
  parsing. Actual bytes are counted even for chunked requests or misleading length headers.
  Decoding also enforces a 16,777,216-pixel cap. Temporary multipart files may exist during a
  bounded request; they are closed afterward. Uploaded images are not added to the corpus.
- Bad images and unknown queries show an error with a route back to the explorer. Oversized
  requests return HTTP 413 before parsing. `/healthz` reports readiness after catalog startup.

## Findings page

`--results-dir docs/results` adds a `/findings` route alongside the explorer, linked from its
header. The page presents what the published analysis found — agreement between representations,
consistency across places, failure structure, and clear-sky availability — with every figure read
from the committed reports at startup rather than written into the template. A deployment that
supplies no directory serves no findings page; a directory missing a required section is refused
at startup rather than rendering a chart with holes in it.

## Container launch

The Dockerfile installs only the locked core and `app` dependencies and runs as an unprivileged
user. Its allowlisted build context excludes all data, generated stores, and local configuration.
Compose mounts the prepared corpus and stores read-only, binds localhost port 8000, and supplies
a bounded temporary filesystem. Stop a local server on that port first.

```powershell
docker compose up --build
# Stop the container when finished:
docker compose down
```

Container execution is **validated locally**. On 2026-09-04 the image built from this definition
and served the prepared EuroSAT corpus on Docker Engine 29.7.2 under Windows 11: all five stores
loaded, the comparison and thumbnail routes answered, PCA upload ranked a new image, and the
upload limits and error paths behaved as documented. The container ran as uid 10001 with a
read-only root filesystem, all capabilities dropped, and both mounts read-only; writes to `/app`
and `/stores` failed while the bounded `/tmp` stayed writable. Startup took about 71 seconds,
most of it verifying 2,000 image hashes and re-projecting the PCA vectors across a bind mount.
The executed checks are in [validation](validation.md).

That is evidence of a working local container, not of a deployment. Load, hosting, and TLS remain
unmeasured.

Public hosting is not configured. Choose a destination, transfer the prepared subset separately,
and verify the container there before publishing a URL. Hosting costs, TLS, concurrency limits,
and upload traffic policy depend on that destination; no free-tier or production claim is made.

The viewer does not acquire BigEarthNet or access its final partition. Full BigEarthNet S2
acquisition remains paused, and no BigEarthNet score exists. Executed checks live in
[validation](validation.md).
