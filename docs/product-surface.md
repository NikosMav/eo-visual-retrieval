# Representation explorer

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

Container execution is **not yet validated**: the local Docker engine failed to start during the
2026-09-04 check. The Compose configuration was parsed successfully. This is a launch definition,
not evidence of a built image or running deployment.

Public hosting is not configured. Choose a destination, transfer the prepared subset separately,
and verify the container there before publishing a URL. Hosting costs, TLS, concurrency limits,
and upload traffic policy depend on that destination; no free-tier or production claim is made.

The viewer does not acquire BigEarthNet or access its final partition. Full BigEarthNet S2
acquisition remains paused, and no BigEarthNet score exists. Executed checks live in
[validation](validation.md).
