# Project checkpoint review — 2026-09-04

The current Python architecture suits the local EuroSAT explorer. The useful changes at this
checkpoint are stronger data validation, browser coverage of the packaged app, and clearer
provenance and evidence boundaries. A frontend rewrite or vector database would not resolve
the correctness issues found in this review.

Review scope covered the acquisition and dataset modules, geographic preparation/audits,
representations and persisted stores, exact and approximate retrieval, evaluation and tracking,
CLI, serving, tests, scripts, dependency profiles, CI, packaging, and documentation. Source
inspection and synthetic regressions complement the bounded real-corpus startup check; this is
not a claim of an exhaustive security audit. Executed checks are in [validation](validation.md).

## Current technology stack

| Layer | Implemented tools and role |
|---|---|
| Language and packaging | Python 3.11/3.12, Hatchling wheel, uv lockfile and optional environments |
| Numerical processing | NumPy; scikit-learn fits PCA on index pixels only |
| Neural inference | PyTorch and torchvision; TerraTorch constructs TerraMind; no TensorFlow pipeline |
| Model access | DINOv2 uses Torch Hub; SSL4EO/TerraMind accept local checksum-verified weights. Hugging Face hosts model files used by documented acquisition steps; Transformers is not the project's inference API |
| EO data | PySTAC Client and Planetary Computer signing; Rasterio for bands/georeferencing; PyArrow and Zstandard for BigEarthNet preparation |
| Retrieval | Exact NumPy cosine search; Faiss Flat/HNSW in the separate systems benchmark |
| Persistence | Local images, sanitized JSONL manifests, compressed NPZ vectors and projections; no deployed vector database |
| Product | FastAPI/Starlette, Uvicorn, Jinja2, plain CSS and JavaScript; server loads existing vectors and performs PCA uploads |
| Quality and tracking | Ruff, Mypy, Pytest/coverage, Playwright browser tests, on-demand deptry, optional aggregate-only local MLflow |
| Delivery | GitHub Actions and a Docker/Compose definition; container execution and public hosting remain unvalidated |

The thumbnail shows RGB. It does not show every input band used by a multispectral model.
Coordinates, timestamps, labels, hashes, and model provenance constrain preparation or explain
results; they are not appended to the embedding or used as extra cosine-ranking features.
There is currently no geographic radius control in the explorer.

## Findings remediated

| Finding | Change and verification boundary |
|---|---|
| Unrelated PCA fits could pass startup based on shape | Catalog startup projects every corpus row in batches and checks agreement with stored vectors. This establishes numerical compatibility on that corpus, not cryptographic identity of the original fitted basis |
| Invalid vectors or mismatched labels/splits could be served | Catalog rejects non-finite, zero/overflow-norm vectors and verifies rows against the manifest |
| Image provenance could outlive changed pixels | Catalog and RGB embedding commands verify declared image hashes; the catalog also checks files and path containment. Legacy manifests without hashes are still accepted without claiming verified pixel identity |
| Non-finite geographic values could make separation checks pass with NaN | Coordinate and policy validation now rejects non-finite and out-of-range inputs before distance/audit calculations |
| API credentials could enter persisted provenance | STAC API identity rejects userinfo, query strings, fragments, and missing hostnames before persistence or chip writes |
| Sanitized preview IDs could overwrite one another | Names include a source-identity digest; previews record their downloaded content hashes; duplicate item IDs across sources are rejected |
| Browser interactions and packaged assets lacked a repeatable gate | Separate installed-wheel Chromium job with synthetic data covers filtering, navigation, upload, HTML errors, keyboard controls, and narrow layout |
| Dependency and documentation drift | Removed unused Optuna installation, declared direct Starlette usage, consolidated content hashing, added deptry configuration, and refreshed installation/testing instructions |
| Comparison wording overstated causal evidence | Distinct RGB and 13-band pretrained checkpoints are described as a pipeline comparison. Published measurements and decision records are unchanged |

The old merged streaming checkout contained only reproducible test caches beyond committed files.
Raw imagery, pilot files, and generated embeddings are not housekeeping targets. Historical plans,
ADRs, and result records remain useful provenance and are retained.

## Third-party options researched

These are documentation-based assessments checked on 2026-09-04, not locally benchmarked vendor
comparisons. Adoption follows a concrete requirement and validation, rather than library popularity.

| Option | Recommendation and reason |
|---|---|
| FastAPI/Jinja, Gradio, Streamlit | Keep the existing [FastAPI templates](https://fastapi.tiangolo.com/advanced/templates/) surface. [Gradio can mount in FastAPI](https://gradio.app/docs/gradio/mount_gradio_app) if a separate model experiment needs it. [Streamlit's rerun model](https://docs.streamlit.io/develop/api-reference/execution-flow) offers an alternative for exploratory dashboards, with no demonstrated benefit from rewriting this app |
| Qdrant | First database prototype when shared persistence or metadata filtering becomes a requirement. [Local mode](https://github.com/qdrant/qdrant-client#local-mode) can run without Docker; its [geo-radius filter](https://qdrant.tech/documentation/search/filtering/#geo-radius) uses metres. A radius predicate over 1,600 local candidates can also precede exact ranking without adding a database |
| PostGIS + pgvector | Consider when relational data and spatial joins dominate. [ST_DWithin geography](https://postgis.net/docs/manual-dev/en/ST_DWithin.html) uses metres. Check [pgvector index limits](https://github.com/pgvector/pgvector#hnsw): ordinary vector ANN indexes support up to 2,000 dimensions, below SSL4EO's 2,048. Half precision changes the numerical contract and needs recall validation; filtered ANN also needs enough candidates |
| FiftyOne | Optional qualitative inspection of existing embeddings and outliers via its [embedding visualizations](https://docs.voxel51.com/tutorials/image_embeddings.html). It introduces [database/runtime dependencies](https://docs.voxel51.com/installation/index.html). A two-dimensional plot is not retrieval-quality evidence |
| TorchGeo / TerraTorch / Hugging Face Hub | Consult [TorchGeo's BigEarthNetV2 adapter](https://docs.torchgeo.org/en/stable/api/datasets/bigearthnet.html) and [TerraMind modality guidance](https://github.com/torchgeo/terratorch/blob/main/docs/guide/terramind.md) when implementing frozen model inputs. [Hub named-file downloads and revisions](https://huggingface.co/docs/huggingface_hub/guides/download) can improve acquisition provenance. Adapters must preserve this project's IDs, band ordering, scaling, partitions, and checkpoint checksums |
| Playwright and deptry | Adopted for focused browser regression and dependency inspection. [Playwright assertions](https://playwright.dev/python/docs/test-assertions) wait for browser state; [deptry rules](https://deptry.com/rules-violations/) flag undeclared, transitive, or unused dependencies. Neither replaces application tests or a security advisory review |
| MLflow and DVC | Keep the existing opt-in [local MLflow backend](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/). Revisit [DVC artifact access](https://github.com/treeverse/dvc.org/blob/main/content/docs/user-guide/data-management/discovering-and-accessing-data.md) when distributing approved artifacts becomes a repeated task; it does not solve the initial archive transfer |

For a database or ANN experiment, first specify the actual corpus, filter predicates, concurrency,
memory budget, latency objective, and required agreement with exact search. The existing synthetic
50k-vector experiment does not establish performance on 50k independent EO images.

### BigEarthNet transfer alternatives

The [official distribution](https://bigearth.net/) still points to large archive files. The inspected
[TorchGeo Hugging Face V2 mirror](https://huggingface.co/datasets/torchgeo/bigearthnet/tree/main/V2)
contains large split archive objects, not a verified per-patch replacement. No alternative provider
was established as suitable for the frozen 5,000 IDs and 2 GiB local-storage ceiling.

[fsspec range access and caching](https://filesystem-spec.readthedocs.io/en/latest/features.html)
does not restore a lost decompressor state; whole-file caches can also breach the storage ceiling.
The [Zstandard format](https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md)
allows dependencies on previously decoded blocks. Its separate
[seekable format](https://github.com/facebook/zstd/blob/dev/contrib/seekable_format/README.md)
requires appropriately framed data and a seek table; it cannot be assumed for the published file.

Full acquisition remains paused. An alternate source must prove selected identity, all 12 L2A
bands, native geometry agreement, integrity, bounded storage, and restart behavior before adoption.
An HTTP Range header alone is not a process-restart solution for a sequential compressed archive.

## Remaining work, in priority order

1. **CLI correctness beyond serving.** `embeddings/encode.py` still accepts a same-shaped unrelated
   PCA projection for a new CLI image query. `vectors.py` can overflow a float32 norm for extreme
   finite inputs, and the protected exact ranker does not reject every NaN/Inf input. The served
   catalog now guards these inputs, but that protection must not be claimed for every offline
   caller. Fix the shared contracts in a separate change with regression checks against existing
   EuroSAT results; do not silently change the published ranker.
2. **Model code provenance.** DINOv2 still loads the Torch Hub repository without an immutable code
   ref. Recover the code/checkpoint identity used for existing runs, then pin and test the next
   inference environment. Locking Python packages does not pin downloaded executable model code.
3. **Optional Lightning advisory.** Dependabot alert #4 remains open for
   [GHSA-qqmf-gpg7-g8gw](https://github.com/advisories/GHSA-qqmf-gpg7-g8gw). The advisory describes
   `_instantiator` execution through `load_from_checkpoint`; the
   [upstream fix](https://github.com/Lightning-AI/pytorch-lightning/pull/21832) exists, but the
   advisory's date-based affected/patched versions conflict with its description of 2.6.5.
   [Lightning on PyPI](https://pypi.org/project/lightning/) and
   [pytorch-lightning on PyPI](https://pypi.org/project/pytorch-lightning/) both reported 2.6.5 as
   latest during this review. No compatible patched release was verified. The repository's
   adapters use checksum-verified `torch.load(weights_only=True)` rather than this Lightning
   loader, and the serving profile excludes Lightning. Keep the alert open until a tested fix or
   verified non-reachability decision resolves it; do not install an unrelated date-numbered
   package to satisfy faulty version metadata.
4. **Geographic audit independence.** EuroSAT audits validate declared geometry and policy rather
   than recomputing geometry from raster pixels. BigEarthNet's future acquisition must retain its
   inline per-band geometry check. Neither metadata source should silently be adjusted to make a
   mismatch pass.
5. **Delivery evidence.** Validate the container on a working engine, then decide hosting and
   approved data distribution. [Hugging Face Docker Spaces](https://huggingface.co/docs/hub/main/spaces-sdks-docker)
   is one possible evaluation target, not a selected or validated deployment. Load, storage,
   upload policy, and costs need measurements in the chosen environment.
6. **Confirmatory data and model inputs.** Acquisition authorization and a viable transfer strategy
   precede adapters and development comparisons. Preserve final queries until the separately
   authorized frozen-configuration gate. No BigEarthNet score exists.

The evaluator, exact ranker, frozen selections, `docs/decisions/`, and `docs/results/` remain
unchanged in this checkpoint. No neural embeddings or retrieval metrics were generated.
