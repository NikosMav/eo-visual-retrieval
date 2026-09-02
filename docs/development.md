# Development guide

## Repository layout

```text
src/eo_visual_retrieval/
  cli.py                 command-line orchestration
  models.py              serializable record types
  hashing.py             streaming content digests (leaf: imports nothing local)
  vectors.py             L2 normalization used by encoders and search (leaf)
  manifests.py           deterministic local-image manifests
  stac.py                catalog discovery and preview materialization
  chips.py               windowed, georeferenced Sentinel-2 chip materialization
  retrieval.py           exact cosine index
  faiss_benchmark.py     exact-versus-HNSW systems benchmark
  evaluation.py          ranked-retrieval metrics
  tracking.py            opt-in local MLflow aggregate tracking
  visualization.py       per-class best/worst result grids
  datasets/
    eurosat.py           EuroSAT identity, band order, and archive access
  benchmarks/
    eurosat.py           spatially separated benchmark preparation and audit
  embeddings/
    pca.py               classical pixel/PCA baseline
    projection.py        persisted PCA basis for embedding unseen images
    dinov2.py            frozen DINOv2 baseline
    ssl4eo.py            frozen 13-band SSL4EO-S12 encoder
    terramind.py         pinned frozen TerraMind-Tiny experiment
    encode.py            embed one new image with a store's own backend
    store.py             NPZ embedding persistence
scripts/                 executed validation utilities outside the package
tests/                   lightweight unit tests
docs/                    concepts, workflow, decisions, and evidence
```

Import direction is enforced by `tests/test_architecture.py`: `hashing` and `vectors`
import nothing else in the package, `datasets/` holds dataset identity, and `embeddings/`
must not import `benchmarks/`. A representation should not inherit the benchmark that
first happened to use it. The same test asserts that content digests and vector
normalization each have exactly one implementation.

## Environment setup

Python 3.11 is recommended for local ML work; the package supports Python 3.11 and 3.12.

The canonical reproducible setup now uses `uv.lock`; follow
[Evaluation foundations](evaluation-foundations.md) for locked CPU/CUDA environments and local
MLflow tracking. The pip workflow below is retained for editable-install compatibility, not exact
dependency reproduction.

On Windows, create the virtual environment at a short path. Deep PyTorch package paths can exceed
older Windows path limits when the environment lives inside an already long checkout path.

```powershell
py -3.11 -m venv C:\Users\<you>\.venvs\eovr
C:\Users\<you>\.venvs\eovr\Scripts\python -m pip install --upgrade pip
C:\Users\<you>\.venvs\eovr\Scripts\python -m pip install -e ".[dev,stac,geo,ml,search]"
```

An editable install points Python at the current checkout. If the repository is moved or another
copy is opened, rerun the editable-install command and verify the import location:

```powershell
C:\Users\<you>\.venvs\eovr\Scripts\python -c `
  "import eo_visual_retrieval; print(eo_visual_retrieval.__file__)"
```

## Dependency groups

Core dependencies are intentionally light:

- NumPy for matrices and vector operations;
- Pillow for RGB image loading.

Optional groups add:

- `stac`: PySTAC Client, Planetary Computer signing, and HTTP downloads;
- `geo`: Rasterio for windowed, aligned geospatial raster processing;
- `pca`: scikit-learn alone, so the deterministic PCA path is testable without PyTorch;
- `ml`: scikit-learn PCA and PyTorch/torchvision DINOv2 execution;
- `search`: Faiss CPU indexes and psutil process-memory observations;
- `dev`: Ruff, Mypy, Pytest, and pytest-cov.
- `cpu` / `cuda`: mutually exclusive official PyTorch wheel selections when using uv;
- `experiments`: local MLflow and Optuna (tuning still requires independent development data);
- `foundation`: TerraTorch for the TerraMind model experiment.

This lets lightweight CI test the deterministic core without downloading model checkpoints.

## Required checks

Run from the repository root:

```powershell
C:\Users\<you>\.venvs\eovr\Scripts\python -m ruff check .
C:\Users\<you>\.venvs\eovr\Scripts\python -m mypy
C:\Users\<you>\.venvs\eovr\Scripts\python -m pytest
C:\Users\<you>\.venvs\eovr\Scripts\python -m pip check
```

Mypy checks `src`, `tests`, and `scripts` with `disallow_untyped_defs`. Optional heavy
dependencies are exempted from missing-stub errors in `pyproject.toml`: their absence from a
lightweight environment is not a defect in this repository's annotations. Mypy is deliberately
not pinned to one interpreter version, because it must agree with the stubs installed for
whichever Python runs it.

To inspect coverage:

```powershell
C:\Users\<you>\.venvs\eovr\Scripts\python -m pytest `
  --cov=eo_visual_retrieval `
  --cov-report=term-missing
```

GitHub Actions uses the committed lockfile on Linux and Windows, Python 3.11 and 3.12, running
Ruff, Mypy, the test suite, and `uv pip check`. Coverage must stay at or above 75%; the threshold
is a regression guard, not a target. CI installs `dev`, `geo`, `search`, and `pca`, so tests that
require PyTorch skip there and run locally in the full environment.

## Testing strategy

The current tests cover:

- deterministic manifests and JSONL round trips;
- Sentinel-2 reflectance scaling, grid alignment, SCL masking, and pixel bounds;
- rejection of duplicate content with conflicting labels;
- embedding-store persistence, including unlabeled rows and stores written before
  label presence was recorded;
- exact-cosine ranking, self-exclusion, and zero-vector handling;
- Faiss normalization, deterministic scale expansion, ANN overlap, and exact/HNSW contracts;
- skipped queries, partial relevance, and the metric denominators;
- STAC query bounds, media-type handling, output-filename safety, and manifest sanitization;
- EuroSAT band mapping, archive-member validation, and checksum verification;
- PCA fitting on index rows only, and reuse of a saved projection;
- DINOv2 guards, device selection, and preprocessing against a stubbed checkpoint;
- backend dispatch when embedding an image that is not already in a store;
- CLI command integration, written artifacts, and error formatting;
- the package's import direction and single-implementation rules.

High-value missing tests include:

- HTTP retries, byte limits, and partial-file cleanup during preview materialization;
- STAC item resolution and signing, which currently require the network;
- malformed embedding-store archives;
- CLI-level Faiss output and argument validation;
- the multispectral encoders' torch execution paths, which need real checkpoints;
- future geospatial windows, transforms, scaling, nodata, and cloud masks.

Network and large-model tests should not make the default unit suite slow or unreliable. Use small
local fixtures and dependency injection or mocks for deterministic behavior, then record bounded
real-service smoke runs separately in `docs/validation.md`.

## Adding behavior

1. Read the project boundaries in `docs/project-context.md`.
2. Define the input/output contract before adding a new CLI command.
3. Keep provider access separate from embedding and evaluation logic.
4. Add typed, testable functions below the CLI layer.
5. Add tests for success, validation errors, and cleanup behavior.
6. Run Ruff, Mypy, Pytest, and `pip check`.
7. Execute a bounded smoke validation when external services or models are involved.
8. Update `docs/validation.md` only with evidence produced by that execution.
9. Update conceptual documentation when assumptions or data contracts change.

## Generated and sensitive files

The following stay local and are ignored by Git:

- `data/`: downloaded or prepared imagery and manifests used for local experiments;
- `artifacts/`: generated vectors and model/index outputs;
- `outputs/`: reports, figures, or temporary run results;
- `.env*` and `configs/local/`: secrets and private local configuration;
- common raster and NumPy artifact extensions.

Before committing, check staged files explicitly:

```powershell
git status --short
git diff --cached --stat
git diff --cached
```

Never commit signed URLs, tokens, proprietary imagery, private areas of interest, or generated
embedding stores.

## Hardware notes

`--device auto` uses CUDA only when `torch.cuda.is_available()` returns true. An NVIDIA GPU can be
present while the installed PyTorch wheel is CPU-only. Use the official PyTorch installation
selector when GPU execution is required, and record the actual device in validation evidence.

DINOv2 model code and weights may be downloaded on first use. Treat successful download and model
execution as a smoke gate, not a retrieval-quality result.
