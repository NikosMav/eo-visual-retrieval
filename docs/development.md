# Development guide

## Repository layout

```text
src/eo_visual_retrieval/
  cli.py                 command-line orchestration
  models.py              serializable record types
  manifests.py           deterministic local-image manifests
  stac.py                catalog discovery and preview materialization
  retrieval.py           exact cosine index
  faiss_benchmark.py     exact-versus-HNSW systems benchmark
  evaluation.py          ranked-retrieval metrics
  embeddings/
    pca.py               classical pixel/PCA baseline
    dinov2.py            frozen DINOv2 baseline
    store.py             NPZ embedding persistence
tests/                   lightweight unit tests
docs/                    concepts, workflow, decisions, and evidence
```

## Environment setup

Python 3.11 is recommended for local ML work; the package supports Python 3.11 and 3.12.

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
- `ml`: scikit-learn PCA and PyTorch/torchvision DINOv2 execution;
- `search`: Faiss CPU indexes and psutil process-memory observations;
- `dev`: Ruff, Pytest, and pytest-cov.

This lets lightweight CI test the deterministic core without downloading model checkpoints.

## Required checks

Run from the repository root:

```powershell
C:\Users\<you>\.venvs\eovr\Scripts\python -m ruff check .
C:\Users\<you>\.venvs\eovr\Scripts\python -m pytest
C:\Users\<you>\.venvs\eovr\Scripts\python -m pip check
```

To inspect coverage:

```powershell
C:\Users\<you>\.venvs\eovr\Scripts\python -m pytest `
  --cov=eo_visual_retrieval `
  --cov-report=term-missing
```

GitHub Actions repeats Ruff and coverage-enabled tests on Python 3.11 and 3.12. Coverage is
reported but is not currently enforced by a percentage threshold.

## Testing strategy

The current tests cover:

- deterministic manifests and JSONL round trips;
- Sentinel-2 reflectance scaling, grid alignment, SCL masking, and pixel bounds;
- rejection of duplicate content with conflicting labels;
- embedding-store persistence;
- exact-cosine ranking, self-exclusion, and zero-vector handling;
- Faiss normalization, deterministic scale expansion, ANN overlap, and exact/HNSW contracts;
- perfect synthetic label retrieval;
- STAC query bounds and manifest sanitization.

High-value missing tests include:

- CLI command integration and error formatting;
- PCA fitting only on index data and output normalization;
- DINOv2 preprocessing with a stubbed model;
- HTTP retries, byte limits, partial-file cleanup, and media-type fallback;
- malformed embedding-store archives;
- skipped-query and non-perfect metric examples;
- CLI-level Faiss output and argument validation;
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
6. Run Ruff, Pytest, and `pip check`.
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
