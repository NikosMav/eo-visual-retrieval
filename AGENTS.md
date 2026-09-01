# Repository guidance

## Mission

Build a public, educational Earth-observation visual-retrieval project that demonstrates
retrieval-engineering practice: reproducible data discovery, embeddings, ranking, evaluation,
and eventually approximate search and a small product surface.

This repository is also a learning environment. Explain unfamiliar EO choices in plain language,
record consequential decisions, and distinguish verified evidence from planned work.

## Read first

Before making a substantial change, read:

1. `README.md` for the supported workflow.
2. `docs/project-context.md` for goals, current state, and priorities.
3. `docs/architecture.md` for system boundaries and component responsibilities.
4. `docs/models-and-metrics.md` for representation and evaluation assumptions.
5. `docs/validation.md` before making performance or quality claims.
6. `docs/learning-stac.md` for the STAC data boundary.

## Non-negotiable boundaries

- Keep the project generic and public. Do not add employer or job-role details.
- Never commit credentials, signed URLs, private areas of interest, proprietary imagery, raw EO
  datasets, or generated embedding artifacts.
- Persist stable STAC identities and allowlisted metadata only. Resolve and sign asset URLs in
  memory at access time.
- Do not present preview imagery as analysis-ready data.
- Treat DINOv2 as an RGB baseline, not as a multispectral EO model.
- Fit learned preprocessing, including PCA, on the index/training partition only.
- Prevent spatial and temporal leakage before publishing benchmark results.
- Keep exact cosine search as the quality reference when approximate search is introduced.

## Engineering expectations

- Prefer small, typed, testable modules over exploratory-only implementation.
- Make runs deterministic where practical and record the data/split/model configuration.
- Add or update tests with behavior changes.
- Run `python -m ruff check .` and `python -m pytest` before committing.
- Update `docs/validation.md` only with evidence produced by an executed validation.
- Do not rename the repository until the labeled EO benchmark gate described in
  `docs/project-context.md` is satisfied.

On Windows, use a short virtual-environment path such as
`C:\Users\<you>\.venvs\eovr` because PyTorch packages can exceed Windows path limits.
