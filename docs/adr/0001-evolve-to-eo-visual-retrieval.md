# ADR-0001: Evolve the existing repository into EO visual retrieval

**Status:** Accepted

**Date:** 2026-08-27

**Decider:** Repository owner

## Context

The repository began as a notebook-only machine-learning assignment using a small, unavailable image collection, PCA, k-NN, and NMF. It is not reproducible from a clean clone. The new objective is to build practical retrieval-engineering experience around public Earth-observation imagery, STAC, DINOv2, evaluation, and eventually approximate nearest-neighbour search.

The work must remain generic and public. It must not contain employer information, proprietary imagery, private areas of interest, credentials, or signed data URLs.

## Decision

Evolve the existing repository while preserving its original state at the `legacy-pca-v1` tag. Keep the notebook under `legacy/`, build a tested Python package on a modernization branch, and rename the repository only after the v0.1 baseline is validated.

Use a provider-neutral, two-stage data boundary:

1. STAC discovery produces a sanitized manifest containing stable item identity and public metadata.
2. Embedding and retrieval operate on local RGB chips described by a separate image manifest.

The initial retrieval comparison is PCA versus frozen DINOv2 using exact cosine search. Faiss and multispectral encoders are later decisions.

## Options considered

### Create a separate repository

| Dimension | Assessment |
|---|---|
| Complexity | Medium |
| Historical continuity | Low |
| Portfolio clarity | Medium |
| Rollback | High |

**Pros:** Clean history and complete isolation from the homework artifact.

**Cons:** Produces another repository and loses the visible progression from classical features to modern retrieval.

### Evolve the existing repository

| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Historical continuity | High |
| Portfolio clarity | High after migration |
| Rollback | High with tag and branch |

**Pros:** Preserves provenance, consolidates related work, and tells a stronger engineering story.

**Cons:** Requires a careful legacy boundary and a staged rename.

## Trade-off analysis

The concepts are continuous: both versions represent images as vectors and compare neighbours. A tagged legacy snapshot and branch-based migration remove most overwrite risk. The existing repository therefore provides more value than a new repository, provided the public/private data boundary is explicit.

## Consequences

- The original notebook remains inspectable but is not presented as the current implementation.
- STAC provider details are isolated from retrieval code.
- Signed asset URLs are resolved at access time and never written to manifests.
- Initial relevance uses public benchmark labels and is only a proxy for user relevance.
- DINOv2 is an RGB baseline; multispectral EO modeling remains a separate experiment.
- Repository renaming is deferred until tests and an evaluated baseline pass.

## Action items

1. [x] Create the `legacy-pca-v1` tag.
2. [x] Move the original notebook under `legacy/`.
3. [x] Validate STAC discovery against a bounded public query.
4. [ ] Run PCA and DINOv2 on a labeled EO retrieval dataset.
5. [ ] Publish an evaluated v0.1 release.
6. [ ] Rename the repository after the release gate.
