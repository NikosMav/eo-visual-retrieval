# ADR 0004: Faiss exact reference and HNSW scaling experiment

- Status: accepted
- Date: 2026-09-02
- Decision owners: project maintainers

## Context

The evaluated retrieval pipeline contains 1,600 searchable vectors and uses exact cosine
similarity. Milestone 4 asks whether an approximate index is justified as the corpus grows. The
experiment must preserve the existing representation and quality evidence, keep exact search as
the reference, and distinguish search-system recall from label-based semantic recall.

Approximate search changes how existing vectors are ranked. It does not train or fine-tune PCA,
DINOv2, or SSL4EO-S12.

## Decision

Use two Faiss indexes on L2-normalized float32 vectors:

- `IndexFlatIP` is the exact reference. Inner product on unit vectors produces the same ranking
  as cosine similarity.
- `IndexHNSWFlat` is the first approximate candidate. It stores full vectors and builds a
  navigable proximity graph as vectors are added.

The first fixed configuration uses `M=32`, `efConstruction=200`, and an `efSearch` sweep of
`16, 32, 64, 128` at `k=10`. Runs use one CPU thread, two warmups, seven timed query batches, and
the same 400 real EuroSAT queries. The benchmark records neighbor recall against `IndexFlatIP`,
median and p95 batch-derived latency per query, build time, serialized bytes, and approximate
process RSS changes.

The actual 1,600-vector stores are measured for all representations. DINOv2 is additionally
expanded deterministically to 10,000 and 50,000 vectors with seeded perturbed copies. Those rows
test index scaling only; they are not new EO observations and cannot support retrieval-quality
claims.

Exact search remains the operational default at the current 1,600-vector scale. HNSW is available
as an evaluated experiment, not an automatic replacement.

## Parameters in plain language

| Parameter | What it controls | Trade-off |
|---|---|---|
| `M` | Approximate number of graph connections per item | More connections usually improve navigation but increase memory and build work |
| `efConstruction` | Search effort while constructing the graph | Higher values usually improve graph quality but make building slower |
| `efSearch` | Search effort for each query | Higher values generally recover more exact neighbors but increase query time |

HNSW has index construction but no learned training stage. This differs from Faiss IVF or product
quantization indexes, which require a representative training sample before vectors are added.

## Options considered

| Option | Fit for this phase | Main trade-off |
|---|---|---|
| Keep exact search only | Correct and already sufficient for 1,600 items | Does not teach or measure scale-oriented indexing |
| Faiss HNSW Flat | No separate training set; direct recall/latency controls | Extra graph memory and potentially expensive construction |
| Faiss IVF Flat | Useful for larger corpora with tunable probes | Requires index training and adds sampling choices |
| Faiss product quantization | Can reduce vector memory substantially | Adds compression error and a larger tuning/validation surface |
| Hosted vector service | Product-like operations and filtering | Hides core mechanics and introduces cost/infrastructure too early |

## Consequences

Positive consequences:

- exact and approximate rankings share one metric and normalization contract;
- the quality loss from approximation is measured directly per workload;
- the CLI emits complete machine-readable provenance;
- unit tests cover normalization, deterministic expansion, overlap recall, and both index types;
- the current evidence can reject ANN when it does not pay for its complexity.

Limitations and risks:

- timing is hardware-, OS-, Faiss-, thread-, and batch-shape-specific;
- the 10k and 50k corpora contain synthetic expansions and do not mimic a natural embedding
  distribution perfectly;
- RSS is a process-level observation affected by native allocators and object lifetime;
- HNSW results can vary with configuration and insertion order;
- filtering, updates, deletion, concurrency, persistence reload time, and million-scale workloads
  are not evaluated.

## Follow-up actions

- Keep `IndexFlatIP` as the current default and quality oracle.
- Re-run the benchmark on deployment-like hardware before choosing a production index.
- Add a real larger corpus before making scale claims about EO data.
- Consider IVF or compression only when corpus size or memory creates a demonstrated need.
- Carry the exact-versus-approximate choice into the Milestone 5 product surface explicitly.

## References

- [Faiss metric and cosine guidance](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
- [Faiss index summary](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [Faiss index-selection guidance](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)

