"""Reproducible exact-versus-HNSW search benchmarks for embedding stores."""

from __future__ import annotations

import gc
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.hashing import file_sha256
from eo_visual_retrieval.vectors import l2_normalize

__all__ = [
    "BuildStats",
    "LatencyStats",
    "ann_recall_at_k",
    "benchmark_faiss",
    "expand_corpus",
    "file_sha256",
    "l2_normalize",
]


@dataclass(frozen=True)
class LatencyStats:
    """Repeated whole-query-batch latency, expressed per query."""

    median_ms_per_query: float
    p95_ms_per_query: float
    repetitions: int
    warmups: int


@dataclass(frozen=True)
class BuildStats:
    """Index construction measurements."""

    seconds: float
    serialized_bytes: int
    rss_before_bytes: int
    rss_after_bytes: int
    rss_delta_bytes: int


def expand_corpus(
    vectors: NDArray[np.float32],
    *,
    target_size: int,
    seed: int,
    noise_std: float,
) -> NDArray[np.float32]:
    """Deterministically expand a corpus for search-scaling experiments.

    Existing vectors are retained. Additional rows cycle through the source vectors,
    receive seeded Gaussian perturbations, and are normalized. This is a systems-only
    workload: it must not be interpreted as additional EO observations.
    """
    base = l2_normalize(vectors)
    if target_size < len(base):
        raise ValueError("target_size must be at least the source corpus size")
    if noise_std <= 0:
        raise ValueError("noise_std must be positive")
    if target_size == len(base):
        return base

    extra_count = target_size - len(base)
    source_positions = np.arange(extra_count) % len(base)
    generator = np.random.default_rng(seed)
    noise = generator.normal(0.0, noise_std, size=(extra_count, base.shape[1])).astype(
        np.float32
    )
    extras = l2_normalize(base[source_positions] + noise)
    return np.ascontiguousarray(np.vstack((base, extras)), dtype=np.float32)


def ann_recall_at_k(
    exact_ids: NDArray[np.int64], approximate_ids: NDArray[np.int64]
) -> float:
    """Measure mean top-k neighbor overlap with the exact index."""
    if exact_ids.ndim != 2 or approximate_ids.shape != exact_ids.shape:
        raise ValueError("exact and approximate neighbor matrices must have the same 2D shape")
    if exact_ids.shape[1] == 0:
        raise ValueError("neighbor matrices must contain at least one result")
    overlaps = [
        len(set(exact_row.tolist()) & set(approximate_row.tolist())) / exact_ids.shape[1]
        for exact_row, approximate_row in zip(exact_ids, approximate_ids, strict=True)
    ]
    return float(np.mean(overlaps))


def _latency(
    index: Any,
    queries: NDArray[np.float32],
    k: int,
    *,
    warmups: int,
    repeats: int,
) -> LatencyStats:
    for _ in range(warmups):
        index.search(queries, k)
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        index.search(queries, k)
        samples.append((time.perf_counter() - started) * 1000 / len(queries))
    return LatencyStats(
        median_ms_per_query=float(np.median(samples)),
        p95_ms_per_query=float(np.percentile(samples, 95)),
        repetitions=repeats,
        warmups=warmups,
    )


def _build_stats(
    *, seconds: float, rss_before: int, rss_after: int, serialized_bytes: int
) -> BuildStats:
    return BuildStats(
        seconds=seconds,
        serialized_bytes=serialized_bytes,
        rss_before_bytes=rss_before,
        rss_after_bytes=rss_after,
        rss_delta_bytes=rss_after - rss_before,
    )


def benchmark_faiss(
    store: EmbeddingStore,
    *,
    corpus_size: int,
    k: int = 10,
    m: int = 32,
    ef_construction: int = 200,
    ef_search_values: tuple[int, ...] = (16, 32, 64, 128),
    threads: int = 1,
    warmups: int = 2,
    repeats: int = 7,
    seed: int = 42,
    noise_std: float = 0.01,
) -> dict[str, Any]:
    """Benchmark Faiss exact inner product against HNSW on normalized embeddings."""
    try:
        import faiss
        import psutil
    except ImportError as error:
        raise RuntimeError("Faiss benchmark dependencies are missing; install .[search]") from error

    if k <= 0 or m <= 0 or ef_construction <= 0 or threads <= 0:
        raise ValueError("k, m, ef_construction, and threads must be positive")
    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    if not ef_search_values or any(value < k for value in ef_search_values):
        raise ValueError("every ef_search value must be at least k")
    if len(set(ef_search_values)) != len(ef_search_values):
        raise ValueError("ef_search values must be unique")

    index_positions = [position for position, split in enumerate(store.splits) if split == "index"]
    query_positions = [position for position, split in enumerate(store.splits) if split == "query"]
    if not index_positions or not query_positions:
        raise ValueError("embedding store must contain both index and query rows")
    if corpus_size < len(index_positions):
        raise ValueError("corpus_size must be at least the number of index rows")
    if k > corpus_size:
        raise ValueError("k must not exceed corpus_size")

    base_index = l2_normalize(store.vectors[index_positions])
    queries = l2_normalize(store.vectors[query_positions])
    corpus = expand_corpus(
        base_index,
        target_size=corpus_size,
        seed=seed,
        noise_std=noise_std,
    )
    process = psutil.Process()
    dimension = int(corpus.shape[1])
    previous_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(threads)
    try:
        gc.collect()
        exact_rss_before = process.memory_info().rss
        started = time.perf_counter()
        exact = faiss.IndexFlatIP(dimension)
        exact.add(corpus)
        exact_build_seconds = time.perf_counter() - started
        exact_rss_after = process.memory_info().rss
        exact_build = _build_stats(
            seconds=exact_build_seconds,
            rss_before=exact_rss_before,
            rss_after=exact_rss_after,
            serialized_bytes=int(faiss.serialize_index(exact).nbytes),
        )
        exact_latency = _latency(exact, queries, k, warmups=warmups, repeats=repeats)
        _, exact_ids = exact.search(queries, k)
        del exact
        gc.collect()

        hnsw_rss_before = process.memory_info().rss
        started = time.perf_counter()
        hnsw = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = ef_construction
        hnsw.add(corpus)
        hnsw_build_seconds = time.perf_counter() - started
        hnsw_rss_after = process.memory_info().rss
        hnsw_build = _build_stats(
            seconds=hnsw_build_seconds,
            rss_before=hnsw_rss_before,
            rss_after=hnsw_rss_after,
            serialized_bytes=int(faiss.serialize_index(hnsw).nbytes),
        )

        searches: list[dict[str, Any]] = []
        for ef_search in ef_search_values:
            hnsw.hnsw.efSearch = ef_search
            latency = _latency(hnsw, queries, k, warmups=warmups, repeats=repeats)
            _, approximate_ids = hnsw.search(queries, k)
            searches.append(
                {
                    "ef_search": ef_search,
                    "ann_recall_at_k": ann_recall_at_k(exact_ids, approximate_ids),
                    "latency": asdict(latency),
                    "speedup_vs_exact_median": (
                        exact_latency.median_ms_per_query / latency.median_ms_per_query
                    ),
                }
            )
    finally:
        faiss.omp_set_num_threads(previous_threads)

    return {
        "schema_version": 1,
        "benchmark": "faiss-exact-vs-hnsw",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "embedding_metadata": store.metadata,
            "base_index_rows": len(index_positions),
            "query_rows": len(query_positions),
        },
        "workload": {
            "corpus_rows": corpus_size,
            "dimension": dimension,
            "k": k,
            "synthetic_expansion": corpus_size > len(index_positions),
            "synthetic_rows": corpus_size - len(index_positions),
            "expansion_seed": seed,
            "expansion_noise_std_per_dimension": noise_std,
        },
        "configuration": {
            "metric": "cosine_via_normalized_inner_product",
            "threads": threads,
            "m": m,
            "ef_construction": ef_construction,
            "ef_search_values": list(ef_search_values),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "faiss_cpu": version("faiss-cpu"),
            "numpy": version("numpy"),
            "psutil": version("psutil"),
        },
        "exact": {
            "index": "IndexFlatIP",
            "build": asdict(exact_build),
            "latency": asdict(exact_latency),
        },
        "hnsw": {
            "index": "IndexHNSWFlat",
            "build": asdict(hnsw_build),
            "searches": searches,
        },
        "measurement_notes": [
            "ANN recall is top-k overlap with IndexFlatIP, not semantic class recall.",
            "Latency is measured for the full query batch and divided by its row count.",
            (
                "RSS deltas are approximate process-level observations and may include "
                "allocator effects."
            ),
            "Synthetic rows are perturbed copies for systems scaling only, not EO evidence.",
        ],
    }
