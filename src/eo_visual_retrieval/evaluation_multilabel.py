"""Development-only retrieval evaluation with binary and graded Jaccard relevance.

Final queries are deliberately not scored here. A separate frozen-configuration
gate is required before adding a confirmatory evaluation entry point.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import numpy as np

from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.evaluation import MetricSummary
from eo_visual_retrieval.relevance import RelevanceManifest
from eo_visual_retrieval.retrieval import ExactCosineIndex
from eo_visual_retrieval.vectors import l2_normalize

DEVELOPMENT_THRESHOLDS = (0.3, 0.5, 0.7)
DEFAULT_THRESHOLD = 0.5


def jaccard(left: Collection[str], right: Collection[str]) -> float:
    """Intersection over union; empty judgments carry zero relevance."""
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


@dataclass(frozen=True)
class MultiLabelSummary(MetricSummary):
    k: int
    threshold: float
    skipped_queries: int
    queries_without_binary_positives: int
    final_queries_held_out: int
    per_label: dict[str, MetricSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "k": self.k,
            "threshold": self.threshold,
            "evaluation_partition": "development",
            "binary_relevance": "jaccard >= threshold",
            "ndcg_gain": "raw-jaccard",
            "skipped_queries": self.skipped_queries,
            "queries_without_binary_positives": self.queries_without_binary_positives,
            "final_queries_held_out": self.final_queries_held_out,
            "per_label": {
                label: metrics.to_dict() for label, metrics in sorted(self.per_label.items())
            },
        }


def _mean_metrics(values: list[tuple[float, float, float, float]]) -> MetricSummary:
    precision, recall, ap, ndcg = np.mean(np.asarray(values), axis=0)
    return MetricSummary(len(values), float(precision), float(recall), float(ap), float(ndcg))


def evaluate_multilabel_development(
    store: EmbeddingStore,
    relevance: RelevanceManifest,
    *,
    k: int = 10,
    threshold: float = DEFAULT_THRESHOLD,
) -> MultiLabelSummary:
    """Score development queries only, keeping query eligibility fixed across thresholds.

Unlabeled queries are skipped. Labeled queries with no binary-relevant index
item receive zero binary metrics but retain their independently graded nDCG.
Their count is reported so a threshold cannot improve scores by dropping them.
"""
    if threshold not in DEVELOPMENT_THRESHOLDS:
        raise ValueError("development threshold must be one of 0.3, 0.5, 0.7")
    if store.metadata.get("manifest_sha256") != relevance.image_manifest_sha256:
        raise ValueError("embedding and relevance image-manifest hashes differ")
    if any(label is not None for label in store.labels):
        raise ValueError("multi-label judgments belong in the relevance manifest, not store.labels")
    by_id = {record.item_id: record for record in relevance.records}
    if set(by_id) != set(store.ids):
        raise ValueError("relevance IDs must exactly match embedding IDs")
    records = [by_id[item_id] for item_id in store.ids]
    for record, split in zip(records, store.splits, strict=True):
        expected = "index" if record.partition == "index" else "query"
        if split != expected:
            raise ValueError(
                f"embedding split disagrees with relevance partition: {record.item_id}"
            )

    index_positions = [i for i, record in enumerate(records) if record.partition == "index"]
    queries = [i for i, record in enumerate(records) if record.partition == "development"]
    if not index_positions or not queries:
        raise ValueError("evaluation requires index items and development queries")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(index_positions):
        raise ValueError("k must be between 1 and the number of index items")
    index_labels = [records[i].labels for i in index_positions]
    index = ExactCosineIndex(
        [store.ids[i] for i in index_positions], l2_normalize(store.vectors[index_positions])
    )
    values: list[tuple[float, float, float, float]] = []
    slices: dict[str, list[tuple[float, float, float, float]]] = {}
    skipped = without_positives = 0
    discounts = np.log2(np.arange(2, k + 2, dtype=np.float64))
    for position in queries:
        labels = records[position].labels
        if not labels:
            skipped += 1
            continue
        # Only index and development vectors are inspected; final vectors cannot
        # influence this computation, including normalization or eligibility.
        query = l2_normalize(store.vectors[position : position + 1])[0]
        ranking = index.search(query, k=k)
        gains = np.asarray([jaccard(labels, target) for target in index_labels])
        ranked_gains = gains[[result.position for result in ranking]]
        hits = ranked_gains >= threshold
        total_relevant = int(np.count_nonzero(gains >= threshold))
        retrieved = int(np.count_nonzero(hits))
        ap = recall = 0.0
        if total_relevant:
            recall = retrieved / total_relevant
            ap = float(np.sum(np.cumsum(hits) / np.arange(1, k + 1) * hits)) / min(
                total_relevant, k
            )
        else:
            without_positives += 1
        ideal_dcg = float(np.sum(np.sort(gains)[::-1][:k] / discounts))
        ndcg = float(np.sum(ranked_gains / discounts)) / ideal_dcg if ideal_dcg else 0.0
        metrics = (retrieved / k, recall, ap, ndcg)
        if not all(math.isfinite(value) for value in metrics):
            raise ValueError("evaluation produced non-finite metrics")
        values.append(metrics)
        for label in labels:
            slices.setdefault(label, []).append(metrics)
    if not values:
        raise ValueError("no labeled development queries to evaluate")
    summary = _mean_metrics(values)
    return MultiLabelSummary(
        **summary.to_dict(),
        k=k,
        threshold=threshold,
        skipped_queries=skipped,
        queries_without_binary_positives=without_positives,
        final_queries_held_out=sum(record.partition == "final" for record in records),
        per_label={label: _mean_metrics(rows) for label, rows in sorted(slices.items())},
    )
