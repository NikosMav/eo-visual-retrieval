"""Label-proxy metrics for offline image retrieval evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.retrieval import ExactCosineIndex


@dataclass(frozen=True)
class MetricSummary:
    evaluated_queries: int
    precision_at_k: float
    recall_at_k: float
    map_at_k: float
    ndcg_at_k: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated_queries": self.evaluated_queries,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "map_at_k": self.map_at_k,
            "ndcg_at_k": self.ndcg_at_k,
        }


@dataclass(frozen=True)
class EvaluationSummary(MetricSummary):
    skipped_queries: int
    k: int
    per_class: dict[str, MetricSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "skipped_queries": self.skipped_queries,
            "k": self.k,
            "per_class": {
                label: summary.to_dict() for label, summary in sorted(self.per_class.items())
            },
        }


@dataclass(frozen=True)
class QueryEvaluation:
    query_id: str
    label: str
    precision_at_k: float
    recall_at_k: float
    average_precision_at_k: float
    ndcg_at_k: float
    ranked_ids: tuple[str, ...]
    ranked_scores: tuple[float, ...]
    relevance: tuple[int, ...]


def _metric_summary(values: list[tuple[str, float, float, float, float]]) -> MetricSummary:
    return MetricSummary(
        evaluated_queries=len(values),
        precision_at_k=float(np.mean([value[1] for value in values])),
        recall_at_k=float(np.mean([value[2] for value in values])),
        map_at_k=float(np.mean([value[3] for value in values])),
        ndcg_at_k=float(np.mean([value[4] for value in values])),
    )


def evaluate_queries(store: EmbeddingStore, *, k: int) -> tuple[list[QueryEvaluation], int]:
    """Return auditable per-query rankings and metrics plus the skipped count."""

    index_positions = [i for i, split in enumerate(store.splits) if split == "index"]
    query_positions = [i for i, split in enumerate(store.splits) if split == "query"]
    if not index_positions or not query_positions:
        raise ValueError("evaluation requires both index and query items")
    if k > len(index_positions):
        raise ValueError("k cannot exceed the number of index items")

    index_ids = [store.ids[i] for i in index_positions]
    index_labels = [store.labels[i] for i in index_positions]
    index = ExactCosineIndex(index_ids, store.vectors[index_positions])
    label_by_id = dict(zip(index_ids, index_labels, strict=True))

    evaluations: list[QueryEvaluation] = []
    skipped = 0

    for position in query_positions:
        query_label = store.labels[position]
        if query_label is None:
            skipped += 1
            continue
        total_relevant = sum(label == query_label for label in index_labels)
        if total_relevant == 0:
            skipped += 1
            continue

        results = index.search(store.vectors[position], k=k, exclude_id=store.ids[position])
        relevance = [int(label_by_id[result.item_id] == query_label) for result in results]
        relevant_retrieved = sum(relevance)
        # Excluding the query from its own index can leave fewer than k results.
        # Scoring against the requested k would then penalise a ranking for
        # positions that could not exist, so use what was actually retrievable.
        retrieved = len(results)
        precision = relevant_retrieved / retrieved
        recall = relevant_retrieved / total_relevant

        running_relevant = 0
        precision_sum = 0.0
        dcg = 0.0
        for rank, is_relevant in enumerate(relevance, start=1):
            if is_relevant:
                running_relevant += 1
                precision_sum += running_relevant / rank
                dcg += 1.0 / math.log2(rank + 1)
        ideal_relevant = min(total_relevant, retrieved)
        average_precision = precision_sum / ideal_relevant
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
        ndcg = dcg / ideal_dcg
        evaluations.append(
            QueryEvaluation(
                query_id=store.ids[position],
                label=query_label,
                precision_at_k=precision,
                recall_at_k=recall,
                average_precision_at_k=average_precision,
                ndcg_at_k=ndcg,
                ranked_ids=tuple(result.item_id for result in results),
                ranked_scores=tuple(result.score for result in results),
                relevance=tuple(relevance),
            )
        )

    return evaluations, skipped


def evaluate_store(store: EmbeddingStore, *, k: int) -> EvaluationSummary:
    evaluations, skipped = evaluate_queries(store, k=k)
    metric_values = [
        (
            evaluation.label,
            evaluation.precision_at_k,
            evaluation.recall_at_k,
            evaluation.average_precision_at_k,
            evaluation.ndcg_at_k,
        )
        for evaluation in evaluations
    ]

    if not metric_values:
        raise ValueError("no labeled queries have relevant index items")

    aggregate = _metric_summary(metric_values)
    labels = sorted({value[0] for value in metric_values})
    return EvaluationSummary(
        evaluated_queries=aggregate.evaluated_queries,
        skipped_queries=skipped,
        k=k,
        precision_at_k=aggregate.precision_at_k,
        recall_at_k=aggregate.recall_at_k,
        map_at_k=aggregate.map_at_k,
        ndcg_at_k=aggregate.ndcg_at_k,
        per_class={
            label: _metric_summary([value for value in metric_values if value[0] == label])
            for label in labels
        },
    )
