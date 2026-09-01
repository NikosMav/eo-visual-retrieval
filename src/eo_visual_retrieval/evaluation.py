"""Label-proxy metrics for offline image retrieval evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.retrieval import ExactCosineIndex


@dataclass(frozen=True)
class EvaluationSummary:
    evaluated_queries: int
    skipped_queries: int
    k: int
    precision_at_k: float
    recall_at_k: float
    map_at_k: float
    ndcg_at_k: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "evaluated_queries": self.evaluated_queries,
            "skipped_queries": self.skipped_queries,
            "k": self.k,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "map_at_k": self.map_at_k,
            "ndcg_at_k": self.ndcg_at_k,
        }


def evaluate_store(store: EmbeddingStore, *, k: int) -> EvaluationSummary:
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

    precision_values: list[float] = []
    recall_values: list[float] = []
    ap_values: list[float] = []
    ndcg_values: list[float] = []
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
        precision_values.append(relevant_retrieved / k)
        recall_values.append(relevant_retrieved / total_relevant)

        running_relevant = 0
        precision_sum = 0.0
        dcg = 0.0
        for rank, is_relevant in enumerate(relevance, start=1):
            if is_relevant:
                running_relevant += 1
                precision_sum += running_relevant / rank
                dcg += 1.0 / math.log2(rank + 1)
        ap_values.append(precision_sum / min(total_relevant, k))
        ideal_relevant = min(total_relevant, k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
        ndcg_values.append(dcg / ideal_dcg)

    if not precision_values:
        raise ValueError("no labeled queries have relevant index items")

    return EvaluationSummary(
        evaluated_queries=len(precision_values),
        skipped_queries=skipped,
        k=k,
        precision_at_k=float(np.mean(precision_values)),
        recall_at_k=float(np.mean(recall_values)),
        map_at_k=float(np.mean(ap_values)),
        ndcg_at_k=float(np.mean(ndcg_values)),
    )
