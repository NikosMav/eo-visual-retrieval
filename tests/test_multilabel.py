"""Numerical, partition, and provenance contracts for development evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eo_visual_retrieval.cli import build_parser
from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.evaluation_multilabel import evaluate_multilabel_development, jaccard
from eo_visual_retrieval.hashing import file_sha256
from eo_visual_retrieval.relevance import RelevanceManifest, RelevanceRecord


def _fixture() -> tuple[EmbeddingStore, RelevanceManifest]:
    # Query {a,b} ranks labels {a}, {c}, {a,b}: binary gains 1,0,1 at tau=0.5.
    records = (
        RelevanceRecord("half", ("a",), "index"),
        RelevanceRecord("miss", ("c",), "index"),
        RelevanceRecord("full", ("a", "b"), "index"),
        RelevanceRecord("dev", ("a", "b"), "development"),
        RelevanceRecord("held-out", ("c",), "final"),
    )
    store = EmbeddingStore(
        ids=tuple(record.item_id for record in records),
        vectors=np.asarray([[1, 0], [0.8, 0.6], [0, 1], [1, 0], [0, 1]], dtype=np.float32),
        labels=(None,) * 5,
        splits=("index", "index", "index", "query", "query"),
        metadata={"manifest_sha256": "a" * 64},
    )
    return store, RelevanceManifest("synthetic-test", "a" * 64, records)


def test_jaccard_uses_sets_and_handles_missing_judgments() -> None:
    assert jaccard(("a", "b"), ("b", "c")) == pytest.approx(1 / 3)
    assert jaccard(("a", "a"), ("a",)) == 1
    assert jaccard((), ()) == 0
    assert jaccard(("a",), ()) == 0


def test_binary_metrics_and_raw_graded_ndcg_are_independent() -> None:
    store, relevance = _fixture()
    result = evaluate_multilabel_development(store, relevance, k=3)
    assert result.precision_at_k == pytest.approx(2 / 3)
    assert result.recall_at_k == 1
    assert result.map_at_k == pytest.approx((1 + 2 / 3) / 2)
    # DCG uses gains [0.5, 0, 1]; the ideal order is [1, 0.5, 0].
    assert result.ndcg_at_k == pytest.approx(1 / (1 + 0.5 / math.log2(3)))
    assert result.evaluated_queries == 1
    assert result.final_queries_held_out == 1
    assert set(result.per_label) == {"a", "b"}
    assert result.to_dict()["ndcg_gain"] == "raw-jaccard"


def test_recall_and_ap_use_all_relevant_index_items_beyond_k() -> None:
    store, relevance = _fixture()
    result = evaluate_multilabel_development(store, relevance, k=1)
    assert result.precision_at_k == 1
    assert result.recall_at_k == 0.5
    assert result.map_at_k == 1
    assert result.ndcg_at_k == 0.5


def test_threshold_changes_do_not_change_query_eligibility_or_graded_ndcg() -> None:
    store, relevance = _fixture()
    # Only the two partially overlapping items are now judged relevant at 0.5.
    changed = tuple(
        replace(record, labels=("a",)) if record.item_id == "full" else record
        for record in relevance.records
    )
    relevance = replace(relevance, records=changed)
    low = evaluate_multilabel_development(store, relevance, k=3, threshold=0.5)
    high = evaluate_multilabel_development(store, relevance, k=3, threshold=0.7)
    assert low.evaluated_queries == high.evaluated_queries == 1
    assert high.queries_without_binary_positives == 1
    assert (high.precision_at_k, high.recall_at_k, high.map_at_k) == (0, 0, 0)
    assert high.ndcg_at_k == low.ndcg_at_k > 0


def test_final_vectors_and_labels_cannot_influence_development_metrics() -> None:
    store, relevance = _fixture()
    before = evaluate_multilabel_development(store, relevance, k=3)
    vectors = store.vectors.copy()
    vectors[-1] = np.nan
    changed = replace(
        relevance,
        records=(*relevance.records[:-1], replace(relevance.records[-1], labels=("unseen",))),
    )
    after = evaluate_multilabel_development(replace(store, vectors=vectors), changed, k=3)
    assert after == before


def test_unlabeled_queries_are_counted_and_no_overlap_queries_score_zero() -> None:
    store, relevance = _fixture()
    records = tuple(
        replace(record, labels=(), partition="development")
        if record.item_id == "held-out" else record
        for record in relevance.records
    )
    result = evaluate_multilabel_development(store, replace(relevance, records=records), k=3)
    assert result.skipped_queries == 1
    assert result.evaluated_queries == 1
    orphan = tuple(
        replace(record, labels=("orphan",)) if record.item_id == "dev" else record
        for record in records
    )
    result = evaluate_multilabel_development(store, replace(relevance, records=orphan), k=3)
    assert result.queries_without_binary_positives == 1
    assert result.ndcg_at_k == result.map_at_k == result.recall_at_k == 0


@pytest.mark.parametrize("threshold", [0, 1, 0.4, float("nan"), float("inf")])
def test_unregistered_thresholds_are_rejected(threshold: float) -> None:
    store, relevance = _fixture()
    with pytest.raises(ValueError, match="threshold"):
        evaluate_multilabel_development(store, relevance, threshold=threshold)


@pytest.mark.parametrize("k", [0, -1, 4, True])
def test_invalid_k_is_rejected(k: int) -> None:
    store, relevance = _fixture()
    with pytest.raises(ValueError, match="k must"):
        evaluate_multilabel_development(store, relevance, k=k)


def test_hash_id_and_partition_mismatches_are_rejected() -> None:
    store, relevance = _fixture()
    with pytest.raises(ValueError, match="hashes differ"):
        evaluate_multilabel_development(replace(store, metadata={}), relevance, k=1)
    with pytest.raises(ValueError, match="IDs must exactly"):
        evaluate_multilabel_development(
            store, replace(relevance, records=relevance.records[:-1]), k=1
        )
    with pytest.raises(ValueError, match="split disagrees"):
        evaluate_multilabel_development(
            replace(store, splits=("query", *store.splits[1:])), relevance, k=1
        )
    with pytest.raises(ValueError, match="store.labels"):
        evaluate_multilabel_development(
            replace(store, labels=("a", *store.labels[1:])), relevance, k=1
        )


@pytest.mark.parametrize("position", [0, 3])
def test_invalid_index_or_development_vectors_are_rejected(position: int) -> None:
    store, relevance = _fixture()
    vectors = store.vectors.copy()
    vectors[position] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_multilabel_development(replace(store, vectors=vectors), relevance, k=1)


def test_relevance_manifest_roundtrip_and_duplicate_rejection(tmp_path: Path) -> None:
    _, relevance = _fixture()
    path = tmp_path / "nested" / "relevance.json"
    relevance.save(path)
    assert RelevanceManifest.load(path) == relevance
    assert RelevanceRecord("canonical", ("b", "a"), "index").labels == ("a", "b")
    with pytest.raises(ValueError, match="IDs must be unique"):
        replace(relevance, records=(*relevance.records, relevance.records[0]))
    with pytest.raises(ValueError, match="duplicates"):
        RelevanceRecord("duplicate", ("a", "a"), "index")
    with pytest.raises(ValueError, match="index items"):
        RelevanceRecord("unlabeled", (), "index")


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema": "wrong"},
        {"image_manifest_sha256": "bad"},
        {"records": "not-a-list"},
        {"records": [{"item_id": "a", "labels": "forest", "partition": "index"}]},
        {"records": [{"item_id": "a", "labels": ["forest"], "partition": "query"}]},
    ],
)
def test_malformed_relevance_json_is_rejected(tmp_path: Path, mutation: dict[str, Any]) -> None:
    _, relevance = _fixture()
    value = {**relevance.to_dict(), **mutation}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        RelevanceManifest.load(path)


def test_cli_writes_bound_development_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store, relevance = _fixture()
    vectors, judgments, output = (
        tmp_path / name for name in ("vectors.npz", "labels.json", "r.json")
    )
    store.save(vectors)
    relevance.save(judgments)
    args = build_parser().parse_args([
        "evaluate-multilabel", "--embeddings", str(vectors), "--relevance", str(judgments),
        "--output", str(output), "--k", "3",
    ])
    args.handler(args)
    result = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == result
    assert result["evaluation_partition"] == "development"
    assert result["embedding_store_sha256"] == file_sha256(vectors)
    assert result["relevance_manifest_sha256"] == file_sha256(judgments)
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "evaluate-multilabel", "--embeddings", str(vectors), "--relevance", str(judgments),
            "--output", str(output), "--partition", "final",
        ])
    args.output = judgments
    with pytest.raises(ValueError, match="must not overwrite"):
        args.handler(args)
