"""Opt-in local experiment tracking with an explicit metadata allowlist."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.evaluation import EvaluationSummary


def evaluation_parameters(store: EmbeddingStore, *, store_sha256: str) -> dict[str, Any]:
    """Never copy arbitrary provider metadata into the experiment record."""

    backend = store.metadata.get("backend")
    if not isinstance(backend, str) or backend not in {"pca", "dinov2", "ssl4eo-s12", "terramind"}:
        backend = "custom"
    parameters: dict[str, Any] = {
        "backend": backend,
        "embedding_dimension": store.vectors.shape[1],
        "index_items": store.splits.count("index"),
        "query_items": store.splits.count("query"),
        "embedding_store_sha256": store_sha256,
        "ranker": "exact-cosine",
        "relevance": "class-label-proxy",
    }
    for key in ("manifest_sha256", "checkpoint_sha256"):
        value = store.metadata.get(key)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
            parameters[key] = value.lower()
    return parameters


def log_evaluation(
    store: EmbeddingStore,
    summary: EvaluationSummary,
    *,
    embeddings_path: Path,
    tracking_dir: Path,
) -> str:
    """Log aggregate metrics/content hashes to local SQLite, not a remote tracking URI.

    No fluent/global tracking state or autologging is used. Artifacts never include
    vectors, imagery, image IDs, labels, paths, or arbitrary provider metadata.
    """

    if str(tracking_dir).startswith(("\\\\", "//")):
        raise ValueError("tracking directory must be local, not a network share")
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as error:
        raise RuntimeError("local tracking requires the 'experiments' dependency group") from error

    digest = hashlib.sha256()
    with embeddings_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    parameters = evaluation_parameters(store, store_sha256=digest.hexdigest())
    parameters["k"] = summary.k
    metrics = {
        key: float(value)
        for key, value in summary.to_dict().items()
        if isinstance(value, (int, float))
    }
    root = tracking_dir.resolve()
    if str(root).startswith(("\\\\", "//")):
        raise ValueError("tracking directory must resolve to a local path")
    root.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    name = "eovr-offline-evaluation"
    experiment = client.get_experiment_by_name(name)
    artifact_location = (root / "artifacts").as_uri()
    if experiment is not None and experiment.artifact_location != artifact_location:
        raise ValueError("existing experiment does not use the expected local artifact directory")
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(name, artifact_location=artifact_location)
    )
    run_id = client.create_run(
        experiment_id,
        tags={"eovr.evidence": "offline-regression", "eovr.network_training": "false"},
    ).info.run_id
    try:
        for key, value in parameters.items():
            client.log_param(run_id, key, value)
        for key, value in metrics.items():
            client.log_metric(run_id, key, value)
        client.log_dict(run_id, {"parameters": parameters, "metrics": metrics}, "evaluation.json")
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    client.set_terminated(run_id, status="FINISHED")
    return run_id
