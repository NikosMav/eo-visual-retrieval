from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from eo_visual_retrieval.cli import build_parser
from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.evaluation import evaluate_store
from eo_visual_retrieval.tracking import evaluation_parameters, log_evaluation


def sample_store() -> EmbeddingStore:
    return EmbeddingStore(
        ids=("index", "query"),
        vectors=np.asarray([[1, 0], [1, 0]], dtype=np.float32),
        labels=("private-label", "private-label"),
        splits=("index", "query"),
        metadata={
            "backend": "pca",
            "manifest_sha256": "A" * 64,
            "checkpoint_sha256": "https://example.invalid/?token=secret",
            "signed_url": "https://example.invalid/?token=secret",
            "private_aoi": [1, 2, 3, 4],
        },
    )


class FakeClient:
    def __init__(self) -> None:
        self.uri = ""
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.artifact: dict[str, Any] = {}
        self.status = ""
        self.experiment: Any = None
        self.fail = False

    def get_experiment_by_name(self, name: str) -> Any:
        return self.experiment

    def create_experiment(self, name: str, *, artifact_location: str) -> str:
        self.experiment = SimpleNamespace(experiment_id="1", artifact_location=artifact_location)
        return "1"

    def create_run(self, experiment_id: str, *, tags: dict[str, str]) -> Any:
        return SimpleNamespace(info=SimpleNamespace(run_id="test-run"))

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        self.params[key] = value

    def log_metric(self, run_id: str, key: str, value: float) -> None:
        if self.fail:
            raise RuntimeError("simulated failure")
        self.metrics[key] = value

    def log_dict(self, run_id: str, value: dict[str, Any], filename: str) -> None:
        self.artifact = value

    def set_terminated(self, run_id: str, *, status: str) -> None:
        self.status = status


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    fake = FakeClient()

    def constructor(*, tracking_uri: str) -> FakeClient:
        fake.uri = tracking_uri
        return fake

    module = ModuleType("mlflow.tracking")
    module.MlflowClient = constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", ModuleType("mlflow"))
    monkeypatch.setitem(sys.modules, "mlflow.tracking", module)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.invalid")
    return fake


def test_parameters_exclude_arbitrary_metadata_and_labels() -> None:
    params = evaluation_parameters(sample_store(), store_sha256="b" * 64)
    assert params["manifest_sha256"] == "a" * 64
    assert params["embedding_dimension"] == 2
    assert "checkpoint_sha256" not in params
    assert "secret" not in json.dumps(params)
    assert "private" not in json.dumps(params)


def test_tracking_is_local_and_reuses_experiment(tmp_path: Path, client: FakeClient) -> None:
    store = sample_store()
    path = tmp_path / "vectors.npz"
    store.save(path)
    for _ in range(2):
        run_id = log_evaluation(
            store, evaluate_store(store, k=1), embeddings_path=path, tracking_dir=tmp_path / "runs"
        )
        assert run_id == "test-run"
    assert client.uri.startswith("sqlite:///")
    assert client.status == "FINISHED"
    assert client.metrics["map_at_k"] == 1.0
    assert len(client.params["embedding_store_sha256"]) == 64
    assert "secret" not in json.dumps(client.artifact)
    assert "private-label" not in json.dumps(client.artifact)


def test_tracking_failure_marks_run_failed(tmp_path: Path, client: FakeClient) -> None:
    store = sample_store()
    path = tmp_path / "vectors.npz"
    store.save(path)
    client.fail = True
    with pytest.raises(RuntimeError, match="simulated failure"):
        log_evaluation(
            store, evaluate_store(store, k=1), embeddings_path=path, tracking_dir=tmp_path
        )
    assert client.status == "FAILED"


def test_tracking_rejects_existing_remote_artifact_store(
    tmp_path: Path, client: FakeClient
) -> None:
    store = sample_store()
    path = tmp_path / "vectors.npz"
    store.save(path)
    client.experiment = SimpleNamespace(experiment_id="1", artifact_location="s3://remote-bucket")
    with pytest.raises(ValueError, match="expected local artifact"):
        log_evaluation(
            store, evaluate_store(store, k=1), embeddings_path=path, tracking_dir=tmp_path
        )


def test_cli_evaluation_tracking_is_opt_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "vectors.npz"
    sample_store().save(path)
    args = build_parser().parse_args(["evaluate", "--embeddings", str(path), "--k", "1"])
    assert args.tracking_dir is None
    args.handler(args)
    assert json.loads(capsys.readouterr().out)["map_at_k"] == 1.0


def test_cli_evaluation_can_track(tmp_path: Path, client: FakeClient) -> None:
    path = tmp_path / "vectors.npz"
    sample_store().save(path)
    output = tmp_path / "report.json"
    args = build_parser().parse_args(
        [
            "evaluate",
            "--embeddings",
            str(path),
            "--k",
            "1",
            "--tracking-dir",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )
    args.handler(args)
    assert json.loads(output.read_text())["mlflow_run_id"] == "test-run"
