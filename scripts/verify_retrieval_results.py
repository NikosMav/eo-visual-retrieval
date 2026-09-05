"""Recompute recorded label-retrieval metrics from existing stores, without rewriting evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.evaluation import evaluate_store


def verify(results: Path, artifacts: Path) -> list[str]:
    checked = []
    for report in sorted(results.glob("*.json")):
        expected: dict[str, Any] = json.loads(report.read_text(encoding="utf-8"))
        match = re.fullmatch(r"(.+)-k(\d+)", report.stem)
        if match and "per_class" in expected:
            store_path = artifacts / f"{match[1]}.npz"
            k = int(match[2])
        elif report.name == "terramind-v1.json":
            expected = expected["metrics"]
            expected.pop("mlflow_run_id", None)  # Tracking identity is not a retrieval metric.
            store_path = artifacts / "eurosat-v1-terramind-tiny.npz"
            k = int(expected["k"])
        else:
            continue
        actual = evaluate_store(EmbeddingStore.load(store_path), k=k).to_dict()
        if actual != expected:
            raise ValueError(f"recomputed metrics differ from {report.name}")
        checked.append(report.name)
    if not checked:
        raise ValueError("no retrieval result records found")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("docs/results"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    print(json.dumps({"exact_matches": verify(args.results, args.artifacts)}, indent=2))


if __name__ == "__main__":
    main()
