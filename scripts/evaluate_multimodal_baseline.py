"""Evaluate frozen RemoteCLIP on the existing development temporal split, no training."""

from __future__ import annotations

import json
from pathlib import Path

from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.evaluation import evaluate_store
from eo_visual_retrieval.hashing import file_sha256


def main() -> None:
    path = Path("artifacts/temporal-v1g-remoteclip-vit-b32.npz")
    store = EmbeddingStore.load(path)
    results = Path("docs/results")
    for k in (1, 5):
        metrics = evaluate_store(store, k=k).to_dict()
        (results / f"{path.stem}-k{k}.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"k": k, "top_k_precision": metrics["precision_at_k"],
                          "map": metrics["map_at_k"]}))
    provenance = {
        "evidence_role": "development image-to-image same-place proxy; no text judgments",
        "store_sha256": file_sha256(path),
        "manifest_sha256": store.metadata["manifest_sha256"],
        "model": store.metadata["model"],
        "checkpoint_sha256": store.metadata["checkpoint_sha256"],
        "training": "Frozen checkpoint; no fine-tuning or hyperparameter selection",
        "limitations": ["36 queries at 12 places in 2024; queries within a place are correlated",
                        "Not a semantic text or hybrid relevance benchmark",
                        "Not an unseen-place or change-detection evaluation"],
    }
    (results / "multimodal-temporal-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
