"""Bounded CPU/CUDA SSL4EO parity check on existing local EuroSAT inputs.

This is a correctness smoke test, not a throughput benchmark. No download occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from eo_visual_retrieval.embeddings.ssl4eo import CHECKPOINT_SHA256, ssl4eo_embeddings
from eo_visual_retrieval.manifests import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: use the separate locked CUDA environment")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    counts: Counter[tuple[str | None, str]] = Counter()
    selected = []
    for record in sorted(read_jsonl(args.manifest), key=lambda row: row.item_id):
        group = (record.label, record.split)
        if counts[group] < 2:
            counts[group] += 1
            selected.append(record)
    if not selected or len(selected) > 40:
        raise SystemExit("this smoke test expects 1-40 selected EuroSAT items")
    outputs = {}
    seconds = {}
    for device in ("cpu", "cuda"):
        started = time.perf_counter()
        outputs[device] = ssl4eo_embeddings(
            selected,
            archive=args.archive,
            checkpoint=args.checkpoint,
            batch_size=2,
            device=device,
        )
        seconds[device] = time.perf_counter() - started
    cpu, cuda = outputs["cpu"], outputs["cuda"]
    norms = np.linalg.norm(cuda, axis=1)
    passed = bool(
        np.isfinite(cuda).all()
        and np.allclose(cpu, cuda, rtol=1e-4, atol=1e-5)
        and np.allclose(norms, 1, atol=1e-5)
    )
    result = {
        "evidence": "bounded-cpu-cuda-correctness-smoke-not-throughput",
        "passed": passed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(row.item_id for row in selected).encode()
        ).hexdigest(),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "shape": list(cuda.shape),
        "batch_size": 2,
        "threads": 4,
        "tolerance": {"rtol": 1e-4, "atol": 1e-5},
        "max_absolute_difference": float(np.abs(cpu - cuda).max()),
        "minimum_paired_cosine": float(np.sum(cpu * cuda, axis=1).min()),
        "cuda_norm_range": [float(norms.min()), float(norms.max())],
        "elapsed_seconds_including_load_and_checksums": seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("CPU/CUDA parity gate failed; inspect the report")


if __name__ == "__main__":
    main()
