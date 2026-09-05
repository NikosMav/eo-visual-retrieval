# Frozen RemoteCLIP temporal diagnostic — 2026-09-05

The image encoder that enables text search is not automatically the strongest image retriever.
On the existing guarded temporal development corpus, RemoteCLIP ViT-B/32 returns a same-place
first result for 15 of 36 queries (41.7%). DINOv2 returns 29 (80.6%) and PCA-32 returns 24 (66.7%).
The comparison uses exactly the same 61 index images, 36 query images, 12 places, and 90-day
minimum same-place temporal separation. No imagery was downloaded and no model was trained.

| Representation | Top-1 accuracy | mAP@5 |
|---|---:|---:|
| RemoteCLIP ViT-B/32 | 0.416667 | 0.427485 |
| DINOv2 ViT-S/14 | 0.805556 | 0.597006 |

Reproduce with `python scripts/evaluate_multimodal_baseline.py`, then
`python scripts/verify_retrieval_results.py`. The first command uses the frozen local RemoteCLIP
store and the shared evaluator; it writes k=1/k=5 reports and model/store/manifest identities in
`multimodal-temporal-provenance.json`. The second recomputes all recorded retrieval metrics,
including the earlier baselines, without changing them.

This finding motivates a separate visual retrieval branch and a controlled rank-fusion experiment.
It does not establish which model best answers descriptions or combined image-text requests.
There are no human semantic judgments here. Queries at each place are correlated, per-place
accuracy has only three observations, and the comparison cannot isolate architecture, pretraining,
or preprocessing effects. These data cannot establish unseen-place or multi-year generalization.
The corpus has already been inspected; it is not a final set for tuning a replacement model.
