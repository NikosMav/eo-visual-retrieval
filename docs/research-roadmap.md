# A measured path to better search

The implemented product is a local research explorer: search, model comparison, findings, and
corpus/experiment visibility. This document proposes the next research stages. It does not report
new acquisitions, fine-tuning, or automated jobs as completed.

## 1. Build semantic evaluation data

Create a versioned query specification covering scene appearance, objects, spatial relationships,
metadata constraints, paraphrases and impossible/ambiguous requests. Each query records intent,
input image identity when present, expected filters, and its spatial/temporal evaluation group.
Pool candidates from multiple frozen models and rankers, randomize their order, and collect 0–3
relevance judgments with a written rubric. Review disagreements without revealing model identity.
Unjudged results are unknown, not negatives. Report judgment coverage and pool-relative recall;
exhaustive recall requires an exhaustively judged corpus.

Use nDCG@10, judged Precision@10, Recall@10 within the pool, constraint extraction accuracy,
filter compliance, no-result rate, and p50/p95 latency under specified hardware/concurrency.
Keep appearance relevance and metadata compliance as separate fields. Existing same-class and
same-place proxies remain useful regression tests but cannot substitute for these judgments.

## 2. Acquire data for the gaps

Prioritize independent places, multiple years, and chip-level cloud/snow annotations. Use STAC
discovery and stable source identities; inspect small batches before scaling. Record license,
bands, resolution, processing, spatial footprint, acquisition date and checksums. Distinguish
RGB previews from analysis-ready reflectance chips. Freeze spatial and temporal splits before
learning or choosing hyperparameters; separate training, development and final geography/time.

The existing BigEarthNet acquisition remains a separately controlled run. Its frozen confirmatory
model roster and final set are unchanged by this product experiment. Record any new semantic
experiment as its own protocol; do not silently reuse or expand the existing final gate.

## 3. Test ranking and models before training

Keep frozen RemoteCLIP as the initial text-capable baseline. Compare GeoRSCLIP/RS5M and SigLIP2
only under matched data and judged queries before selecting a replacement. Test a DINOv2 visual
branch alongside text retrieval; vectors from different encoders cannot be directly compared or
averaged. Rank fusion is one candidate when score scales differ. Tune fusion only on development
judgments, with image-only and text-only ablations and a fixed search space.

For the current small local corpus, exact NumPy cosine search is a useful reference with no database
operations to maintain. The existing Faiss experiments remain available. Qdrant becomes worth
testing when persistent multi-vector storage, metadata filtering or scale require it; add measured
recall against the exact reference and latency before changing the serving default.

## 4. Fine-tune conditionally

Fine-tuning is feasible with suitable licensed image–text pairs or relevance supervision. First
freeze the protocol and data version. Compare a frozen baseline, a lightweight adapter/LoRA run,
and only then full contrastive fine-tuning if justified by the data and compute budget. Include
hard negatives and visually similar scenes with different meanings. Inspect caption quality;
automatically generated captions are not independent ground truth for evaluating their generator.

OpenCLIP supplies training workflows; PEFT can reduce trainable parameter counts through LoRA.
The actual module targets and compatibility with the pinned OpenCLIP checkpoint must be tested
before a run. Do not assume a generic LLM LoRA recipe works unchanged for the vision/text towers.
Keep early stopping and model selection on development data. Log seeds, preprocessing, frozen
modules, optimizer, checkpoints, hardware, memory, elapsed time and retrieval regressions. Publish
the final result only once the chosen protocol is locked, including failures and uncertainty
computed over independent spatial/temporal groups.

## 5. Add explanations and change evidence carefully

Today's result explanations expose the exact score calculation and metadata checks. Region
occlusion is a possible next diagnostic: mask patches and observe score changes, checking multiple
patch sizes/fills. Its heatmap describes sensitivity to perturbations, not causality or verified
object localization. Evaluate usefulness with analysts before presenting it as a default explanation.

Urban expansion needs aligned before/after observations, comparable radiometry, quality masks and
separate change labels. A temporal comparison view should show dates, alignment and change evidence.
Text-image retrieval can find candidate scenes; it cannot certify expansion by itself.

## Primary research and implementation references

- [RemoteCLIP authors and official implementation](https://github.com/ChenDelong1999/RemoteCLIP)
- [RS5M / GeoRSCLIP authors and implementation](https://github.com/om-ai-lab/RS5M)
- [SigLIP2 model documentation](https://huggingface.co/docs/transformers/model_doc/siglip2)
- [OpenCLIP training workflows](https://github.com/mlfoundations/open_clip)
- [PEFT LoRA conceptual guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
- [Qdrant hybrid queries and rank fusion](https://qdrant.tech/documentation/search/hybrid-queries/)
- [Captum attribution methods](https://captum.ai/docs/attribution_algorithms)

Reviewed 2026-09-05. Tool capabilities are not evidence that they improve this corpus; the staged
experiments above are our proposed way to establish that.
