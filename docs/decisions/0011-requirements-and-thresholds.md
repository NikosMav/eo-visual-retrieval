# ADR 0011: Requirements, and which ones this project will not state

- Status: accepted
- Date: 2026-09-04
- Scope: turn quality and performance intentions into statements that can fail

## Context

Two kinds of requirement were proposed: a retrieval-quality target ("top 5 retrieval accuracy
above 85%") and a set of non-functional adjectives ("low latency", "responsive", "scalable",
"high-performance").

A requirement must be falsifiable. Someone has to be able to run a stated procedure and report
that it failed. "Responsive" cannot fail. Neither, as written, can "top 5 accuracy above 85%",
because the phrase has at least three meanings that disagree about which representation passes.

Measured on EuroSAT v1 at the time of this record:

| Representation | Precision@5 | Hit@5 | Top-1 |
|---|---|---|---|
| PCA-64 | 0.325 | 0.698 | 0.333 |
| DINOv2 | 0.712 | 0.943 | 0.743 |
| SSL4EO-S12 RGB | 0.823 | 0.975 | 0.868 |
| SSL4EO-S12 13-band | 0.878 | 0.970 | 0.910 |
| TerraMind-Tiny | 0.783 | 0.965 | 0.833 |

Under Precision@5 an 85% bar admits one representation. Under Hit@5 it admits four, including one
that returns four wrong images out of five. The threshold does not decide the outcome; the
unstated definition does.

## Decision

### 1. Quality requirements name a metric, a set, and a purpose

Every quality requirement in this project states the metric, the evaluation set, and whether it is
a regression guard or a target. The two are not interchangeable.

**R1 — retrieval-quality regression guard.** On EuroSAT v1 at k=5, the selected representation
must hold `precision_at_k >= 0.85`, measured by `eovr evaluate --k 5`. Observed 0.878 for
SSL4EO-S12 13-band.

**R2 — published-result reproduction.** Re-running `eovr evaluate --k 10` from the committed
stores must reproduce every file in `docs/results/` exactly. This is the stronger guard and has
caught real changes; R1 exists for the case where a future change is intended to alter rankings.

Both are **regression guards**. EuroSAT v1 was used to select the model roster, so no score on it
is evidence of generalization, and neither R1 nor R2 may be quoted as a quality claim.

### 2. The confirmatory set carries a pre-registered hypothesis, not a number

A pass/fail threshold invented before seeing a dataset is arbitrary. A directional prediction is
not. Registered here, before any confirmatory imagery exists:

**R3 — pre-registered confirmatory hypothesis.** On the frozen final partition, scored once under
[ADR 0009](0009-confirmatory-model-roster.md)'s roster, we predict the EuroSAT v1 ordering holds:
SSL4EO-S12 RGB and TerraMind-Tiny both rank above DINOv2, and DINOv2 ranks above PCA-64, on mAP@10.

If the ordering does not hold, that is a finding to publish, not a failure to fix. Recording the
prediction now is what makes the result informative either way.

### 3. Non-functional requirements state metric, threshold, workload, and environment

Each is a regression guard against the measured behaviour recorded in
[validation](../validation.md), with generous headroom. None is a service-level objective: there
is no deployment, no users, and no load profile, and an objective without a consumer is decoration.

**R4 — container startup budget.** The container reaches its health check within **180 s** when
serving the five EuroSAT v1 stores from read-only mounts. Observed 70.7 s over a Windows bind
mount, 19.0 s for the equivalent checks on the host. The budget exists because startup verifies
2,000 image hashes and re-projects the PCA vectors, and a host with a shorter deadline would kill
the container before it is ready.

**R5 — interactive query latency.** p95 end-to-end latency for `GET /compare` with a stored query,
over the 1,600-image corpus, in one container at 10 concurrent requests, is **at most 150 ms**.
Observed 31–40 ms single-request, and 20 requests at concurrency 10 completed in 1,031 ms.

**R6 — serving memory ceiling.** Steady resident memory serving all five stores is **at most
512 MiB**. Observed 139.9 MiB.

**R7 — exact-search cost.** Median exact cosine search over 1,600 vectors, single-threaded, is
**at most 1 ms per query**. Observed 0.0123 ms. The ceiling is three orders of magnitude above the
measurement on purpose: it is a guard against an accidental quadratic, not a performance target.

Each requirement is checked by re-running the procedure named beside it. A requirement whose
measurement procedure is not written down is not adopted.

### 4. Requirements this project declines to state

- **"Scalable."** Meaningless without a target corpus size, concurrency, and latency objective.
  The recorded 50k-vector experiment used deterministic synthetic expansion and establishes
  nothing about 50,000 independent EO images.
- **"High-performance", "responsive", "low latency"** as unqualified terms. Where a real
  constraint exists it is stated as R4 through R7.
- **Availability, throughput, or concurrency objectives.** These require a deployment and observed
  traffic. Neither exists.

Declining these is a decision, not an oversight. They can be adopted when a destination and a
workload make them measurable.

## Consequences

- Quality claims must name their metric and set. "Top-5 accuracy" alone is not usable in this
  repository's documentation.
- R1 and R2 can fail a change. R3 cannot be checked until confirmatory imagery exists, and must
  not be quietly revised after the data arrives.
- R4 through R7 constrain hosting: a platform that cannot tolerate a 180 s startup is unsuitable
  without first reducing the startup verification, which is a deliberate integrity guard.
- A future user-facing evaluation needs a relevance notion finer than the class label. The
  [structure analysis](../results/eurosat-v1-analysis.md) shows representations sharing as little
  as one retrieved image in thirty can post comparable scores, so class relevance measures whether
  the retrieved thing is the right kind and never whether it is the right one.

## Boundary

These thresholds encode behaviour measured on one machine on 2026-09-04, on a 1,600-image corpus.
They are floors and ceilings for detecting regression, not predictions about other hardware, other
corpus sizes, or a deployed service.
