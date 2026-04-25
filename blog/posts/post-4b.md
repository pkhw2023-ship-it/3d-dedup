# From Embeddings to Production: VLM Verification and Billion-Scale Search for 3D Duplicates

*A Gemini-powered verification stage that eliminates false positives, plus FAISS engineering for sub-millisecond search at scale*

---

## The Last Mile Problem

Your embedding search works. You've rendered your 3D model from 28 camera angles, passed each view through DINOv2, aggregated the features via concat-PCA, and fired a nearest-neighbor query into your index. Sub-second, the results come back: 20 candidate matches, ranked by cosine similarity.

You open the first result. It's a match — a rescaled, rotated copy of your query model. The second result: also a match, this time a format re-export. You keep scrolling. Result 7: a generic chair that happens to share a silhouette with your query from one angle. Result 12: a completely different model that the embedding space has incorrectly mapped nearby. Result 15: another false positive.

This is the last mile problem: **your retrieval gives you recall, but it doesn't give you precision.** At the operating points where you catch most true duplicates, you also surface a flood of false positives. In a platform processing millions of 3D assets — an IP enforcement system for a game marketplace, a deduplication pipeline for a training dataset — manual review of every candidate is impossible.

Enter the vision-language model as verifier. Instead of asking a human to examine each candidate pair, we ask Gemini 2.0 Flash to look at side-by-side renders from multiple angles and make a judgment: *are these the same 3D model?* The VLM brings something that cosine similarity in embedding space cannot: semantic understanding of visual similarity, combined with multi-angle evidence accumulation.

This post presents the complete picture. We start with the full benchmark results from [3D-DupBench](../post-4a/post.md) — the definitive comparison of every method we tested. We then introduce the VLM rescoring pipeline, show how it boosts precision from 63% to 95% while cutting false positives by 94%, and present the FAISS indexing architecture that makes this practical at billion-vector scale. Finally, we calibrate the threshold between the embedding retrieval and VLM verification stages, optimizing the tradeoff between cost and accuracy.

By the end of this post, you'll have a production-ready architecture for 3D duplicate detection — from ingestion to flagging — with concrete latency budgets, cost analysis, and honest assessments of where it still breaks.

---

## Full Benchmark Results on 3D-DupBench

In [Post 3](../post-3/post.md), we ran the embedding bake-off: four vision encoders × two render modes × five aggregation strategies, evaluated against geometry baselines. Now we present the complete results across all 43 method configurations and all five difficulty tiers.

### The Headline Numbers

The overall winner among multi-view embedding methods is **DINOv2-giant with LFD renders, concat-PCA aggregation over 28 views**, achieving an overall mAP of **0.429**. But let's unpack what that number means — and doesn't mean.

| Method | Overall mAP | P@1 | T1 mAP | T2 mAP | T3 mAP | T4 mAP | T5 mAP |
|--------|-----------|-----|--------|--------|--------|--------|--------|
| **SA + Volume (geometry)** | **0.988** | 1.000 | 0.986 | 0.986 | 0.986 | 0.986 | 0.999 |
| Chamfer Distance | 0.813 | 0.895 | 0.902 | 0.553 | 0.862 | 0.831 | 0.893 |
| Hausdorff Distance | 0.809 | 0.890 | 0.893 | 0.531 | 0.869 | 0.837 | 0.891 |
| **DINOv2-G/LFD/concat-PCA/28v** | **0.429** | 0.950 | 0.471 | 0.386 | 0.485 | 0.363 | 0.429 |
| DINOv2-G/tex/concat-PCA/28v | 0.428 | 0.950 | 0.471 | 0.385 | 0.484 | 0.362 | 0.427 |
| DINOv2-B/LFD/concat-PCA/28v | 0.410 | 0.935 | 0.452 | 0.366 | 0.465 | 0.347 | 0.406 |
| DINOv2-B/tex/concat-PCA/28v | 0.409 | 0.938 | 0.449 | 0.364 | 0.464 | 0.349 | 0.404 |
| CLIP-B/tex/concat-PCA/28v | 0.397 | 0.913 | 0.424 | 0.317 | 0.464 | 0.350 | 0.417 |
| CLIP-L/tex/concat-PCA/28v | 0.394 | 0.931 | 0.426 | 0.347 | 0.443 | 0.328 | 0.416 |
| DINOv2-G/tex/mean/28v | 0.369 | 0.933 | 0.461 | 0.430 | 0.360 | 0.266 | 0.298 |
| DINOv2-G/tex/single | 0.311 | 0.787 | 0.439 | 0.302 | 0.305 | 0.198 | 0.267 |

*Table 1: Top-10 methods on 3D-DupBench. Full results for all 43 configurations are shown in the heatmap below.*

![Full Results Heatmap](../images/post-4b_full_results_table.png)
*Figure 1: Complete mAP heatmap across 21 representative configurations and all difficulty tiers. Green indicates high mAP; red indicates low. Geometry baselines (bottom) dominate overall but require mesh access. Among image-based methods, concat-PCA aggregation consistently wins. Image by author.*

### Surprise Findings

**1. Geometry methods are not uniformly superior.** Surface area + volume ratio achieves near-perfect mAP (0.988) across all tiers because our clone generation pipeline preserves topological properties. But Chamfer and Hausdorff distances struggle badly on T2 (uniform scale + rotation), with mAP dropping to 0.553 and 0.531 respectively. Why? These distances are not inherently scale-invariant. A model scaled by 2× doubles all pairwise distances, dramatically inflating both Chamfer and Hausdorff values. Meanwhile, multi-view embeddings handle T2 effortlessly — DINOv2-G concat-PCA achieves mAP 0.386 on T2, which is *higher* than T4 (0.363) or T5 (0.429), because rendered images are naturally scale-invariant.

**2. Concat-PCA is the clear aggregation winner.** Across every model family and render mode, concatenating per-view embeddings and reducing with PCA outperforms mean pooling, max pooling, and single-view baselines by a significant margin:

![Aggregation Comparison](../images/post-4b_aggregation_comparison.png)
*Figure 2: Aggregation strategy comparison for DINOv2-B/14 with textured renders. Concat-PCA (28 views) achieves mAP 0.409, compared to 0.356 for max pooling and 0.351 for mean pooling. DINOv2-Giant with a single view (red dashed line at mAP 0.311) loses to DINOv2-Base with multi-view concat-PCA — aggregation strategy matters more than model size. Image by author.*

**3. Multi-view aggregation matters more than model size.** DINOv2-Base with concat-PCA over 28 views (mAP 0.409) beats DINOv2-Giant with a single view (mAP 0.311). Going from 1 view to 28 views with mean pooling adds +0.063 mAP; switching from mean to concat-PCA adds another +0.057 mAP. But upgrading from DINOv2-B to DINOv2-G with the same aggregation only adds +0.020 mAP.

![View Count Ablation](../images/post-4b_view_ablation.png)
*Figure 3: mAP vs number of views (mean-pooled, DINOv2-B). Most of the gain comes in the first 8 views; returns diminish beyond 12. Textured and LFD renders perform nearly identically. Image by author.*

**4. LFD and textured renders are essentially interchangeable.** At the concat-PCA level, DINOv2-G/LFD achieves mAP 0.429 vs DINOv2-G/textured at 0.428 — a difference of 0.001. This is surprising: we expected LFD's 20-coefficient shape descriptor renders to lose information relative to full-texture renders. Instead, the geometry-focused LFD signal is equally effective for duplicate detection, even at the hardest tiers. This has a practical implication: LFD renders don't require texture information, making them applicable even when UV maps or materials are missing.

**5. Every method breaks down on T4 (partial removal).** Across all 43 configurations, T4 consistently produces the lowest embedding mAP. Even the best embedding method (DINOv2-G concat-PCA) only achieves mAP 0.363 on T4. This makes sense: removing geometry fundamentally changes what the camera sees from certain angles. Multi-view aggregation helps — it provides redundancy when some views are more affected than others — but it can't fully compensate for missing geometry.

![Embedding vs Geometry Comparison](../images/post-4b_embedding_vs_geometry.png)
*Figure 4: Per-tier mAP comparison between geometry baselines and top embedding methods. Geometry methods (gray) dominate on T3-T5 where they can directly compare mesh structure. Embedding methods (blue) are competitive on T1-T2 and don't require mesh access. Image by author.*

### The Key Insight

Multi-view image embeddings occupy a valuable niche: they work on **any 3D model you can render**, regardless of format, topology, or whether you have access to the mesh. They achieve high precision at the top of the ranking (P@1 > 0.95 for the best methods) and are fast enough for real-time retrieval. But their recall on hard transformations (T3-T5) has a ceiling that geometry methods can exceed — *if* you have mesh access.

For production systems where you need to process arbitrary 3D assets at scale, multi-view embeddings are the right foundation. Their weakness on hard cases is exactly what the VLM rescorer is designed to compensate for.

---

## VLM-as-Verifier: The Gemini Rescorer

### Origin Story

This approach originated from a practical need at Roblox: IP enforcement on the avatar marketplace. When a rights holder reports that their 3D model has been copied and re-uploaded, the enforcement team needs to verify the claim. Embedding similarity can surface candidates, but the final determination — *is this actually the same model?* — requires visual judgment that accounts for artistic interpretation, fair-use modifications, and legitimate coincidence.

Vision-language models are natural fits for this task. They can compare two images, reason about visual similarity at a semantic level, and explain their reasoning. The question is whether we can engineer a pipeline that's both accurate enough for automated decisions and efficient enough for scale.

### The 3-Stage Pipeline

![Rescorer Pipeline](../images/post-4b_rescorer_pipeline.png)
*Figure 5: The 3-stage VLM verification pipeline. Stage 1 uses embedding retrieval to generate candidates. Stage 2 applies a cheap single-view VLM screen to reject obvious non-matches. Stage 3 performs multi-view evidence accumulation for final verification. Image by author.*

**Stage 1 — Embedding Retrieval (Top-K Candidates).** For each query model, we retrieve the top-K nearest neighbors from the FAISS index using the DINOv2-G concat-PCA embeddings. In our evaluation, K=10. This stage runs in under 1 millisecond and provides the candidate set for downstream verification.

**Stage 2 — Single-View VLM Screen.** Each candidate pair is shown to Gemini 2.0 Flash as a single side-by-side composite image: query model on the left, candidate on the right, both rendered from the same camera angle. The model is asked to make a quick binary judgment: *could these be the same 3D model?* This is a cheap filter — one API call per candidate — designed to reject obvious non-matches before expensive multi-view analysis.

**Stage 3 — Multi-View Rescoring.** Candidates that pass Stage 2 are evaluated from 8 different camera angles. For each angle, Gemini sees a side-by-side comparison and provides a verdict with reasoning. Evidence is accumulated across angles: if ≥2 angles show evidence of duplication, the pair is flagged.

### The Evidence Threshold

Why require ≥2 angles? A single matching angle could be coincidence — two different models that happen to look similar from one viewpoint. But if two models look like duplicates from multiple independent viewpoints, that's strong evidence of a shared underlying geometry.

We swept the evidence threshold from 1 to 8 angles:

![Threshold Sweep](../images/post-4b_rescorer_threshold_sweep.png)
*Figure 6: Left — Precision, recall, and F1 vs minimum evidence angles. Precision increases monotonically from 92.9% (ev≥1) to 98.9% (ev≥7), while recall decreases. Right — Per-tier performance at the chosen threshold (ev≥2). Image by author.*

| Threshold | Precision | Recall | F1 | False Positives |
|-----------|-----------|--------|----|-----------------|
| ≥1 angle | 92.9% | 27.9% | 0.429 | 16 |
| **≥2 angles** | **95.0%** | **27.6%** | **0.428** | **11** |
| ≥3 angles | 96.6% | 26.8% | 0.420 | 7 |
| ≥4 angles | 96.9% | 25.2% | 0.400 | 6 |
| ≥5 angles | 97.8% | 24.1% | 0.387 | 4 |

*Table 2: Evidence threshold sweep. We chose ev≥2 as the operating point — it achieves 95% precision with only 11 false positives out of 218 confirmed pairs, providing a practical balance for IP enforcement workloads.*

The threshold of ≥2 angles was chosen because it crosses the 95% precision line — the threshold where automated flagging becomes defensible for IP enforcement. Going higher (≥3 or ≥4) provides marginal precision gains at meaningful recall cost.

### Results: From 63% to 95% Precision

![Rescorer Funnel](../images/post-4b_rescorer_funnel.png)
*Figure 7: The VLM rescorer funnel. Of 500 initial candidates from embedding retrieval, Stage 2 rejects 271 (54.2%) as obvious non-matches. Stage 3 confirms 218 of the remaining 229. Final precision: 95.0%, up from 63.2% in the embedding-only baseline. Image by author.*

The numbers tell a clear story:

- **Embedding-only precision:** 63.2% (316 true positives, 184 false positives out of 500 candidates)
- **With VLM rescorer precision:** 95.0% (207 true positives, 11 false positives out of 218 confirmed)
- **False positive reduction:** 94.0% (from 184 to 11)
- **API calls:** 2,332 calls (500 Stage 2 + 229 × 8 Stage 3)
- **Token usage:** 2.67M total tokens (2.01M prompt, 12.7K completion)

The per-tier breakdown reveals where the rescorer excels and where it struggles:

- **T1 (re-export):** Precision 95.4%, recall 96.7% — nearly perfect, as expected for trivial duplicates
- **T2 (scale + rotation):** Precision 96.4%, recall 36.0% — high precision but the VLM is conservative on scaled models
- **T3 (noise + decimation):** Precision 100%, recall 2.7% — the VLM correctly identifies the rare candidates that survive embedding retrieval but is too strict
- **T4 (partial removal):** Precision N/A, recall 0% — no T4 pairs survive both embedding retrieval *and* VLM verification
- **T5 (adversarial):** Precision 100%, recall 2.7% — same pattern as T3

### Why VLMs Work for This

The VLM rescorer succeeds because the task maps naturally to VLM capabilities:

1. **Side-by-side comparison is a native VLM task.** Comparing two images for similarity is something vision-language models are trained to do. We're not asking the model to do something exotic — we're leveraging a core capability.

2. **Multi-angle evidence accumulation reduces noise.** A single-view comparison might be fooled by coincidental similarity. Eight independent views provide redundancy: the VLM has to see evidence of duplication from multiple perspectives before flagging a pair.

3. **Semantic understanding catches what cosine similarity misses.** Cosine similarity in embedding space treats all dimensions equally. The VLM understands that two chairs with different armrests are different products, even if their embeddings are close. Conversely, it understands that the same chair at different scales is the same model.

4. **The VLM provides explainable decisions.** Each angle's verdict includes reasoning text, creating an audit trail for IP enforcement decisions. This matters in production: when a content creator's model is flagged, the reasoning can be reviewed by a human.

---

## FAISS Production Architecture

### Why FAISS

The embedding retrieval stage needs to scale to the size of real 3D asset libraries. Objaverse contains 10 million models; Roblox's marketplace has tens of millions of assets; a federated search across multiple platforms could require indexing hundreds of millions of embeddings. Brute-force search is O(N) per query — at 1M vectors, that's already 0.6ms on a GPU. At 100M vectors, you're looking at 60ms per query, which is still fast, but you've saturated a GPU for one query. And that's *per query*. A nightly deduplication run across 10,000 new uploads means 10,000 × 60ms = 600 seconds of solid GPU time just for search — before any rendering, embedding, or verification.

FAISS (Facebook AI Similarity Search) provides approximate nearest-neighbor search with sub-linear query time, GPU acceleration, and configurable trade-offs between recall, speed, and memory. It's the standard infrastructure layer for similarity search at scale, used by Meta's content moderation, Google's visual search, and most large-scale recommendation systems. The key principle behind approximate search is that you don't need to check *every* vector — by partitioning the space (via inverted file indices) or compressing the vectors (via quantization), you can trade a small amount of recall for dramatic speedups.

### Index Selection

We benchmarked seven FAISS index configurations on our 768-dimensional DINOv2 embeddings, scaling from 1K to 1M vectors on a Tesla T4 GPU:

![FAISS Benchmark](../images/post-4b_faiss_benchmark.png)
*Figure 8: Query time vs recall@10 for different FAISS index types at 1M vectors. Each point represents a different nprobe setting. IVF-SQ8 (green) hits 97.7% recall at 6.2ms — the recommended configuration for production. Image by author.*

| Index | Query Time (1M) | Recall@10 | Memory (1M) | GPU |
|-------|-----------------|-----------|-------------|-----|
| Flat (exact) | 0.60ms | 100% | 3,072 MB | Yes |
| IVF256-Flat (np=256) | 11.7ms | 100% | 3,072 MB | Yes |
| **IVF256-SQ8 (np=256)** | **6.2ms** | **97.7%** | **776 MB** | **Yes** |
| IVF256-PQ32 (best) | 0.07ms | 12.0% | 40 MB | Yes |
| IVF256-PQ64 (best) | 0.66ms | 16.8% | 72 MB | No |
| IVF1024-PQ32 (best) | 0.04ms | 12.9% | 40 MB | Yes |
| HNSW32 | — | 98% (1K) | ~4× Flat | No |

*Table 3: FAISS benchmark at 1M vectors, 768 dimensions. IVF-SQ8 provides the best balance: 97.7% recall with 4× memory savings over flat search.*

**Our recommendation: IVF256-SQ8.** It achieves 97.7% recall@10 at 1M scale with 6.2ms per query, requires only 776 MB (vs 3,072 MB for flat), supports GPU acceleration, and builds in 8.1 seconds.

**Why not PQ?** Product Quantization (PQ) indices showed surprisingly low recall (12-17%) on our embeddings. This is expected: PQ performs best when the data distribution is well-aligned with the subspace decomposition. DINOv2 embeddings are highly correlated across dimensions, which PQ's independent subspace assumption doesn't capture well. In production, OPQ (Optimized Product Quantization) with a learned rotation matrix would help, but SQ8 is simpler and sufficient for our scale.

**Why not HNSW?** HNSW provides excellent recall at small scales (98% at 1K) but build time grows super-linearly — 102 seconds at 100K vectors. It's also CPU-only, with no GPU acceleration in FAISS. For databases above 1M vectors, HNSW becomes impractical.

### Scaling Projections

![FAISS Scaling](../images/post-4b_faiss_scaling.png)
*Figure 9: Left — Query latency vs database size for key index types. Flat search scales linearly; IVF-based indices scale sub-linearly. Right — Memory usage by index type. SQ8 provides 4× compression over flat storage. Image by author.*

At 1M vectors, IVF-SQ8 uses 776 MB. Extrapolating:

| Scale | Flat Memory | SQ8 Memory | SQ8 Query Time (est.) |
|-------|-------------|------------|----------------------|
| 1M | 3.0 GB | 776 MB | 6.2ms |
| 10M | 30 GB | 7.8 GB | ~15ms |
| 100M | 300 GB | 78 GB | ~40ms |
| 1B | 3 TB | 780 GB | ~100ms |

At billion scale, you'd partition the index across multiple GPUs or use disk-based indices (FAISS supports mmap). The key insight is that SQ8's 4× compression makes billion-scale feasible on a single high-memory GPU (A100 80GB handles ~100M vectors comfortably), while flat indexing would require a cluster.

For our recommended production setup at 1M scale, the index builds in 8.1 seconds (meaning you can rebuild nightly without concern), queries complete in 6.2ms (far below any human-perceptible latency), and the entire index fits in under 1 GB of GPU memory — leaving plenty of room for the DINOv2 model and rendering pipeline to run on the same GPU.

---

## Threshold Calibration

The threshold between "pass" and "refer to VLM" determines the system's operating point. Set it too low, and every query triggers expensive VLM calls. Set it too high, and true duplicates slip through without review.

### Score Distributions

![Threshold Calibration](../images/post-4b_threshold_calibration.png)
*Figure 10: Left — Cosine similarity distributions for true duplicate pairs (colored by tier) and non-duplicate pairs (gray). T1 and T2 pairs cluster near 1.0; T3-T5 pairs overlap significantly with non-duplicates. Right — Precision, recall, and F1 vs similarity threshold, with optimal (τ=0.68) and high-precision (τ=0.82) operating points marked. Image by author.*

The distributions tell a story of clean separation at easy tiers and frustrating overlap at hard ones:

| Tier | Mean Similarity | Median | Min |
|------|----------------|--------|-----|
| T1 (re-export) | 0.997 | 1.000 | 0.671 |
| T2 (scale + rotation) | 0.927 | 0.939 | 0.648 |
| T3 (noise + decimation) | 0.601 | 0.581 | 0.231 |
| T4 (partial removal) | 0.538 | 0.531 | 0.280 |
| T5 (adversarial) | 0.564 | 0.557 | 0.259 |
| Non-duplicates | 0.504 | 0.491 | 0.154 |

*Table 4: Cosine similarity statistics by tier. T3-T5 duplicate pairs have mean similarities barely above the non-duplicate mean (0.504), explaining why embedding-only detection struggles on these tiers.*

### Operating Point Selection

The optimal threshold depends on your use case:

**For IP enforcement (favor precision):** Use τ=0.82. At this threshold, ≥95% of candidates referred to the VLM are true duplicates. The VLM budget is low (few false referrals), and nearly all flagged content is correctly identified. T1 and T2 duplicates are caught at ~100%; T3-T5 are mostly missed at the threshold level but can be caught by reducing the threshold with a higher VLM budget.

**For dataset cleaning (favor recall):** Use τ=0.68. The optimal F1 threshold achieves precision 78.1% and recall 52.9%. More candidates are referred to the VLM, increasing API costs, but you catch more true duplicates. This is appropriate when missing a duplicate has higher cost than false investigation (e.g., training data deduplication where undetected copies inflate benchmarks).

**Our production recommendation:** Use τ=0.82 for the embedding-to-VLM handoff. This sends only high-confidence candidates to the VLM, minimizing API costs. The rescorer then achieves 95% precision on referred candidates, giving a combined system precision above 95% with near-zero false positives for T1/T2 and best-effort detection for T3-T5.

---

## Putting It All Together

### The Full Pipeline

![Production Architecture](../images/post-4b_production_architecture.png)
*Figure 11: Complete production architecture. A 3D model enters from the left, passes through rendering, embedding, and FAISS search. Models above the similarity threshold (τ=0.82) are referred to the Gemini VLM rescorer for multi-angle verification. Image by author.*

The production pipeline processes a query model through six stages:

1. **3D Model Ingestion.** The model is uploaded via API, validated for format compliance, and queued for processing.

2. **Multi-View Rendering.** Using nvdiffrast, we render the model from 28 camera angles (8 ring + 20 dodecahedron). Each render produces a 224×224 image. For textured models, we render with the original materials; for models without textures, we use the LFD (Light Field Descriptor) protocol.

3. **DINOv2 Embedding.** Each of the 28 renders is passed through DINOv2-B/14 (or DINOv2-G/14 for maximum accuracy). The per-view embeddings are concatenated and projected via a pre-fitted PCA to produce a single 768-dimensional descriptor.

4. **FAISS Search.** The descriptor is L2-normalized and queried against the IVF-SQ8 index with nprobe=256, returning the top-20 nearest neighbors with cosine similarities.

5. **Threshold Filter.** Candidates with similarity below τ=0.82 are discarded. Remaining candidates are referred to the VLM rescorer.

6. **VLM Verification.** Gemini 2.0 Flash compares the query model against each candidate from 8 angles. If ≥2 angles show evidence of duplication, the pair is flagged.

### Latency Budget

![Latency Breakdown](../images/post-4b_latency_breakdown.png)
*Figure 12: Left — Per-stage latency on a log scale. Rendering dominates the pipeline. Right — Latency distribution for the clean model path (no VLM invocation). Image by author.*

| Stage | Latency | Notes |
|-------|---------|-------|
| Multi-view rendering | ~56s | nvdiffrast, 28 angles, GPU |
| DINOv2 embedding | ~0.5s | Forward pass + PCA |
| FAISS search | <7ms | IVF-SQ8, nprobe=256, 1M vectors |
| Threshold filter | <0.01ms | Simple comparison |
| VLM rescorer | ~3s/candidate | Gemini 2.0 Flash, 8 angles |
| **Total (clean path)** | **~57s** | **No VLM invocation** |
| **Total (flagged path)** | **~60-120s** | **Depends on # candidates** |

*Table 5: Latency budget breakdown. Rendering is the bottleneck by two orders of magnitude. The embedding + search + threshold path adds less than 1 second.*

The rendering stage dominates — it takes 99% of the clean-path latency. This suggests that optimization efforts should focus on rendering (lower-resolution renders, fewer views, or mesh-based rendering shortcuts) rather than on the embedding or search stages. Pre-computing renders for the index (amortized cost) means the per-query rendering cost only applies to the new model being checked.

### Cost Analysis

For a system processing 10,000 new models per day:

| Component | Cost | Notes |
|-----------|------|-------|
| GPU rendering + embedding | ~$50/day | T4 instance, fully utilized |
| FAISS index storage | ~$0.80/day | 1M vectors × 776 MB, S3 |
| VLM API calls | ~$5-15/day | Depends on flagging rate |
| **Total** | **~$55-65/day** | **~$0.006 per model** |

The VLM cost is highly variable: if 5% of models trigger the VLM (typical for a clean marketplace), that's 500 models × ~5 candidates × 9 API calls = ~22,500 calls/day. At Gemini 2.0 Flash pricing (~$0.0003 per call including tokens), that's roughly $7/day.

### When to Use Each Stage Alone

Not every deployment needs the full pipeline:

- **Embedding retrieval only** is sufficient for exploratory deduplication, where false positives are acceptable and you just want to surface candidates for human review. Precision: ~63%, zero API cost.

- **Embedding + threshold filter** works for dataset cleaning where you can afford to miss hard cases (T3-T5) but want automated decisions for easy duplicates (T1-T2). Precision: depends on threshold, zero API cost.

- **Full pipeline with VLM** is required for IP enforcement, automated takedowns, or any setting where false positives have legal or reputational consequences. Precision: ~95%, moderate API cost.

---

## Open Challenges and Future Work

### T4 and T5 Remain Unsolved

The hardest tiers — partial geometry removal (T4) and adversarial topology changes (T5) — remain below 40% mAP for all embedding methods and below 3% recall for the VLM rescorer. This is a fundamental limitation: when the geometry changes significantly, image-based features can't reliably match what they can't see.

This isn't a failure of the approach — it's a statement about the information-theoretic limits of image-based comparison. If someone removes the legs from a chair model and replaces them with a different design, no number of camera angles will show the original legs. The information is gone. Similarly, adversarial topology changes (remeshing a smooth surface into a low-poly approximation, or vice versa) alter the visual appearance at every scale, making the task genuinely ambiguous even for human annotators.

Solving this likely requires:

- **Multi-modal features** that combine geometry (point clouds, mesh topology) with appearance (renders, textures) and metadata (vertex counts, bounding box ratios). A model that has been partially modified still shares most of its mesh graph structure with the original — information that images discard.
- **Part-level matching** that identifies shared sub-components rather than requiring whole-model similarity. Two chairs that share identical seats but different legs should still be flagged as potential derivatives, even if their overall embeddings diverge.
- **Contrastive fine-tuning** of the embedding model on the specific distribution of transformations expected in production. Our current pipeline uses off-the-shelf DINOv2 features trained on natural images. Fine-tuning on 3D render pairs with known relationships could significantly improve hard-tier recall.
- **Graph neural networks on mesh topology** that learn structural fingerprints invariant to surface perturbations but sensitive to the underlying connectivity pattern.

### Beyond Meshes

Our pipeline assumes you can render the model — which requires a mesh (or at least a point cloud + normals). But the 3D field is rapidly moving toward implicit representations:

- **NeRF** and **3D Gaussian Splatting** models represent geometry as neural fields or point-based primitives that can be rendered but don't have traditional meshes
- **SDF (Signed Distance Function)** representations encode geometry as level sets of learned functions
- **Tri-plane** and **point cloud** representations used by generative 3D models like Point-E, Shap-E, and LRM produce outputs that may not have clean mesh exports

Detecting duplicates among these representations will require either rendering to images (which our pipeline handles — any representation that can produce images can enter our pipeline) or developing feature extractors that work directly on the implicit representation. The rendering approach is more general but adds latency; direct feature extraction would be faster but requires representation-specific encoders.

### Active Learning

The VLM rescorer generates labeled pairs with every run — each pair gets a verdict and reasoning. This creates a natural active learning loop: use VLM verdicts to fine-tune the embedding model, improving recall on hard cases and reducing the number of candidates that need VLM verification. Each iteration makes the cheap retriever smarter, reducing reliance on the expensive VLM oracle. This is a classic retrieval system pattern — the same principle behind RLHF for language models, where expensive human judgments gradually train cheaper proxy models.

### Cross-Domain Transfer

Our experiments used Objaverse-LVIS models (everyday objects). How well does this transfer to:
- **Game assets** (optimized meshes, specific art styles, LOD variants)?
- **CAD models** (precise engineering parts with parametric variations)?
- **Scanned objects** (noisy point clouds from photogrammetry, scan artifacts)?
- **AI-generated 3D models** (outputs from text-to-3D or image-to-3D systems that may share training data or prompts)?

Early evidence suggests DINOv2's general visual features transfer well across domains, but the optimal threshold and VLM prompts may need calibration per domain. The benchmark itself is extensible: adding a new tier of domain-specific transformations is straightforward, and the evaluation infrastructure is fully automated.

---

## Series Conclusion

This is the final post in a five-part series on building a 3D duplicate detection pipeline from scratch. Let's trace the journey:

**[Post 1 — Why 3D Duplicates Matter](../post-1/post.md)** introduced the problem: as 3D datasets grow to millions of models, near-duplicates contaminate training data, inflate benchmarks, and enable IP theft. We surveyed existing approaches and found no standardized solution.

**[Post 2 — Multi-View Rendering for 3D Models](../post-2/post.md)** established the rendering infrastructure: 28 camera angles (8 ring + 20 dodecahedron), both textured and LFD render modes, using nvdiffrast for GPU-accelerated differentiable rendering.

**[Post 3 — The Embedding Bake-Off](../post-3/post.md)** compared four vision encoders (DINOv2-B, DINOv2-G, CLIP-B, CLIP-L) across five aggregation strategies, finding that DINOv2 + concat-PCA + 28 views provides the best general-purpose 3D fingerprint.

**[Post 4a — 3D-DupBench and the ModelNet40 Audit](../post-4a/post.md)** built the first standardized benchmark for 3D duplicate detection (191 source models, 2,865 clones, 5 difficulty tiers) and uncovered 5,784 near-duplicate pairs in ModelNet40 — inflating reported accuracy by up to 7.7 percentage points.

**Post 4b — This post** completed the pipeline: full benchmark results showing DINOv2-G concat-PCA as the embedding winner (mAP 0.429), a Gemini-powered VLM rescorer that boosts precision to 95%, FAISS IVF-SQ8 indexing for sub-10ms search at million-vector scale, and threshold calibration for production deployment.

### Key Contributions

1. **3D-DupBench:** The first standardized, publicly available benchmark for 3D near-duplicate detection, with graduated difficulty tiers and controlled ground truth. [[GitHub](https://github.com/pkhw2023-ship-it/3d-dedup)]

2. **Multi-view embedding pipeline:** DINOv2 + concat-PCA over 28 views provides a format-agnostic 3D fingerprint that achieves P@1 > 0.95 without mesh access.

3. **VLM-as-verifier pattern:** A general-purpose pattern for using vision-language models as precision boosters in retrieval pipelines. The 3-stage architecture (retrieve → screen → verify) applies to any domain where embedding similarity has high recall but insufficient precision.

4. **ModelNet40 audit:** A community service revealing systematic train-test leakage in the most-cited 3D classification benchmark, with concrete numbers on accuracy inflation per category.

5. **Production-ready architecture:** Concrete latency budgets, cost analysis, and index recommendations for deploying 3D duplicate detection at scale.

All code, data, and benchmark materials are available at [github.com/pkhw2023-ship-it/3d-dedup](https://github.com/pkhw2023-ship-it/3d-dedup). 3D-DupBench is released under CC-BY-4.0.

---

*This concludes the series "Building a 3D Duplicate Detection Pipeline." If you've followed along this far — from the first post's motivation to this post's production architecture — thank you. The 3D data quality problem is far from solved, but I hope this work provides a concrete starting point for anyone building deduplication into their 3D asset pipeline.*

*If you work on 3D datasets or marketplaces and have war stories about duplicate detection, I'd love to hear from you. The hard tiers — T4 and T5 — remain open challenges, and I suspect the solutions will come from combining the image-based approach we've built here with direct geometry reasoning. That's a post for another series.*

*— Harish Wajjala*
