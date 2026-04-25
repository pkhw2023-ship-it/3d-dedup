---
title: "Which Vision Model Sees 3D Shapes Best? A DINOv2, CLIP, and Geometry Bake-Off"
subtitle: "Comparing 4 foundation models, 6 aggregation strategies, and classical geometry baselines for 3D near-duplicate retrieval"
author: "Harish Wajjala"
date: 2026-04-24
series: "Detecting 3D Near-Duplicates at Scale"
part: 3 of 5
tags: ["3D Computer Vision", "Machine Learning", "Near-Duplicate Detection", "DINOv2", "Multi-View Rendering"]
estimated_read_time: "19 min"
---

# Which Vision Model Sees 3D Shapes Best? A DINOv2, CLIP, and Geometry Bake-Off

*Comparing 4 foundation models, 6 aggregation strategies, and classical geometry baselines for 3D near-duplicate retrieval*

---

## The Paradox: 2D Models That Understand 3D

Here's a claim that should bother you: a model that has never seen a 3D mesh — never processed a vertex buffer, never computed a face normal, never traversed a scene graph — can understand 3D shape similarity better than algorithms specifically designed for geometry comparison.

DINOv2 was trained on 142 million curated photographs of the real world. CLIP was trained on 400 million image-text pairs scraped from the internet. Neither model knows what a triangle mesh is. Neither has a concept of vertex count, watertightness, or UV coordinates. And yet, when you render a 3D model from multiple viewpoints and feed those images through these models, the resulting embeddings capture geometric similarity with remarkable fidelity.

Why? The answer is embarrassingly simple: **scale beats specialization**. The largest 3D shape datasets contain hundreds of thousands of models. The largest 2D image datasets contain *billions*. A model trained on billions of diverse images learns visual structure — edges, symmetries, part relationships, spatial configurations — that transfers directly to rendered views of 3D objects. No 3D-native model comes close to this scale of pre-training.

But "it works well" isn't an engineering answer. *How* well? Which model? With how many views? Does the aggregation strategy matter? When should you skip embeddings entirely and just compare geometry?

This post answers those questions with numbers. I evaluated **4 foundation models** (DINOv2-base, DINOv2-giant, CLIP ViT-L/14, CLIP ViT-B/32) across **5 aggregation strategies** (single-view, mean-8, mean-28, max-28, concat+PCA) with **2 rendering modes** (textured, silhouette) — that's 40 embedding configurations — plus **3 classical geometry baselines** (Chamfer distance, Hausdorff distance, surface area + volume). Everything is evaluated on the [3D-DupBench](/post-1) dataset: 191 source models from Objaverse with 2,865 synthetic clones across 5 difficulty tiers.

The headline result: **DINOv2-giant with concat+PCA aggregation over 28 LFD views** achieves the highest mAP of **0.429**, but the margins between configurations tell a more nuanced story. The choice of aggregation strategy often matters more than the choice of model.

*This is Part 3 of 5 in the series "[Detecting 3D Near-Duplicates at Scale](/post-1)." [Part 1](/post-1) introduced the pipeline and benchmark. [Part 2](/post-2) covered multi-view rendering. Part 4a will address benchmark design, and Part 4b will cover search and threshold tuning at scale.*

---

## The Contenders

### Vision Foundation Models

I tested four models spanning two architectural families: DINOv2 (self-supervised) and CLIP (vision-language). Each brings a different inductive bias to the problem.

**DINOv2-base (ViT-B/14)** is a Vision Transformer with 86M parameters producing 768-dimensional embeddings. Trained via self-supervised distillation on LVD-142M — a curated dataset of 142 million images — DINOv2 learns to represent visual structure without any labels. Its training objective (self-distillation with no labels) forces it to capture geometric and spatial features rather than semantic categories. This makes it a natural fit for shape similarity: two chairs that *look* alike should embed nearby, regardless of whether they're both labeled "chair."

**DINOv2-giant (ViT-g/14)** is the scaled-up sibling: 1.1 billion parameters, 1536-dimensional embeddings, same training data. This is the "does bigger help?" test. With 13× more parameters, DINOv2-G should capture finer-grained visual features — but at 3× the inference cost per image.

**CLIP ViT-L/14** takes a fundamentally different approach. Trained on 400 million image-text pairs via contrastive learning, CLIP aligns images with natural language descriptions. Its 768-dimensional embeddings are optimized for matching images to captions like "a red wooden chair" — an objective that prioritizes semantic category over geometric detail. A CLIP embedding of a chair tells you it's a chair; a DINOv2 embedding tells you what *kind* of chair it is.

**CLIP ViT-B/32** is the efficiency-oriented CLIP variant: 512-dimensional embeddings (projected to 768 for comparison), 4× faster inference than ViT-L/14 due to the larger 32×32 patch size. This is the "good enough?" test — can you trade model quality for speed without losing too much retrieval accuracy?

| Model | Parameters | Embedding Dim | Patch Size | Training Data | Training Objective |
|-------|-----------|--------------|-----------|--------------|-------------------|
| DINOv2-base | 86M | 768 | 14×14 | LVD-142M | Self-distillation |
| DINOv2-giant | 1.1B | 1536 | 14×14 | LVD-142M | Self-distillation |
| CLIP ViT-L/14 | 304M | 768 | 14×14 | WIT-400M | Image-text contrastive |
| CLIP ViT-B/32 | 88M | 512→768 | 32×32 | WIT-400M | Image-text contrastive |

Extracting embeddings is straightforward — a single forward pass per image:

```python
import torch
from transformers import AutoModel, AutoProcessor

# Load DINOv2
model = AutoModel.from_pretrained("facebook/dinov2-base")
processor = AutoProcessor.from_pretrained("facebook/dinov2-base")

def extract_embedding(image):
    """Extract a 768-dim embedding from a single image."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Use CLS token as the image-level representation
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
```

### Geometry Baselines

To contextualize the embedding results, I also evaluated three classical geometry methods that operate directly on the mesh:

**Chamfer Distance** computes the average bidirectional nearest-point distance between two point clouds. For each point in cloud A, find the closest point in cloud B, average those distances, then repeat in reverse and average. It captures overall shape similarity but is sensitive to sampling density.

**Hausdorff Distance** takes the *maximum* nearest-point distance rather than the average — measuring the worst-case deviation between two shapes. It's more sensitive to outlier points and partial geometry changes than Chamfer.

**Surface Area + Volume Ratio** is a simple geometric signature: compare the surface area and volume of two meshes as ratios. If both ratios are close to 1.0, the meshes are geometrically similar. This is fast (O(1) per comparison after preprocessing) but captures only the coarsest shape information.

All geometry baselines require loading and processing each mesh: normalizing to a unit sphere, sampling 2,000 surface points (for Chamfer/Hausdorff), and computing mesh properties. This is far more expensive per pair than a cosine similarity between pre-computed embeddings.

---

## Multi-View Aggregation: How to Combine 28 Views Into One Vector

Rendering 28 views per model is pointless if you can't aggregate 28 per-view embeddings into a single searchable vector. I tested five strategies.

**Single View (baseline):** Use only view 0 — the front-facing horizontal view at 0° azimuth, 30° elevation. This discards 27/28 of the visual information but requires no aggregation and produces the fastest pipeline. It's the "how bad is lazy?" test.

**Mean Pool (8 views):** Average the embeddings from the 8 horizontal-ring views (equally spaced at 30° elevation). This captures the full horizontal profile while keeping computation at 8 forward passes.

**Mean Pool (28 views):** Average all 28 view embeddings — 8 horizontal ring + 20 dodecahedron vertices for near-uniform spherical coverage. The simplest possible full-coverage aggregation.

**Max Pool (28 views):** Take the element-wise maximum across all 28 embeddings. The intuition is that max pooling should capture the most "activated" features from any view, emphasizing distinctive features over average appearance.

**Concat+PCA (28 views):** Concatenate all 28 embeddings into a single 28×D vector (e.g., 28×768 = 21,504 dimensions for DINOv2-base), then reduce to 768 dimensions via PCA. This preserves the most information by allowing PCA to learn which cross-view feature combinations matter.

```python
import numpy as np
from sklearn.decomposition import PCA

def aggregate_views(per_view_embeddings, strategy="mean"):
    """Aggregate per-view embeddings into a single vector.

    Args:
        per_view_embeddings: (n_views, dim) array
        strategy: 'mean', 'max', or 'concat_pca'

    Returns: (out_dim,) L2-normalized vector
    """
    if strategy == "mean":
        agg = per_view_embeddings.mean(axis=0)
    elif strategy == "max":
        agg = per_view_embeddings.max(axis=0)
    elif strategy == "concat_pca":
        # Concatenate then reduce — needs to be fit on the corpus
        flat = per_view_embeddings.reshape(1, -1)  # (1, n_views * dim)
        return flat  # PCA fit separately on full corpus
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # L2 normalize for cosine similarity
    agg = agg / np.linalg.norm(agg)
    return agg
```

### The Key Finding: PCA Dominates, Mean Is the Runner-Up

The aggregation comparison reveals a clear hierarchy:

![Aggregation comparison across models](../images/post-3_03_aggregation_comparison.png)
*Figure 1: mAP by aggregation strategy for each model. Concat+PCA consistently outperforms other methods, with mean pooling as the strongest simple baseline.*

**Concat+PCA wins across all four models.** For DINOv2-base (textured), PCA reaches 0.409 mAP vs. 0.352 for mean-28 — a 16% improvement. For CLIP-B/32, the gap is even larger: 0.397 vs. 0.335 (18% improvement). PCA's advantage comes from preserving view-specific features that mean pooling averages away. When a chair looks like a table from one angle but distinctly chair-like from another, PCA can learn to weight those discriminative views.

**Mean and max pooling perform similarly, with no consistent winner.** Max pooling for DINOv2-base (textured) achieves 0.356 vs. 0.352 for mean-28 — a slight edge for max — while for CLIP-L, mean actually wins (0.334 vs. 0.332). For DINOv2-giant, they're tied at 0.369. Neither approach offers a meaningful advantage over the other, and both trail PCA by a wide margin. The intuition that max pooling captures "the best" features from each view doesn't hold in practice — it also captures noise. If one view produces an unusually high activation in some feature dimension due to a rendering artifact or an unusual viewpoint, max pooling propagates that noise into the final embedding.

**The jump from 1 view to 8 views is massive; 8 to 28 is marginal.** Single-view DINOv2-base gets 0.289 mAP; adding 7 more horizontal views (mean-8) jumps to 0.345 — a 19% improvement. Going from 8 to 28 views (mean-28) adds only 0.007 more. The practical implication: if inference speed matters, 8 views with mean pooling captures most of the multi-view benefit.

| Strategy | DINOv2-B | DINOv2-G | CLIP-L/14 | CLIP-B/32 |
|----------|---------|---------|----------|----------|
| Single View | 0.289 | 0.311 | 0.246 | 0.226 |
| Mean (8 views) | 0.345 | 0.363 | 0.323 | 0.320 |
| Mean (28 views) | 0.351 | 0.369 | 0.334 | 0.335 |
| Max (28 views) | 0.356 | 0.369 | 0.332 | 0.328 |
| Concat+PCA (28) | **0.409** | **0.428** | **0.394** | **0.397** |

*Table 1: mAP by aggregation strategy (textured renders). Best per model in bold.*

---

## Geometry vs. Embeddings: When Does Each Win?

The geometry baselines tell a surprising story — but one that requires a critical caveat.

On the 3D-DupBench clone retrieval task, the geometry baselines achieve very high overall mAP: **Chamfer distance at 0.813**, **Hausdorff at 0.809**, and **SA+Volume at 0.989**. That last number — surface area plus volume achieving near-perfect retrieval — seems too good to be true. And in a sense, it is.

**The sparse evaluation caveat.** Because computing pairwise geometry distances is O(N²) expensive, the evaluation computes distances only within each source group (a source model and its clones) plus a small random sample of cross-group pairs (20 negative samples per model). This means the geometry baselines are evaluated against a far easier negative pool than the embedding methods, which compute a full 3,056 × 3,056 cosine similarity matrix. The SA+Volume metric is particularly inflated: it has so few finite cross-group pairs that nearly every finite comparison is a true positive. **Don't compare these absolute numbers directly to the embedding mAP values.**

What *is* meaningful is the **per-tier breakdown**, where within-group comparisons are the same across all methods:

![Geometry vs. embedding comparison](../images/post-3_06_geometry_vs_embedding.png)
*Figure 2: Geometry baselines achieve high metrics on sparse pairwise evaluation, but the comparison to embeddings requires caution due to different negative pool sizes.*

**T1 (Trivial — re-export):** Geometry baselines hit near-perfect scores (Chamfer mAP: 0.902, Hausdorff: 0.893). Re-exported models have identical geometry, so point cloud distances are essentially zero. Embeddings also perform well here (DINOv2-G PCA: 0.471), but with the full N×N negative pool, they face much harder distractor models.

**T2 (Easy — scale + rotation):** This is where geometry struggles most. Chamfer drops to 0.553, Hausdorff to 0.531. Scale and rotation changes, even after unit-sphere normalization, shift the point cloud enough to confuse nearest-neighbor distance metrics. Embeddings handle this better (DINOv2-G PCA: 0.385) because rendered views are largely invariant to scale and rotation — the rendering pipeline normalizes these away.

**T3–T5 (Medium through Adversarial):** Both Chamfer (0.862) and Hausdorff (0.869) rebound at T3, suggesting that noise and decimation in our T3 clones don't disrupt the point cloud as much as scale/rotation changes. However, these numbers are measured against the sparse negative pool, making direct comparison difficult.

The practical takeaway: **geometry baselines are a useful first-pass filter for trivial duplicates** (exact copies, re-exports), but they can't serve as the primary detection method at scale because they don't scale to millions of pairwise comparisons.

---

## The Results: 40 Configurations, Ranked

This is the core of the bake-off: every model × rendering mode × aggregation strategy, evaluated on 3D-DupBench.

### Overall Ranking

![Method comparison bar chart](../images/post-3_01_method_comparison.png)
*Figure 3: Overall mAP for the best configuration per model family, plus geometry baselines. DINOv2-giant leads among embeddings, but geometry baselines (evaluated on sparse pairs) show the potential of direct comparison for small-scale tasks.*

The top-10 embedding configurations by overall mAP:

| Rank | Method | mAP | P@1 | P@5 |
|------|--------|-----|-----|-----|
| 1 | DINOv2-G + LFD + PCA-28 | **0.429** | 0.950 | 0.837 |
| 2 | DINOv2-G + Tex + PCA-28 | 0.428 | 0.950 | 0.835 |
| 3 | DINOv2-B + LFD + PCA-28 | 0.410 | 0.935 | 0.808 |
| 4 | DINOv2-B + Tex + PCA-28 | 0.409 | 0.938 | 0.806 |
| 5 | CLIP-B/32 + Tex + PCA-28 | 0.397 | 0.913 | 0.769 |
| 6 | CLIP-L/14 + Tex + PCA-28 | 0.394 | 0.931 | 0.801 |
| 7 | CLIP-B/32 + LFD + PCA-28 | 0.392 | 0.906 | 0.763 |
| 8 | CLIP-L/14 + LFD + PCA-28 | 0.389 | 0.930 | 0.795 |
| 9 | DINOv2-G + LFD + Mean-28 | 0.370 | 0.928 | 0.769 |
| 10 | DINOv2-G + LFD + Max-28 | 0.369 | 0.940 | 0.783 |

*Table 2: Top-10 embedding configurations by mAP. All use 28 views; PCA aggregation dominates the top.*

Several patterns emerge:

**DINOv2 > CLIP, consistently.** The best DINOv2-G configuration (0.429) beats the best CLIP configuration (0.397) by 8%. Even the smaller DINOv2-B (0.410) outperforms both CLIP variants. Self-supervised visual features transfer better to shape similarity than vision-language features.

**Giant > Base, but marginally.** DINOv2-G beats DINOv2-B by ~0.02 mAP across all aggregation strategies. For 13× more parameters (1.1B vs. 86M) and 3× more inference time, that's a modest improvement. Unless you're optimizing for the last percentage point, DINOv2-base is the better cost-performance tradeoff.

**CLIP-B/32 ≈ CLIP-L/14.** The smaller CLIP variant actually matches or slightly beats the larger one: CLIP-B/32 PCA-28 (0.397) vs. CLIP-L/14 PCA-28 (0.394). The larger patch size (32 vs. 14) loses spatial resolution but CLIP's representations seem to be robust to this for shape similarity tasks.

**P@1 is remarkably high across the board.** Even the worst embedding method (CLIP-B/32 single view, LFD) achieves P@1 = 0.629 — meaning the top-1 retrieval result is a true clone 64.6% of the time. The best methods exceed P@1 = 0.950. This means the top retrieved result is almost always correct; the challenge is recall — finding *all* the clones.

### The Per-Tier Breakdown: Where Difficulty Exposes Model Differences

![Per-tier heatmap](../images/post-3_02_tier_heatmap.png)
*Figure 4: Per-tier mAP heatmap. Warmer colors indicate higher retrieval accuracy. The difficulty gradient from T1 to T5 clearly degrades all methods, but PCA-based aggregation maintains the strongest performance at every tier.*

The per-tier breakdown reveals how clone difficulty affects each method:

| Method | T1 (Trivial) | T2 (Easy) | T3 (Medium) | T4 (Hard) | T5 (Adversarial) |
|--------|-------------|----------|------------|---------|-----------------|
| DINOv2-G PCA-28 (lfd) | 0.471 | 0.386 | 0.485 | 0.363 | 0.429 |
| DINOv2-G PCA-28 (tex) | 0.471 | 0.385 | 0.484 | 0.362 | 0.427 |
| DINOv2-B PCA-28 (lfd) | 0.452 | 0.366 | 0.465 | 0.347 | 0.406 |
| DINOv2-G Mean-28 (tex) | 0.462 | 0.430 | 0.360 | 0.266 | 0.298 |
| DINOv2-B Mean-28 (tex) | 0.450 | 0.418 | 0.334 | 0.248 | 0.274 |

*Table 3: Per-tier mAP for the top configurations. T3 (noise + decimation) is often easier than T2 (scale + rotation) for PCA methods.*

**T1 (Trivial — re-export):** All methods score 0.42–0.47 mAP. Re-exported models render nearly identically, so every model finds its clones easily. The ceiling here is set by the difficulty of the negative pool (3,056 models with many visually similar shapes).

**T2 (Easy — scale + rotation):** A surprising dip. DINOv2-G PCA drops from 0.471 (T1) to 0.386 (T2). Meanwhile, DINOv2-G Mean-28 actually improves slightly to 0.430. Scale and rotation changes affect PCA and mean aggregation differently: PCA's higher-dimensional learned features are more sensitive to view-angle shifts that rotation induces.

**T3 (Medium — noise + decimation):** PCA methods rebound: DINOv2-G PCA reaches 0.485 — the *highest* per-tier score for any embedding. Adding noise to vertices and decimating faces doesn't change the overall visual appearance much when viewed through a rendered image. The silhouette and major structure remain intact.

**T4 (Hard — partial removal + noise):** The steepest drop. DINOv2-G PCA falls to 0.363. Removing parts of the mesh fundamentally changes the visual appearance from some viewpoints. A chair with one leg removed looks noticeably different in side views but similar from above.

**T5 (Adversarial — remesh + deform):** Moderate performance at 0.429 for DINOv2-G PCA. Remeshing changes topology without changing appearance, so embeddings are relatively robust. The deformation component is what hurts — but if the deformation preserves the overall silhouette, the embedding still captures similarity.

### Textured vs. LFD: Does Color Help?

![Textured vs LFD comparison](../images/post-3_04_textured_vs_lfd.png)
*Figure 5: Textured vs. LFD (Light Field Descriptor / white plastic) rendering comparison with mean-28 aggregation.*

Textured renders preserve material and color information. LFD renders strip everything to white plastic, isolating pure geometry. Which matters more for duplicate detection?

The answer: **it barely matters**. DINOv2-G with PCA-28 achieves 0.428 (textured) vs. 0.429 (LFD) — the LFD version is actually marginally *better*. For DINOv2-B with mean-28: 0.351 (textured) vs. 0.353 (LFD). The differences are within noise.

This makes intuitive sense for our task. Near-duplicate clones share the same geometry; they might differ in texture application but the overall shape is the same. LFD renders eliminate a source of variation (color/texture differences) that isn't relevant to geometric similarity. In a production pipeline with diverse textures, LFD might gain a clearer advantage.

The practical guidance: **use LFD renders for pure duplicate detection** (marginally better, faster to render since no texture loading required). Use textured renders if you also need to match texture-modified duplicates or want embeddings useful for broader visual similarity tasks.

### View Count Ablation: How Many Views Do You Actually Need?

![View count ablation](../images/post-3_05_view_ablation.png)
*Figure 6: mAP vs. number of views for mean-pool aggregation. The biggest jump is from 1 to 4 views; returns diminish sharply after 8.*

The view count ablation quantifies the diminishing returns of more views:

| Views | DINOv2-B (tex) | DINOv2-G (tex) | CLIP-L (tex) | CLIP-B (tex) |
|-------|---------------|---------------|-------------|-------------|
| 1 | 0.289 | 0.311 | 0.246 | 0.226 |
| 4 | 0.336 | 0.354 | 0.310 | 0.299 |
| 8 | 0.345 | 0.363 | 0.323 | 0.320 |
| 12 | 0.347 | 0.365 | 0.324 | 0.320 |
| 20 | 0.349 | 0.367 | 0.329 | 0.327 |
| 28 | 0.352 | 0.369 | 0.334 | 0.335 |

*Table 4: mAP by view count (mean pooling, textured renders). The 1→4 jump is worth +0.047 for DINOv2-B; the 8→28 jump adds only +0.007.*

The data tells a clear story:

- **1 → 4 views:** +16% relative improvement (DINOv2-B). The first three additional views add viewpoints that resolve front-back ambiguity and capture profile information.
- **4 → 8 views:** +3% relative. The full horizontal ring adds some value but with diminishing returns.
- **8 → 28 views:** +2% relative. Adding 20 dodecahedron views for top/bottom/oblique coverage barely moves the needle for mean pooling.

The efficiency sweet spot is **8 views with mean pooling**: you capture 98% of the full-coverage benefit at 29% of the rendering cost. This is the configuration I'd recommend for production systems where inference latency matters.

---

## Embedding Space Visualization

What does the embedding space actually look like? UMAP projections reveal the structure that similarity metrics operate on.

![UMAP visualization](../images/post-3_07_umap_visualization.png)
*Figure 7: UMAP projection of DINOv2-base mean-28 embeddings. Left: colored by clone tier — sources (black) cluster tightly with their T1/T2 clones (green/yellow) but T4/T5 clones (orange/red) drift away. Right: colored by category — clear semantic clustering emerges, with airplanes, cars, and guitars forming tight clusters while chairs and tables overlap.*

**The tier panel (left)** shows exactly the difficulty gradient we measured quantitatively. Source models (black) form cluster centers. T1 and T2 clones (green shades) sit right on top of their sources — trivial duplicates are trivially close in embedding space. T4 and T5 clones (orange, red) drift further, creating halos around each cluster. Some T5 clones have drifted so far they're closer to *other* categories than to their source.

**The category panel (right)** reveals something important about the embedding space: it's organized primarily by semantic category, not by individual identity. Airplanes form a tight cluster. Cars form another. Chairs sprawl across a large region (high intra-class variation — there are many kinds of chairs). This means that duplicate retrieval within a category is the hard problem: your duplicate of a chair is surrounded by hundreds of other chairs, not by random objects from other categories.

---

## Precision-Recall: The Threshold Tradeoff

![Precision-recall curves](../images/post-3_09_precision_recall.png)
*Figure 8: Precision-recall curves for the top 3 embedding methods. Higher curves indicate better performance; all three methods show a smooth precision-recall tradeoff without abrupt cliffs.*

The precision-recall curves show how threshold selection affects the precision/recall tradeoff. All three top methods (DINOv2-G PCA, DINOv2-B PCA, CLIP-B PCA) produce smooth curves without the abrupt precision cliffs that would indicate embedding space discontinuities.

At a precision-oriented operating point (0.80 precision), DINOv2-G PCA achieves ~0.30 recall — meaning it finds 30% of all clones while maintaining 80% precision. At a recall-oriented operating point (0.80 recall), precision drops to ~0.45. The choice of operating point depends on the application: a marketplace enforcing IP requires high precision (low false-positive rate), while a dataset deduplication pipeline can tolerate lower precision if it catches more duplicates.

---

## Summary Table

![Summary results table](../images/post-3_08_summary_table.png)
*Figure 9: Summary results across all key configurations. The heatmap coloring highlights the performance gradient from geometry baselines (high but sparsely evaluated) through PCA-based methods (best among embeddings) down to single-view baselines.*

---

## Statistical Significance

Are the differences between methods statistically significant? I used the Wilcoxon signed-rank test on per-query AP values from the 3,056-model evaluation.

**DINOv2-G PCA vs. DINOv2-B PCA:** Δ mAP = +0.020, p < 0.001 — statistically significant, but the practical difference is small.

**DINOv2-G PCA vs. CLIP-B PCA:** Δ mAP = +0.033, p < 0.001 — significant and practically meaningful. DINOv2's self-supervised features consistently outperform CLIP's vision-language features for geometric similarity.

**PCA vs. Mean (DINOv2-B):** Δ mAP = +0.057, p < 0.001 — the aggregation strategy effect is *larger* than the model choice effect. Upgrading from mean pooling to PCA matters more than upgrading from DINOv2-B to DINOv2-G.

**Mean-28 vs. Mean-8 (DINOv2-B):** Δ mAP = +0.007, p = 0.02 — barely significant, confirming that the marginal value of views 9–28 is minimal for mean aggregation.

---

## Practical Recommendations

Based on the full bake-off, here's my decision framework:

### If You Want Maximum Accuracy

Use **DINOv2-giant + concat+PCA over 28 LFD views** (mAP: 0.429, P@1: 0.950). This is the best configuration we found, but it requires:
- 1.1B parameter model (needs GPU for reasonable throughput)
- 28 renders per model (0.5–2 seconds per model with GPU rendering)
- PCA fitting on the full corpus before indexing
- 1536-dimensional per-view embeddings → 768 after PCA

### If You Want Speed

Use **CLIP-B/32 + mean pool over 8 views** (mAP: 0.320, P@1: 0.857). This trades 25% of the accuracy for:
- 88M parameter model (runs on CPU at ~50ms per image)
- 8 renders instead of 28 (71% less rendering cost)
- No PCA fitting needed — just average and normalize
- 512-dimensional embeddings (2× less storage)

### If You Want the Best Tradeoff

Use **DINOv2-base + concat+PCA over 28 LFD views** (mAP: 0.410, P@1: 0.935). This is 95% of the best accuracy at a fraction of the cost:
- 86M parameters (13× smaller than DINOv2-G, same quality for 95% of cases)
- Full 28-view coverage with PCA captures the information gain
- 768-dimensional embeddings after PCA

### When to Add Geometry Baselines

Use Chamfer distance as a **first-pass filter for trivial duplicates** (T1). If two models have Chamfer distance < 0.01 after unit-sphere normalization, they're almost certainly exact copies or re-exports. This catches the low-hanging fruit before running the more expensive embedding pipeline. But don't rely on geometry alone — it fails on scale/rotation changes (T2) and can't scale to millions of pairwise comparisons.

### How to Combine Approaches

In the production pipeline I built (covered in [Post 4b](/post-4b)):

1. **Geometry pre-filter:** Compute surface area + volume for each model (O(1) per comparison). Flag pairs with ratios > 0.99 as likely trivial duplicates.
2. **Embedding retrieval:** Use DINOv2-base + PCA-28 embeddings in a FAISS index for approximate nearest-neighbor search. Retrieve top-20 candidates per query.
3. **Threshold + verification:** Apply a cosine similarity threshold tuned for your precision/recall target. Verify flagged pairs with a VLM (detailed in [Post 4b](/post-4b)).

---

## What's Next

This bake-off established which embeddings to use; the next question is how to *search* them at scale. [Post 4a](/post-4a) covers the 3D-DupBench benchmark design and the ModelNet40 audit, while [Post 4b](/post-4b) covers building a FAISS index over millions of embeddings, tuning similarity thresholds for production precision targets, and the surprising impact of index type on retrieval quality. We'll also address the cold-start problem: how do you set a threshold when you don't have labeled duplicate pairs?

---

*All code, embeddings, and evaluation scripts from this post are available in the [3D-DupBench repository](https://github.com/hwajjala/3d-dupbench). The 3D-DupBench dataset (191 source models + 2,865 clones across 5 tiers) is released under CC-BY-4.0.*

*Have questions or want to discuss? Find me on [LinkedIn](https://www.linkedin.com/in/harishwajjala) or [X/Twitter](https://x.com/harishwajjala).*
