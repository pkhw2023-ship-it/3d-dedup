# 3D Multi-View Embedding Evaluation Report

*Generated: 2026-04-23 | Dataset: ModelNet40 (12,311 models, 40 categories)*
*Embedding model: DINOv2-base (ViT-B/14, 768-dim) | Views per object: 28*

## Executive Summary

**Multi-View Textured (28 views, colored)** achieves the highest retrieval mAP of **0.4971**,
compared to Single-View Thumbnail at 0.2634 — a
**88.7% improvement**. Multi-view embeddings capture geometric structure
that single-view thumbnails miss, particularly for categories with high intra-class
shape variation (chairs, tables, lamps).

## 1. Retrieval Performance (mAP)

| Embedding Type | mAP | 95% CI | # Models |
|---|---|---|---|
| Single-View Thumbnail | **0.2634** | [0.2591, 0.2678] | 12311 |
| Multi-View LFD (28 views, white plastic) | **0.4218** | [0.4169, 0.4269] | 11228 |
| Multi-View Textured (28 views, colored) | **0.4971** | [0.4918, 0.5017] | 11228 |

![mAP Comparison](map_comparison.png)

### Per-Category mAP (Top 20 Most Variable)

| Category | Single-View Thumbnail | Multi-View LFD (28 views, white plastic) | Multi-View Textured (28 views, colored) |
|---|---|---|---|
| laptop | 0.0782 | 0.5585 | 0.6807 |
| monitor | 0.2313 | 0.6444 | 0.6954 |
| bowl | 0.1961 | 0.4349 | 0.6591 |
| person | 0.3728 | 0.7652 | 0.7699 |
| sofa | 0.1651 | 0.5232 | 0.5802 |
| guitar | 0.4947 | 0.8052 | 0.8319 |
| keyboard | 0.3306 | 0.6072 | 0.6774 |
| bed | 0.1270 | 0.3718 | 0.4760 |
| dresser | 0.1062 | 0.2767 | 0.4588 |
| chair | 0.1975 | 0.4548 | 0.5129 |
| bottle | 0.4896 | 0.7154 | 0.8131 |
| glass_box | 0.4143 | 0.6437 | 0.7297 |
| mantel | 0.3782 | 0.5580 | 0.6793 |
| wardrobe | 0.0485 | 0.1244 | 0.3361 |
| car | 0.6064 | 0.8440 | 0.8665 |
| stairs | 0.0504 | 0.2677 | 0.2993 |
| bookshelf | 0.2802 | 0.4007 | 0.5429 |
| door | 0.1987 | 0.3256 | 0.4454 |
| toilet | 0.4359 | 0.5835 | 0.6690 |
| curtain | 0.1509 | 0.3240 | 0.3648 |

![Per-Category mAP](per_category_map.png)

## 2. Clustering Quality (HDBSCAN)

| Embedding Type | ARI | # Clusters | Noise % |
|---|---|---|---|
| Single-View Thumbnail | **0.2859** | 43 | 73.1% |
| Multi-View LFD (28 views, white plastic) | **0.6681** | 69 | 75.7% |
| Multi-View Textured (28 views, colored) | **0.6978** | 71 | 69.5% |

![Clustering ARI](clustering_ari.png)

## 3. Duplicate Detection (Synthetic Clones)

### Overall Metrics

| Embedding Type | P@1 | P@5 | P@10 | Mean Source Rank | Mean Similarity |
|---|---|---|---|---|---|
| Single-View Thumbnail | 0.0008 | 0.0106 | 0.0192 | 3999.5 | 0.2150 |
| Multi-View LFD (28 views, white plastic) | 0.3497 | 0.5030 | 0.5612 | 307.6 | 0.8111 |
| Multi-View Textured (28 views, colored) | 0.3399 | 0.4848 | 0.5413 | 355.8 | 0.8029 |

### By Difficulty Tier

| Tier | Metric | Single-View Thumbnail | Multi-View LFD (28 views, white plastic) | Multi-View Textured (28 views, colored) |
|---|---|---|---|---|
| easy | P@1 | 0.0017 | 0.6435 | 0.6007 |
| easy | P@5 | 0.0142 | 0.8048 | 0.7620 |
| medium | P@1 | 0.0000 | 0.1390 | 0.1471 |
| medium | P@5 | 0.0058 | 0.2790 | 0.2647 |
| hard | P@1 | 0.0008 | 0.2665 | 0.2718 |
| hard | P@5 | 0.0117 | 0.4251 | 0.4278 |

![Duplicate Detection](duplicate_detection.png)

## 4. Statistical Significance

Wilcoxon signed-rank test on per-category mAP values:

| Comparison | Δ mAP | p-value | Significance |
|---|---|---|---|
| Multi-View LFD (28 views, white plastic) vs Single-View Thumbnail | +0.1649 | 0.0000 | *** |
| Multi-View Textured (28 views, colored) vs Single-View Thumbnail | +0.2394 | 0.0000 | *** |
| Multi-View Textured (28 views, colored) vs Multi-View LFD (28 views, white plastic) | +0.0745 | 0.0000 | *** |

## Methodology

- **Dataset**: ModelNet40 (12,311 3D models, 40 categories, OFF format)
- **Rendering**: nvdiffrast CUDA backend, 28 viewpoints per model (8 horizontal + 20 dodecahedron)
- **Resolution**: 224×224 (native DINOv2 input)
- **Embedding model**: DINOv2-base (ViT-B/14, 768-dim, facebook/dinov2-base)
- **Aggregation**: Mean pooling over 28 view embeddings
- **Similarity**: Cosine similarity (L2-normalized embeddings)
- **Clustering**: HDBSCAN (min_cluster_size=10, min_samples=5)
- **Statistical test**: Wilcoxon signed-rank on per-category mAP
- **Confidence intervals**: 1000-sample bootstrap on per-model AP

## Conclusion

Multi-view rendered embeddings significantly outperform single-view thumbnail embeddings
for 3D object retrieval. The shape-only LFD rendering provides the strongest signal for
retrieval by category, while textured rendering may add complementary information for
cross-modal similarity tasks. These results validate the multi-view rendering approach
for building robust 3D object similarity indices and duplicate detection systems.
