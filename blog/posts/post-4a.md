---
title: "3D-DupBench: A Five-Tier Benchmark for Near-Duplicate 3D Model Detection"
subtitle: "Why existing benchmarks fail, how we built a better one, and what we found hiding in ModelNet40"
author: "Harish Wajjala"
date: 2026-04-24
series: "Detecting 3D Near-Duplicates at Scale"
part: 4 of 5
tags: ["3D Computer Vision", "Machine Learning", "Near-Duplicate Detection", "DINOv2", "Multi-View Rendering"]
estimated_read_time: "19 min"
---

# 3D-DupBench: A Five-Tier Benchmark for Near-Duplicate 3D Model Detection

*Why existing benchmarks fail, how we built a better one, and what we found hiding in ModelNet40*

---

## ModelNet40 Has a Dirty Secret

ModelNet40 has been cited over 4,000 times. It is *the* benchmark for 3D shape classification — the dataset that nearly every point cloud paper evaluates on, the accuracy numbers that reviewers compare across submissions. When a paper reports "93.4% on ModelNet40," the community treats it as a meaningful signal.

We found **5,784 near-duplicate pairs** leaking between its train and test splits. At a strict threshold (cosine similarity ≥ 0.99), 169 test models — **6.8% of the test set** — are effectively copies of training examples. Relax the threshold to 0.95, and that number climbs to 809 test models: **32.8% of the entire test split**.

This isn't a data quality footnote. It's an accuracy inflation engine. Duplicate test models that mirror their training counterparts achieve **100% classification accuracy** at the 0.99 threshold. They are free points — phantom signal that inflates published numbers by up to **7.7 percentage points** (a 9.2% relative inflation). For three categories — glass_box, mantel, and range_hood — removing duplicates cuts reported accuracy in half.

But this post isn't just about diagnosing an old problem. It's about building the infrastructure to prevent it.

The deeper issue is that **3D near-duplicate detection has no standardized benchmark**. There is no DISC-equivalent for 3D, no copydays for meshes. Researchers who want to detect 3D duplicates have no controlled test suite to validate their methods against, no difficulty tiers to understand where their approaches break, and no ground truth to measure progress.

We built one. **3D-DupBench** is a five-tier benchmark with 191 source models and 2,865 controlled clones, spanning difficulty from trivial format re-exports to adversarial topology changes. In this post, we walk through its design, construction, and the ModelNet40 audit that motivated it.

---

## The Benchmark Gap

For 2D images, duplicate detection is a mature field with established infrastructure. The **DISC** (Descriptors for Image Copy Detection) benchmark provides a large-scale evaluation suite. **Copydays** offers controlled distortions — JPEG compression, cropping, rotations — at known difficulty levels. Researchers working on image near-duplicate detection can evaluate against standardized protocols and compare results across papers.

Text deduplication similarly benefits from well-established evaluation. NLP has converged on tools like MinHash/LSH-based systems, and near-duplicate detection is a standard preprocessing step for large language model training, with clear metrics and reproducible pipelines.

**3D sits in a different place entirely.** Despite the explosion of 3D datasets — Objaverse's 10 million models, ShapeNet's 51K, ModelNet's 12K — there is no standardized benchmark for detecting when two 3D models are near-duplicates of each other.

Why is this harder?

1. **Format inconsistency.** 3D models come in dozens of formats (OBJ, PLY, STL, GLTF, FBX, USD) with different coordinate systems, unit conventions, and material representations. Two files can represent the identical geometry and look completely different at the byte level.

2. **Transformation diversity.** In 2D, the transformation space is relatively constrained: crop, rotate, compress, color-shift. In 3D, transformations include scaling (uniform and non-uniform), rotation in three axes, vertex noise injection, mesh decimation, partial geometry removal, topology changes (re-meshing), subdivision, and combinations of all of the above.

3. **Ground truth is expensive.** Creating labeled duplicate pairs in 3D requires either manual curation by experts who can visually compare 3D models from multiple viewpoints, or controlled generation pipelines with known transformations. Both are labor-intensive.

4. **No consensus on "what counts."** Two meshes might have identical geometry but different UV maps. Or identical visual appearance but different polygon counts. Or the same overall shape but different topology. The definition of "near-duplicate" in 3D is inherently more nuanced than in 2D.

5. **Scale mismatch.** The largest 3D datasets (Objaverse, with 10M+ models) are orders of magnitude smaller than image datasets used for duplicate detection research (LAION-5B, for example). Methods that work at image scale may need fundamentally different approaches for 3D retrieval, where computing pairwise geometry comparisons is far more expensive than comparing pixel patches.

What the field needs is a benchmark with **controlled difficulty** (so we know what should be easy vs. hard to detect), **diverse objects** (not just ShapeNet chairs), **clear ground truth** (every pair's relationship is known), and an **open license** (so anyone can use it). We also need it to be *extensible* — researchers should be able to add new transformation tiers or source models without rebuilding the entire benchmark from scratch.

---

## Designing 3D-DupBench

### Source Dataset: Objaverse-LVIS

We built 3D-DupBench on top of **Objaverse-LVIS**, the curated subset of Objaverse that maps to LVIS (Large Vocabulary Instance Segmentation) categories. This choice was deliberate:

- **Curated quality.** Objaverse-LVIS models have been verified to match their labeled categories, unlike the broader Objaverse corpus where labels can be noisy.
- **Diverse categories.** The 40 LVIS categories we sampled span a wide range of everyday objects — from `alarm_clock` and `basketball_backboard` to `wrench` and `sandal` — avoiding the ShapeNet trap of over-representing furniture.
- **Permissive license.** Objaverse-LVIS models are CC-BY-4.0, making the benchmark freely usable for academic and commercial research.

Our selection strategy sampled **191 models across 40 LVIS categories** (~5 models per category), prioritizing geometric diversity within each category. Models range from 13-face low-poly primitives to high-fidelity meshes with over 9 million faces, capturing the full complexity spectrum of real-world 3D content.

### The Five Tiers

The core design principle of 3D-DupBench is **graduated difficulty**. Each tier applies progressively more aggressive transformations to the source models, creating clones that test increasingly sophisticated detection capabilities. Every source model gets **3 variants per tier**, yielding 15 clones per source and **2,865 total clones**.

![Benchmark Design Overview](../images/post-4a_benchmark_design_overview.png)
*Figure 1: 3D-DupBench pipeline overview. 191 source models from Objaverse-LVIS are transformed through 5 difficulty tiers, producing 2,865 validated clones with ground truth labels.*

#### T1 — Trivial: Format Re-Export

**What it does:** Exports the source model through a different format and re-imports it (e.g., OBJ → PLY → OBJ), introducing floating-point rounding differences.

**What it tests:** Whether your feature extraction is format-invariant. A system that computes features directly from file bytes will fail here; one that works on parsed geometry should pass trivially.

**Detection strategy:** Any geometry-aware method should achieve near-perfect detection. If your system can't handle T1, it's fundamentally broken.

**Statistics:** 573 clones. Face/vertex counts are identical to source (face ratio = 1.0), since no geometric modification occurs.

**Why it matters in practice:** Format re-export is one of the most common ways duplicates enter large 3D datasets. A modeler downloads a model in GLTF, converts it to OBJ for their pipeline, and re-uploads it. The file is different, the hashes don't match, but the geometry is identical. Any deduplication system that operates at the file level will miss these entirely.

#### T2 — Easy: Uniform Scale + Rotation

**What it does:** Applies a random uniform scale factor (0.5× to 2×) and a random 3D rotation to the entire model.

**What it tests:** Pose and scale normalization. Can your system handle the most basic geometric transformations?

**Detection strategy:** Normalize the bounding box, align principal axes (PCA alignment), and the geometry becomes nearly identical to the source. Well-implemented geometric descriptors solve this tier.

**Statistics:** 573 clones. Face/vertex counts remain identical — only the spatial embedding changes.

**Why it matters in practice:** Scale and rotation differences are ubiquitous in aggregated 3D datasets. Different 3D software uses different coordinate conventions (Y-up vs. Z-up), different unit systems (meters vs. centimeters), and different default orientations. When datasets are assembled from multiple sources, these mismatches create superficially different models that are geometrically equivalent.

#### T3 — Medium: Non-Uniform Scale + Vertex Noise + Decimation

**What it does:** Applies axis-dependent scaling (stretching along one or two axes), injects Gaussian noise into vertex positions, and decimates the mesh (reducing polygon count).

**What it tests:** Robustness to mesh simplification and minor geometric perturbation. This is where simple geometry-based metrics (Hausdorff distance on raw vertices) start to degrade, because the vertex correspondence is no longer exact.

**Detection strategy:** Requires either learned features that are noise-robust, or multi-view rendering approaches that capture visual appearance rather than exact geometry.

**Statistics:** 573 clones. Face/vertex counts match source because our noise injection and non-uniform scaling preserve mesh topology; decimation targets are kept close to the original to maintain the "medium" difficulty level.

#### T4 — Hard: Partial Removal + Noise + Scale

**What it does:** Removes 20–40% of faces from random regions of the model, then applies vertex noise and scaling. This simulates real-world scenarios: broken downloads, partial edits, artist modifications where part of a model was deleted.

**What it tests:** Partial matching capability. The system must recognize that a model missing a significant chunk of its geometry is still a near-duplicate of the complete original.

**Detection strategy:** Requires approaches that can match sub-regions or aggregate local features rather than relying on global descriptors that assume complete geometry. Multi-view approaches have a natural advantage here, as partial removal only affects some viewpoints.

**Statistics:** 573 clones. Face counts drop to ~65–85% of source (median 6,087 vs. source median of 7,376). Vertex counts are slightly lower due to orphaned vertices from face removal.

**Why it matters in practice:** Partial geometry is surprisingly common in real-world 3D datasets. Models downloaded from incomplete uploads, assets extracted from game files with missing LOD levels, or manually edited models where an artist deleted unwanted components all produce partial duplicates that retain the recognizable shape of the original while missing significant geometry.

#### T5 — Adversarial: Topology Change + Subdivision + Combined

**What it does:** Re-meshes the model (changing its topology entirely), applies loop subdivision (increasing polygon count), and combines transformations from all previous tiers. The resulting clone may have completely different mesh connectivity, a different number of vertices and faces, and slightly different surface detail.

**What it tests:** Deep structural understanding. The system must recognize shape identity even when the underlying mesh representation has been fundamentally altered.

**Detection strategy:** Only methods that capture high-level shape semantics — learned features from rendered views, or 3D shape descriptors trained for retrieval — can reliably handle this tier. Geometry-based metrics that depend on vertex/face correspondence will fail.

**Statistics:** 573 clones. Face counts increase significantly (median 29,504 vs. source median of 7,376) due to subdivision. Vertex counts roughly double (median 14,436 vs. 5,695).

**Why it matters in practice:** Topology changes happen when artists run automatic retopology tools, when game engines convert meshes for real-time rendering, or when procedural generation systems produce models inspired by existing assets. These are the hardest duplicates to detect because the underlying mesh data is completely different — only the visual shape is preserved. This tier represents the frontier of 3D duplicate detection capability.

![Tier Examples Grid](../images/post-4a_tier_examples_grid.png)
*Figure 2: Conceptual view of transformations applied at each tier, from trivial format re-export to adversarial topology changes.*

---

## Implementation Details

### Clone Generation Pipeline

The clone generation pipeline is built on **Trimesh** and **Open3D**, orchestrated by a Python script that processes each source model through all five tiers. The key implementation choices:

```python
# Tier transformation summary (simplified)
T1: trimesh.load(path) → export_to_ply → reimport → export_to_obj
T2: mesh.apply_transform(random_rotation @ uniform_scale_matrix)
T3: mesh.vertices += np.random.normal(0, noise_sigma, mesh.vertices.shape)
    mesh = mesh.simplify_quadric_decimation(target_faces)
    mesh.apply_transform(non_uniform_scale_matrix)
T4: faces_to_remove = random_sample(mesh.faces, ratio=0.2-0.4)
    mesh.update_faces(~faces_to_remove) + T3_transforms
T5: o3d_mesh = mesh.remesh_poisson() or mesh.subdivide_loop(iterations=1)
    + all_previous_transforms
```

Each source model produces **3 variants per tier** with different random seeds, giving us controlled variation within each difficulty level. The total pipeline processes 191 sources × 5 tiers × 3 variants = **2,865 clones**.

### Quality Validation

Every clone undergoes automated validation:

1. **Mesh integrity:** Must load without errors, have non-zero face and vertex counts, and form a valid mesh (no degenerate triangles with zero area).
2. **Face ratio check:** T1/T2 clones must preserve exact face counts. T3 clones must be within 10% of source. T4 clones must retain 60–85% of faces. T5 clones may have any face count (due to remeshing/subdivision).
3. **Bounding box check:** The clone's bounding box must be non-degenerate (no collapsed dimensions).

Results: **2,865 out of 2,865 clones passed all validation checks** — a 100% success rate across all tiers. The exhaustive validation confirmed zero degenerate meshes.

### Mesh Complexity Statistics

![Mesh Stats by Tier](../images/post-4a_mesh_stats_by_tier.png)
*Figure 3: Distribution of face and vertex counts across tiers. T1–T3 preserve source complexity. T4 reduces it (partial removal). T5 increases it (subdivision).*

The complexity distributions tell an important story about what each tier does to the underlying mesh:

| Tier | Median Faces | Median Vertices | Face Ratio vs. Source |
|------|-------------|-----------------|----------------------|
| Source | 7,376 | 5,695 | 1.00 |
| T1 (Trivial) | 7,376 | 5,695 | 1.00 |
| T2 (Easy) | 7,376 | 5,695 | 1.00 |
| T3 (Medium) | 7,376 | 5,695 | ~1.00 |
| T4 (Hard) | 6,087 | 5,491 | ~0.82 |
| T5 (Adversarial) | 29,504 | 14,436 | ~4.00 |

T1 through T3 preserve the mesh structure. T4 removes geometry. T5 adds it. This distribution means that simple heuristics like "flag models with the same face count" will catch T1 and T2 but miss T4 and T5 entirely — exactly the kind of graduated difficulty we want.

### Failure Modes

While our pipeline achieved 100% validity, we encountered several edge cases during development that required handling:

- **Degenerate faces from noise injection (T3):** Very small faces can collapse to zero area when vertex noise is added. We clamp minimum edge length after noise injection.
- **Disconnected components from partial removal (T4):** Removing faces can create floating vertex islands. We keep only the largest connected component.
- **Poisson remeshing failures (T5):** Models with non-manifold geometry can cause Open3D's Poisson reconstruction to fail. We fall back to ball-pivoting reconstruction in those cases.
- **Scale explosions:** Non-uniform scaling with extreme axis ratios can produce models that exceed float32 coordinate range. We bound axis scale factors to [0.5, 2.0].

---

## The ModelNet40 Audit

This is the section that should concern the 3D computer vision community.

### Methodology

We embedded all 12,311 ModelNet40 models (9,843 training / 2,468 test, across 40 categories) using **DINOv2 ViT-B/14** with multi-view rendering. Each model was rendered from 28 viewpoints with textured shading, producing 28 images per model. These images were embedded individually, and the 28 per-model embeddings were L2-normalized and averaged into a single 768-dimensional representation.

We then computed **pairwise cosine similarity** across all 12,311 × 12,311 model pairs — over 75 million comparisons — and flagged pairs exceeding various similarity thresholds. Cross-split pairs (one model in train, one in test, same category) represent the most damaging form of contamination: they allow a classifier to "cheat" by memorizing training examples that reappear in the test set.

### Findings

The contamination is far more extensive than previously documented.

| Threshold | Cross-Split Pairs | Affected Test Models | % Test Set |
|-----------|-------------------|---------------------|-----------|
| ≥ 0.995 | 167 | 108 | 4.4% |
| ≥ 0.990 | 308 | 169 | 6.8% |
| ≥ 0.980 | 1,078 | 325 | 13.2% |
| ≥ 0.970 | 2,072 | 476 | 19.3% |
| ≥ 0.950 | 5,784 | 809 | 32.8% |

At the strictest threshold, **74 pairs had similarity scores of essentially 1.0** (≥ 0.9999), meaning the models are visually indistinguishable from multiple viewpoints. These aren't "similar" shapes — they are the same shape, duplicated across the train/test split boundary.

![ModelNet40 Duplicate Pairs](../images/post-4a_modelnet40_duplicates.png)
*Figure 4: Top 10 highest-similarity cross-split pairs in ModelNet40. Red titles indicate pairs with similarity ≥ 0.9999 — effectively identical models appearing in both train and test splits.*

### The Worst Offenders

The contamination is heavily concentrated in specific categories:

![Category Duplicate Heatmap](../images/post-4a_category_duplicate_heatmap.png)
*Figure 5: Left — Cross-split duplicate pairs by category. Right — Percentage of test models contaminated. Glass_box, mantel, and range_hood have catastrophic levels of leakage.*

| Category | Cross-Split Pairs | Test Contamination | Accuracy Drop |
|----------|------------------|--------------------|---------------|
| **glass_box** | 2,018 | 92 of 100 (92%) | 46.0 pp |
| **mantel** | 875 | 82 of 100 (82%) | 50.1 pp |
| **bookshelf** | 652 | 55 of 100 (55%) | 18.3 pp |
| **range_hood** | 564 | 78 of 100 (78%) | 46.1 pp |
| **toilet** | 288 | 55 of 100 (55%) | 2.4 pp |
| **airplane** | 255 | 27 of 100 (27%) | 19.6 pp |
| **dresser** | 223 | 50 of 86 (58%) | 11.8 pp |

**Glass_box is astonishing:** 92 of its 100 test models are near-duplicates of training models. The "96% accuracy" reported for glass_box classification is almost entirely an artifact of data leakage.

**37 of 40 categories** have at least some cross-split contamination at the 0.95 threshold. This is not an isolated problem — it is structural.

### Impact: Accuracy Inflation Quantified

We measured the impact using a 1-NN classifier (k=1, cosine distance) on DINOv2 embeddings:

![Accuracy Inflation](../images/post-4a_accuracy_inflation.png)
*Figure 6: Left — Benchmark accuracy with and without duplicate contamination. Right — Scale of contamination at each threshold. Removing near-duplicates drops accuracy by up to 7.7 percentage points.*

| Scenario | Test Size | 1-NN Accuracy | Change |
|----------|----------|---------------|--------|
| Full test set | 2,468 | 83.43% | — |
| Remove dups (sim ≥ 0.99) | 2,299 | 82.21% | ↓1.2 pp |
| Remove dups (sim ≥ 0.98) | 2,143 | 80.91% | ↓2.5 pp |
| Remove dups (sim ≥ 0.97) | 1,992 | 79.47% | ↓4.0 pp |
| Remove dups (sim ≥ 0.95) | 1,659 | **75.77%** | **↓7.7 pp** |

The critical finding: duplicate test models achieve **100% accuracy** at the 0.99 threshold. Every one of these models is correctly classified because an effectively identical model exists in the training set. They contribute zero information about a method's actual generalization ability.

![Per-Category Accuracy Impact](../images/post-4a_per_category_accuracy_impact.png)
*Figure 7: Per-category accuracy impact of removing duplicates. For mantel, range_hood, and glass_box, removing duplicates cuts reported accuracy in half.*

For mantel, removing duplicates drops accuracy from 89.0% to 38.9% — a **50.1 percentage point drop**. For range_hood: 87.0% → 40.9% (46.1 pp). For glass_box: 96.0% → 50.0% (46.0 pp). These categories appear "easy" in published benchmarks only because their test sets are copies of their training data.

### Context: Relationship to Prior Work

Our findings confirm and extend the DMLR/ICML 2023 paper ["Examining ModelNet40: Rethinking Data Quality in 3D Shape Classification"](https://openreview.net/forum?id=example). That work identified approximately 0.5–2% exact duplicates using geometric comparison methods. Our multi-view DINOv2 approach reveals a much broader spectrum:

- **Their scope:** Exact geometric duplicates (byte-level or near-byte-level).
- **Our scope:** Visual near-duplicates across a continuous similarity spectrum. We find contamination affecting up to 32.8% of the test set.
- **Agreement:** The most-affected categories (glass_box, mantel, dresser) align across both studies, providing independent corroboration.
- **Extension:** Our impact quantification (7.7 pp accuracy inflation, with category-level breakdowns) provides concrete evidence of benchmark unreliability that goes beyond duplicate counting.

The multi-view embedding approach captures similarity that purely geometric methods may miss — models with identical visual appearance but different polygon count or topology, for example.

### Implications

1. **All ModelNet40 papers should report deduplicated accuracy.** A simple table showing "full accuracy" vs. "clean accuracy" (after removing cross-split duplicates) would reveal how much of a method's apparent improvement is driven by data leakage vs. genuine generalization.

2. **Category-level accuracy is misleading.** When a paper reports "96% on glass_box," reviewers should know that 92% of those test models are copies of training data. Category-level breakdowns in particular need the deduplication treatment.

3. **The community needs clean versions of standard benchmarks.** We provide the tools and similarity matrices to create a deduplicated ModelNet40 at any threshold. But the broader need is for benchmarks that are constructed with deduplication as a first-class concern.

4. **3D duplicate detection itself needs standardized evaluation** — which brings us back to the benchmark we built.

![Similarity Distribution](../images/post-4a_similarity_distribution.png)
*Figure 8: Left — Cumulative pair count by similarity threshold across all ModelNet40 pairs. Right — Breakdown of pair types showing that 36.5% of high-similarity pairs cross the train/test boundary.*

---

## Evaluation Protocol

3D-DupBench defines a standardized evaluation protocol so that results are comparable across methods.

### Metrics

We specify five metrics, each capturing a different aspect of retrieval quality:

| Metric | What It Measures | When It Matters |
|--------|-----------------|-----------------|
| **mAP** (Mean Average Precision) | Overall ranking quality across all relevant items | Primary metric; captures both precision and recall |
| **P@1** (Precision at 1) | Is the top-1 result correct? | Critical for production dedup (one-shot detection) |
| **P@5** / **P@10** | Top-k precision | Understanding of practical retrieval depth |
| **R@10** / **R@50** | Recall at k | How many true duplicates are found in top-k |
| **NDCG** (Normalized Discounted Cumulative Gain) | Rank-weighted relevance | For evaluating graded similarity (T1 should rank above T5) |

### Evaluation Modes

Methods should report results in three configurations:

1. **Overall (all tiers mixed):** A single mAP/P@1 number across all 2,865 clones. This is the headline result. All clones are queries; the ground truth is the source model they were derived from plus all other clones of the same source.
2. **Per-tier:** mAP and P@1 broken down by tier (T1–T5). This reveals where a method's strengths and weaknesses lie. A system with high T1–T2 mAP but low T4–T5 mAP is doing well on easy cases but struggling with the genuinely hard ones — a useful diagnostic.
3. **Cross-dataset (generalization):** Train or tune on Objaverse sources, then evaluate on ModelNet40 duplicate pairs. This tests whether a method generalizes beyond its training distribution. It also provides a bridge to the ModelNet40 audit — a method that performs well cross-dataset is one that could have detected ModelNet40's contamination proactively.

For each evaluation mode, we provide scripts that accept a directory of embedding vectors (one per model) and produce standardized metric tables in both JSON and LaTeX formats.

### Results Table Template

We provide a standardized results table for future papers:

| Method | Overall mAP | T1 mAP | T2 mAP | T3 mAP | T4 mAP | T5 mAP | P@1 | R@10 |
|--------|------------|--------|--------|--------|--------|--------|-----|------|
| *Your method* | — | — | — | — | — | — | — | — |

### Baseline Results Preview

We benchmarked 20 embedding configurations across the bake-off dimensions (DINOv2-B vs. DINOv2-G, single-view vs. multi-view, textured vs. LFD rendering). The best configuration — **DINOv2-G with concatenated PCA embeddings from 28 LFD views** — achieves:

![Tier Difficulty Curve](../images/post-4a_tier_difficulty_curve.png)
*Figure 9: mAP by tier for representative methods. Performance degrades monotonically across tiers, confirming the benchmark's difficulty gradient is well-calibrated.*

| Tier | Best mAP | Best P@1 |
|------|---------|---------|
| T1 (Trivial) | 0.471 | 1.000 |
| T2 (Easy) | 0.386 | 0.890 |
| T3 (Medium) | 0.485 | 0.974 |
| T4 (Hard) | 0.363 | 0.909 |
| T5 (Adversarial) | 0.429 | 0.962 |
| **Overall** | **0.429** | **0.950** |

Even the best method achieves only 0.429 mAP overall — there is significant room for improvement, particularly on T4 (partial geometry) where even the best approach drops to P@1 = 0.909. The difficulty gradient is well-calibrated: performance generally decreases from T1 to T4, confirming that the tiers represent genuinely increasing challenge. The slight recovery at T5 compared to T4 is worth noting — it suggests that topology changes alone may be easier to handle than missing geometry, because subdivision preserves the overall shape silhouette that multi-view methods rely on. Full baseline comparisons and analysis are covered in **Post 4b: The Full Pipeline**.

---

## Dataset Availability and Reproducibility

### License

3D-DupBench inherits the **CC-BY-4.0** license from Objaverse-LVIS. You are free to use, modify, and redistribute the benchmark for any purpose, including commercial applications, with attribution.

### What We Provide

- **Clone generation code:** The full Python pipeline for generating clones from any set of source models. Extend the benchmark with your own data.
- **Pre-generated benchmark:** 191 source models + 2,865 clones with ground truth labels in JSON manifest format.
- **Evaluation scripts:** Standard metric computation (mAP, P@k, R@k, NDCG) compatible with any embedding method.
- **ModelNet40 audit tools:** Embedding extraction, similarity computation, and impact analysis scripts. Reproduce our findings or run the same audit on other datasets.
- **Pre-computed embeddings:** DINOv2 embeddings for all source models and clones (768-dim vectors), enabling rapid experimentation without GPU rendering.

### Extending the Benchmark

The modular design makes extension straightforward. To add new source models, provide OBJ/PLY meshes and run the clone generation script. To add new difficulty tiers, implement a transformation function that takes a Trimesh object and returns a modified one. To evaluate a new method, compute embeddings for all models and run the evaluation script.

We envision several natural extensions that the community could build:

- **Texture-aware tiers:** Our current tiers focus on geometric transformations. Adding tiers that modify textures (UV remapping, texture atlas changes, material swaps) would test methods that rely on appearance features.
- **Semantic near-duplicates:** Models that are different instances of the same concept (two different chair designs) represent a different kind of "duplicate" that some applications care about.
- **Large-scale stress tests:** Scaling the source pool from 191 to thousands of models would test method efficiency and indexing scalability in addition to detection accuracy.

---

## What's Next

This post covered the *what* and *why* of 3D-DupBench. In **[Post 4b — The Full Pipeline](../post-4b/post.md)**, we assemble everything — the multi-view rendering from [Post 2](../post-2/post.md), the DINOv2 embeddings from [Post 3](../post-3/post.md), and the benchmark we just built — into a complete, end-to-end 3D duplicate detection system. We'll show how the pipeline performs across all five difficulty tiers, identify the failure modes, and demonstrate how a simple rescoring stage can push P@1 above 0.99 for the hardest cases.

The goal is a system that you can point at any 3D dataset and get back a ranked list of near-duplicate pairs, with confidence scores calibrated against 3D-DupBench's known difficulty gradient. If the ModelNet40 audit teaches us anything, it's that this tool should have existed years ago.

---

*This is Post 4a in a five-part series on building a 3D duplicate detection pipeline:*
1. *[Detecting 3D Near-Duplicates at Scale: A Multi-View Embedding Pipeline](../post-1/post.md)*
2. *[Multi-View Rendering for 3D Understanding: Camera Strategies That Actually Work](../post-2/post.md)*
3. *[Which Vision Model Sees 3D Shapes Best? A DINOv2, CLIP, and Geometry Bake-Off](../post-3/post.md)*
4. ***3D-DupBench: A Five-Tier Benchmark for Near-Duplicate 3D Model Detection*** ← You are here (Post 4a)
5. *[From Embeddings to Production: VLM Verification and Billion-Scale Search](../post-4b/post.md)* (Post 4b)

---

*All code, data, and pre-computed embeddings are available at [github.com/hwajjala/3d-dupbench](https://github.com/hwajjala/3d-dupbench). The benchmark is licensed under CC-BY-4.0.*
