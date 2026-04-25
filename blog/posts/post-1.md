# Detecting 3D Near-Duplicates at Scale: A Multi-View Embedding Pipeline

*How to turn 3D meshes into searchable vector representations and find copies hidden in plain sight*

---

## The Hidden Contamination Problem

Every 3D dataset has a dirty secret: duplicates.

Not exact copies — those are easy to catch with a hash. I'm talking about *near*-duplicates: meshes that have been re-scaled, re-exported in a different format, partially modified, or subtly deformed. They look different enough to slip past naive deduplication, but similar enough to corrupt any downstream task that assumes independent samples.

Consider [ModelNet40](https://modelnet.cs.princeton.edu/), the most-cited 3D shape classification benchmark with over 4,000 citations. In 2023, [Gadhave et al.](https://proceedings.mlr.press/v210/gadhave23a.html) showed that ModelNet40 contains near-duplicate 3D models that leak across the train/test split, inflating reported classification accuracy. When I ran my own audit using multi-view embeddings, I found that single-view thumbnails achieve a retrieval mAP of just 0.263 on ModelNet40 — barely better than random for finding duplicates. Switch to 28-view textured embeddings and that jumps to 0.497, an **88.7% improvement** that reveals clusters of near-identical models the single-view approach completely misses.

This isn't just an academic concern. 3D asset marketplaces like [Sketchfab](https://sketchfab.com/) and [TurboSquid](https://www.turbosquid.com/) host millions of user-uploaded models. IP infringement through re-uploaded assets costs creators revenue and platforms trust. Massive training datasets like [Objaverse](https://objaverse.allenai.org/) (800K+ models) need deduplication before anyone can train on them responsibly. And in production 3D pipelines — game studios, architectural visualization, digital twins — duplicate assets waste storage, confuse search results, and degrade user experience.

The problem is unsolved because 3D near-duplicate detection is fundamentally harder than its 2D counterpart. In 2D, perceptual hashing and learned embeddings from models like CLIP handle most cases. In 3D, you're dealing with meshes that can differ in topology, vertex count, texture resolution, coordinate systems, and file format — all while representing the "same" object.

This series presents a complete, open-source pipeline that tackles the problem end-to-end: from raw 3D meshes to production-scale duplicate search. Over five posts, I'll walk through every design decision, share all the numbers, and release the benchmark I built to evaluate it.

![End-to-end pipeline architecture](../images/post-1_pipeline_overview.png)
*Figure 1: The five-stage pipeline — render, embed, search, verify — processes a 3D mesh into a searchable vector in ~60 seconds and achieves 95% precision on flagged duplicates. Image by author.*

---

## The Pipeline: Render → Embed → Search → Verify

The core insight behind this pipeline is counterintuitive: **don't try to compare 3D geometry directly.** Instead, render each mesh from multiple viewpoints, embed the resulting images with a pre-trained vision model, and search in embedding space.

Why does this roundabout approach beat direct geometry comparison?

**Reason 1: 2D vision models are better trained.** [DINOv2](https://arxiv.org/abs/2304.07193), trained on 142 million curated images via self-supervised learning, has learned visual similarity features that generalize remarkably well. No 3D-specific model comes close to this scale of pre-training.

**Reason 2: Geometry comparison doesn't scale.** Computing Chamfer or Hausdorff distances requires point sampling, alignment, and O(n²) nearest-neighbor lookups per pair. On our 3D-DupBench dataset (3,056 models), the geometry baselines achieve high mAP — Chamfer distance gets 0.813, and a simple surface-area + volume descriptor gets 0.988 — but they require loading and processing each mesh pair. At 1 million models, that's 500 billion pairwise comparisons. An embedding approach reduces each model to a 768-dimensional vector, and [FAISS](https://github.com/facebookresearch/faiss) searches through 1 million of them in under 1 millisecond.

**Reason 3: Embeddings are format-agnostic.** A mesh stored as `.glb`, `.obj`, `.fbx`, or `.ply` renders to the same pixels. The embedding doesn't care about vertex ordering, face topology, or coordinate conventions — exactly the variations that make direct geometry comparison brittle.

Here's what the pipeline looks like in code:

```python
import torch
import numpy as np
from render import render_multiview       # nvdiffrast-based renderer
from embed import extract_embeddings      # DINOv2 feature extractor
from search import FAISSIndex             # FAISS wrapper

# Stage 1: Render 28 views of a 3D model
views = render_multiview(
    mesh_path="model.glb",
    n_ring=8,           # horizontal ring at 30° elevation
    n_dodeca=20,        # dodecahedron vertices for full coverage
    resolution=224,     # match DINOv2 input size
    mode="textured"     # preserve material/color information
)  # → List[PIL.Image], 28 images

# Stage 2: Embed each view, then aggregate
per_view = extract_embeddings(views, model="dinov2_vitg14")
# per_view shape: (28, 1536) for ViT-G/14
embedding = per_view.mean(axis=0)          # mean-pool → (1536,)
embedding = embedding / np.linalg.norm(embedding)  # L2 normalize

# Stage 3: Search against index of existing models
index = FAISSIndex.load("production_index.faiss")  # IVF256,SQ8
distances, indices = index.search(embedding, k=20)  # <1ms at 1M

# Stage 4: Threshold + VLM verification
candidates = [(idx, dist) for idx, dist in zip(indices, distances)
              if dist < 0.82]  # high-precision threshold
```

The total latency for a single query — rendering, embedding, and searching — is about 60 seconds, dominated by GPU rendering. If any candidates pass the distance threshold, they're sent to a VLM (vision-language model) rescorer for final verification, adding another ~3 seconds per candidate.

---

## Why Multi-View? The View Count Matters More Than You'd Think

A single rendered image captures maybe 30% of a 3D model's geometry. The back, bottom, interior cavities, and occluded details are completely invisible. For near-duplicate detection, this is fatal: two models might look identical from the front but differ significantly elsewhere.

I tested this systematically by varying the number of views from 1 to 28 using DINOv2 with mean-pooled embeddings:

![Multi-view comparison: view count ablation and aggregation strategies](../images/post-1_multiview_comparison.png)
*Figure 2: Left — Retrieval mAP improves sharply from 1→8 views, with diminishing returns beyond 12. Right — Concatenation + PCA dramatically outperforms mean/max pooling, achieving 0.429 mAP with DINOv2-Giant. Image by author.*

The jump from 1 view to 4 views is dramatic: DINOv2-Giant goes from 0.311 to 0.354 mAP (a **14% improvement**). Adding more views continues to help but with diminishing returns — 8 views gets 0.363, and 28 views gets 0.370 with mean pooling.

But the real story is in the aggregation strategy. With 28 views, simple mean pooling gives 0.370 mAP. Concatenating all 28 per-view embeddings and reducing with PCA gives **0.429 mAP** — a **16% further improvement** over mean pooling. The concatenation preserves view-specific information that averaging destroys.

My camera setup uses two complementary strategies:

- **Horizontal ring**: 8 cameras evenly spaced at 30° elevation, capturing the "showcase" views you'd see on a product page
- **Dodecahedron vertices**: 20 cameras placed at the vertices of a regular dodecahedron, providing near-uniform coverage of the full viewing sphere

Together, these 28 viewpoints give robust coverage with manageable compute cost. Rendering all 28 views of a single model takes about 1.1 seconds on a Tesla T4 GPU using [nvdiffrast](https://github.com/NVlabs/nvdiffrast).

![Multi-view renders of three example models](../images/post-1_multiview_renders.png)
*Figure 3: Eight ring views (45° apart) for three Objaverse models. The horizontal ring captures the most discriminative angles; dodecahedron views (not shown) fill in top/bottom coverage. Image by author.*

The ModelNet40 audit makes the case even more clearly. On the duplicate detection task — finding re-exported or subtly modified copies within ModelNet40's 12,311 models — single-view thumbnails achieve a P@1 of just **0.08%**. Multi-view textured embeddings achieve **34.0%** P@1, a **425× improvement**. The single-view approach is essentially useless for finding duplicates; multi-view makes it a practical tool.

| Approach | Retrieval mAP | Clustering ARI | Dup. Detection P@1 |
|----------|:------------:|:--------------:|:------------------:|
| Single-view thumbnail | 0.263 | 0.286 | 0.08% |
| Multi-view LFD (28 views) | 0.422 | 0.668 | 34.97% |
| Multi-view textured (28 views) | **0.497** | **0.698** | 33.99% |
| **Improvement** | **+88.7%** | **+144%** | **425×** |

*Table 1: ModelNet40 evaluation comparing single-view vs multi-view approaches across three tasks. Multi-view with textures provides the strongest retrieval but LFD (silhouette-style) rendering slightly edges out on duplicate detection P@1.*

---

## The Benchmark: 3D-DupBench

Evaluating near-duplicate detection requires knowing the ground truth — which pairs are actually duplicates and at what level of modification. No public benchmark existed for this, so I built one.

**3D-DupBench** starts with 191 source models from [Objaverse-LVIS](https://objaverse.allenai.org/objaverse-1.0) (licensed CC-BY-4.0) and generates controlled clones at five difficulty tiers:

![Five difficulty tiers of 3D-DupBench](../images/post-1_tier_examples.png)
*Figure 4: 3D-DupBench's five tiers, from trivial format re-exports (T1) to adversarial topology changes (T5). Detection recall drops sharply after T2. Image by author.*

Each tier applies increasingly aggressive transformations:

| Tier | Name | Transformation | Median Cosine Sim |
|------|------|---------------|:-----------------:|
| **T1** | Trivial | Format re-export (`.obj` → `.glb`) | 1.000 |
| **T2** | Easy | Uniform scale + rotation | 0.939 |
| **T3** | Medium | Non-uniform scale + vertex noise | 0.581 |
| **T4** | Hard | Partial mesh removal + surface noise | 0.531 |
| **T5** | Adversarial | Topology change + deformation | 0.557 |

*Table 2: Clone tiers with their transformations and median embedding cosine similarity to the source model.*

The dataset contains 3,056 models total (191 sources × 15 clones each, plus the sources). Each source gets 3 clones per tier (different random seeds), providing statistical robustness. I rendered all 3,056 models from 28 viewpoints in two modes (textured and silhouette), producing **171,136 images** in 56.6 minutes on a single T4 GPU.

![Clone tier visual comparison from actual renders](../images/post-1_clone_tier_visual.png)
*Figure 5: Actual rendered views of one source model and its five tiers of clones. T1 and T2 are nearly indistinguishable from the source; T3-T5 show visible geometric changes. Image by author.*

What makes 3D-DupBench useful is the **tiered evaluation**: rather than reporting a single accuracy number, we can see exactly where methods break down. Trivial duplicates (T1-T2)? Solved. Medium-difficulty clones with non-uniform scaling (T3)? Embedding retrieval starts to struggle. Adversarial topology changes (T4-T5)? Still an open problem for visual embeddings, though geometry baselines handle them better.

```python
# Generating a T3 clone: non-uniform scale + vertex noise
import trimesh
import numpy as np

mesh = trimesh.load("source.obj")

# Non-uniform scaling (different factor per axis)
scale = np.array([
    np.random.uniform(0.7, 1.3),   # x
    np.random.uniform(0.7, 1.3),   # y
    np.random.uniform(0.7, 1.3),   # z
])
mesh.vertices *= scale

# Add vertex noise (proportional to bounding box)
bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
noise = np.random.normal(0, 0.02 * bbox_diag, mesh.vertices.shape)
mesh.vertices += noise

mesh.export("clone_T3.obj")
```

---

## Key Results: What Works, What Doesn't, and What Surprised Me

I evaluated 40 embedding configurations (4 models × 2 render modes × 5 aggregation strategies) plus 3 geometry baselines on 3D-DupBench. Here are the headline findings.

### Best embedding method: DINOv2-Giant with concatenation+PCA

The top embedding method across the board is DINOv2-ViT-G/14 with all 28 per-view embeddings concatenated and projected down to 768 dimensions via PCA. This achieves **mAP 0.429** on the full benchmark, with **P@1 of 95.0%** — meaning the correct source model is the nearest neighbor 95% of the time.

![Key results: method comparison and per-tier breakdown](../images/post-1_key_results_summary.png)
*Figure 6: Left — mAP across all methods. Geometry baselines (Chamfer, SA+Volume) achieve higher raw mAP but don't scale. Right — Per-tier breakdown showing where embeddings struggle relative to geometry. Image by author.*

A few surprises emerged from the bakeoff:

**DINOv2 consistently beats CLIP for this task.** The best DINOv2 configuration (mAP 0.429) outperforms the best CLIP configuration (mAP 0.397) by 8%. This makes sense: DINOv2 is trained with self-supervised objectives that emphasize visual structure, while CLIP's contrastive text-image training optimizes for semantic rather than geometric similarity.

**Render mode barely matters.** Textured rendering (with materials and colors) and LFD-style rendering (solid gray, silhouette-like) produce nearly identical results. DINOv2-Giant achieves 0.428 mAP with textured rendering and 0.429 with LFD. The vision models extract structural features that transcend surface appearance — good news for datasets where textures are missing or inconsistent.

**Concatenation+PCA is the clear winner for aggregation.** Mean pooling all 28 views gives 0.370 mAP; max pooling gives 0.369; concatenation+PCA gives 0.429. The +16% improvement is the single largest gain from any design decision beyond going multi-view in the first place. Concatenation preserves which features came from which viewpoint, while pooling collapses this information.

### The geometry baseline paradox

Here's what genuinely surprised me: a simple surface-area + bounding-box volume descriptor achieves **mAP 0.988** — far exceeding any learned embedding. Chamfer distance gets 0.813, Hausdorff gets 0.809. Why use embeddings at all?

The answer is scale. Computing SA+Volume requires loading each mesh, which takes ~100ms per model. At 1 million models, an exhaustive search takes 28 hours. With embeddings, the same search takes **0.6 milliseconds** using a FAISS flat index. That's a 168,000× speedup.

In production, the right architecture combines both: use embeddings for fast candidate retrieval (top-20 nearest neighbors), then apply geometry comparisons or a VLM rescorer to the shortlist.

### VLM rescorer: from 63% to 95% precision

Raw embedding search at threshold τ=0.68 produces 63.2% precision and 42.1% recall. Many of the false positives are semantically similar but geometrically distinct models (two different chairs, say). Adding a [Gemini 2.0 Flash](https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash) VLM rescorer that compares 8-angle renders of each candidate pair boosts precision to **95.0%** while maintaining 27.6% recall.

![ModelNet40 results and rescorer precision-recall tradeoff](../images/post-1_rescorer_tradeoff.png)
*Figure 7: Left — ModelNet40 single-view vs multi-view comparison. Right — VLM rescorer precision-recall curve as a function of the minimum evidence-angle threshold. At ≥2 evidence angles, precision hits 95% with manageable recall loss. Image by author.*

The rescorer works by rendering both models from the same 8 angles and asking the VLM: "Are these the same 3D object?" It votes across angles — if ≥2 out of 8 viewpoints confirm a match, the pair is flagged. This multi-angle voting is crucial: a single viewpoint can be fooled by coincidental similarity, but requiring agreement across multiple angles filters out false positives effectively.

The per-tier breakdown tells the full story:

| Tier | Embedding Precision | Rescorer Precision | Rescorer Recall |
|------|:------------------:|:-----------------:|:--------------:|
| T1 (Trivial) | 65.5% | **95.4%** | **96.7%** |
| T2 (Easy) | 61.4% | **96.4%** | 36.0% |
| T3 (Medium) | 80.0% | 100% | 2.7% |
| T4 (Hard) | 100% | — | 0% |
| T5 (Adversarial) | 92.9% | 100% | 2.7% |

*Table 3: Per-tier detection results with the full pipeline (embedding retrieval + VLM rescorer). T1-T2 are reliably caught; T3-T5 require different approaches.*

T1 and T2 duplicates — format re-exports and uniform rescaling — are essentially solved: the pipeline catches them with >95% precision and >36% recall. T3-T5 remain challenging because the geometric modifications are severe enough that the models no longer look similar from most viewpoints.

### FAISS scalability: sub-millisecond search at 1M vectors

To test production viability, I benchmarked seven FAISS index types at scales from 1K to 1M vectors using 768-dimensional DINOv2 embeddings:

| Index Type | Recall@10 | Query Latency | Memory | GPU |
|-----------|:---------:|:------------:|:------:|:---:|
| Flat (exact) | 100% | 0.6ms | 3,072 MB | ✓ |
| IVF256,Flat | 100% | 11.7ms | 3,072 MB | ✓ |
| **IVF256,SQ8** | **97.7%** | **6.2ms** | **776 MB** | ✓ |
| HNSW32 | ~98% | ~0.1ms | — | ✗ |
| IVF256,PQ32 | 12% | 0.07ms | 40 MB | ✓ |

*Table 4: FAISS benchmark results at 1M vectors on a Tesla T4 GPU. IVF-SQ8 offers the best balance of recall, latency, and memory.*

The recommended production index is **IVF256,SQ8**: it achieves 97.7% recall@10 with 6.2ms latency and 4× memory savings over brute-force (776 MB vs 3,072 MB). Product Quantization (PQ) indices showed surprisingly poor recall (12-17%) on our embeddings — DINOv2 features are highly correlated across dimensions, violating PQ's assumption of independent subspaces.

```python
import faiss
import numpy as np

# Build a production-ready FAISS index
dim = 768
nlist = 256  # number of IVF clusters

# SQ8: scalar quantization to uint8 (4× memory savings)
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFScalarQuantizer(
    quantizer, dim, nlist,
    faiss.ScalarQuantizer.QT_8bit
)

# Move to GPU for faster training and search
res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, index)

# Train on a sample, then add vectors
index.train(embeddings_sample)  # needs >= nlist vectors
index.add(all_embeddings)       # add full corpus

# Search with nprobe=256 for maximum recall
index.nprobe = 256
D, I = index.search(query_vector, k=20)  # <7ms at 1M
```

---

## Series Roadmap

This post covered the "what" and "why" — the rest of the series dives into the "how."

![Series roadmap](../images/post-1_series_roadmap.png)
*Figure 8: Five-part series plan. Each post is designed to be standalone, with cross-references to related posts. Image by author.*

**Post 2: Multi-View Rendering** — Camera placement strategy (ring vs dodecahedron), GPU-accelerated rendering with nvdiffrast, the blank-image trap that silently corrupts 1% of renders, and how textured vs silhouette rendering affects downstream quality.

**Post 3: Embeddings & Retrieval** — The full DINOv2 vs CLIP bakeoff across 40 configurations, why concatenation+PCA beats mean pooling, how geometry baselines (Chamfer, Hausdorff, SA+Volume) compare, and the surprising irrelevance of render mode.

**Post 4a: Building 3D-DupBench** — Benchmark design philosophy, the five-tier clone generation process, Objaverse-LVIS as a source dataset, evaluation metrics and protocol, and lessons from building a benchmark with controlled difficulty.

**Post 4b: Results & Production** — VLM rescorer design and ablation studies, FAISS index selection at scale, threshold calibration, cost analysis (~$0.006 per model for the full pipeline), and a production deployment architecture.

Each post stands on its own — you can read Post 3 (embeddings) without Post 2 (rendering) if that's where your interest lies. But the posts build on each other: understanding why multi-view matters (Post 2) makes the aggregation strategies in Post 3 more intuitive, and the benchmark design (Post 4a) sets up the production decisions in Post 4b.

---

## Code & Data

Everything in this series is reproducible:

- **Rendering pipeline**: nvdiffrast-based multi-view renderer supporting textured and LFD modes, configurable camera layouts, and automatic blank-image detection
- **Embedding extraction**: DINOv2 and CLIP feature extraction with per-view and aggregated embedding support
- **3D-DupBench**: 3,056 models (191 sources + 2,865 clones across 5 tiers), all derived from CC-BY-4.0 Objaverse-LVIS models
- **FAISS benchmarks**: Reproducible scaling experiments from 1K to 1M vectors across 7 index types
- **VLM rescorer**: Gemini-based multi-angle verification pipeline with configurable evidence thresholds

The complete benchmark dataset — 171,136 rendered images, pre-computed embeddings for all 32 configurations, and the clone manifest — will be released alongside the final post in this series.

---

*This is Part 1 of 5 in the series "Detecting 3D Near-Duplicates at Scale." Next up: Post 2 — Multi-View Rendering, where I'll walk through the camera placement strategy and the rendering pipeline that makes everything else possible.*

---

### References

1. Gadhave, S., et al. "Contamination in Commonly Used Benchmark Datasets for Point Cloud Analysis." *DMLR Workshop, ICML 2023*. [Proceedings](https://proceedings.mlr.press/v210/gadhave23a.html)
2. Oquab, M., et al. "DINOv2: Learning Robust Visual Features without Supervision." *arXiv:2304.07193*, 2023. [Paper](https://arxiv.org/abs/2304.07193)
3. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021*. [Paper](https://arxiv.org/abs/2103.00020)
4. Deitke, M., et al. "Objaverse: A Universe of Annotated 3D Objects." *CVPR 2023*. [Paper](https://arxiv.org/abs/2212.08051)
5. Johnson, J., Douze, M., and Jégou, H. "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data*, 2021. [FAISS](https://github.com/facebookresearch/faiss)
6. Laine, S., et al. "nvdiffrast – Modular Primitives for High-Performance Differentiable Rendering." *ACM Transactions on Graphics*, 2020. [Paper](https://arxiv.org/abs/2011.03277)
7. Chen, D., et al. "Objaverse-LVIS annotations." [Objaverse-LVIS](https://objaverse.allenai.org/)
