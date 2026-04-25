# 3D-DupBench: Detecting 3D Near-Duplicates at Scale

A complete pipeline for detecting near-duplicate 3D models using multi-view rendering, vision foundation model embeddings, and VLM-based verification. Includes **3D-DupBench**, the first standardized benchmark for 3D near-duplicate detection with graduated difficulty tiers.

## Key Findings

- **Multi-view embeddings beat single-view by 88.7%** on ModelNet40 retrieval (mAP 0.497 vs 0.263)
- **DINOv2 + concat-PCA aggregation** is the best frozen-encoder configuration (mAP 0.429 on 3D-DupBench)
- **ModelNet40 has 32.8% test set contamination** — near-duplicate models leak across train/test splits, inflating reported classification accuracy by 7.7 percentage points
- **VLM rescoring (Gemini 2.0 Flash)** pushes duplicate detection precision from 63% to 95%
- **FAISS IVF256-SQ8** searches 1M vectors in 6.2ms with 97.7% recall@10

## Pipeline Architecture

```
Raw 3D Mesh → Multi-View Render (28 views) → Embed (DINOv2/CLIP) → Search (FAISS) → Verify (VLM)
                  nvdiffrast                    frozen encoder        ANN index      Gemini Flash
                  ~1.1s/model                   ~0.3s/model          <1ms/query      ~3s/flagged
```

## 3D-DupBench Benchmark

191 source models from Objaverse-LVIS with 2,865 synthetic clones across 5 difficulty tiers:

| Tier | Difficulty | Transformation | Clone Count |
|------|-----------|----------------|-------------|
| T1 | Trivial | Format re-export (identical geometry) | 573 |
| T2 | Easy | Uniform scale + rotation | 573 |
| T3 | Medium | Non-uniform scale + vertex noise + decimation | 573 |
| T4 | Hard | Partial mesh removal + noise + scale | 573 |
| T5 | Adversarial | Topology change + subdivision + deformation | 573 |

## Repository Structure

```
├── scripts/
│   ├── data_prep/              # Download, convert, generate clones
│   │   ├── download_lvis.py
│   │   ├── convert_to_obj.py
│   │   ├── generate_clones_5tier.py
│   │   └── validate_clones.py
│   ├── rendering/              # Multi-view rendering (nvdiffrast)
│   │   └── render_multiview.py
│   ├── embeddings/             # DINOv2/CLIP embedding extraction
│   │   └── compute_embeddings.py
│   ├── evaluation/             # Retrieval evaluation & baselines
│   │   ├── evaluate_bakeoff.py
│   │   ├── compute_geometry_baselines.py
│   │   ├── generalized_rescorer.py
│   │   ├── run_ablations.py
│   │   └── offline_ablations.py
│   └── visualization/          # Figure generation
│       ├── generate_plots.py
│       └── generate_rescorer_plots.py
├── results/                    # Evaluation results and benchmark data
│   ├── bakeoff/                # 40-config embedding bake-off results
│   ├── faiss/                  # FAISS scaling benchmarks
│   ├── rescorer/               # VLM rescorer ablation studies
│   ├── modelnet40_audit/       # ModelNet40 contamination audit
│   └── geometry/               # Geometry baseline results
├── blog/                       # TDS blog series (5 posts + figures)
│   ├── posts/
│   └── figures/
├── docs/                       # Additional documentation
├── requirements.txt
├── LICENSE                     # MIT (code)
└── DATA_LICENSE                # CC-BY-4.0 (benchmark data)
```

## Quick Start

### Setup

```bash
conda create -n 3d-dedup python=3.10
conda activate 3d-dedup
pip install -r requirements.txt
```

### Hardware Requirements

- **Minimum:** 1x GPU with 8GB VRAM (rendering + embeddings)
- **Recommended:** 1x T4/A10G (16-24GB VRAM for FAISS-GPU at scale)
- Full pipeline runs in ~2 hours on a single T4

### Run the Pipeline

```bash
# 1. Download Objaverse-LVIS source models
python scripts/data_prep/download_lvis.py --output data/objaverse-lvis/

# 2. Convert to OBJ format
python scripts/data_prep/convert_to_obj.py --input data/objaverse-lvis/ --output data/obj/

# 3. Generate 5-tier clones
python scripts/data_prep/generate_clones_5tier.py --input data/obj/ --output data/clones/

# 4. Render 28 multi-view images per model
python scripts/rendering/render_multiview.py --input data/obj/ --output data/renders/

# 5. Compute embeddings (DINOv2-Giant, concat+PCA)
python scripts/embeddings/compute_embeddings.py --renders data/renders/ --output data/embeddings/

# 6. Evaluate retrieval performance
python scripts/evaluation/evaluate_bakeoff.py --embeddings data/embeddings/ --output results/
```

## Blog Series

This repository accompanies a 5-part blog series on Towards Data Science:

1. **Pipeline Overview** — End-to-end architecture and key results
2. **Multi-View Rendering** — Camera strategies, nvdiffrast, view count ablation
3. **Embedding Bake-Off** — DINOv2 vs CLIP, aggregation strategies, 40 configurations
4. **3D-DupBench + ModelNet40 Audit** — Benchmark design and contamination analysis
5. **Production Pipeline** — VLM verification, FAISS scaling, cost analysis

## Citation

If you use 3D-DupBench or this pipeline in your research, please cite:

```bibtex
@misc{wajjala2026threedupbench,
  title={3D-DupBench: Detecting Near-Duplicate 3D Models at Scale with Multi-View Embeddings},
  author={Wajjala, Harish},
  year={2026},
  url={https://github.com/pkhw2023-ship-it/3d-dedup}
}
```

## License

- **Code:** MIT License
- **Benchmark Data:** CC-BY-4.0 (source models retain original Objaverse-LVIS licenses)

## Acknowledgments

- [Objaverse-LVIS](https://objaverse.allenai.org/) for source 3D models
- [DINOv2](https://github.com/facebookresearch/dinov2) and [CLIP](https://github.com/openai/CLIP) for vision embeddings
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast) for GPU-accelerated differentiable rendering
- [FAISS](https://github.com/facebookresearch/faiss) for scalable nearest-neighbor search
