# ModelNet40 Natural Duplicate Audit Report

**Date:** 2026-04-24  
**Method:** DINOv2 multi-view textured embeddings (768-dim), cosine similarity  
**Dataset:** 12,311 models across 40 categories (9,843 train / 2,468 test)  
**Reference:** "Examining ModelNet40: Rethinking Data Quality in 3D Shape Classification" (DMLR @ ICML 2023)

---

## Executive Summary

ModelNet40 contains **widespread near-duplicate contamination** between its train and test splits. Using DINOv2 multi-view embeddings, we identified **5,784 cross-split duplicate pairs** (similarity > 0.95), affecting **809 of 2,468 test models (32.8%)**. Many pairs have similarity = 1.0000, indicating visually identical models leaked across splits. This contamination inflates 1-NN benchmark accuracy by **up to 7.7 percentage points** (9.2% relative).

---

## Key Findings

### 1. Scale of Contamination

| Threshold | Total Pairs | Cross-Split Pairs | Affected Test Models |
|-----------|------------|-------------------|---------------------|
| >= 0.995  | 498        | ~308              | 169 (6.8%)          |
| >= 0.990  | 905        | 308               | 169 (6.8%)          |
| >= 0.980  | 2,896      | 1,078             | 325 (13.2%)         |
| >= 0.970  | 5,622      | 2,072             | 476 (19.3%)         |
| >= 0.950  | 15,849     | 5,784             | 809 (32.8%)         |
| >= 0.900  | 169,827    | —                 | —                   |

### 2. Most Affected Categories

| Category    | Cross-Split Dup Pairs | Test Models | Dup Test Models | Accuracy Drop |
|-------------|----------------------|-------------|-----------------|---------------|
| **glass_box**   | 2,018                | 100         | 92 (92%)        | 46.0 pp       |
| **mantel**      | 875                  | 100         | 82 (82%)        | 50.1 pp       |
| **bookshelf**   | 652                  | 100         | 55 (55%)        | 18.3 pp       |
| **range_hood**  | 564                  | 100         | 78 (78%)        | 46.1 pp       |
| **toilet**      | 288                  | 100         | 55 (55%)        | 2.4 pp        |
| **airplane**    | 255                  | 100         | 27 (27%)        | 19.6 pp       |
| **dresser**     | 223                  | 86          | 50 (58%)        | 11.8 pp       |

- **glass_box** is the worst offender: 92 of 100 test models are near-duplicates of training models
- **mantel** has 82% test set contamination and the largest accuracy drop (50 pp)
- **range_hood** shows 78% contamination with 46 pp accuracy impact

### 3. Pair Type Breakdown (sim > 0.95)

- **Cross-split, same category** (train/test contamination): 5,784 pairs (36.5%)
- **Same-split, same category**: 9,751 pairs (61.5%)
- **Cross-category**: 314 pairs (2.0%)

Cross-category pairs are dominated by **flower_pot ↔ plant** confusion (top similarity: 0.999), suggesting these categories share extremely similar 3D shapes.

### 4. Benchmark Impact

Using a 1-NN classifier (cosine distance) on DINOv2 embeddings:

| Scenario | Test Set Size | Accuracy |
|----------|--------------|----------|
| Full test set | 2,468 | **83.43%** |
| Remove dups (sim >= 0.99) | 2,299 | 82.21% (↓1.2 pp) |
| Remove dups (sim >= 0.98) | 2,143 | 80.91% (↓2.5 pp) |
| Remove dups (sim >= 0.97) | 1,992 | 79.47% (↓4.0 pp) |
| Remove dups (sim >= 0.95) | 1,659 | **75.77%** (↓7.7 pp, **9.2% relative**) |

**Critical finding:** Test models that are near-duplicates of training models achieve **100% accuracy** at the 0.99 threshold and 99.1% at the 0.95 threshold. These are effectively "free points" that inflate reported accuracy.

### 5. Per-Category Accuracy Impact (Top 5)

| Category | Full Accuracy | Clean Accuracy | Drop |
|----------|-------------|----------------|------|
| mantel | 89.0% | 38.9% | **50.1 pp** |
| range_hood | 87.0% | 40.9% | **46.1 pp** |
| glass_box | 96.0% | 50.0% | **46.0 pp** |
| airplane | 47.0% | 27.4% | **19.6 pp** |
| bookshelf | 85.0% | 66.7% | **18.3 pp** |

For mantel, range_hood, and glass_box, **removing duplicates cuts accuracy in half**, revealing that the high reported accuracy on these categories is almost entirely driven by train/test leakage.

---

## Methodology Notes

- **Embeddings:** DINOv2 ViT-B/14, multi-view (28 views per model), textured renders, averaged into 768-dim representation
- **Similarity:** Cosine similarity on L2-normalized embeddings
- **Impact measurement:** 1-NN classifier (k=1, cosine metric) trained on full training set; accuracy measured with and without duplicate test models
- **"Duplicate" definition:** A test model is flagged as a duplicate if it has cosine similarity >= threshold with any training model in the same category

---

## Comparison to ICML 2023 Paper

Our findings **confirm and extend** the DMLR/ICML 2023 paper's findings:
- They identified ~0.5-2% exact duplicates; we find a much broader spectrum of near-duplicates (up to 32.8% of test set at sim >= 0.95)
- Our multi-view DINOv2 approach captures visual similarity that geometric-only methods might miss
- The most affected categories (glass_box, mantel, dresser) align with their reported findings
- Our impact quantification (7.7 pp accuracy inflation) provides concrete evidence of benchmark unreliability

---

## Implications

1. **Benchmark scores on ModelNet40 are inflated** by data leakage, particularly for categories like glass_box, mantel, and range_hood
2. **Category-level accuracy is misleading** — some categories appear "easy" only because their test sets are copies of training data
3. **Fair comparisons require duplicate-aware evaluation** — either removing duplicates or reporting both full and clean-set accuracy
4. **New benchmarks are needed** — or at minimum, a cleaned version of ModelNet40 with cross-split duplicates removed

---

## Output Files

| File | Description |
|------|-------------|
| `audit_results.json` | All duplicate pairs with similarity scores and category stats |
| `impact_results.json` | KNN accuracy impact at multiple thresholds, per-category breakdown |
| `model_meta.json` | Model metadata (ID, category, split) |
| `embeddings_norm.npy` | L2-normalized embeddings for downstream use |
| `sim_matrix_stats.npz` | Per-model max/mean similarity statistics |
| `comparisons/` | Side-by-side render images of top 20 duplicate pairs |
| `plot_similarity_histogram.png` | Distribution of max pairwise similarity |
| `plot_category_duplicates.png` | Category-level duplicate pair counts |
| `plot_impact_barchart.png` | Accuracy with/without duplicates by threshold |
| `plot_per_category_impact.png` | Per-category accuracy drop |
| `plot_pair_breakdown.png` | Pair type breakdown (donut chart) |
| `plot_duplicate_grid.png` | Example duplicate pairs grid |
