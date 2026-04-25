#!/usr/bin/env python3
"""evaluate_bakeoff.py — Aggregation, geometry baselines, and retrieval evaluation.

Loads per-view embeddings from compute_embeddings.py, applies aggregation strategies,
computes geometry baselines, evaluates retrieval quality, and saves results JSON.

Usage:
    python evaluate_bakeoff.py
"""

import json
import numpy as np
import time
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────────────
DATA_DIR = Path("/home/lightsail-user/3d-dataset-storage/tds-blog/data")
EMB_DIR = DATA_DIR / "embeddings"
CLONE_MANIFEST = DATA_DIR / "clones-objaverse" / "clone_manifest_5tier.json"
SOURCE_MANIFEST = DATA_DIR / "clones-objaverse" / "source_manifest.json"
SOURCE_DIR = DATA_DIR / "clones-objaverse" / "sources"
CLONE_DIR = DATA_DIR / "clones-objaverse" / "clones"
RESULTS_FILE = DATA_DIR / "bakeoff_results.json"

EMBEDDING_MODELS = ["dinov2_base", "dinov2_giant", "clip_large", "clip_base"]
RENDER_MODES = ["textured", "lfd"]


# ═══════════════════════════════════════════════════════════════════════
# 1. Ground Truth Setup
# ═══════════════════════════════════════════════════════════════════════

def load_ground_truth():
    """Load clone manifest and build ground truth mappings."""
    with open(CLONE_MANIFEST) as f:
        clones = json.load(f)
    with open(SOURCE_MANIFEST) as f:
        sources = json.load(f)

    # Build mappings
    clone_to_source = {}  # clone_id -> source_uid
    clone_to_tier = {}    # clone_id -> tier
    source_uids = set(sources["uids"])

    for c in clones:
        clone_to_source[c["clone_id"]] = c["source_uid"]
        clone_to_tier[c["clone_id"]] = c["tier"]

    return clone_to_source, clone_to_tier, source_uids, sources


def get_source_id(model_id, clone_to_source, source_uids):
    """Get the source UID for any model ID (source or clone)."""
    if model_id in source_uids:
        return model_id
    return clone_to_source.get(model_id, model_id)


def build_relevance_groups(model_ids, clone_to_source, source_uids):
    """Build groups of models that share the same source.

    Returns: dict mapping source_uid -> list of indices in model_ids
    """
    groups = defaultdict(list)
    for idx, mid in enumerate(model_ids):
        src = get_source_id(mid, clone_to_source, source_uids)
        groups[src].append(idx)
    return groups


# ═══════════════════════════════════════════════════════════════════════
# 2. Aggregation Strategies
# ═══════════════════════════════════════════════════════════════════════

def aggregate_embeddings(per_view_embs, strategy, n_views=None):
    """Apply aggregation strategy to per-view embeddings.

    Args:
        per_view_embs: (N_models, 28, dim)
        strategy: str
        n_views: int, number of views to use (None = all 28)

    Returns: (N_models, out_dim)
    """
    if n_views is not None:
        per_view_embs = per_view_embs[:, :n_views, :]

    N, V, D = per_view_embs.shape

    if strategy == "single":
        result = per_view_embs[:, 0, :]
    elif strategy == "mean":
        result = per_view_embs.mean(axis=1)
    elif strategy == "max":
        result = per_view_embs.max(axis=1)
    elif strategy == "concat_pca":
        from sklearn.decomposition import PCA
        flat = per_view_embs.reshape(N, -1)  # (N, V*D)
        target_dim = min(768, flat.shape[1], N - 1)
        pca = PCA(n_components=target_dim)
        result = pca.fit_transform(flat)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # L2 normalize
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    result = result / norms

    return result


# ═══════════════════════════════════════════════════════════════════════
# 3. Distance / Similarity Computation
# ═══════════════════════════════════════════════════════════════════════

def compute_cosine_distance_matrix(embeddings):
    """Compute pairwise cosine distance matrix.

    Args:
        embeddings: (N, D), L2-normalized

    Returns: (N, N) distance matrix where 0 = identical, 2 = opposite
    """
    # For normalized vectors: cosine_sim = dot product
    sim = embeddings @ embeddings.T
    dist = 1.0 - sim
    return dist


# ═══════════════════════════════════════════════════════════════════════
# 4. Geometry Baselines
# ═══════════════════════════════════════════════════════════════════════

def compute_geometry_baselines(model_ids, clone_to_source, source_uids):
    """Compute Chamfer and Hausdorff distance matrices using mesh geometry."""
    import trimesh
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import directed_hausdorff

    N_SAMPLES = 2000  # 2K points sufficient for unit-sphere-normalized meshes
    N = len(model_ids)

    print(f"Loading {N} meshes for geometry baselines...")
    point_clouds = []
    mesh_stats = []  # (surface_area, volume)
    valid_mask = np.ones(N, dtype=bool)

    for idx, mid in enumerate(tqdm(model_ids, desc="Loading meshes")):
        mesh_path = None
        if mid in source_uids:
            mesh_path = SOURCE_DIR / f"{mid}.obj"
        else:
            tier = mid.split("_")[-2] if "_T" in mid else None
            if tier:
                mesh_path = CLONE_DIR / tier / f"{mid}.obj"

        if mesh_path is None or not mesh_path.exists():
            point_clouds.append(None)
            mesh_stats.append((0, 0))
            valid_mask[idx] = False
            continue

        try:
            mesh = trimesh.load(str(mesh_path), force='mesh', process=True)
            # Normalize to unit sphere
            centroid = mesh.bounding_box.centroid
            mesh.vertices -= centroid
            scale = mesh.bounding_box.extents.max()
            if scale > 0:
                mesh.vertices /= scale

            points = mesh.sample(N_SAMPLES)
            point_clouds.append(points)
            try:
                sa = mesh.area
                vol = abs(mesh.volume) if mesh.is_watertight else 0
            except Exception:
                sa = 0
                vol = 0
            mesh_stats.append((sa, vol))
        except Exception as e:
            point_clouds.append(None)
            mesh_stats.append((0, 0))
            valid_mask[idx] = False

    n_valid = valid_mask.sum()
    print(f"  Loaded {n_valid}/{N} meshes successfully")

    # Compute pairwise distances
    chamfer_dist = np.full((N, N), np.inf)
    hausdorff_dist = np.full((N, N), np.inf)
    sa_vol_dist = np.full((N, N), np.inf)

    np.fill_diagonal(chamfer_dist, 0)
    np.fill_diagonal(hausdorff_dist, 0)
    np.fill_diagonal(sa_vol_dist, 0)

    # We only need to compute distances for pairs that share a source
    # (for efficiency — we still compute a full matrix for evaluation)
    # But for a full bake-off, let's compute all pairs within reason
    # With 3056 models, full pairwise is ~4.7M pairs — too many
    # Instead: compute distances only within each source group

    # For the evaluation, we only need distances between each model and
    # all other models sharing the same source. For non-related models,
    # we'll sample a subset for negative pairs.

    from scipy.spatial import cKDTree

    # Build source groups
    groups = defaultdict(list)
    for idx, mid in enumerate(model_ids):
        if valid_mask[idx]:
            src = get_source_id(mid, clone_to_source, source_uids)
            groups[src].append(idx)

    # Compute within-group distances
    print("Computing geometry distances within source groups...")
    for src, indices in tqdm(groups.items(), desc="Geometry distances"):
        for i_pos, i in enumerate(indices):
            if point_clouds[i] is None:
                continue
            tree_i = cKDTree(point_clouds[i])
            for j in indices[i_pos + 1:]:
                if point_clouds[j] is None:
                    continue
                tree_j = cKDTree(point_clouds[j])

                # Chamfer
                d_ij = tree_i.query(point_clouds[j])[0].mean()
                d_ji = tree_j.query(point_clouds[i])[0].mean()
                chamfer_dist[i, j] = chamfer_dist[j, i] = (d_ij + d_ji) / 2

                # Hausdorff
                h_ij = directed_hausdorff(point_clouds[i], point_clouds[j])[0]
                h_ji = directed_hausdorff(point_clouds[j], point_clouds[i])[0]
                hausdorff_dist[i, j] = hausdorff_dist[j, i] = max(h_ij, h_ji)

                # SA+Volume ratio
                sa_i, vol_i = mesh_stats[i]
                sa_j, vol_j = mesh_stats[j]
                if sa_i > 0 and sa_j > 0:
                    sa_ratio = min(sa_i, sa_j) / max(sa_i, sa_j)
                    vol_ratio = min(vol_i + 1e-10, vol_j + 1e-10) / max(vol_i + 1e-10, vol_j + 1e-10)
                    sa_vol_dist[i, j] = sa_vol_dist[j, i] = 1.0 - (sa_ratio + vol_ratio) / 2

    # For negative pairs, sample random cross-group distances
    print("Computing cross-group geometry distances (sample)...")
    all_valid = [i for i in range(N) if valid_mask[i]]
    rng = np.random.RandomState(42)

    # For each model, compute distance to a random sample of non-group models
    n_neg_samples = min(20, len(all_valid) - 1)
    for i in tqdm(all_valid, desc="Cross-group distances"):
        if point_clouds[i] is None:
            continue
        src_i = get_source_id(model_ids[i], clone_to_source, source_uids)
        tree_i = cKDTree(point_clouds[i])

        non_group = [j for j in all_valid if get_source_id(model_ids[j], clone_to_source, source_uids) != src_i]
        if len(non_group) > n_neg_samples:
            non_group = rng.choice(non_group, n_neg_samples, replace=False)

        for j in non_group:
            if chamfer_dist[i, j] < np.inf:
                continue  # already computed
            if point_clouds[j] is None:
                continue
            tree_j = cKDTree(point_clouds[j])

            d_ij = tree_i.query(point_clouds[j])[0].mean()
            d_ji = tree_j.query(point_clouds[i])[0].mean()
            chamfer_dist[i, j] = chamfer_dist[j, i] = (d_ij + d_ji) / 2

            h_ij = directed_hausdorff(point_clouds[i], point_clouds[j])[0]
            h_ji = directed_hausdorff(point_clouds[j], point_clouds[i])[0]
            hausdorff_dist[i, j] = hausdorff_dist[j, i] = max(h_ij, h_ji)

    return chamfer_dist, hausdorff_dist, sa_vol_dist, valid_mask


# ═══════════════════════════════════════════════════════════════════════
# 5. Retrieval Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_retrieval(distance_matrix, model_ids, clone_to_source, clone_to_tier,
                       source_uids, tier_filter=None):
    """Evaluate retrieval quality using standard metrics.

    Args:
        distance_matrix: (N, N) pairwise distances
        model_ids: list of model IDs
        clone_to_source, clone_to_tier, source_uids: ground truth
        tier_filter: if set, only evaluate queries from this tier

    Returns: dict with mAP, P@1, P@5, P@10, R@10, R@50
    """
    N = len(model_ids)
    id_to_idx = {mid: i for i, mid in enumerate(model_ids)}

    # Build relevance: for each model, its relevant set = same source (excluding self)
    relevance_groups = build_relevance_groups(model_ids, clone_to_source, source_uids)

    aps = []
    p_at_k = {1: [], 5: [], 10: []}
    r_at_k = {10: [], 50: []}

    for i in range(N):
        mid = model_ids[i]

        # Apply tier filter
        if tier_filter is not None:
            t = clone_to_tier.get(mid)
            if t != tier_filter and mid in source_uids:
                continue  # Skip sources when filtering by tier
            if t is not None and t != tier_filter:
                continue

        src = get_source_id(mid, clone_to_source, source_uids)
        group = relevance_groups.get(src, [])
        relevant = [j for j in group if j != i]

        if len(relevant) == 0:
            continue

        # Get distances to all other models
        dists = distance_matrix[i].copy()
        dists[i] = np.inf  # exclude self

        # Check for inf distances (geometry baselines may have sparse matrices)
        finite_mask = np.isfinite(dists)
        if finite_mask.sum() < 10:
            continue  # skip if too few finite distances

        # Rank by distance (ascending)
        ranked = np.argsort(dists)
        # Only keep indices with finite distances
        ranked = [j for j in ranked if np.isfinite(dists[j])]

        if len(ranked) == 0:
            continue

        # Average Precision
        relevant_set = set(relevant)
        n_relevant = len(relevant_set)
        hits = 0
        precision_sum = 0
        for rank, j in enumerate(ranked):
            if j in relevant_set:
                hits += 1
                precision_sum += hits / (rank + 1)
        ap = precision_sum / n_relevant if n_relevant > 0 else 0
        aps.append(ap)

        # Precision@k
        for k in p_at_k:
            top_k = ranked[:min(k, len(ranked))]
            n_hit = sum(1 for j in top_k if j in relevant_set)
            p_at_k[k].append(n_hit / k)

        # Recall@k
        for k in r_at_k:
            top_k = ranked[:min(k, len(ranked))]
            n_hit = sum(1 for j in top_k if j in relevant_set)
            r_at_k[k].append(n_hit / n_relevant if n_relevant > 0 else 0)

    results = {
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "n_queries": len(aps),
    }
    for k, vals in p_at_k.items():
        results[f"P@{k}"] = float(np.mean(vals)) if vals else 0.0
    for k, vals in r_at_k.items():
        results[f"R@{k}"] = float(np.mean(vals)) if vals else 0.0

    return results


# ═══════════════════════════════════════════════════════════════════════
# 6. Main Pipeline
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-geometry", action="store_true", help="Skip geometry baselines")
    args = parser.parse_args()

    print("=" * 70)
    print("EMBEDDING BAKE-OFF — Evaluation Pipeline")
    print("=" * 70)

    # Load ground truth
    print("\nLoading ground truth...")
    clone_to_source, clone_to_tier, source_uids, sources = load_ground_truth()
    print(f"  Sources: {len(source_uids)}")
    print(f"  Clones:  {len(clone_to_source)}")

    all_results = {}

    # ─── Embedding-based evaluation ──────────────────────────────
    aggregation_configs = [
        ("single", None),      # single view (view 0)
        ("mean", 8),           # mean pool of 8 horizontal views
        ("mean", 28),          # mean pool of all 28 views
        ("max", 28),           # max pool of all 28 views
        ("concat_pca", 28),    # concat + PCA
    ]

    for model_key in EMBEDDING_MODELS:
        for mode in RENDER_MODES:
            emb_file = EMB_DIR / f"{model_key}_{mode}_perview.npz"
            if not emb_file.exists():
                print(f"\n  SKIP: {emb_file.name} not found")
                continue

            print(f"\n{'─' * 60}")
            print(f"Evaluating: {model_key} × {mode}")
            print(f"{'─' * 60}")

            data = np.load(emb_file, allow_pickle=True)
            model_ids = list(data["model_ids"])
            per_view_embs = data["embeddings"]  # (N, 28, dim)

            print(f"  Models: {len(model_ids)}, Views: {per_view_embs.shape[1]}, Dim: {per_view_embs.shape[2]}")

            for strategy, n_views in aggregation_configs:
                if n_views:
                    key = f"{model_key}_{mode}_{strategy}{n_views}"
                else:
                    key = f"{model_key}_{mode}_{strategy}"
                print(f"\n  Strategy: {key}")

                # Aggregate
                agg_embs = aggregate_embeddings(per_view_embs, strategy, n_views)
                print(f"    Aggregated shape: {agg_embs.shape}")

                # Compute distance matrix
                dist_matrix = compute_cosine_distance_matrix(agg_embs)

                # Overall evaluation
                results = evaluate_retrieval(
                    dist_matrix, model_ids,
                    clone_to_source, clone_to_tier, source_uids
                )
                print(f"    mAP={results['mAP']:.4f}  P@1={results['P@1']:.4f}  P@5={results['P@5']:.4f}  P@10={results['P@10']:.4f}")

                # Per-tier evaluation
                for tier in ["T1", "T2", "T3", "T4", "T5"]:
                    tier_results = evaluate_retrieval(
                        dist_matrix, model_ids,
                        clone_to_source, clone_to_tier, source_uids,
                        tier_filter=tier
                    )
                    results[f"mAP_{tier}"] = tier_results["mAP"]
                    results[f"P@1_{tier}"] = tier_results["P@1"]

                print(f"    Per-tier mAP: T1={results.get('mAP_T1',0):.4f}  T2={results.get('mAP_T2',0):.4f}  "
                      f"T3={results.get('mAP_T3',0):.4f}  T4={results.get('mAP_T4',0):.4f}  T5={results.get('mAP_T5',0):.4f}")

                all_results[key] = results

    # ─── View count ablation ─────────────────────────────────────
    # For the best model, compute mAP vs number of views
    print(f"\n{'─' * 60}")
    print("View Count Ablation")
    print(f"{'─' * 60}")

    ablation_results = {}
    for model_key in EMBEDDING_MODELS:
        for mode in RENDER_MODES:
            emb_file = EMB_DIR / f"{model_key}_{mode}_perview.npz"
            if not emb_file.exists():
                continue

            data = np.load(emb_file, allow_pickle=True)
            model_ids = list(data["model_ids"])
            per_view_embs = data["embeddings"]

            for n_views in [1, 4, 8, 12, 20, 28]:
                agg_embs = aggregate_embeddings(per_view_embs, "mean", n_views)
                dist_matrix = compute_cosine_distance_matrix(agg_embs)
                results = evaluate_retrieval(
                    dist_matrix, model_ids,
                    clone_to_source, clone_to_tier, source_uids
                )
                ablation_key = f"{model_key}_{mode}_mean{n_views}"
                ablation_results[ablation_key] = results["mAP"]
                print(f"  {ablation_key}: mAP={results['mAP']:.4f}")

    all_results["view_ablation"] = ablation_results

    # ─── Geometry baselines ──────────────────────────────────────
    if args.skip_geometry:
        print(f"\n{'─' * 60}")
        print("Geometry Baselines — SKIPPED (--skip-geometry)")
        print(f"{'─' * 60}")
    else:
        print(f"\n{'─' * 60}")
        print("Geometry Baselines")
        print(f"{'─' * 60}")

        # Use model_ids from the first available embedding file
        ref_emb_file = None
        for mk in EMBEDDING_MODELS:
            for m in RENDER_MODES:
                f = EMB_DIR / f"{mk}_{m}_perview.npz"
                if f.exists():
                    ref_emb_file = f
                    break
            if ref_emb_file:
                break

        if ref_emb_file:
            data = np.load(ref_emb_file, allow_pickle=True)
            model_ids = list(data["model_ids"])

            chamfer_dist, hausdorff_dist, sa_vol_dist, valid_mask = compute_geometry_baselines(
                model_ids, clone_to_source, source_uids
            )

            # Save geometry distances for plotting
            np.savez_compressed(
                DATA_DIR / "geometry_distances.npz",
                chamfer=chamfer_dist,
                hausdorff=hausdorff_dist,
                sa_vol=sa_vol_dist,
                model_ids=np.array(model_ids),
                valid_mask=valid_mask,
            )

            # Evaluate Chamfer
            chamfer_results = evaluate_retrieval(
                chamfer_dist, model_ids,
                clone_to_source, clone_to_tier, source_uids
            )
            print(f"  Chamfer:   mAP={chamfer_results['mAP']:.4f}  P@1={chamfer_results['P@1']:.4f}")
            for tier in ["T1", "T2", "T3", "T4", "T5"]:
                tr = evaluate_retrieval(chamfer_dist, model_ids, clone_to_source, clone_to_tier, source_uids, tier_filter=tier)
                chamfer_results[f"mAP_{tier}"] = tr["mAP"]
                chamfer_results[f"P@1_{tier}"] = tr["P@1"]
            all_results["chamfer"] = chamfer_results

            # Evaluate Hausdorff
            hausdorff_results = evaluate_retrieval(
                hausdorff_dist, model_ids,
                clone_to_source, clone_to_tier, source_uids
            )
            print(f"  Hausdorff: mAP={hausdorff_results['mAP']:.4f}  P@1={hausdorff_results['P@1']:.4f}")
            for tier in ["T1", "T2", "T3", "T4", "T5"]:
                tr = evaluate_retrieval(hausdorff_dist, model_ids, clone_to_source, clone_to_tier, source_uids, tier_filter=tier)
                hausdorff_results[f"mAP_{tier}"] = tr["mAP"]
                hausdorff_results[f"P@1_{tier}"] = tr["P@1"]
            all_results["hausdorff"] = hausdorff_results

            # Evaluate SA+Volume
            savol_results = evaluate_retrieval(
                sa_vol_dist, model_ids,
                clone_to_source, clone_to_tier, source_uids
            )
            print(f"  SA+Vol:    mAP={savol_results['mAP']:.4f}  P@1={savol_results['P@1']:.4f}")
            for tier in ["T1", "T2", "T3", "T4", "T5"]:
                tr = evaluate_retrieval(sa_vol_dist, model_ids, clone_to_source, clone_to_tier, source_uids, tier_filter=tier)
                savol_results[f"mAP_{tier}"] = tr["mAP"]
                savol_results[f"P@1_{tier}"] = tr["P@1"]
            all_results["sa_volume"] = savol_results

    # ─── Save results ────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"Total configurations evaluated: {len(all_results)}")
    print(f"{'=' * 70}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Method':<45} {'mAP':>6} {'P@1':>6} {'P@5':>6} {'P@10':>6}")
    print("-" * 90)
    for key, res in sorted(all_results.items()):
        if key == "view_ablation":
            continue
        print(f"{key:<45} {res.get('mAP',0):>6.4f} {res.get('P@1',0):>6.4f} {res.get('P@5',0):>6.4f} {res.get('P@10',0):>6.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
