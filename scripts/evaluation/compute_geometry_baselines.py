#!/usr/bin/env python3
"""compute_geometry_baselines.py — Fast geometry-only baselines.

Loads OBJ meshes using a lightweight vertex-only parser, computes
Chamfer/Hausdorff/SA+Volume distances, and appends to bakeoff_results.json.

Designed to run faster than the full trimesh approach by:
  1. Using a simple OBJ vertex+face parser (no textures/materials)
  2. Reducing point cloud samples to 2000
  3. Only computing within-group + 20 random cross-group distances
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

N_SAMPLES = 2000
N_NEG_SAMPLES = 20


def load_obj_fast(filepath):
    """Fast OBJ loader — only reads vertices and faces, ignores textures/materials."""
    vertices = []
    faces = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.split()[1:]
                    # Handle f v, f v/vt, f v/vt/vn, f v//vn
                    face_verts = []
                    for p in parts:
                        idx = int(p.split('/')[0]) - 1  # OBJ is 1-indexed
                        face_verts.append(idx)
                    if len(face_verts) >= 3:
                        # Triangulate polygons
                        for i in range(1, len(face_verts) - 1):
                            faces.append([face_verts[0], face_verts[i], face_verts[i+1]])
    except Exception:
        return None, None

    if len(vertices) < 3 or len(faces) < 1:
        return None, None

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def normalize_to_unit_sphere(vertices):
    """Center and scale vertices to fit in unit sphere."""
    centroid = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices = vertices - centroid
    scale = np.max(vertices.max(axis=0) - vertices.min(axis=0))
    if scale > 0:
        vertices = vertices / scale
    return vertices


def sample_surface(vertices, faces, n_samples):
    """Sample points uniformly on mesh surface."""
    # Compute face areas
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = areas.sum()

    if total_area < 1e-10:
        # Degenerate mesh — sample from vertices
        idx = np.random.choice(len(vertices), min(n_samples, len(vertices)), replace=True)
        return vertices[idx]

    # Sample faces weighted by area
    probs = areas / total_area
    face_idx = np.random.choice(len(faces), n_samples, p=probs, replace=True)

    # Sample random point within each triangle
    r1 = np.random.rand(n_samples, 1)
    r2 = np.random.rand(n_samples, 1)
    sqrt_r1 = np.sqrt(r1)

    v0_s = vertices[faces[face_idx, 0]]
    v1_s = vertices[faces[face_idx, 1]]
    v2_s = vertices[faces[face_idx, 2]]

    points = (1 - sqrt_r1) * v0_s + sqrt_r1 * (1 - r2) * v1_s + sqrt_r1 * r2 * v2_s
    return points.astype(np.float64), total_area


def load_ground_truth():
    with open(CLONE_MANIFEST) as f:
        clones = json.load(f)
    with open(SOURCE_MANIFEST) as f:
        sources = json.load(f)
    clone_to_source = {}
    clone_to_tier = {}
    source_uids = set(sources["uids"])
    for c in clones:
        clone_to_source[c["clone_id"]] = c["source_uid"]
        clone_to_tier[c["clone_id"]] = c["tier"]
    return clone_to_source, clone_to_tier, source_uids


def get_source_id(model_id, clone_to_source, source_uids):
    if model_id in source_uids:
        return model_id
    return clone_to_source.get(model_id, model_id)


def evaluate_retrieval(distance_matrix, model_ids, clone_to_source, clone_to_tier,
                       source_uids, tier_filter=None):
    """Evaluate retrieval quality using standard metrics."""
    N = len(model_ids)
    groups = defaultdict(list)
    for idx, mid in enumerate(model_ids):
        src = get_source_id(mid, clone_to_source, source_uids)
        groups[src].append(idx)

    aps = []
    p_at_k = {1: [], 5: [], 10: []}
    r_at_k = {10: [], 50: []}

    for i in range(N):
        mid = model_ids[i]
        if tier_filter is not None:
            t = clone_to_tier.get(mid)
            if t != tier_filter and mid in source_uids:
                continue
            if t is not None and t != tier_filter:
                continue

        src = get_source_id(mid, clone_to_source, source_uids)
        group = groups.get(src, [])
        relevant = [j for j in group if j != i]

        if len(relevant) == 0:
            continue

        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        finite_mask = np.isfinite(dists)
        if finite_mask.sum() < 10:
            continue

        ranked = np.argsort(dists)
        ranked = [j for j in ranked if np.isfinite(dists[j])]
        if len(ranked) == 0:
            continue

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

        for k in p_at_k:
            top_k = ranked[:min(k, len(ranked))]
            n_hit = sum(1 for j in top_k if j in relevant_set)
            p_at_k[k].append(n_hit / k)
        for k in r_at_k:
            top_k = ranked[:min(k, len(ranked))]
            n_hit = sum(1 for j in top_k if j in relevant_set)
            r_at_k[k].append(n_hit / n_relevant if n_relevant > 0 else 0)

    results = {"mAP": float(np.mean(aps)) if aps else 0.0, "n_queries": len(aps)}
    for k, vals in p_at_k.items():
        results[f"P@{k}"] = float(np.mean(vals)) if vals else 0.0
    for k, vals in r_at_k.items():
        results[f"R@{k}"] = float(np.mean(vals)) if vals else 0.0
    return results


def main():
    print("=" * 60)
    print("GEOMETRY BASELINES — Fast Computation")
    print("=" * 60)

    clone_to_source, clone_to_tier, source_uids = load_ground_truth()

    # Get model IDs from embedding file
    ref_file = None
    for mk in ["dinov2_base", "dinov2_giant", "clip_large", "clip_base"]:
        for mode in ["textured", "lfd"]:
            f = EMB_DIR / f"{mk}_{mode}_perview.npz"
            if f.exists():
                ref_file = f
                break
        if ref_file:
            break

    data = np.load(ref_file, allow_pickle=True)
    model_ids = list(data["model_ids"])
    N = len(model_ids)
    print(f"Models: {N}")

    # Load meshes
    print(f"\nLoading {N} meshes (fast OBJ parser)...")
    start = time.time()
    point_clouds = [None] * N
    mesh_areas = np.zeros(N)
    valid_mask = np.ones(N, dtype=bool)
    np.random.seed(42)

    for idx, mid in enumerate(tqdm(model_ids, desc="Loading meshes")):
        mesh_path = None
        if mid in source_uids:
            mesh_path = SOURCE_DIR / f"{mid}.obj"
        else:
            # Parse tier from clone ID (e.g., "abc123_T1_v0")
            if "_T" in mid:
                parts = mid.split("_")
                tier = parts[-2]  # T1, T2, etc.
                mesh_path = CLONE_DIR / tier / f"{mid}.obj"

        if mesh_path is None or not mesh_path.exists():
            valid_mask[idx] = False
            continue

        # Skip overly complex meshes (file size > 5MB)
        file_size = mesh_path.stat().st_size
        if file_size > 5 * 1024 * 1024:
            valid_mask[idx] = False
            continue

        vertices, faces = load_obj_fast(str(mesh_path))
        if vertices is None:
            valid_mask[idx] = False
            continue

        # Skip meshes with too many vertices (>200K)
        if len(vertices) > 200_000:
            valid_mask[idx] = False
            continue

        vertices = normalize_to_unit_sphere(vertices)
        result = sample_surface(vertices, faces, N_SAMPLES)
        if isinstance(result, tuple):
            points, area = result
        else:
            points = result
            area = 0

        point_clouds[idx] = points.astype(np.float32)
        mesh_areas[idx] = area

    elapsed = time.time() - start
    n_valid = valid_mask.sum()
    print(f"Loaded {n_valid}/{N} meshes in {elapsed:.1f}s ({N/elapsed:.0f} meshes/sec)")

    # Compute distances
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import directed_hausdorff

    chamfer_dist = np.full((N, N), np.inf, dtype=np.float32)
    hausdorff_dist = np.full((N, N), np.inf, dtype=np.float32)
    sa_vol_dist = np.full((N, N), np.inf, dtype=np.float32)
    np.fill_diagonal(chamfer_dist, 0)
    np.fill_diagonal(hausdorff_dist, 0)
    np.fill_diagonal(sa_vol_dist, 0)

    # Build source groups
    groups = defaultdict(list)
    for idx, mid in enumerate(model_ids):
        if valid_mask[idx]:
            src = get_source_id(mid, clone_to_source, source_uids)
            groups[src].append(idx)

    # Within-group distances
    print("\nComputing within-group distances...")
    start = time.time()
    n_pairs = 0
    for src, indices in tqdm(groups.items(), desc="Within-group"):
        trees = {}
        for i in indices:
            if point_clouds[i] is not None:
                trees[i] = cKDTree(point_clouds[i])

        for p, i in enumerate(indices):
            if i not in trees:
                continue
            for j in indices[p+1:]:
                if j not in trees:
                    continue

                d_ij = trees[i].query(point_clouds[j])[0].mean()
                d_ji = trees[j].query(point_clouds[i])[0].mean()
                chamfer_dist[i, j] = chamfer_dist[j, i] = (d_ij + d_ji) / 2

                h_ij = directed_hausdorff(point_clouds[i], point_clouds[j])[0]
                h_ji = directed_hausdorff(point_clouds[j], point_clouds[i])[0]
                hausdorff_dist[i, j] = hausdorff_dist[j, i] = max(h_ij, h_ji)

                a_i, a_j = mesh_areas[i], mesh_areas[j]
                if a_i > 0 and a_j > 0:
                    sa_ratio = min(a_i, a_j) / max(a_i, a_j)
                    sa_vol_dist[i, j] = sa_vol_dist[j, i] = 1.0 - sa_ratio

                n_pairs += 1

    elapsed = time.time() - start
    print(f"  {n_pairs} within-group pairs in {elapsed:.1f}s")

    # Cross-group distances (sampled)
    print("\nComputing cross-group distances (sampled)...")
    start = time.time()
    all_valid = [i for i in range(N) if valid_mask[i]]
    rng = np.random.RandomState(42)
    n_cross = 0

    for i in tqdm(all_valid, desc="Cross-group"):
        if point_clouds[i] is None:
            continue
        src_i = get_source_id(model_ids[i], clone_to_source, source_uids)
        tree_i = cKDTree(point_clouds[i])

        non_group = [j for j in all_valid if get_source_id(model_ids[j], clone_to_source, source_uids) != src_i]
        if len(non_group) > N_NEG_SAMPLES:
            non_group = rng.choice(non_group, N_NEG_SAMPLES, replace=False)

        for j in non_group:
            if chamfer_dist[i, j] < np.inf:
                continue
            if point_clouds[j] is None:
                continue
            tree_j = cKDTree(point_clouds[j])

            d_ij = tree_i.query(point_clouds[j])[0].mean()
            d_ji = tree_j.query(point_clouds[i])[0].mean()
            chamfer_dist[i, j] = chamfer_dist[j, i] = (d_ij + d_ji) / 2

            h_ij = directed_hausdorff(point_clouds[i], point_clouds[j])[0]
            h_ji = directed_hausdorff(point_clouds[j], point_clouds[i])[0]
            hausdorff_dist[i, j] = hausdorff_dist[j, i] = max(h_ij, h_ji)

            n_cross += 1

    elapsed = time.time() - start
    print(f"  {n_cross} cross-group pairs in {elapsed:.1f}s")

    # Save geometry distances
    np.savez_compressed(
        DATA_DIR / "geometry_distances.npz",
        chamfer=chamfer_dist,
        hausdorff=hausdorff_dist,
        sa_vol=sa_vol_dist,
        model_ids=np.array(model_ids),
        valid_mask=valid_mask,
    )
    print(f"Saved geometry distances to {DATA_DIR / 'geometry_distances.npz'}")

    # Evaluate
    print("\nEvaluating geometry baselines...")
    geo_results = {}

    for name, dist_matrix in [("chamfer", chamfer_dist), ("hausdorff", hausdorff_dist), ("sa_volume", sa_vol_dist)]:
        results = evaluate_retrieval(dist_matrix, model_ids, clone_to_source, clone_to_tier, source_uids)
        print(f"  {name}: mAP={results['mAP']:.4f}  P@1={results['P@1']:.4f}")
        for tier in ["T1", "T2", "T3", "T4", "T5"]:
            tr = evaluate_retrieval(dist_matrix, model_ids, clone_to_source, clone_to_tier, source_uids, tier_filter=tier)
            results[f"mAP_{tier}"] = tr["mAP"]
            results[f"P@1_{tier}"] = tr["P@1"]
        geo_results[name] = results

    # Merge into existing results
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results.update(geo_results)
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nUpdated results saved to {RESULTS_FILE}")

    print(f"\n{'=' * 60}")
    print("GEOMETRY BASELINES COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
