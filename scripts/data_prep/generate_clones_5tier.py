#!/usr/bin/env python3
"""
Step 3: Generate clones at 5 difficulty tiers for 3D-DupBench.

Tier structure:
  T1 (Trivial)      — Re-export (format round-trip)
  T2 (Easy)         — Uniform scale + random rotation
  T3 (Medium)       — Non-uniform scale + vertex noise + decimation
  T4 (Hard)         — Partial mesh removal + noise + non-uniform scale
  T5 (Adversarial)  — Topology change (subdivision/remesh) + combined transforms
"""

import trimesh
import numpy as np
import json
import os
import io
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
CLONES_PER_TIER = 3  # 3 clones of each source model per tier
OUTPUT_BASE = "/home/lightsail-user/3d-dataset-storage/tds-blog/data/clones-objaverse/"


def tier1_trivial(mesh, variant_idx):
    """Re-export: round-trip through PLY format (introduces floating-point drift)."""
    # Export to PLY bytes
    ply_data = trimesh.exchange.ply.export_ply(mesh)
    # Reimport from PLY
    reimported = trimesh.load(io.BytesIO(ply_data), file_type='ply', force='mesh')
    return reimported


def tier2_easy(mesh, variant_idx):
    """Uniform scale + random rotation."""
    clone = mesh.copy()
    # Random uniform scale between 0.5 and 2.0
    scale = np.random.uniform(0.5, 2.0)
    clone.vertices *= scale
    # Random rotation around a random axis
    angle = np.random.uniform(0, 2 * np.pi)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    rot = trimesh.transformations.rotation_matrix(angle, axis)
    clone.apply_transform(rot)
    return clone


def tier3_medium(mesh, variant_idx):
    """Non-uniform scale + vertex noise + decimation."""
    clone = mesh.copy()
    # Non-uniform scale
    scale = np.random.uniform(0.7, 1.5, size=3)
    clone.vertices *= scale
    # Vertex noise
    sigma = np.random.uniform(0.01, 0.05)
    noise = np.random.normal(0, sigma, clone.vertices.shape)
    clone.vertices += noise
    # Decimation to 50-80% of original faces
    target_faces = int(len(clone.faces) * np.random.uniform(0.5, 0.8))
    if target_faces > 10:
        try:
            clone = clone.simplify_quadric_decimation(target_faces)
        except Exception:
            pass  # Keep undecimated if simplification fails
    return clone


def tier4_hard(mesh, variant_idx):
    """Partial mesh removal + noise + non-uniform scale."""
    clone = mesh.copy()
    # Remove 10-30% of faces (randomly)
    n_faces = len(clone.faces)
    n_remove = int(n_faces * np.random.uniform(0.1, 0.3))
    keep_mask = np.ones(n_faces, dtype=bool)
    remove_indices = np.random.choice(n_faces, n_remove, replace=False)
    keep_mask[remove_indices] = False
    clone.update_faces(keep_mask)
    clone.remove_unreferenced_vertices()

    if len(clone.faces) < 4:
        # If too few faces remain, fall back to a gentler removal
        clone = mesh.copy()
        n_remove = int(n_faces * 0.05)
        keep_mask = np.ones(n_faces, dtype=bool)
        remove_indices = np.random.choice(n_faces, n_remove, replace=False)
        keep_mask[remove_indices] = False
        clone.update_faces(keep_mask)
        clone.remove_unreferenced_vertices()

    # Non-uniform scale
    scale = np.random.uniform(0.6, 1.6, size=3)
    clone.vertices *= scale
    # Vertex noise
    sigma = np.random.uniform(0.02, 0.06)
    clone.vertices += np.random.normal(0, sigma, clone.vertices.shape)
    return clone


def tier5_adversarial(mesh, variant_idx):
    """Topology change (subdivision) + combined transforms."""
    clone = mesh.copy()

    # Subdivide to change topology — but cap face count to avoid memory issues
    max_faces_for_subdiv = 50000
    if len(clone.faces) <= max_faces_for_subdiv:
        try:
            v, f = trimesh.remesh.subdivide(clone.vertices, clone.faces)
            clone = trimesh.Trimesh(vertices=v, faces=f)
        except Exception:
            pass  # If subdivision fails, continue with original topology

    # Apply tier3-like transforms on top
    scale = np.random.uniform(0.7, 1.4, size=3)
    clone.vertices *= scale
    sigma = np.random.uniform(0.01, 0.04)
    clone.vertices += np.random.normal(0, sigma, clone.vertices.shape)

    # Decimate back down (aggressive)
    target_faces = int(len(clone.faces) * np.random.uniform(0.4, 0.7))
    if target_faces > 10:
        try:
            clone = clone.simplify_quadric_decimation(target_faces)
        except Exception:
            pass
    return clone


TIER_FNS = {
    "T1": tier1_trivial,
    "T2": tier2_easy,
    "T3": tier3_medium,
    "T4": tier4_hard,
    "T5": tier5_adversarial,
}

TIER_NAMES = {
    "T1": "Trivial (re-export)",
    "T2": "Easy (uniform scale + rotation)",
    "T3": "Medium (non-uniform scale + noise + decimation)",
    "T4": "Hard (partial removal + noise + scale)",
    "T5": "Adversarial (topology change + combined)",
}


def generate_all_clones():
    source_dir = os.path.join(OUTPUT_BASE, "sources")
    clone_manifest = []
    errors = []

    # Ensure tier directories exist
    for tier in TIER_FNS:
        os.makedirs(os.path.join(OUTPUT_BASE, "clones", tier), exist_ok=True)

    source_files = sorted(Path(source_dir).glob("*.obj"))
    print(f"Generating clones for {len(source_files)} source models...")
    print(f"Tiers: {', '.join(TIER_FNS.keys())}")
    print(f"Clones per tier per model: {CLONES_PER_TIER}")
    print(f"Expected total: {len(source_files) * len(TIER_FNS) * CLONES_PER_TIER}")
    print()

    for src_idx, src_path in enumerate(tqdm(source_files, desc="Sources")):
        uid = src_path.stem
        try:
            mesh = trimesh.load(str(src_path), force='mesh')
        except Exception as e:
            errors.append({"uid": uid, "error": f"Load failed: {e}"})
            continue

        if len(mesh.faces) < 4:
            errors.append({"uid": uid, "error": f"Too few faces: {len(mesh.faces)}"})
            continue

        for tier_name, tier_fn in TIER_FNS.items():
            tier_dir = os.path.join(OUTPUT_BASE, "clones", tier_name)

            for i in range(CLONES_PER_TIER):
                clone_id = f"{uid}_{tier_name}_v{i}"
                try:
                    clone = tier_fn(mesh, i)
                    out_path = os.path.join(tier_dir, f"{clone_id}.obj")
                    clone.export(out_path)
                    clone_manifest.append({
                        "clone_id": clone_id,
                        "source_uid": uid,
                        "tier": tier_name,
                        "tier_name": TIER_NAMES[tier_name],
                        "variant": i,
                        "faces": int(len(clone.faces)),
                        "vertices": int(len(clone.vertices)),
                    })
                except Exception as e:
                    errors.append({
                        "uid": uid,
                        "tier": tier_name,
                        "variant": i,
                        "error": str(e)[:200],
                    })

    # Save manifest
    manifest_path = os.path.join(OUTPUT_BASE, "clone_manifest_5tier.json")
    with open(manifest_path, "w") as f:
        json.dump(clone_manifest, f, indent=2)

    # Save errors
    if errors:
        error_path = os.path.join(OUTPUT_BASE, "clone_errors.json")
        with open(error_path, "w") as f:
            json.dump(errors, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Clone Generation Summary")
    print(f"{'=' * 60}")
    print(f"  Total clones generated: {len(clone_manifest)}")
    print(f"  Total errors: {len(errors)}")
    print()
    for tier in ["T1", "T2", "T3", "T4", "T5"]:
        count = sum(1 for x in clone_manifest if x["tier"] == tier)
        tier_faces = [x["faces"] for x in clone_manifest if x["tier"] == tier]
        if tier_faces:
            print(f"  {tier} ({TIER_NAMES[tier]}):")
            print(f"    Count: {count}")
            print(f"    Faces: min={min(tier_faces)}, max={max(tier_faces)}, median={sorted(tier_faces)[len(tier_faces)//2]}")

    if errors:
        print(f"\n  First 5 errors:")
        for e in errors[:5]:
            print(f"    {e}")

    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    generate_all_clones()
