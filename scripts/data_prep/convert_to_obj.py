#!/usr/bin/env python3
"""Step 2: Convert all Objaverse source models to OBJ format."""

import trimesh
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

OUTPUT_BASE = "/home/lightsail-user/3d-dataset-storage/tds-blog/data/clones-objaverse/"
SOURCE_DIR = os.path.join(OUTPUT_BASE, "sources")
os.makedirs(SOURCE_DIR, exist_ok=True)

print("=" * 60)
print("Step 2: Convert Source Models to OBJ")
print("=" * 60)

manifest_path = os.path.join(OUTPUT_BASE, "source_manifest.json")
with open(manifest_path) as f:
    manifest = json.load(f)

conversion_log = []
total = len(manifest["model_paths"])

for idx, (uid, path) in enumerate(manifest["model_paths"].items()):
    if (idx + 1) % 20 == 0 or idx == 0:
        print(f"  Processing {idx+1}/{total}...")

    try:
        # Load the mesh, handling various formats
        scene_or_mesh = trimesh.load(path, force=None)

        if isinstance(scene_or_mesh, trimesh.Scene):
            # Concatenate all geometry in the scene
            meshes = []
            for geom in scene_or_mesh.geometry.values():
                if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
                    meshes.append(geom)
            if not meshes:
                conversion_log.append({"uid": uid, "status": "error", "error": "No valid geometry in scene"})
                continue
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(scene_or_mesh, trimesh.Trimesh):
            mesh = scene_or_mesh
        else:
            # Try forcing to mesh
            mesh = trimesh.load(path, force='mesh')

        if len(mesh.faces) < 4:
            conversion_log.append({"uid": uid, "status": "error", "error": f"Too few faces: {len(mesh.faces)}"})
            continue

        # Normalize to unit bounding box centered at origin
        mesh.vertices -= mesh.bounding_box.centroid
        scale = mesh.bounding_box.extents.max()
        if scale > 0:
            mesh.vertices /= scale

        out_path = os.path.join(SOURCE_DIR, f"{uid}.obj")
        mesh.export(out_path)

        conversion_log.append({
            "uid": uid,
            "status": "ok",
            "faces": int(len(mesh.faces)),
            "vertices": int(len(mesh.vertices)),
            "category": manifest.get("category_map", {}).get(uid, "unknown"),
            "original_format": Path(path).suffix,
        })

    except Exception as e:
        conversion_log.append({"uid": uid, "status": "error", "error": str(e)})

# Save conversion log
log_path = os.path.join(OUTPUT_BASE, "conversion_log.json")
with open(log_path, "w") as f:
    json.dump(conversion_log, f, indent=2)

successes = [x for x in conversion_log if x["status"] == "ok"]
failures = [x for x in conversion_log if x["status"] == "error"]

print(f"\n{'=' * 60}")
print(f"Conversion Results")
print(f"{'=' * 60}")
print(f"  Total attempted: {len(conversion_log)}")
print(f"  Successful: {len(successes)}")
print(f"  Failed: {len(failures)}")

if failures:
    print(f"\n  Failures:")
    for f_entry in failures[:10]:
        print(f"    {f_entry['uid']}: {f_entry['error'][:80]}")
    if len(failures) > 10:
        print(f"    ... and {len(failures) - 10} more")

if successes:
    faces = [x["faces"] for x in successes]
    verts = [x["vertices"] for x in successes]
    print(f"\n  Face count stats: min={min(faces)}, max={max(faces)}, median={sorted(faces)[len(faces)//2]}")
    print(f"  Vertex count stats: min={min(verts)}, max={max(verts)}, median={sorted(verts)[len(verts)//2]}")

print(f"\nConversion log saved to {log_path}")
