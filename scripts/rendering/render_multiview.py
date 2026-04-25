#!/usr/bin/env python3
"""render_multiview.py — Multi-view renderer for Objaverse-LVIS source models and clones.

Renders 28 viewpoints (8 horizontal ring + 20 dodecahedron) x 2 styles
(textured + LFD white plastic) per model using nvdiffrast CUDA backend.

Usage:
    python render_multiview.py                    # Render all
    python render_multiview.py --sources-only     # Render sources only
    python render_multiview.py --clones-only      # Render clones only
    python render_multiview.py --validate         # Validate existing renders
    python render_multiview.py --comparison       # Generate blog comparison images
"""

import os
import sys
import json
import math
import time
import argparse
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────
BLOG_DIR = Path("/home/lightsail-user/3d-dataset-storage/tds-blog")
DATA_DIR = BLOG_DIR / "data" / "clones-objaverse"
RENDER_DIR = BLOG_DIR / "data" / "renders"
IMAGE_DIR = BLOG_DIR / "post-2" / "images"
SOURCE_DIR = DATA_DIR / "sources"
CLONE_DIR = DATA_DIR / "clones"
SOURCE_MANIFEST = DATA_DIR / "source_manifest.json"
CLONE_MANIFEST = DATA_DIR / "clone_manifest_5tier.json"
CHECKPOINT = RENDER_DIR / "render_checkpoint.json"

IMG_SIZE = 224
DEVICE = "cuda"
MAX_FACES = 80000  # Decimate meshes above this to avoid OOM on T4

# ─── Camera setup: 28 viewpoints ─────────────────────────────────────


def compute_ring_cameras(n_views=8, elevation_deg=30.0):
    """8 cameras at 45deg intervals along a horizontal ring, fixed elevation."""
    cameras = []
    for i in range(n_views):
        az = i * (360.0 / n_views)
        cameras.append((az, elevation_deg))
    return cameras


def compute_dodecahedron_cameras():
    """20 cameras at vertices of a regular dodecahedron.

    Dodecahedron vertices derived from the dual of an icosahedron.
    """
    phi = (1 + math.sqrt(5)) / 2  # golden ratio

    # 20 vertices of a dodecahedron
    verts = []
    # 8 cube vertices
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                verts.append((s1, s2, s3))
    # 4 vertices on each coordinate plane
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            verts.append((0, s1 * phi, s2 / phi))
            verts.append((s1 / phi, 0, s2 * phi))
            verts.append((s1 * phi, s2 / phi, 0))

    cameras = []
    for v in verts:
        x, y, z = v
        r = math.sqrt(x * x + y * y + z * z)
        el = math.degrees(math.asin(y / r))
        az = math.degrees(math.atan2(x, z))
        cameras.append((az, el))

    return cameras


def compute_all_cameras():
    """Combined 28 cameras: 8 ring + 20 dodecahedron."""
    ring = compute_ring_cameras(n_views=8, elevation_deg=30.0)
    dodec = compute_dodecahedron_cameras()
    return ring + dodec  # 28 total


# ─── Camera matrix computation (mesh-adaptive) ───────────────────────


def spherical_to_cartesian(az_deg, el_deg, radius=1.0):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = radius * math.cos(el) * math.sin(az)
    y = radius * math.sin(el)
    z = radius * math.cos(el) * math.cos(az)
    return np.array([x, y, z], dtype=np.float32)


def look_at_matrix(eye, target, up):
    """Compute a view matrix using look-at convention."""
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up_vec = np.cross(right, forward)

    mat = np.eye(4, dtype=np.float32)
    mat[0, :3] = right
    mat[1, :3] = up_vec
    mat[2, :3] = -forward
    mat[0, 3] = -np.dot(right, eye)
    mat[1, 3] = -np.dot(up_vec, eye)
    mat[2, 3] = np.dot(forward, eye)
    return mat


def projection_matrix(fov_deg=45, aspect=1.0, near=0.01, far=100.0):
    """Perspective projection matrix."""
    fov = math.radians(fov_deg)
    f = 1.0 / math.tan(fov / 2.0)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat


def compute_mvp_matrices(vertices, cameras):
    """Compute MVP matrices for a specific mesh.

    CRITICAL: Cameras look at the mesh bounding-box center, not the origin.
    Camera distance is based on bounding-box diagonal to ensure the full
    model is visible.
    """
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    diagonal = np.linalg.norm(bbox_max - bbox_min)
    radius = diagonal * 1.5  # distance from center so whole mesh is visible

    # Clamp radius to reasonable range
    radius = max(radius, 0.5)

    proj = projection_matrix(fov_deg=45, aspect=1.0, near=radius * 0.01, far=radius * 20.0)

    mvps = []
    for az, el in cameras:
        direction = spherical_to_cartesian(az, el, radius=radius)
        eye = center + direction
        view = look_at_matrix(eye, center, np.array([0, 1, 0], dtype=np.float32))
        mvps.append(proj @ view)

    return np.array(mvps, dtype=np.float32)


# ─── Mesh processing ─────────────────────────────────────────────────


def normalize_mesh(vertices):
    """Center mesh at origin and scale to unit bounding box."""
    centroid = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    vertices = vertices - centroid
    max_extent = np.abs(vertices).max()
    if max_extent > 1e-8:
        vertices = vertices / max_extent
    return vertices


def load_mesh(path, max_faces=MAX_FACES):
    """Load and prepare an OBJ mesh for rendering.

    Handles:
    - Simple meshes loaded with force='mesh'
    - Scenes (multi-geometry OBJs) — concatenates all sub-meshes
    - Very large meshes — decimates to max_faces
    - Various visual types (ColorVisuals, TextureVisuals)

    Returns: (vertices, faces, vertex_colors_or_None)
    - vertices: (V, 3) float32, normalized to unit bbox
    - faces: (F, 3) int32
    - vertex_colors: (V, 3) float32 in [0,1] or None
    """
    try:
        # First try force='mesh' (fast path)
        try:
            result = trimesh.load(str(path), force='mesh')
            if result.vertices.shape[0] > 0 and result.faces.shape[0] > 0:
                mesh = result
            else:
                raise ValueError("Empty mesh")
        except Exception:
            # Fallback: load without forcing, handle scenes
            result = trimesh.load(str(path), process=False)
            if isinstance(result, trimesh.Scene):
                # Concatenate all mesh geometries in the scene
                meshes = [g for g in result.geometry.values()
                          if isinstance(g, trimesh.Trimesh) and g.faces.shape[0] > 0]
                if not meshes:
                    return None, None, None
                mesh = trimesh.util.concatenate(meshes)
            elif isinstance(result, trimesh.Trimesh):
                mesh = result
            else:
                return None, None, None

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            return None, None, None

        # Decimate if too large
        if mesh.faces.shape[0] > max_faces:
            try:
                mesh = mesh.simplify_quadric_decimation(max_faces)
            except Exception:
                # If decimation fails, subsample faces
                idx = np.random.choice(mesh.faces.shape[0], max_faces, replace=False)
                mesh = mesh.submesh([idx], append=True)

        # Extract vertex colors if available
        vertex_colors = None
        vis = mesh.visual
        if isinstance(vis, trimesh.visual.ColorVisuals):
            try:
                vc = np.array(vis.vertex_colors, dtype=np.float32)
                if vc.ndim == 2 and vc.shape[1] >= 3:
                    vertex_colors = vc[:, :3] / 255.0 if vc.max() > 1.0 else vc[:, :3]
            except Exception:
                pass
        elif isinstance(vis, trimesh.visual.TextureVisuals):
            # Try to bake texture to vertex colors
            try:
                color_vis = vis.to_color()
                vc = np.array(color_vis.vertex_colors, dtype=np.float32)
                if vc.ndim == 2 and vc.shape[1] >= 3:
                    vertex_colors = vc[:, :3] / 255.0 if vc.max() > 1.0 else vc[:, :3]
            except Exception:
                pass

        vertices = normalize_mesh(mesh.vertices.copy().astype(np.float32))
        faces = mesh.faces.copy().astype(np.int32)

        return vertices, faces, vertex_colors
    except Exception as e:
        return None, None, None


# ─── Rendering ────────────────────────────────────────────────────────


def compute_vertex_normals(verts, faces_t, device):
    """Compute smooth vertex normals from face normals."""
    v0 = verts[faces_t[:, 0]]
    v1 = verts[faces_t[:, 1]]
    v2 = verts[faces_t[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_normals = face_normals / (face_normals.norm(dim=1, keepdim=True) + 1e-8)

    vert_normals = torch.zeros_like(verts)
    for j in range(3):
        vert_normals.index_add_(0, faces_t[:, j], face_normals)
    vert_normals = vert_normals / (vert_normals.norm(dim=1, keepdim=True) + 1e-8)
    return vert_normals


def render_views(glctx, vertices_np, faces_np, mvp_matrices,
                 vertex_colors_np=None, mode="lfd", img_size=224):
    """Render model from all viewpoints.

    Args:
        mode: "lfd" = white plastic (Lambertian), "textured" = with vertex colors

    Returns: list of (H, W, 4) uint8 numpy arrays (RGBA).
    """
    verts = torch.tensor(vertices_np, dtype=torch.float32, device=DEVICE)
    faces_t = torch.tensor(faces_np, dtype=torch.int32, device=DEVICE)

    # Homogeneous coordinates
    verts_homo = torch.cat([verts, torch.ones(verts.shape[0], 1, device=DEVICE)], dim=1)

    # Vertex normals
    vert_normals = compute_vertex_normals(verts, faces_t, DEVICE)
    vert_normals_4 = torch.cat([vert_normals,
                                torch.ones(vert_normals.shape[0], 1, device=DEVICE)], dim=1)

    # Vertex colors for textured mode
    if mode == "textured" and vertex_colors_np is not None:
        vert_colors = torch.tensor(vertex_colors_np, dtype=torch.float32, device=DEVICE)
    else:
        # Default: mid-gray for textured without colors, or white plastic for LFD
        if mode == "lfd":
            vert_colors = torch.full((verts.shape[0], 3), 0.82, device=DEVICE)
        else:
            vert_colors = torch.full((verts.shape[0], 3), 0.7, device=DEVICE)

    vert_colors_4 = torch.cat([vert_colors,
                               torch.ones(vert_colors.shape[0], 1, device=DEVICE)], dim=1)

    images = []
    for i in range(len(mvp_matrices)):
        mvp = torch.tensor(mvp_matrices[i], dtype=torch.float32, device=DEVICE)

        # Transform vertices
        clip_verts = verts_homo @ mvp.T
        clip_verts = clip_verts.unsqueeze(0)  # (1, V, 4)

        # Rasterize
        rast, _ = dr.rasterize(glctx, clip_verts, faces_t, resolution=[img_size, img_size])

        # Interpolate normals
        normals, _ = dr.interpolate(vert_normals_4.unsqueeze(0), rast, faces_t)
        normals = normals[0, :, :, :3]

        # Interpolate colors
        colors, _ = dr.interpolate(vert_colors_4.unsqueeze(0), rast, faces_t)
        colors = colors[0, :, :, :3]

        # Lighting: directional + ambient (Lambertian)
        light_dir = torch.tensor([0.3, 0.5, 0.8], device=DEVICE)
        light_dir = light_dir / light_dir.norm()
        diffuse = torch.clamp(torch.sum(normals * light_dir, dim=-1, keepdim=True), 0, 1)

        ambient = 0.35
        color = colors * (ambient + (1 - ambient) * diffuse)

        # Alpha mask from rasterization
        mask = (rast[0, :, :, 3:4] > 0).float()

        # RGBA output: transparent background
        alpha = mask * 255
        rgb = (color.clamp(0, 1) * 255)
        img = torch.cat([rgb, alpha], dim=-1)

        img_np = img.byte().cpu().numpy()
        images.append(img_np)

    return images


# ─── Model discovery ─────────────────────────────────────────────────


def discover_models():
    """Find all source and clone models to render.

    Returns dict: {model_id: {"path": str, "type": "source"|"clone", "tier": str|None, ...}}
    """
    models = {}

    # Source models
    if SOURCE_DIR.exists():
        for f in sorted(SOURCE_DIR.iterdir()):
            if f.suffix == '.obj':
                uid = f.stem
                models[uid] = {
                    "path": str(f),
                    "type": "source",
                    "tier": None,
                    "source_uid": uid,
                }

    # Clone models (organized by tier: T1/ T2/ T3/ T4/ T5/)
    if CLONE_DIR.exists():
        for tier_dir in sorted(CLONE_DIR.iterdir()):
            if tier_dir.is_dir() and tier_dir.name.startswith("T"):
                tier = tier_dir.name
                for f in sorted(tier_dir.iterdir()):
                    if f.suffix == '.obj':
                        clone_id = f.stem
                        # Parse source_uid from clone_id: {source_uid}_{tier}_v{n}
                        parts = clone_id.rsplit("_", 2)
                        source_uid = parts[0] if len(parts) >= 3 else clone_id
                        models[clone_id] = {
                            "path": str(f),
                            "type": "clone",
                            "tier": tier,
                            "source_uid": source_uid,
                        }

    return models


# ─── Validation ───────────────────────────────────────────────────────


def check_render(path):
    """Validate a single rendered image.

    Returns: "ok", "blank_alpha", "flat_color", or "missing"
    """
    if not os.path.exists(path):
        return "missing"
    try:
        img = np.array(Image.open(path))
        if img.ndim < 3:
            return "flat_color"
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            if alpha.max() == 0:
                return "blank_alpha"
            # Check if foreground pixels have any variation
            fg_mask = alpha > 0
            if fg_mask.sum() < 10:
                return "blank_alpha"
            rgb_fg = img[:, :, :3][fg_mask]
            if rgb_fg.std() < 1.0:
                return "flat_color"
        else:
            if img[:, :, :3].std() < 1.0:
                return "flat_color"
        return "ok"
    except Exception:
        return "corrupt"


def validate_renders(render_dir, models, n_views=28):
    """Validate all renders. Returns dict of stats."""
    stats = {"total": 0, "ok": 0, "blank_alpha": 0, "flat_color": 0,
             "missing": 0, "corrupt": 0, "by_mode": {}, "by_tier": {}}

    for mode in ["textured", "lfd"]:
        mode_dir = render_dir / mode
        mode_stats = {"ok": 0, "issues": []}

        for model_id, info in tqdm(models.items(), desc=f"Validating {mode}"):
            model_dir = mode_dir / model_id
            for v in range(n_views):
                path = model_dir / f"view_{v:02d}.png"
                result = check_render(str(path))
                stats["total"] += 1
                stats[result] = stats.get(result, 0) + 1

                if result == "ok":
                    mode_stats["ok"] += 1
                else:
                    mode_stats["issues"].append({
                        "model": model_id,
                        "view": v,
                        "status": result,
                    })

                # Track by tier
                tier = info.get("tier", "source") or "source"
                if tier not in stats["by_tier"]:
                    stats["by_tier"][tier] = {"ok": 0, "issues": 0}
                if result == "ok":
                    stats["by_tier"][tier]["ok"] += 1
                else:
                    stats["by_tier"][tier]["issues"] += 1

        stats["by_mode"][mode] = mode_stats

    return stats


# ─── Comparison image generation ─────────────────────────────────────


def create_comparison_strip(render_dir, model_id, n_ring_views=8, out_path=None):
    """Create a side-by-side comparison: textured row vs LFD row for ring views."""
    tex_dir = render_dir / "textured" / model_id
    lfd_dir = render_dir / "lfd" / model_id

    if not tex_dir.exists() or not lfd_dir.exists():
        return None

    tex_images = []
    lfd_images = []
    for v in range(n_ring_views):
        tp = tex_dir / f"view_{v:02d}.png"
        lp = lfd_dir / f"view_{v:02d}.png"
        if tp.exists() and lp.exists():
            tex_images.append(np.array(Image.open(tp).convert("RGB")))
            lfd_images.append(np.array(Image.open(lp).convert("RGB")))

    if len(tex_images) < n_ring_views:
        return None

    # Create strip: two rows
    h, w = tex_images[0].shape[:2]
    gap = 4  # pixels between images
    label_h = 32  # height for row labels

    strip_w = n_ring_views * w + (n_ring_views - 1) * gap
    strip_h = 2 * h + gap + 2 * label_h

    canvas = np.ones((strip_h, strip_w, 3), dtype=np.uint8) * 255

    # Row 1: textured
    y_off = label_h
    for i, img in enumerate(tex_images):
        x = i * (w + gap)
        canvas[y_off:y_off + h, x:x + w] = img

    # Row 2: LFD
    y_off = label_h + h + gap + label_h
    for i, img in enumerate(lfd_images):
        x = i * (w + gap)
        canvas[y_off:y_off + h, x:x + w] = img

    # Add row labels using simple text (PIL)
    result = Image.fromarray(canvas)
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(result)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        draw.text((8, 4), "Textured", fill=(0, 0, 0), font=font)
        draw.text((8, label_h + h + gap + 4), "LFD (White Plastic)", fill=(0, 0, 0), font=font)
    except ImportError:
        pass

    if out_path:
        result.save(out_path)
    return result


# ─── Main rendering pipeline ─────────────────────────────────────────


def render_all(models, render_dir, cameras, glctx, modes=("textured", "lfd"),
               checkpoint_path=None, batch_checkpoint=200):
    """Render all models in both modes with checkpointing."""

    # Load checkpoint
    rendered_ids = set()
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            cp = json.load(f)
            rendered_ids = set(cp.get("rendered", []))
        print(f"  Checkpoint: {len(rendered_ids)} already rendered")

    # Filter to unrendered
    to_render = {k: v for k, v in models.items() if k not in rendered_ids}
    print(f"  Remaining to render: {len(to_render)}")

    if not to_render:
        print("  All models already rendered!")
        return {"rendered": len(rendered_ids), "failed": 0, "skipped": 0}

    # Create output dirs
    for mode in modes:
        (render_dir / mode).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    success = 0
    failed = 0
    skipped = 0
    render_times = []

    for idx, (model_id, info) in enumerate(tqdm(to_render.items(), desc="Rendering")):
        model_start = time.time()

        vertices, faces, vertex_colors = load_mesh(info["path"])
        if vertices is None:
            failed += 1
            tqdm.write(f"  SKIP (load fail): {model_id}")
            continue

        try:
            # Compute camera matrices for THIS mesh's bounding box
            mvp_matrices = compute_mvp_matrices(vertices, cameras)

            for mode in modes:
                out_dir = render_dir / mode / model_id
                out_dir.mkdir(parents=True, exist_ok=True)

                images = render_views(
                    glctx, vertices, faces, mvp_matrices,
                    vertex_colors_np=vertex_colors,
                    mode=mode, img_size=IMG_SIZE
                )

                for i, img in enumerate(images):
                    Image.fromarray(img, mode="RGBA").save(out_dir / f"view_{i:02d}.png")

            rendered_ids.add(model_id)
            success += 1
            render_times.append(time.time() - model_start)

        except Exception as e:
            tqdm.write(f"  ERROR rendering {model_id}: {e}")
            failed += 1
            torch.cuda.empty_cache()
            continue

        # Periodic checkpoint
        if (idx + 1) % batch_checkpoint == 0:
            if checkpoint_path:
                with open(checkpoint_path, 'w') as f:
                    json.dump({"rendered": sorted(rendered_ids)}, f)

            elapsed = time.time() - start_time
            rate = success / elapsed if elapsed > 0 else 0

            # Disk space check
            st = os.statvfs(str(render_dir))
            free_gb = (st.f_bavail * st.f_bsize) / (1024 ** 3)
            tqdm.write(
                f"  Checkpoint: {success} done, {failed} fail, "
                f"{rate:.1f} models/sec, disk: {free_gb:.0f}GB free"
            )
            if free_gb < 20:
                tqdm.write("  WARNING: <20GB free! Stopping.")
                break

    # Final checkpoint
    if checkpoint_path:
        with open(checkpoint_path, 'w') as f:
            json.dump({"rendered": sorted(rendered_ids)}, f)

    elapsed = time.time() - start_time
    stats = {
        "rendered": success,
        "failed": failed,
        "skipped": skipped,
        "total_time_sec": elapsed,
        "mean_time_per_model_sec": np.mean(render_times) if render_times else 0,
        "median_time_per_model_sec": float(np.median(render_times)) if render_times else 0,
        "max_time_per_model_sec": max(render_times) if render_times else 0,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Multi-view renderer for Objaverse-LVIS")
    parser.add_argument("--sources-only", action="store_true", help="Render sources only")
    parser.add_argument("--clones-only", action="store_true", help="Render clones only")
    parser.add_argument("--validate", action="store_true", help="Validate existing renders")
    parser.add_argument("--comparison", action="store_true", help="Generate blog comparison images")
    parser.add_argument("--tier", type=str, default=None, help="Render a specific tier only (e.g., T1)")
    args = parser.parse_args()

    print("=" * 70)
    print("Multi-View Rendering Pipeline (nvdiffrast CUDA)")
    print("=" * 70)

    # ── Discover models ──
    all_models = discover_models()
    sources = {k: v for k, v in all_models.items() if v["type"] == "source"}
    clones = {k: v for k, v in all_models.items() if v["type"] == "clone"}

    print(f"Discovered: {len(sources)} sources, {len(clones)} clones")
    tier_counts = defaultdict(int)
    for v in clones.values():
        tier_counts[v["tier"]] += 1
    for t in sorted(tier_counts):
        print(f"  {t}: {tier_counts[t]} clones")

    # ── Camera setup ──
    cameras = compute_all_cameras()
    print(f"Camera viewpoints: {len(cameras)} (8 ring + 20 dodecahedron)")

    # ── Select models to render ──
    if args.sources_only:
        models = sources
        print(f"Rendering SOURCES only: {len(models)}")
    elif args.clones_only:
        models = clones
        if args.tier:
            models = {k: v for k, v in models.items() if v["tier"] == args.tier}
        print(f"Rendering CLONES only: {len(models)}")
    elif args.tier:
        models = {k: v for k, v in all_models.items()
                  if v["tier"] == args.tier or v["type"] == "source"}
        print(f"Rendering tier {args.tier} + sources: {len(models)}")
    else:
        models = all_models
        print(f"Rendering ALL: {len(models)}")

    # ── Validate mode ──
    if args.validate:
        print("\n--- Validation ---")
        val_stats = validate_renders(RENDER_DIR, models, n_views=len(cameras))
        print(f"Total images checked: {val_stats['total']}")
        print(f"  OK: {val_stats['ok']}")
        print(f"  Blank (alpha): {val_stats.get('blank_alpha', 0)}")
        print(f"  Flat color: {val_stats.get('flat_color', 0)}")
        print(f"  Missing: {val_stats.get('missing', 0)}")
        print(f"  Corrupt: {val_stats.get('corrupt', 0)}")
        print("\nBy tier:")
        for tier in sorted(val_stats["by_tier"]):
            ts = val_stats["by_tier"][tier]
            print(f"  {tier}: {ts['ok']} ok, {ts['issues']} issues")

        # Save validation results
        val_out = RENDER_DIR / "validation_results.json"
        with open(val_out, 'w') as f:
            json.dump(val_stats, f, indent=2, default=str)
        print(f"Validation saved to {val_out}")
        return

    # ── Comparison image mode ──
    if args.comparison:
        print("\n--- Generating comparison images ---")
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        # Pick diverse source models
        source_ids = sorted(sources.keys())
        # Select 5 evenly spaced
        step = max(1, len(source_ids) // 5)
        selected = [source_ids[i] for i in range(0, len(source_ids), step)][:5]

        for sid in selected:
            out_path = IMAGE_DIR / f"comparison_{sid}.png"
            result = create_comparison_strip(RENDER_DIR, sid, n_ring_views=8, out_path=str(out_path))
            if result:
                print(f"  Created: {out_path}")
            else:
                print(f"  SKIP: {sid} (renders not found)")

        # Also create clone comparisons: source vs T1 vs T5
        print("\nSource vs clone comparisons:")
        for sid in selected[:3]:
            # Find clones for this source
            src_clones = {k: v for k, v in clones.items()
                          if v["source_uid"] == sid}
            t1_clones = [k for k, v in src_clones.items() if v["tier"] == "T1"]
            t5_clones = [k for k, v in src_clones.items() if v["tier"] == "T5"]

            if t1_clones and t5_clones:
                # Create strip: source / T1 clone / T5 clone
                strips = []
                for mid in [sid, t1_clones[0], t5_clones[0]]:
                    s = create_comparison_strip(RENDER_DIR, mid, n_ring_views=8)
                    if s:
                        strips.append((mid, np.array(s)))

                if len(strips) == 3:
                    # Stack vertically with labels
                    gap = 8
                    label_h = 28
                    max_w = max(s[1].shape[1] for s in strips)
                    total_h = sum(s[1].shape[0] for s in strips) + 2 * gap + 3 * label_h

                    canvas = np.ones((total_h, max_w, 3), dtype=np.uint8) * 255
                    y = 0
                    labels = ["Source", "T1 Clone (Trivial)", "T5 Clone (Hard)"]
                    for i, (mid, img) in enumerate(strips):
                        y += label_h
                        h = img.shape[0]
                        canvas[y:y + h, :img.shape[1]] = img
                        y += h + gap

                    out = IMAGE_DIR / f"clone_comparison_{sid}.png"
                    result_img = Image.fromarray(canvas)
                    try:
                        from PIL import ImageDraw, ImageFont
                        draw = ImageDraw.Draw(result_img)
                        try:
                            font = ImageFont.truetype(
                                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                        except Exception:
                            font = ImageFont.load_default()
                        y = 0
                        for i, (mid, img) in enumerate(strips):
                            draw.text((8, y + 4), labels[i], fill=(200, 0, 0), font=font)
                            y += label_h + img.shape[0] + gap
                    except ImportError:
                        pass
                    result_img.save(str(out))
                    print(f"  Created: {out}")

        return

    # ── Render mode ──
    print("\nInitializing nvdiffrast CUDA context...")
    glctx = dr.RasterizeCudaContext()
    print("nvdiffrast ready!")

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")

    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    stats = render_all(
        models, RENDER_DIR, cameras, glctx,
        modes=("textured", "lfd"),
        checkpoint_path=CHECKPOINT,
        batch_checkpoint=200,
    )

    # ── Summary ──
    elapsed = stats.get("total_time_sec", 0)
    print(f"\n{'=' * 70}")
    print(f"RENDERING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Models rendered: {stats['rendered']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total time: {elapsed / 60:.1f} minutes")
    if stats['rendered'] > 0:
        print(f"Mean time/model: {stats['mean_time_per_model_sec']:.3f}s")
        print(f"Median time/model: {stats['median_time_per_model_sec']:.3f}s")
        print(f"Max time/model: {stats['max_time_per_model_sec']:.3f}s")
    total_images = stats['rendered'] * len(cameras) * 2
    print(f"Total images: {total_images}")

    # Disk usage
    st = os.statvfs(str(RENDER_DIR))
    free_gb = (st.f_bavail * st.f_bsize) / (1024 ** 3)
    print(f"Disk free: {free_gb:.0f}GB")

    # GPU memory
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak GPU memory: {peak_mem:.2f}GB")
        stats["peak_gpu_memory_gb"] = peak_mem

    # Compute disk usage of renders
    try:
        result = os.popen(f"du -sh {RENDER_DIR}").read().strip()
        disk_used = result.split("\t")[0]
        stats["disk_used"] = disk_used
        print(f"Render disk usage: {disk_used}")
    except Exception:
        pass

    # Save render stats
    stats["total_images"] = total_images
    stats["n_cameras"] = len(cameras)
    stats["img_size"] = IMG_SIZE
    stats["n_sources"] = len(sources)
    stats["n_clones"] = len(clones)
    stats["n_models_total"] = len(models)
    stats_path = RENDER_DIR / "render_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
