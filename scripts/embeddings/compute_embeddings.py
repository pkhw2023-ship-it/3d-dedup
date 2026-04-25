#!/usr/bin/env python3
"""compute_embeddings.py — Embedding Bake-Off: Multi-model per-view embeddings.

Computes per-view embeddings for all rendered models using 4 vision models:
  1. DINOv2-base  (ViT-B/14, 768-dim)
  2. DINOv2-giant (ViT-g/14, 1536-dim)
  3. CLIP ViT-L/14 (768-dim)
  4. CLIP ViT-B/32 (512-dim)

For each model, computes embeddings on both textured and LFD renders.
Saves per-view embeddings as .npz files for downstream aggregation.

Usage:
    python compute_embeddings.py [--model MODEL_KEY] [--mode MODE] [--batch-size BS]

    MODEL_KEY: dinov2_base | dinov2_giant | clip_large | clip_base | all (default: all)
    MODE: textured | lfd | all (default: all)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# ─── Config ──────────────────────────────────────────────────────────
DATA_DIR = Path("/home/lightsail-user/3d-dataset-storage/tds-blog/data")
RENDER_DIR = DATA_DIR / "renders"
EMB_DIR = DATA_DIR / "embeddings"
CLONE_MANIFEST = DATA_DIR / "clones-objaverse" / "clone_manifest_5tier.json"
SOURCE_MANIFEST = DATA_DIR / "clones-objaverse" / "source_manifest.json"
CHECKPOINT_FILE = EMB_DIR / "embed_checkpoint.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_VIEWS = 28

# Model definitions
MODELS = {
    "dinov2_base": {
        "hf_name": "facebook/dinov2-base",
        "type": "dinov2",
        "dim": 768,
    },
    "dinov2_giant": {
        "hf_name": "facebook/dinov2-giant",
        "type": "dinov2",
        "dim": 1536,
    },
    "clip_large": {
        "hf_name": "openai/clip-vit-large-patch14",
        "type": "clip",
        "dim": 768,
    },
    "clip_base": {
        "hf_name": "openai/clip-vit-base-patch32",
        "type": "clip",
        "dim": 512,
    },
}


def load_model(model_key):
    """Load a vision model and its processor."""
    cfg = MODELS[model_key]
    hf_name = cfg["hf_name"]
    model_type = cfg["type"]

    if model_type == "dinov2":
        from transformers import AutoModel, AutoImageProcessor
        print(f"Loading {hf_name}...")
        processor = AutoImageProcessor.from_pretrained(hf_name)
        model = AutoModel.from_pretrained(hf_name, torch_dtype=torch.float16)
        model = model.to(DEVICE).eval()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Loaded {n_params:.0f}M params, dim={cfg['dim']}")
        return model, processor, model_type

    elif model_type == "clip":
        from transformers import CLIPModel, CLIPProcessor
        print(f"Loading {hf_name}...")
        processor = CLIPProcessor.from_pretrained(hf_name)
        model = CLIPModel.from_pretrained(hf_name, torch_dtype=torch.float16)
        model = model.to(DEVICE).eval()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Loaded {n_params:.0f}M params, dim={cfg['dim']}")
        return model, processor, model_type


def compute_batch_embeddings(model, processor, image_paths, model_type, batch_size=32):
    """Compute embeddings for a batch of images.

    Returns: numpy array of shape (N, dim), float32
    """
    all_embeddings = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception:
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

        if model_type == "dinov2":
            inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            # Cast pixel_values to float16 to match model dtype
            inputs["pixel_values"] = inputs["pixel_values"].half()
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        elif model_type == "clip":
            inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
            inputs["pixel_values"] = inputs["pixel_values"].half()
            with torch.no_grad():
                out = model.get_image_features(pixel_values=inputs["pixel_values"])
                # transformers >=5.x returns BaseModelOutputWithPooling
                if hasattr(out, "pooler_output"):
                    emb = out.pooler_output
                elif hasattr(out, "shape"):
                    emb = out  # older transformers returns tensor directly
                else:
                    raise RuntimeError(f"Unexpected CLIP output type: {type(out)}")

        all_embeddings.append(emb.float().cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def get_model_dirs(render_dir, mode):
    """Get sorted list of model directories for a render mode."""
    mode_dir = render_dir / mode
    if not mode_dir.exists():
        print(f"WARNING: {mode_dir} does not exist!")
        return []
    dirs = sorted([d for d in mode_dir.iterdir() if d.is_dir()])
    return dirs


def compute_per_view_embeddings(model_key, mode, batch_size=32):
    """Compute per-view embeddings for one model × one render mode.

    Saves: EMB_DIR / f"{model_key}_{mode}_perview.npz"
      - model_ids: list of model IDs
      - embeddings: (N_models, 28, dim) array
    """
    output_path = EMB_DIR / f"{model_key}_{mode}_perview.npz"

    # Check checkpoint
    checkpoint = load_checkpoint()
    ckpt_key = f"{model_key}_{mode}"
    if checkpoint.get(ckpt_key, False):
        print(f"  [{ckpt_key}] Already computed (checkpoint). Skipping.")
        return output_path

    cfg = MODELS[model_key]
    dim = cfg["dim"]

    # Load model
    model, processor, model_type = load_model(model_key)

    # Get model directories
    model_dirs = get_model_dirs(RENDER_DIR, mode)
    print(f"  Found {len(model_dirs)} model directories in {mode}/")

    model_ids = []
    all_embeddings = []
    skipped = 0

    for model_dir in tqdm(model_dirs, desc=f"{model_key}/{mode}"):
        mid = model_dir.name
        view_paths = [model_dir / f"view_{v:02d}.png" for v in range(N_VIEWS)]
        existing = [p for p in view_paths if p.exists()]

        if len(existing) < N_VIEWS:
            # Pad missing views with the first available view
            if len(existing) == 0:
                skipped += 1
                continue

        # Use actual view paths (sorted by view index)
        actual_paths = []
        for v in range(N_VIEWS):
            vp = model_dir / f"view_{v:02d}.png"
            if vp.exists():
                actual_paths.append(str(vp))
            else:
                actual_paths.append(str(existing[0]))  # fallback to first view

        view_emb = compute_batch_embeddings(model, processor, actual_paths, model_type, batch_size)
        # Normalize embeddings to unit sphere
        norms = np.linalg.norm(view_emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        view_emb = view_emb / norms

        model_ids.append(mid)
        all_embeddings.append(view_emb)

    if skipped > 0:
        print(f"  Skipped {skipped} models with no views")

    embeddings_array = np.stack(all_embeddings)  # (N, 28, dim)
    print(f"  Embeddings shape: {embeddings_array.shape}")

    # Save
    np.savez_compressed(
        output_path,
        model_ids=np.array(model_ids),
        embeddings=embeddings_array.astype(np.float32),
    )
    print(f"  Saved to {output_path}")

    # Update checkpoint
    checkpoint[ckpt_key] = True
    save_checkpoint(checkpoint)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return output_path


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(data):
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Embedding Bake-Off: Compute per-view embeddings")
    parser.add_argument("--model", default="all", choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--mode", default="all", choices=["textured", "lfd", "all"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and recompute")
    args = parser.parse_args()

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset and CHECKPOINT_FILE.exists():
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint reset.")

    # Determine which models and modes to run
    model_keys = list(MODELS.keys()) if args.model == "all" else [args.model]
    modes = ["textured", "lfd"] if args.mode == "all" else [args.mode]

    print("=" * 70)
    print("EMBEDDING BAKE-OFF — Per-View Computation")
    print("=" * 70)
    print(f"Models: {model_keys}")
    print(f"Modes:  {modes}")
    print(f"Batch:  {args.batch_size}")
    print(f"Device: {DEVICE}")
    print(f"Output: {EMB_DIR}")
    print("=" * 70)

    total_start = time.time()

    for model_key in model_keys:
        for mode in modes:
            print(f"\n{'─' * 60}")
            print(f"Computing: {model_key} × {mode}")
            print(f"{'─' * 60}")
            start = time.time()

            output_path = compute_per_view_embeddings(model_key, mode, args.batch_size)

            elapsed = time.time() - start
            print(f"  Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"ALL DONE — Total time: {total_elapsed / 60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
