#!/usr/bin/env python3
"""Step 1: Download Objaverse-LVIS subset for 3D-DupBench."""

import objaverse
import random
import json
import os

OUTPUT_BASE = "/home/lightsail-user/3d-dataset-storage/tds-blog/data/clones-objaverse/"
SEED = 42
random.seed(SEED)

print("=" * 60)
print("Step 1: Download Objaverse-LVIS Subset")
print("=" * 60)

# Get LVIS annotations — these are the high-quality curated subset
print("\nLoading LVIS annotations...")
lvis_annotations = objaverse.load_lvis_annotations()
print(f"Total LVIS categories: {len(lvis_annotations)}")
print(f"Total LVIS objects: {sum(len(v) for v in lvis_annotations.values())}")

# Select 200 models across diverse categories for the benchmark
# Pick 5 models from each of 40 randomly sampled categories
categories = list(lvis_annotations.keys())
selected_cats = random.sample(categories, min(40, len(categories)))

selected_uids = []
category_map = {}
for cat in selected_cats:
    uids = lvis_annotations[cat]
    chosen = random.sample(uids, min(5, len(uids)))
    selected_uids.extend(chosen)
    for uid in chosen:
        category_map[uid] = cat

print(f"\nSelected {len(selected_uids)} models from {len(selected_cats)} categories")
print("Categories:", sorted(selected_cats)[:10], "...")

# Download the selected models
print(f"\nDownloading {len(selected_uids)} models (this may take a while)...")
objects = objaverse.load_objects(uids=selected_uids, download_processes=8)
print(f"Downloaded {len(objects)} models")

# Save the selection manifest
manifest = {
    "total_models": len(selected_uids),
    "total_downloaded": len(objects),
    "categories": len(selected_cats),
    "category_list": sorted(selected_cats),
    "category_map": category_map,
    "uids": selected_uids,
    "model_paths": {uid: str(path) for uid, path in objects.items()}
}

manifest_path = os.path.join(OUTPUT_BASE, "source_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest saved to {manifest_path}")
print(f"  Total selected: {len(selected_uids)}")
print(f"  Total downloaded: {len(objects)}")
print(f"  Missing: {len(selected_uids) - len(objects)}")

# Print category distribution
print("\nCategory distribution:")
for cat in sorted(selected_cats):
    count = sum(1 for uid in selected_uids if category_map.get(uid) == cat)
    print(f"  {cat}: {count}")
