#!/usr/bin/env python3
"""Step 4: Validate clone quality across all tiers."""

import trimesh
import json
import os
import random
import warnings

warnings.filterwarnings("ignore")

OUTPUT_BASE = "/home/lightsail-user/3d-dataset-storage/tds-blog/data/clones-objaverse/"
random.seed(42)

print("=" * 60)
print("Step 4: Clone Quality Validation")
print("=" * 60)

manifest = json.load(open(os.path.join(OUTPUT_BASE, "clone_manifest_5tier.json")))

validation_results = []
n_samples = 10  # samples per tier

for tier in ["T1", "T2", "T3", "T4", "T5"]:
    tier_clones = [c for c in manifest if c["tier"] == tier]
    samples = random.sample(tier_clones, min(n_samples, len(tier_clones)))

    print(f"\n{'='*40}")
    print(f"  {tier} — {len(tier_clones)} total clones, validating {len(samples)} samples")
    print(f"{'='*40}")

    valid = 0
    invalid = 0

    for s in samples:
        path = os.path.join(OUTPUT_BASE, "clones", tier, f"{s['clone_id']}.obj")
        try:
            mesh = trimesh.load(path, force='mesh')
            is_valid = len(mesh.faces) >= 4 and len(mesh.vertices) >= 4

            # Also load the source to compare
            src_path = os.path.join(OUTPUT_BASE, "sources", f"{s['source_uid']}.obj")
            src_mesh = trimesh.load(src_path, force='mesh')

            face_ratio = len(mesh.faces) / max(len(src_mesh.faces), 1)

            result = {
                "clone_id": s['clone_id'],
                "tier": tier,
                "valid": is_valid,
                "clone_faces": len(mesh.faces),
                "clone_verts": len(mesh.vertices),
                "source_faces": len(src_mesh.faces),
                "face_ratio": round(face_ratio, 3),
                "watertight": mesh.is_watertight,
            }

            if is_valid:
                valid += 1
            else:
                invalid += 1

            status = "OK" if is_valid else "INVALID"
            print(f"  [{status}] {s['clone_id']}: "
                  f"{len(mesh.faces)} faces (ratio={face_ratio:.2f}), "
                  f"watertight={mesh.is_watertight}")

            validation_results.append(result)

        except Exception as e:
            invalid += 1
            print(f"  [ERROR] {s['clone_id']}: {e}")
            validation_results.append({
                "clone_id": s['clone_id'],
                "tier": tier,
                "valid": False,
                "error": str(e)[:200],
            })

    print(f"\n  Summary: {valid}/{valid+invalid} valid")

# Now do an exhaustive check: just try loading every single clone
print(f"\n{'='*60}")
print(f"Exhaustive Load Check (all {len(manifest)} clones)")
print(f"{'='*60}")

load_ok = 0
load_fail = 0
tier_ok = {t: 0 for t in ["T1", "T2", "T3", "T4", "T5"]}
tier_fail = {t: 0 for t in ["T1", "T2", "T3", "T4", "T5"]}

for i, entry in enumerate(manifest):
    if (i + 1) % 500 == 0:
        print(f"  Checked {i+1}/{len(manifest)}...")

    path = os.path.join(OUTPUT_BASE, "clones", entry["tier"], f"{entry['clone_id']}.obj")
    try:
        mesh = trimesh.load(path, force='mesh')
        if len(mesh.faces) >= 4:
            load_ok += 1
            tier_ok[entry["tier"]] += 1
        else:
            load_fail += 1
            tier_fail[entry["tier"]] += 1
    except:
        load_fail += 1
        tier_fail[entry["tier"]] += 1

print(f"\n  Total: {load_ok}/{load_ok+load_fail} loadable ({load_fail} failures)")
for t in ["T1", "T2", "T3", "T4", "T5"]:
    print(f"    {t}: {tier_ok[t]}/{tier_ok[t]+tier_fail[t]} ok")

# Save validation results
with open(os.path.join(OUTPUT_BASE, "validation_results.json"), "w") as f:
    json.dump({
        "sample_validations": validation_results,
        "exhaustive_check": {
            "total_ok": load_ok,
            "total_fail": load_fail,
            "per_tier_ok": tier_ok,
            "per_tier_fail": tier_fail,
        }
    }, f, indent=2)

print(f"\nValidation results saved.")
