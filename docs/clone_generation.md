# 3D-DupBench: Objaverse-LVIS Clone Dataset

Synthetic near-duplicate benchmark for 3D model deduplication evaluation,
generated from [Objaverse-LVIS](https://objaverse.allenai.org/) (CC-BY-4.0).

**Generated:** 2026-04-24
**Pipeline:** lightsail-shapenet (Tesla T4 GPU)
**Conda env:** `3d-dedup`

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Source models (selected) | 192 |
| Source models (downloaded) | 192 |
| Source models (converted to OBJ) | 191 |
| Conversion failures | 1 |
| LVIS categories | 40 |
| Difficulty tiers | 5 (T1–T5) |
| Clones per source per tier | 3 |
| **Total clones** | **2,865** |
| Total disk usage | ~38 GB |

## Directory Structure

```
clones-objaverse/
├── sources/              # 191 normalized OBJ source models
│   ├── {uid}.obj
│   └── ...
├── clones/
│   ├── T1/               # 573 trivial clones
│   ├── T2/               # 573 easy clones
│   ├── T3/               # 573 medium clones
│   ├── T4/               # 573 hard clones
│   └── T5/               # 573 adversarial clones
├── source_manifest.json  # UID → category + download path mapping
├── conversion_log.json   # Per-model conversion status + geometry stats
├── clone_manifest_5tier.json  # Complete clone registry with metadata
├── validation_results.json    # Quality validation report
└── README.md
```

## Difficulty Tiers

| Tier | Name | Transformations | Face Count (min/median/max) | Detection Difficulty |
|------|------|-----------------|----------------------------|---------------------|
| T1 | Trivial | Re-export (OBJ → PLY → OBJ round-trip) | 13 / 7,376 / 9.5M | Near-identical; floating-point drift only |
| T2 | Easy | Uniform scale (0.5×–2.0×) + random rotation | 13 / 7,376 / 9.5M | Simple normalization solves it |
| T3 | Medium | Non-uniform scale + vertex noise (σ=0.01–0.05) + decimation (50–80%) | 13 / 7,376 / 9.5M | Requires robust features |
| T4 | Hard | Partial mesh removal (10–30%) + noise + non-uniform scale | 11 / 6,087 / 8.3M | Significant geometry change |
| T5 | Adversarial | Subdivision + non-uniform scale + noise + aggressive decimation | 52 / 29,504 / 9.5M | Defeats naive approaches |

## Source Model Statistics

- **Face count:** min=13, max=9,482,790, median=7,376, mean=184,077
- **Vertex count:** min=15, max=4,976,857, median=7,630, mean=120,082
- **Normalization:** All source models centered at origin, scaled to unit bounding box
- **Format:** Wavefront OBJ

## Clone Quality Validation

All 2,865 clones passed exhaustive load validation:

| Tier | Total | Loadable | Failures |
|------|-------|----------|----------|
| T1 | 573 | 573 | 0 |
| T2 | 573 | 573 | 0 |
| T3 | 573 | 573 | 0 |
| T4 | 573 | 573 | 0 |
| T5 | 573 | 573 | 0 |
| **Total** | **2,865** | **2,865** | **0** |

## Disk Usage

| Component | Size |
|-----------|------|
| Sources (191 OBJ) | 2.8 GB |
| T1 clones (573 OBJ) | 7.2 GB |
| T2 clones (573 OBJ) | 7.1 GB |
| T3 clones (573 OBJ) | 7.2 GB |
| T4 clones (573 OBJ) | 6.3 GB |
| T5 clones (573 OBJ) | 7.4 GB |
| **Total** | **~38 GB** |

## Category Distribution (40 LVIS categories)

5 models each from 38 categories, 1 model each from 2 categories (pacifier, poker):

alarm_clock, bandage, banner, baseball_cap, basketball_backboard, belt,
camper_(vehicle), cap_(headwear), cappuccino, carrot, cellular_telephone,
chopstick, cocoa_(beverage), cougar, cushion, deck_chair, fireplug,
fume_hood, gargle, gazelle, ginger, green_onion, handsaw, icecream, key,
kite, orange_(fruit), pacifier, pan_(for_cooking), pencil_sharpener,
poker_(fire_stirring_tool), salad_plate, saltshaker, sandal_(type_of_shoe),
ski_parka, soup, thimble, vinegar, watch, wrench

## Clone Naming Convention

```
{source_uid}_{tier}_{variant}.obj
```

Example: `2f3803f9694b4db88a41d75e37daf75b_T3_v2.obj`
- Source UID: `2f3803f9694b4db88a41d75e37daf75b`
- Tier: T3 (Medium)
- Variant: v2 (3rd clone variant)

## Usage

```python
import json
import trimesh

# Load the clone manifest
manifest = json.load(open("clone_manifest_5tier.json"))

# Get all T3 clones and their source UIDs
t3_pairs = [(c["clone_id"], c["source_uid"]) for c in manifest if c["tier"] == "T3"]

# Load a clone and its source
clone = trimesh.load("clones/T3/abc123_T3_v0.obj", force='mesh')
source = trimesh.load("sources/abc123.obj", force='mesh')
```

## Reproduction

Scripts in this directory (run with `3d-dedup` conda env):

1. `01_download_lvis.py` — Download Objaverse-LVIS subset
2. `02_convert_to_obj.py` — Convert to normalized OBJ
3. `03_generate_clones_5tier.py` — Generate all 5-tier clones
4. `04_validate_clones.py` — Validate all clones

Random seed: 42 (deterministic selection and transformations)

## License

Source models from Objaverse-LVIS are licensed under CC-BY-4.0.
Clone transformations and benchmark metadata are original work.
