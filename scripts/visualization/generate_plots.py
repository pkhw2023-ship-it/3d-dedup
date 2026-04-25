#!/usr/bin/env python3
"""generate_plots.py — Publication-quality plots for the Embedding Bake-Off blog post.

Creates 7+ plots from bakeoff_results.json and saves to tds-blog/post-3/images/.

Usage:
    python generate_plots.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# ─── Config ──────────────────────────────────────────────────────────
DATA_DIR = Path("/home/lightsail-user/3d-dataset-storage/tds-blog/data")
EMB_DIR = DATA_DIR / "embeddings"
IMG_DIR = Path("/home/lightsail-user/3d-dataset-storage/tds-blog/post-3/images")
RESULTS_FILE = DATA_DIR / "bakeoff_results.json"
CLONE_MANIFEST = DATA_DIR / "clones-objaverse" / "clone_manifest_5tier.json"
SOURCE_MANIFEST = DATA_DIR / "clones-objaverse" / "source_manifest.json"

IMG_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette
MODEL_COLORS = {
    "dinov2_base": "#2196F3",
    "dinov2_giant": "#1565C0",
    "clip_large": "#FF9800",
    "clip_base": "#FFB74D",
    "chamfer": "#4CAF50",
    "hausdorff": "#66BB6A",
    "sa_volume": "#81C784",
}

MODEL_LABELS = {
    "dinov2_base": "DINOv2-B",
    "dinov2_giant": "DINOv2-G",
    "clip_large": "CLIP-L/14",
    "clip_base": "CLIP-B/32",
    "chamfer": "Chamfer Dist.",
    "hausdorff": "Hausdorff Dist.",
    "sa_volume": "SA+Volume",
}

TIER_NAMES = {
    "T1": "Trivial\n(re-export)",
    "T2": "Easy\n(scale+rot)",
    "T3": "Medium\n(noise+decimate)",
    "T4": "Hard\n(partial+noise)",
    "T5": "Adversarial\n(remesh+deform)",
}

TIER_SHORT = {"T1": "T1: Trivial", "T2": "T2: Easy", "T3": "T3: Medium", "T4": "T4: Hard", "T5": "T5: Adversarial"}


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# Plot 1: Method Comparison Bar Chart (mAP across all methods)
# ═══════════════════════════════════════════════════════════════════════

def plot1_method_comparison(results):
    """Bar chart comparing mAP across best config per model + all strategies for winner."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Show best strategy (concat_pca28) for each model + geometry baselines
    target_keys = []

    # Best aggregation for each model, textured only
    for model in ["dinov2_giant", "dinov2_base", "clip_large", "clip_base"]:
        key = f"{model}_textured_concat_pca28"
        if key in results:
            target_keys.append(key)

    # Geometry baselines
    for geo in ["chamfer", "hausdorff", "sa_volume"]:
        if geo in results:
            target_keys.append(geo)

    labels = []
    mAPs = []
    colors = []

    for key in target_keys:
        res = results[key]
        parts = key.split("_")
        model = "_".join(parts[:2]) if parts[0] in ("dinov2", "clip") else key
        strategy = "_".join(parts[3:]) if len(parts) > 3 else ""

        label = MODEL_LABELS.get(model, model)
        if strategy:
            label += f"\n({strategy})"

        labels.append(label)
        mAPs.append(res.get("mAP", 0))
        colors.append(MODEL_COLORS.get(model, "#999999"))

    x = np.arange(len(labels))
    bars = ax.bar(x, mAPs, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    # Add value labels
    for bar, val in zip(bars, mAPs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Average Precision (mAP)")
    ax.set_title("Clone Retrieval: Best Embedding per Model vs. Geometry Baselines",
                 fontweight="bold", fontsize=14)
    ax.set_ylim(0, min(1.08, max(mAPs) * 1.12))

    # Add legend for model families
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=MODEL_COLORS["dinov2_giant"], label="DINOv2"),
        Patch(facecolor=MODEL_COLORS["clip_large"], label="CLIP"),
        Patch(facecolor=MODEL_COLORS.get("chamfer", "#4CAF50"), label="Geometry"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "01_method_comparison.png", bbox_inches="tight")
    plt.close()
    print("  Saved: 01_method_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 2: Per-Tier Heatmap
# ═══════════════════════════════════════════════════════════════════════

def plot2_tier_heatmap(results):
    """Heatmap showing mAP by tier for all methods."""
    tiers = ["T1", "T2", "T3", "T4", "T5"]

    # Collect methods that have per-tier results
    methods = []
    tier_data = []
    for key, res in sorted(results.items()):
        if key == "view_ablation":
            continue
        if f"mAP_T1" not in res:
            continue
        # Focus on concat_pca28, mean28 + geometry
        if not ("concat_pca28" in key or "mean28" in key or key in ("chamfer", "hausdorff", "sa_volume") or "single" in key):
            continue
        methods.append(key)
        row = [res.get(f"mAP_{t}", 0) for t in tiers]
        tier_data.append(row)

    if not methods:
        print("  SKIP: plot2 — no per-tier data")
        return

    data = np.array(tier_data)

    # Sort by overall mAP (sum of tiers)
    order = np.argsort(-data.sum(axis=1))
    data = data[order]
    methods = [methods[i] for i in order]

    # Clean up labels
    clean_labels = []
    for m in methods:
        parts = m.split("_")
        model = "_".join(parts[:2]) if parts[0] in ("dinov2", "clip") else m
        label = MODEL_LABELS.get(model, model)
        rest = "_".join(parts[2:]) if len(parts) > 2 else ""
        if rest:
            label += f" ({rest})"
        clean_labels.append(label)

    fig, ax = plt.subplots(figsize=(10, max(6, len(methods) * 0.4 + 1)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Add text annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([TIER_SHORT[t] for t in tiers], fontsize=10)
    ax.set_yticks(range(len(clean_labels)))
    ax.set_yticklabels(clean_labels, fontsize=9)
    ax.set_title("Per-Tier mAP: How Difficulty Affects Detection", fontweight="bold", fontsize=13)

    plt.colorbar(im, ax=ax, label="mAP", shrink=0.8)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "02_tier_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  Saved: 02_tier_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 3: Aggregation Strategy Comparison
# ═══════════════════════════════════════════════════════════════════════

def plot3_aggregation_comparison(results):
    """Grouped bar chart: aggregation strategies for each model."""
    strategies = ["single", "mean8", "mean28", "max28", "concat_pca28"]
    strategy_labels = ["Single View", "Mean (8)", "Mean (28)", "Max (28)", "Concat+PCA"]

    models = ["dinov2_base", "dinov2_giant", "clip_large", "clip_base"]
    mode = "textured"  # Focus on textured

    fig, ax = plt.subplots(figsize=(12, 6))
    n_strategies = len(strategies)
    n_models = len(models)
    bar_width = 0.15
    x = np.arange(n_models)

    for s_idx, (strat, strat_label) in enumerate(zip(strategies, strategy_labels)):
        mAPs = []
        for model in models:
            key = f"{model}_{mode}_{strat}"
            if key in results:
                mAPs.append(results[key].get("mAP", 0))
            else:
                mAPs.append(0)

        offset = (s_idx - n_strategies / 2 + 0.5) * bar_width
        cmap = plt.cm.viridis(s_idx / n_strategies)
        bars = ax.bar(x + offset, mAPs, bar_width, label=strat_label, color=cmap, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.set_ylabel("mAP")
    ax.set_title("Aggregation Strategy Comparison (Textured Renders)", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fancybox=True)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "03_aggregation_comparison.png", bbox_inches="tight")
    plt.close()
    print("  Saved: 03_aggregation_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 4: Textured vs LFD
# ═══════════════════════════════════════════════════════════════════════

def plot4_textured_vs_lfd(results):
    """Side-by-side comparison of textured vs LFD rendering."""
    models = ["dinov2_base", "dinov2_giant", "clip_large", "clip_base"]
    strategy = "mean28"

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    bar_width = 0.35

    textured_mAPs = []
    lfd_mAPs = []
    for model in models:
        key_tex = f"{model}_textured_{strategy}"
        key_lfd = f"{model}_lfd_{strategy}"
        textured_mAPs.append(results.get(key_tex, {}).get("mAP", 0))
        lfd_mAPs.append(results.get(key_lfd, {}).get("mAP", 0))

    bars1 = ax.bar(x - bar_width / 2, textured_mAPs, bar_width, label="Textured", color="#2196F3", edgecolor="white")
    bars2 = ax.bar(x + bar_width / 2, lfd_mAPs, bar_width, label="LFD (white plastic)", color="#FF9800", edgecolor="white")

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.set_ylabel("mAP")
    ax.set_title("Textured vs. LFD Rendering (Mean-28 Aggregation)", fontweight="bold")
    ax.legend(frameon=True, fancybox=True)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "04_textured_vs_lfd.png", bbox_inches="tight")
    plt.close()
    print("  Saved: 04_textured_vs_lfd.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 5: View Count Ablation
# ═══════════════════════════════════════════════════════════════════════

def plot5_view_ablation(results):
    """Line chart: mAP vs number of views."""
    ablation = results.get("view_ablation", {})
    if not ablation:
        print("  SKIP: plot5 — no view ablation data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by model+mode
    series = defaultdict(dict)
    for key, mAP in ablation.items():
        parts = key.split("_")
        model = "_".join(parts[:2])
        mode = parts[2]
        n_views = int(parts[3].replace("mean", ""))
        series[f"{model}_{mode}"][n_views] = mAP

    for name, data in sorted(series.items()):
        model = "_".join(name.split("_")[:2])
        mode = name.split("_")[2]
        label = f"{MODEL_LABELS.get(model, model)} ({mode})"
        color = MODEL_COLORS.get(model, "#999999")
        linestyle = "-" if mode == "textured" else "--"
        marker = "o" if mode == "textured" else "s"

        views = sorted(data.keys())
        mAPs = [data[v] for v in views]
        ax.plot(views, mAPs, marker=marker, linestyle=linestyle, label=label, color=color, linewidth=2, markersize=6)

    ax.set_xlabel("Number of Views")
    ax.set_ylabel("mAP")
    ax.set_title("View Count Ablation: Diminishing Returns?", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fancybox=True, ncol=2)
    ax.set_xticks([1, 4, 8, 12, 20, 28])
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "05_view_ablation.png", bbox_inches="tight")
    plt.close()
    print("  Saved: 05_view_ablation.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 6: Geometry vs. Embedding Methods
# ═══════════════════════════════════════════════════════════════════════

def plot6_geometry_vs_embedding(results):
    """Radar chart or grouped comparison of geometry vs embedding methods."""
    methods = []
    categories = ["mAP", "P@1", "P@5", "P@10", "R@10", "R@50"]

    # Best embedding method overall
    best_emb = None
    best_emb_map = 0
    for k, r in results.items():
        if k == "view_ablation":
            continue
        if k in ("chamfer", "hausdorff", "sa_volume"):
            continue
        if r.get("mAP", 0) > best_emb_map:
            best_emb_map = r.get("mAP", 0)
            best_emb = (k, r)

    target_methods = []
    if best_emb:
        target_methods.append(best_emb)
    for geo in ["chamfer", "hausdorff", "sa_volume"]:
        if geo in results:
            target_methods.append((geo, results[geo]))

    if len(target_methods) < 2:
        print("  SKIP: plot6 — insufficient data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))
    n_methods = len(target_methods)
    bar_width = 0.8 / n_methods

    for idx, (key, res) in enumerate(target_methods):
        model = "_".join(key.split("_")[:2]) if key.split("_")[0] in ("dinov2", "clip") else key
        label = MODEL_LABELS.get(model, model)
        rest = "_".join(key.split("_")[2:]) if len(key.split("_")) > 2 else ""
        if rest:
            label += f" ({rest})"
        color = MODEL_COLORS.get(model, "#999999")

        vals = [res.get(cat, 0) for cat in categories]
        offset = (idx - n_methods / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=label, color=color, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Score")
    ax.set_title("Embedding vs. Geometry Baselines", fontweight="bold")
    ax.legend(frameon=True, fancybox=True)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "06_geometry_vs_embedding.png", bbox_inches="tight")
    plt.close()
    print("  Saved: 06_geometry_vs_embedding.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 7: t-SNE / UMAP Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot7_embedding_visualization(results):
    """t-SNE or UMAP visualization of embeddings colored by source + tier."""
    import umap

    # Use the best model
    best_key = None
    best_map = 0
    for k in results:
        if k == "view_ablation":
            continue
        if "mean28" in k and "textured" in k:
            if results[k].get("mAP", 0) > best_map:
                best_map = results[k].get("mAP", 0)
                best_key = k

    if not best_key:
        print("  SKIP: plot7 — no embedding data")
        return

    model_key = "_".join(best_key.split("_")[:2])
    mode = best_key.split("_")[2]
    emb_file = EMB_DIR / f"{model_key}_{mode}_perview.npz"

    if not emb_file.exists():
        print(f"  SKIP: plot7 — {emb_file} not found")
        return

    data = np.load(emb_file, allow_pickle=True)
    model_ids = list(data["model_ids"])
    per_view_embs = data["embeddings"]

    # Aggregate
    agg_embs = per_view_embs.mean(axis=1)
    norms = np.linalg.norm(agg_embs, axis=1, keepdims=True)
    agg_embs = agg_embs / np.maximum(norms, 1e-8)

    # Load ground truth
    with open(CLONE_MANIFEST) as f:
        clones = json.load(f)
    with open(SOURCE_MANIFEST) as f:
        sources = json.load(f)

    clone_to_source = {c["clone_id"]: c["source_uid"] for c in clones}
    clone_to_tier = {c["clone_id"]: c["tier"] for c in clones}
    source_uids = set(sources["uids"])
    category_map = sources.get("category_map", {})

    # Assign colors
    source_ids = []
    tiers = []
    categories = []
    for mid in model_ids:
        if mid in source_uids:
            source_ids.append(mid)
            tiers.append("Source")
        else:
            source_ids.append(clone_to_source.get(mid, mid))
            tiers.append(clone_to_tier.get(mid, "Unknown"))
        categories.append(category_map.get(source_ids[-1], "unknown"))

    # UMAP
    print("  Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric="cosine")
    embedding_2d = reducer.fit_transform(agg_embs)

    # Plot colored by tier
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel A: Color by tier
    ax = axes[0]
    tier_colors = {"Source": "#333333", "T1": "#4CAF50", "T2": "#8BC34A", "T3": "#FFC107", "T4": "#FF9800", "T5": "#F44336"}
    tier_order = ["Source", "T1", "T2", "T3", "T4", "T5"]

    for tier in tier_order:
        mask = [t == tier for t in tiers]
        if sum(mask) == 0:
            continue
        idx = [i for i, m in enumerate(mask) if m]
        ax.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1],
                   c=tier_colors.get(tier, "#999"), s=8, alpha=0.6, label=tier)

    ax.set_title("UMAP — Colored by Clone Tier", fontweight="bold")
    ax.legend(markerscale=3, frameon=True, fancybox=True)
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel B: Color by category (top 10 categories)
    ax = axes[1]
    cat_counts = defaultdict(int)
    for c in categories:
        cat_counts[c] += 1
    top_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:10]
    top_cat_names = [c[0] for c in top_cats]

    cat_cmap = plt.cm.tab10
    for cat_idx, cat_name in enumerate(top_cat_names):
        mask = [c == cat_name for c in categories]
        idx = [i for i, m in enumerate(mask) if m]
        ax.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1],
                   c=[cat_cmap(cat_idx)], s=8, alpha=0.6, label=cat_name)

    # Plot remaining as grey
    other_mask = [c not in top_cat_names for c in categories]
    other_idx = [i for i, m in enumerate(other_mask) if m]
    if other_idx:
        ax.scatter(embedding_2d[other_idx, 0], embedding_2d[other_idx, 1],
                   c="#cccccc", s=4, alpha=0.3, label="other")

    ax.set_title("UMAP — Colored by Category", fontweight="bold")
    ax.legend(markerscale=3, frameon=True, fancybox=True, ncol=2, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.suptitle(f"Embedding Space Visualization ({MODEL_LABELS.get(model_key, model_key)}, {mode}, mean-28)",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "07_umap_visualization.png", bbox_inches="tight")
    plt.close()
    print("  Saved: 07_umap_visualization.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 8: Summary Table as Figure
# ═══════════════════════════════════════════════════════════════════════

def plot8_summary_table(results):
    """Render the key results as a clean table figure."""
    # Collect key methods
    rows = []
    for key, res in sorted(results.items()):
        if key == "view_ablation":
            continue
        # Only show key configurations
        if not ("concat_pca28" in key or "mean28" in key or "single" in key or key in ("chamfer", "hausdorff", "sa_volume")):
            continue
        parts = key.split("_")
        model = "_".join(parts[:2]) if parts[0] in ("dinov2", "clip") else key
        mode = parts[2] if len(parts) > 2 else "-"
        strategy = "_".join(parts[3:]) if len(parts) > 3 else "-"

        rows.append({
            "Method": MODEL_LABELS.get(model, model),
            "Mode": mode,
            "Agg": strategy,
            "mAP": res.get("mAP", 0),
            "P@1": res.get("P@1", 0),
            "P@5": res.get("P@5", 0),
            "T1": res.get("mAP_T1", 0),
            "T5": res.get("mAP_T5", 0),
        })

    if not rows:
        print("  SKIP: plot8 — no data")
        return

    rows.sort(key=lambda r: -r["mAP"])

    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.35 + 2)))
    ax.axis("off")

    col_labels = ["Method", "Mode", "Agg", "mAP", "P@1", "P@5", "T1 mAP", "T5 mAP"]
    cell_text = []
    cell_colors = []

    cmap = plt.cm.RdYlGn

    for row in rows:
        text = [
            row["Method"], row["Mode"], row["Agg"],
            f"{row['mAP']:.4f}", f"{row['P@1']:.4f}", f"{row['P@5']:.4f}",
            f"{row['T1']:.4f}", f"{row['T5']:.4f}",
        ]
        cell_text.append(text)

        # Color intensity based on mAP
        intensity = row["mAP"]
        colors = ["white", "white", "white",
                  cmap(intensity), cmap(row["P@1"]), cmap(row["P@5"]),
                  cmap(row["T1"]), cmap(row["T5"])]
        cell_colors.append(colors)

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=cell_colors,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#E0E0E0")

    ax.set_title("Embedding Bake-Off: Summary Results", fontweight="bold", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "08_summary_table.png", bbox_inches="tight")
    plt.close()
    print("  Saved: 08_summary_table.png")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("GENERATING PUBLICATION-QUALITY PLOTS")
    print("=" * 60)

    results = load_results()
    print(f"Loaded {len(results)} result entries")

    plot1_method_comparison(results)
    plot2_tier_heatmap(results)
    plot3_aggregation_comparison(results)
    plot4_textured_vs_lfd(results)
    plot5_view_ablation(results)
    plot6_geometry_vs_embedding(results)
    plot7_embedding_visualization(results)
    plot8_summary_table(results)

    print(f"\n{'=' * 60}")
    print(f"All plots saved to: {IMG_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
