#!/usr/bin/env python3
"""
generate_rescorer_plots.py — Generate blog-quality visualizations for the VLM rescorer.

Produces:
1. Pipeline diagram: 3-stage flowchart with filtering ratios
2. Precision-recall curve: threshold sweep
3. Stage funnel: how many pairs survive each stage
4. Example verdicts: side-by-side comparisons with VLM reasoning
5. Ablation plots: impact of views, render mode, stage 2 skip

Usage:
    python generate_rescorer_plots.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image
from collections import defaultdict

# ─── Config ──────────────────────────────────────────────────────────
DATA_DIR = Path("/home/lightsail-user/3d-dataset-storage/tds-blog/data")
RESCORER_DIR = DATA_DIR / "rescorer"
RENDER_DIR = DATA_DIR / "renders"
IMG_DIR = Path("/home/lightsail-user/3d-dataset-storage/tds-blog/post-4b/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.alpha": 0.3,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "primary": "#2563EB",
    "secondary": "#10B981",
    "accent": "#F59E0B",
    "danger": "#EF4444",
    "purple": "#8B5CF6",
    "gray": "#6B7280",
    "light_blue": "#DBEAFE",
    "light_green": "#D1FAE5",
    "light_orange": "#FEF3C7",
    "T1": "#10B981",
    "T2": "#3B82F6",
    "T3": "#F59E0B",
    "T4": "#EF4444",
    "T5": "#8B5CF6",
}


def load_ablation_summary():
    """Load the ablation summary JSON."""
    path = RESCORER_DIR / "ablation_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_pipeline_result(tag):
    """Load a single pipeline result JSON."""
    path = RESCORER_DIR / f"pipeline_{tag}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════
# Plot 1: Pipeline Diagram
# ═══════════════════════════════════════════════════════════════════════

def plot_pipeline_diagram(stats):
    """Create a flowchart showing the 3-stage pipeline with filtering ratios."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Stage boxes
    stages = [
        {"x": 1, "y": 1.5, "w": 2.2, "h": 1.8,
         "title": "Stage 1\nEmbedding\nRetrieval",
         "count": stats.get("stage1_pairs", 0),
         "color": COLORS["light_blue"],
         "edge": COLORS["primary"]},
        {"x": 4, "y": 1.5, "w": 2.2, "h": 1.8,
         "title": "Stage 2\nSingle-View\nVLM Screen",
         "count": stats.get("stage2_passed", 0),
         "color": COLORS["light_orange"],
         "edge": COLORS["accent"]},
        {"x": 7, "y": 1.5, "w": 2.2, "h": 1.8,
         "title": "Stage 3\nMulti-View\nRescoring",
         "count": stats.get("stage3_confirmed", 0),
         "color": COLORS["light_green"],
         "edge": COLORS["secondary"]},
    ]

    for s in stages:
        rect = mpatches.FancyBboxPatch(
            (s["x"], s["y"]), s["w"], s["h"],
            boxstyle=mpatches.BoxStyle("Round", pad=0.15),
            facecolor=s["color"], edgecolor=s["edge"], linewidth=2.5,
        )
        ax.add_patch(rect)
        ax.text(s["x"] + s["w"]/2, s["y"] + s["h"]*0.65, s["title"],
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="#1F2937")
        ax.text(s["x"] + s["w"]/2, s["y"] + 0.25,
                f"{s['count']:,} pairs",
                ha="center", va="center", fontsize=10, color=s["edge"],
                fontweight="bold")

    # Arrows
    for i in range(2):
        x_start = stages[i]["x"] + stages[i]["w"]
        x_end = stages[i+1]["x"]
        y_mid = stages[i]["y"] + stages[i]["h"] / 2
        ax.annotate("", xy=(x_end, y_mid), xytext=(x_start, y_mid),
                     arrowprops=dict(arrowstyle="->", color="#4B5563",
                                     lw=2, connectionstyle="arc3"))

    # Rejection labels
    s2_rejected = stats.get("stage2_rejected", 0)
    s3_rejected = stats.get("stage3_evaluated", 0) - stats.get("stage3_confirmed", 0)

    if s2_rejected > 0:
        ax.annotate(f"✗ {s2_rejected}\nrejected",
                     xy=(5.1, 1.3), fontsize=9, color=COLORS["danger"],
                     ha="center", va="top", fontweight="bold")
        ax.annotate("", xy=(5.1, 0.6), xytext=(5.1, 1.3),
                     arrowprops=dict(arrowstyle="->", color=COLORS["danger"],
                                     lw=1.5, linestyle="--"))

    if s3_rejected > 0:
        ax.annotate(f"✗ {s3_rejected}\nrejected",
                     xy=(8.1, 1.3), fontsize=9, color=COLORS["danger"],
                     ha="center", va="top", fontweight="bold")
        ax.annotate("", xy=(8.1, 0.6), xytext=(8.1, 1.3),
                     arrowprops=dict(arrowstyle="->", color=COLORS["danger"],
                                     lw=1.5, linestyle="--"))

    # Title
    ax.text(5, 3.7, "3-Stage VLM Rescoring Pipeline",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="#1F2937")

    fig.tight_layout()
    fig.savefig(IMG_DIR / "pipeline_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: pipeline_diagram.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 2: Precision-Recall Curve (Threshold Sweep)
# ═══════════════════════════════════════════════════════════════════════

def plot_precision_recall_curve(summary):
    """Plot precision vs recall at different evidence thresholds."""
    thresholds = []
    precisions = []
    recalls = []

    for tag, data in sorted(summary.items()):
        if not tag.startswith("threshold_ev"):
            continue
        ev = data["config"]["min_angle_evidence"]
        p = data["with_rescorer"].get("precision", 0)
        r = data["with_rescorer"].get("recall", 0)
        thresholds.append(ev)
        precisions.append(p)
        recalls.append(r)

    if not thresholds:
        print("  SKIP: No threshold sweep data found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # PR curve
    ax1.plot(recalls, precisions, "o-", color=COLORS["primary"], lw=2,
             markersize=8, zorder=5)
    for t, p, r in zip(thresholds, precisions, recalls):
        ax1.annotate(f"t={t}", (r, p), textcoords="offset points",
                      xytext=(8, 5), fontsize=8, color=COLORS["gray"])

    # Highlight default threshold
    if 2 in thresholds:
        idx = thresholds.index(2)
        ax1.plot(recalls[idx], precisions[idx], "s", color=COLORS["accent"],
                 markersize=12, zorder=6, label="Default (t=2)")

    # Add embedding-only baseline
    emb_data = list(summary.values())[0].get("embedding_only", {})
    if emb_data:
        ax1.axhline(y=emb_data.get("precision", 0), color=COLORS["danger"],
                     linestyle="--", alpha=0.7, label="Embedding-only P")
        ax1.axvline(x=emb_data.get("recall", 0), color=COLORS["danger"],
                     linestyle=":", alpha=0.7, label="Embedding-only R")

    ax1.set_xlabel("Recall", fontsize=12)
    ax1.set_ylabel("Precision", fontsize=12)
    ax1.set_title("Precision-Recall: Threshold Sweep", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # Threshold vs metrics
    ax2.plot(thresholds, precisions, "o-", color=COLORS["primary"], lw=2,
             markersize=8, label="Precision")
    ax2.plot(thresholds, recalls, "s-", color=COLORS["secondary"], lw=2,
             markersize=8, label="Recall")

    # F1
    f1s = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(precisions, recalls)]
    ax2.plot(thresholds, f1s, "^-", color=COLORS["accent"], lw=2,
             markersize=8, label="F1")

    ax2.set_xlabel("Evidence Threshold", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title("Metrics vs. Evidence Threshold", fontsize=13, fontweight="bold")
    ax2.set_xticks(thresholds)
    ax2.legend(fontsize=10)
    ax2.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "precision_recall_threshold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: precision_recall_threshold.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 3: Stage Funnel
# ═══════════════════════════════════════════════════════════════════════

def plot_stage_funnel(stats):
    """Bar chart showing how many pairs survive each stage."""
    stages = ["Stage 1\n(Embedding)", "Stage 2\n(VLM Screen)", "Stage 3\n(Multi-View)"]
    counts = [
        stats.get("stage1_pairs", 0),
        stats.get("stage2_passed", 0),
        stats.get("stage3_confirmed", 0),
    ]

    if all(c == 0 for c in counts):
        print("  SKIP: No stage funnel data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [COLORS["primary"], COLORS["accent"], COLORS["secondary"]]
    bars = ax.bar(stages, counts, color=colors, edgecolor="white", linewidth=1.5,
                  width=0.6)

    # Labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                f"{count:,}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Filtering percentages
    if counts[0] > 0 and counts[1] > 0:
        pct_s2 = counts[1] / counts[0] * 100
        ax.annotate(f"→ {pct_s2:.0f}% pass", xy=(0.5, (counts[0]+counts[1])/2),
                     fontsize=10, color=COLORS["gray"], ha="center")
    if counts[1] > 0 and counts[2] > 0:
        pct_s3 = counts[2] / counts[1] * 100
        ax.annotate(f"→ {pct_s3:.0f}% pass", xy=(1.5, (counts[1]+counts[2])/2),
                     fontsize=10, color=COLORS["gray"], ha="center")

    ax.set_ylabel("Candidate Pairs", fontsize=12)
    ax.set_title("VLM Rescorer: Stage Funnel", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(IMG_DIR / "stage_funnel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: stage_funnel.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 4: Example Verdicts
# ═══════════════════════════════════════════════════════════════════════

def plot_example_verdicts(pipeline_results, render_mode="textured", n_examples=6):
    """Show example pairs with VLM verdicts per angle."""
    if not pipeline_results:
        print("  SKIP: No pipeline results for example verdicts")
        return

    # Find interesting examples: confirmed duplicates and rejected pairs
    confirmed = []
    rejected = []

    for qr in pipeline_results:
        for dup in qr.get("confirmed_duplicates", []):
            if dup.get("angle_details"):
                confirmed.append((qr["query"], dup))
        for rej in qr.get("rejected", []):
            if rej.get("angle_details") and rej.get("rejected_at") == "stage3":
                rejected.append((qr["query"], rej))

    examples = []
    for item in confirmed[:3]:
        examples.append(("✓ Confirmed Duplicate", item))
    for item in rejected[:3]:
        examples.append(("✗ Rejected", item))

    if not examples:
        print("  SKIP: No examples with angle details found")
        return

    n_examples = min(n_examples, len(examples))
    fig, axes = plt.subplots(n_examples, 4, figsize=(16, 3 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    render_base = RENDER_DIR / render_mode

    for row, (verdict_label, (query_id, pair_data)) in enumerate(examples[:n_examples]):
        cand_id = pair_data["candidate"]
        evidence = pair_data.get("evidence_count", 0)
        sim = pair_data.get("sim_score", 0)

        # Show 4 angles
        angle_details = pair_data.get("angle_details", [])
        for col in range(4):
            ax = axes[row, col]
            if col < len(angle_details):
                ad = angle_details[col]
                view_idx = ad.get("view_idx", col)
                angle = ad.get("angle", 0)
                has_ev = ad.get("evidence", False)

                # Load side-by-side
                img_a_path = render_base / query_id / f"view_{view_idx:02d}.png"
                img_b_path = render_base / cand_id / f"view_{view_idx:02d}.png"

                if img_a_path.exists() and img_b_path.exists():
                    img_a = Image.open(img_a_path).convert("RGB").resize((112, 112))
                    img_b = Image.open(img_b_path).convert("RGB").resize((112, 112))
                    composite = Image.new("RGB", (224, 112))
                    composite.paste(img_a, (0, 0))
                    composite.paste(img_b, (112, 0))
                    ax.imshow(composite)

                ev_marker = "✓" if has_ev else "✗"
                ev_color = COLORS["secondary"] if has_ev else COLORS["danger"]
                ax.set_title(f"{angle}° {ev_marker}", fontsize=10, color=ev_color,
                            fontweight="bold")
            ax.axis("off")

        # Row label
        color = COLORS["secondary"] if "Confirmed" in verdict_label else COLORS["danger"]
        axes[row, 0].set_ylabel(
            f"{verdict_label}\nEv: {evidence}/8\nSim: {sim:.3f}",
            fontsize=9, color=color, fontweight="bold", rotation=0,
            labelpad=80, ha="right", va="center",
        )

    fig.suptitle("VLM Rescorer: Example Verdicts (4 of 8 angles shown)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "example_verdicts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: example_verdicts.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 5: Ablation Plots
# ═══════════════════════════════════════════════════════════════════════

def plot_ablation_views(summary):
    """Impact of number of views on precision/recall."""
    views = []
    precisions = []
    recalls = []
    f1s = []

    for tag, data in sorted(summary.items()):
        if not tag.startswith("views_"):
            continue
        n = data["config"]["n_views"]
        p = data["with_rescorer"].get("precision", 0)
        r = data["with_rescorer"].get("recall", 0)
        views.append(n)
        precisions.append(p)
        recalls.append(r)
        f1s.append(2*p*r/(p+r) if (p+r) > 0 else 0)

    if not views:
        print("  SKIP: No view count ablation data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(views, precisions, "o-", color=COLORS["primary"], lw=2,
            markersize=8, label="Precision")
    ax.plot(views, recalls, "s-", color=COLORS["secondary"], lw=2,
            markersize=8, label="Recall")
    ax.plot(views, f1s, "^-", color=COLORS["accent"], lw=2,
            markersize=8, label="F1")
    ax.set_xlabel("Number of Views", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Ablation: Number of Views in Stage 3", fontsize=13, fontweight="bold")
    ax.set_xticks(views)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "ablation_views.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: ablation_views.png")


def plot_ablation_render_mode(summary):
    """Textured vs LFD for VLM input."""
    modes = {}
    for tag, data in summary.items():
        if tag.startswith("render_"):
            mode = data["config"]["render_mode"]
            modes[mode] = data

    if len(modes) < 2:
        print("  SKIP: Need both render modes for comparison")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(3)
    width = 0.3
    labels = ["Precision", "Recall", "F1"]

    for i, (mode, data) in enumerate(modes.items()):
        r = data["with_rescorer"]
        p = r.get("precision", 0)
        rec = r.get("recall", 0)
        f1 = 2*p*rec/(p+rec) if (p+rec) > 0 else 0
        vals = [p, rec, f1]
        color = COLORS["primary"] if mode == "textured" else COLORS["accent"]
        bars = ax.bar(x + i*width, vals, width, label=mode.capitalize(),
                      color=color, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width/2)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Ablation: Render Mode for VLM Input", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "ablation_render_mode.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: ablation_render_mode.png")


def plot_ablation_stage2(summary):
    """Impact of Stage 2 screening: with vs without."""
    with_s2 = summary.get("stage2_enabled")
    without_s2 = summary.get("stage2_disabled")

    if not with_s2 or not without_s2:
        print("  SKIP: Need both Stage 2 variants")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Metrics comparison
    labels = ["Precision", "Recall", "F1"]
    for i, (tag, data, color, label) in enumerate([
        ("with", with_s2, COLORS["primary"], "With Stage 2"),
        ("without", without_s2, COLORS["accent"], "Without Stage 2"),
    ]):
        r = data["with_rescorer"]
        p = r.get("precision", 0)
        rec = r.get("recall", 0)
        f1 = 2*p*rec/(p+rec) if (p+rec) > 0 else 0
        vals = [p, rec, f1]
        x = np.arange(3)
        ax1.bar(x + i*0.3, vals, 0.3, label=label, color=color, edgecolor="white")
        for j, (xx, val) in enumerate(zip(x + i*0.3, vals)):
            ax1.text(xx, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax1.set_xticks(x + 0.15)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Metrics: Stage 2 Impact", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.15)

    # Cost comparison
    cost_labels = ["API Calls", "Total Tokens (K)"]
    with_costs = [
        with_s2["stats"]["api_calls"],
        with_s2["stats"]["total_tokens"] / 1000,
    ]
    without_costs = [
        without_s2["stats"]["api_calls"],
        without_s2["stats"]["total_tokens"] / 1000,
    ]

    x = np.arange(2)
    ax2.bar(x - 0.15, with_costs, 0.3, label="With Stage 2",
            color=COLORS["primary"], edgecolor="white")
    ax2.bar(x + 0.15, without_costs, 0.3, label="Without Stage 2",
            color=COLORS["accent"], edgecolor="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels(cost_labels, fontsize=11)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Cost: Stage 2 Impact", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "ablation_stage2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: ablation_stage2.png")


def plot_per_tier_comparison(summary):
    """Per-tier precision/recall: embedding-only vs rescorer."""
    # Use render_textured which has the full per-tier format with both emb and rescorer
    data = None
    for tag in ["render_textured", "baseline", "stage2_enabled", "threshold_ev2"]:
        if tag in summary and "per_tier" in summary[tag]:
            pt = summary[tag]["per_tier"]
            # Check if this has the embedding vs rescorer format
            if pt and any("embedding_precision" in v or "rescorer_precision" in v
                         for v in pt.values() if isinstance(v, dict)):
                data = summary[tag]
                break
            elif pt and any("precision" in v for v in pt.values() if isinstance(v, dict)):
                data = summary[tag]
                break

    if not data or not data.get("per_tier"):
        print("  SKIP: No per-tier data available")
        return

    tiers = ["T1", "T2", "T3", "T4", "T5"]
    tier_labels = ["T1\nTrivial", "T2\nEasy", "T3\nMedium", "T4\nHard", "T5\nAdversarial"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(tiers))
    width = 0.35

    # Handle both per-tier formats
    pt = data["per_tier"]
    emb_p = [pt[t].get("embedding_precision", pt[t].get("precision", 0)) for t in tiers]
    res_p = [pt[t].get("rescorer_precision", pt[t].get("precision", 0)) for t in tiers]
    emb_r = [pt[t].get("embedding_recall", pt[t].get("recall", 0)) for t in tiers]
    res_r = [pt[t].get("rescorer_recall", pt[t].get("recall", 0)) for t in tiers]

    ax1.bar(x - width/2, emb_p, width, label="Embedding Only",
            color=COLORS["gray"], edgecolor="white")
    ax1.bar(x + width/2, res_p, width, label="+ VLM Rescorer",
            color=COLORS["primary"], edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tier_labels, fontsize=10)
    ax1.set_ylabel("Precision", fontsize=12)
    ax1.set_title("Precision by Clone Difficulty Tier", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.15)

    ax2.bar(x - width/2, emb_r, width, label="Embedding Only",
            color=COLORS["gray"], edgecolor="white")
    ax2.bar(x + width/2, res_r, width, label="+ VLM Rescorer",
            color=COLORS["secondary"], edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tier_labels, fontsize=10)
    ax2.set_ylabel("Recall", fontsize=12)
    ax2.set_title("Recall by Clone Difficulty Tier", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 1.15)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "per_tier_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: per_tier_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
# Combined Summary Table
# ═══════════════════════════════════════════════════════════════════════

def plot_summary_table(summary):
    """Create a summary comparison table as an image."""
    if not summary:
        return

    # Collect key rows
    rows = []
    for tag in ["baseline", "threshold_ev1", "threshold_ev2",
                "threshold_ev4", "threshold_ev6", "threshold_ev8",
                "views_4", "views_6", "views_8",
                "render_textured", "render_lfd",
                "stage2_enabled", "stage2_disabled"]:
        if tag not in summary:
            continue
        d = summary[tag]
        emb = d.get("embedding_only", {})
        res = d.get("with_rescorer", {})
        s = d.get("stats", {})
        rows.append({
            "Run": tag,
            "Emb P": f"{emb.get('precision', 0):.3f}",
            "Emb R": f"{emb.get('recall', 0):.3f}",
            "Res P": f"{res.get('precision', 0):.3f}",
            "Res R": f"{res.get('recall', 0):.3f}",
            "API": str(s.get("api_calls", "?")),
            "Tokens": f"{s.get('total_tokens', 0)/1000:.0f}K",
        })

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(14, max(4, 0.4 * len(rows) + 2)))
    ax.axis("off")

    col_labels = list(rows[0].keys())
    cell_text = [[r[c] for c in col_labels] for r in rows]

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor(COLORS["primary"])
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Ablation Study Summary", fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "ablation_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: ablation_summary_table.png")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("GENERATING RESCORER BLOG ASSETS")
    print("=" * 70)

    # Load data
    summary = load_ablation_summary()

    # Find baseline pipeline result for detailed data
    baseline_data = None
    for tag in ["baseline_textured_top10_ev2_v8", "baseline_textured_top20_ev2_v8",
                "baseline", "stage2_enabled", "render_textured", "threshold_ev2"]:
        d = load_pipeline_result(tag)
        if d:
            baseline_data = d
            break

    # Get stats from baseline
    stats = baseline_data.get("stats", {}) if baseline_data else {}
    pipeline_results = baseline_data.get("pipeline_results", []) if baseline_data else []

    # Generate plots
    print("\n1. Pipeline diagram...")
    if stats:
        plot_pipeline_diagram(stats)
    else:
        print("  SKIP: No stats data")

    print("\n2. Precision-recall curve...")
    if summary:
        plot_precision_recall_curve(summary)
    else:
        print("  SKIP: No ablation summary")

    print("\n3. Stage funnel...")
    if stats:
        plot_stage_funnel(stats)
    else:
        print("  SKIP: No stats")

    print("\n4. Example verdicts...")
    plot_example_verdicts(pipeline_results, render_mode="textured")

    print("\n5. Ablation: views...")
    if summary:
        plot_ablation_views(summary)

    print("\n6. Ablation: render mode...")
    if summary:
        plot_ablation_render_mode(summary)

    print("\n7. Ablation: Stage 2...")
    if summary:
        plot_ablation_stage2(summary)

    print("\n8. Per-tier comparison...")
    if summary:
        plot_per_tier_comparison(summary)

    print("\n9. Summary table...")
    if summary:
        plot_summary_table(summary)

    print(f"\n{'='*70}")
    print(f"All assets saved to: {IMG_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
