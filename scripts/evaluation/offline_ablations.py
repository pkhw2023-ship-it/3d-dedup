#!/usr/bin/env python3
"""
offline_ablations.py — Compute ablation results from a single pipeline run.

The key insight: once we have multi-view evidence scores from a full 8-view
run, we can retroactively apply different thresholds and view subsets without
re-running the VLM. This saves thousands of API calls.

From a single pipeline_*.json with 8-view Stage 3 results, we can compute:
1. Threshold sweep: min_angle_evidence from 1 to 8
2. View count ablation: what if we only used 4 or 6 views?
3. Stage 2 skip: what if we sent everything to Stage 3?

The render mode ablation still requires a separate run.

Usage:
    python offline_ablations.py --input pipeline_baseline.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("/home/lightsail-user/3d-dataset-storage/tds-blog/data")
RESCORER_DIR = DATA_DIR / "rescorer"
CLONE_MANIFEST = DATA_DIR / "clones-objaverse" / "clone_manifest_5tier.json"
SOURCE_MANIFEST = DATA_DIR / "clones-objaverse" / "source_manifest.json"
EMB_DIR = DATA_DIR / "embeddings"


def load_ground_truth():
    with open(CLONE_MANIFEST) as f:
        clones = json.load(f)
    with open(SOURCE_MANIFEST) as f:
        sources = json.load(f)

    clone_to_source = {c["clone_id"]: c["source_uid"] for c in clones}
    clone_to_tier = {c["clone_id"]: c["tier"] for c in clones}
    source_uids = set(sources["uids"])
    return clone_to_source, clone_to_tier, source_uids


def get_source_id(model_id, clone_to_source, source_uids):
    if model_id in source_uids:
        return model_id
    return clone_to_source.get(model_id, model_id)


def evaluate_with_threshold(pipeline_results, threshold, n_views,
                            clone_to_source, clone_to_tier, source_uids,
                            include_stage2_rejects=False):
    """Re-evaluate pipeline results with a different threshold / view count.

    Args:
        pipeline_results: list of per-query results from the rescorer
        threshold: min_angle_evidence for "confirmed duplicate"
        n_views: only consider the first n_views angles
        include_stage2_rejects: if True, re-evaluate Stage 2 rejects in Stage 3
                                (simulates skipping Stage 2)
    """
    source_to_clones = defaultdict(set)
    for clone_id, src in clone_to_source.items():
        source_to_clones[src].add(clone_id)
    for src in source_uids:
        source_to_clones[src].add(src)

    def get_relevant(model_id):
        src = get_source_id(model_id, clone_to_source, source_uids)
        return source_to_clones.get(src, set()) - {model_id}

    # Per-tier tracking
    tier_metrics = {t: {"tp": 0, "fp": 0, "relevant": 0}
                    for t in ["T1", "T2", "T3", "T4", "T5"]}

    tp = 0
    fp = 0
    total_relevant = 0

    for qr in pipeline_results:
        query_id = qr["query"]
        relevant = get_relevant(query_id)
        if not relevant:
            continue
        total_relevant += len(relevant)

        # Count relevant per tier
        for rel_id in relevant:
            tier = clone_to_tier.get(rel_id)
            if tier and tier in tier_metrics:
                tier_metrics[tier]["relevant"] += 1

        # Collect all candidates with angle details
        all_candidates = []

        # Confirmed duplicates (already have angle_details)
        for dup in qr.get("confirmed_duplicates", []):
            all_candidates.append(dup)

        # Rejected at Stage 3 (already have angle_details)
        for rej in qr.get("rejected", []):
            if rej.get("angle_details"):
                all_candidates.append(rej)
            elif include_stage2_rejects and rej.get("rejected_at") == "stage2":
                # For Stage 2 skip ablation: these don't have angle_details
                # We can only count them as "would need Stage 3" but can't evaluate
                pass

        # Re-evaluate with new threshold and view count
        for cand in all_candidates:
            cand_id = cand["candidate"]
            angle_details = cand.get("angle_details", [])

            # Subset to first n_views
            subset = angle_details[:n_views]
            evidence = sum(1 for a in subset if a.get("evidence", False))
            is_dup = evidence >= threshold

            if is_dup:
                if cand_id in relevant:
                    tp += 1
                    tier = clone_to_tier.get(cand_id)
                    if tier and tier in tier_metrics:
                        tier_metrics[tier]["tp"] += 1
                else:
                    fp += 1
                    tier = clone_to_tier.get(cand_id)
                    if tier and tier in tier_metrics:
                        tier_metrics[tier]["fp"] += 1

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, total_relevant)
    f1 = safe_div(2 * tp, 2 * tp + fp + (total_relevant - tp))

    per_tier = {}
    for tier, tm in tier_metrics.items():
        per_tier[tier] = {
            "precision": safe_div(tm["tp"], tm["tp"] + tm["fp"]),
            "recall": safe_div(tm["tp"], tm["relevant"]),
        }

    return {
        "threshold": threshold,
        "n_views": n_views,
        "true_positives": tp,
        "false_positives": fp,
        "total_relevant": total_relevant,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_tier": per_tier,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Pipeline result JSON file")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Load pipeline results
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = RESCORER_DIR / input_path

    with open(input_path) as f:
        data = json.load(f)

    pipeline_results = data.get("pipeline_results", [])
    stats = data.get("stats", {})
    config = data.get("config", {})

    print(f"Loaded {len(pipeline_results)} query results from {input_path.name}")
    print(f"API calls: {stats.get('api_calls')}, Tokens: {stats.get('total_tokens')}")

    # Load ground truth
    clone_to_source, clone_to_tier, source_uids = load_ground_truth()

    # ─── Ablation 1: Threshold sweep ────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION 1: Evidence threshold sweep (offline)")
    print("=" * 60)

    threshold_results = {}
    for t in range(1, 9):
        result = evaluate_with_threshold(
            pipeline_results, threshold=t, n_views=8,
            clone_to_source=clone_to_source,
            clone_to_tier=clone_to_tier,
            source_uids=source_uids,
        )
        threshold_results[f"threshold_ev{t}"] = result
        print(f"  t={t}: P={result['precision']:.4f}  R={result['recall']:.4f}  "
              f"F1={result['f1']:.4f}  TP={result['true_positives']}  FP={result['false_positives']}")

    # ─── Ablation 2: View count ─────────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION 2: View count (offline)")
    print("=" * 60)

    view_results = {}
    for n_views in [2, 4, 6, 8]:
        result = evaluate_with_threshold(
            pipeline_results, threshold=2, n_views=n_views,
            clone_to_source=clone_to_source,
            clone_to_tier=clone_to_tier,
            source_uids=source_uids,
        )
        view_results[f"views_{n_views}"] = result
        print(f"  views={n_views}: P={result['precision']:.4f}  R={result['recall']:.4f}  "
              f"F1={result['f1']:.4f}")

    # ─── Ablation 3: Stage 2 impact ─────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION 3: Stage 2 impact analysis")
    print("=" * 60)

    # Count how many Stage 2 rejects were actually true/false negatives
    s2_true_negatives = 0
    s2_false_negatives = 0
    s2_total_rejects = 0

    for qr in pipeline_results:
        query_id = qr["query"]
        src = get_source_id(query_id, clone_to_source, source_uids)
        relevant = defaultdict(set)
        for clone_id, s in clone_to_source.items():
            if s == src:
                relevant[src].add(clone_id)

        for rej in qr.get("rejected", []):
            if rej.get("rejected_at") == "stage2":
                s2_total_rejects += 1
                cand_id = rej["candidate"]
                cand_src = get_source_id(cand_id, clone_to_source, source_uids)
                if cand_src == src:
                    s2_false_negatives += 1
                else:
                    s2_true_negatives += 1

    print(f"  Stage 2 total rejects: {s2_total_rejects}")
    print(f"  Stage 2 true negatives:  {s2_true_negatives} (correctly rejected)")
    print(f"  Stage 2 false negatives: {s2_false_negatives} (wrongly rejected duplicates)")

    stage2_analysis = {
        "total_rejects": s2_total_rejects,
        "true_negatives": s2_true_negatives,
        "false_negatives": s2_false_negatives,
        "saved_api_calls": s2_total_rejects * 8,  # 8 views saved per rejection
    }

    # ─── Combine results ────────────────────────────────────────
    ablation_summary = {
        "source_run": input_path.name,
        "n_queries": len(pipeline_results),
        "original_config": config,
        "original_stats": stats,
        "threshold_sweep": threshold_results,
        "view_count": view_results,
        "stage2_analysis": stage2_analysis,
    }

    # Build the format expected by plotting script
    plot_summary = {}

    # Threshold sweep entries
    for tag, result in threshold_results.items():
        plot_summary[tag] = {
            "config": {
                "render_mode": config.get("render_mode"),
                "min_angle_evidence": result["threshold"],
                "n_views": 8,
                "skip_stage2": False,
                "top_k": config.get("top_k"),
            },
            "embedding_only": data.get("evaluation", {}).get("embedding_only", {}),
            "with_rescorer": {
                "true_positives": result["true_positives"],
                "false_positives": result["false_positives"],
                "total_relevant": result["total_relevant"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            },
            "per_tier": result["per_tier"],
            "stats": stats,
        }

    # View count entries
    for tag, result in view_results.items():
        plot_summary[tag] = {
            "config": {
                "render_mode": config.get("render_mode"),
                "min_angle_evidence": 2,
                "n_views": result["n_views"],
                "skip_stage2": False,
                "top_k": config.get("top_k"),
            },
            "embedding_only": data.get("evaluation", {}).get("embedding_only", {}),
            "with_rescorer": {
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            },
            "per_tier": result["per_tier"],
            "stats": stats,
        }

    # Stage 2 entries
    # "with stage2" = the baseline result (threshold=2, views=8)
    baseline_result = threshold_results.get("threshold_ev2", {})
    plot_summary["stage2_enabled"] = {
        "config": {
            "render_mode": config.get("render_mode"),
            "min_angle_evidence": 2,
            "n_views": 8,
            "skip_stage2": False,
            "top_k": config.get("top_k"),
        },
        "embedding_only": data.get("evaluation", {}).get("embedding_only", {}),
        "with_rescorer": {
            "precision": baseline_result.get("precision", 0),
            "recall": baseline_result.get("recall", 0),
            "f1": baseline_result.get("f1", 0),
        },
        "per_tier": baseline_result.get("per_tier", {}),
        "stats": stats,
    }

    # Baseline entry
    plot_summary["baseline"] = plot_summary["stage2_enabled"].copy()

    # Save
    output_path = RESCORER_DIR / "ablation_summary.json"
    with open(output_path, "w") as f:
        json.dump(plot_summary, f, indent=2)
    print(f"\nSaved ablation summary to: {output_path}")

    detail_path = RESCORER_DIR / "ablation_detailed.json"
    with open(detail_path, "w") as f:
        json.dump(ablation_summary, f, indent=2)
    print(f"Saved detailed results to: {detail_path}")


if __name__ == "__main__":
    main()
