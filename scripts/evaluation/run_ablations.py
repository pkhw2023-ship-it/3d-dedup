#!/usr/bin/env python3
"""
run_ablations.py — Ablation studies for the VLM rescorer.

Runs the rescorer with different configurations to measure the impact of:
1. Evidence threshold sweep (min_angle_evidence: 1–8)
2. View count (4, 6, 8 views)
3. Render mode (textured vs LFD)
4. Stage 2 skip (bypass single-view screening)

Each ablation run is saved separately, then combined for plotting.

Usage:
    python run_ablations.py                # full ablation suite
    python run_ablations.py --quick        # quick mode (10 queries)
"""

import json
import sys
import time
import argparse
import subprocess
from pathlib import Path

CODE_DIR = Path("/home/lightsail-user/3d-dataset-storage/tds-blog/code")
OUTPUT_DIR = Path("/home/lightsail-user/3d-dataset-storage/tds-blog/data/rescorer")
PYTHON = "/home/lightsail-user/3d-dataset-storage/miniconda3/envs/3d-dedup/bin/python3"

def run_rescorer(run_tag: str, extra_args: list, n_queries: int = 50) -> dict:
    """Run the generalized rescorer with specific arguments."""
    cmd = [
        PYTHON, str(CODE_DIR / "generalized_rescorer.py"),
        "--n-queries", str(n_queries),
        "--run-tag", run_tag,
        "--output-dir", str(OUTPUT_DIR),
    ] + extra_args

    print(f"\n{'='*70}")
    print(f"Running: {run_tag}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.0f}s")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        return None

    # Load results
    out_file = OUTPUT_DIR / f"pipeline_{run_tag}.json"
    if out_file.exists():
        with open(out_file) as f:
            data = json.load(f)
        eval_data = data.get("evaluation", {})
        stats = data.get("stats", {})
        print(f"  API calls: {stats.get('api_calls', '?')}")
        print(f"  Tokens: {stats.get('total_tokens', '?')}")
        emb = eval_data.get("embedding_only", {})
        res = eval_data.get("with_rescorer", {})
        print(f"  Embedding-only: P={emb.get('precision',0):.4f}, R={emb.get('recall',0):.4f}")
        print(f"  With rescorer:  P={res.get('precision',0):.4f}, R={res.get('recall',0):.4f}")
        return data
    else:
        print(f"  ERROR: output file not found: {out_file}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10 queries instead of 50")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the baseline run")
    args = parser.parse_args()

    n_queries = 10 if args.quick else 50
    all_results = {}

    # ─── Baseline run ────────────────────────────────────────────
    if not args.skip_baseline:
        base = run_rescorer("baseline_textured_top20_ev2_v8", [
            "--render-mode", "textured",
            "--top-k", "20",
            "--min-evidence", "2",
            "--n-views", "8",
        ], n_queries)
        if base:
            all_results["baseline"] = base

    # ─── Ablation 1: Evidence threshold sweep (1–8) ──────────────
    print("\n" + "#" * 70)
    print("ABLATION 1: Evidence threshold sweep")
    print("#" * 70)
    for threshold in [1, 2, 3, 4, 5, 6, 7, 8]:
        tag = f"threshold_ev{threshold}"
        r = run_rescorer(tag, [
            "--render-mode", "textured",
            "--top-k", "20",
            "--min-evidence", str(threshold),
            "--n-views", "8",
        ], n_queries)
        if r:
            all_results[tag] = r

    # ─── Ablation 2: View count ──────────────────────────────────
    print("\n" + "#" * 70)
    print("ABLATION 2: View count")
    print("#" * 70)
    for n_views in [4, 6, 8]:
        tag = f"views_{n_views}"
        r = run_rescorer(tag, [
            "--render-mode", "textured",
            "--top-k", "20",
            "--min-evidence", "2",
            "--n-views", str(n_views),
        ], n_queries)
        if r:
            all_results[tag] = r

    # ─── Ablation 3: Render mode ─────────────────────────────────
    print("\n" + "#" * 70)
    print("ABLATION 3: Render mode")
    print("#" * 70)
    for mode in ["textured", "lfd"]:
        tag = f"render_{mode}"
        r = run_rescorer(tag, [
            "--render-mode", mode,
            "--top-k", "20",
            "--min-evidence", "2",
            "--n-views", "8",
        ], n_queries)
        if r:
            all_results[tag] = r

    # ─── Ablation 4: Stage 2 skip ────────────────────────────────
    print("\n" + "#" * 70)
    print("ABLATION 4: Stage 2 skip")
    print("#" * 70)
    # With Stage 2
    tag_with = "stage2_enabled"
    r_with = run_rescorer(tag_with, [
        "--render-mode", "textured",
        "--top-k", "20",
        "--min-evidence", "2",
        "--n-views", "8",
    ], n_queries)
    if r_with:
        all_results[tag_with] = r_with

    # Without Stage 2
    tag_without = "stage2_disabled"
    r_without = run_rescorer(tag_without, [
        "--render-mode", "textured",
        "--top-k", "20",
        "--min-evidence", "2",
        "--n-views", "8",
        "--skip-stage2",
    ], n_queries)
    if r_without:
        all_results[tag_without] = r_without

    # ─── Save combined results ───────────────────────────────────
    # Extract summary for each run
    summary = {}
    for tag, data in all_results.items():
        evaluation = data.get("evaluation", {})
        stats = data.get("stats", {})
        config = data.get("config", {})
        summary[tag] = {
            "config": {
                "render_mode": config.get("render_mode"),
                "min_angle_evidence": config.get("min_angle_evidence"),
                "n_views": config.get("n_views"),
                "skip_stage2": config.get("skip_stage2"),
                "top_k": config.get("top_k"),
            },
            "embedding_only": evaluation.get("embedding_only", {}),
            "with_rescorer": evaluation.get("with_rescorer", {}),
            "per_tier": evaluation.get("per_tier", {}),
            "stats": {
                "api_calls": stats.get("api_calls"),
                "total_tokens": stats.get("total_tokens"),
                "stage2_passed": stats.get("stage2_passed"),
                "stage2_rejected": stats.get("stage2_rejected"),
                "stage3_confirmed": stats.get("stage3_confirmed"),
                "stage3_evaluated": stats.get("stage3_evaluated"),
            },
            "runtime_seconds": data.get("runtime_seconds"),
        }

    summary_path = OUTPUT_DIR / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Ablation summary saved to: {summary_path}")
    print(f"Total runs: {len(summary)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
