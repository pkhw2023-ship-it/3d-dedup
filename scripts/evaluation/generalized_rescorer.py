#!/usr/bin/env python3
"""
Generalized Multi-View VLM Rescorer for 3D Near-Duplicate Detection

3-stage pipeline:
  Stage 1 — Embedding retrieval (cosine similarity from pre-computed embeddings)
  Stage 2 — Single-view VLM screening (quick reject via front view)
  Stage 3 — Multi-view rescoring (8-angle evidence accumulation, high-precision)

Uses a local ML Gateway proxy at http://127.0.0.1:8080 (OpenAI-compatible API)
to access Gemini models. The proxy handles auth — no API key needed.

Adapted from Roblox avatar IP enforcement rescorer for general 3D models.

Usage:
    python generalized_rescorer.py                    # default run
    python generalized_rescorer.py --n-queries 100    # more queries
    python generalized_rescorer.py --model google/gemini-2.5-flash  # different model
    python generalized_rescorer.py --skip-stage2      # ablation: skip stage 2
"""

import json
import base64
import io
import time
import random
import logging
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict

# ─── Constants ──────────────────────────────────────────────────────
# 8 horizontal ring cameras at 45° intervals (views 0–7 in our render set)
RING_VIEW_INDICES = list(range(8))  # view_00 through view_07
RING_ANGLES = [i * 45 for i in range(8)]  # 0°, 45°, ..., 315°

# Proxy configuration
PROXY_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "google/gemini-2.5-flash"  # fast + cheap; pro for quality

# Paths
DATA_DIR = Path("/home/lightsail-user/3d-dataset-storage/tds-blog/data")
RENDER_DIR = DATA_DIR / "renders"
EMB_DIR = DATA_DIR / "embeddings"
OUTPUT_DIR = DATA_DIR / "rescorer"
CLONE_MANIFEST = DATA_DIR / "clones-objaverse" / "clone_manifest_5tier.json"
SOURCE_MANIFEST = DATA_DIR / "clones-objaverse" / "source_manifest.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rescorer")


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RescorerConfig:
    """Configuration for the 3-stage rescoring pipeline."""
    model_name: str = DEFAULT_MODEL
    proxy_base_url: str = PROXY_BASE_URL
    render_dir: str = str(RENDER_DIR)
    render_mode: str = "textured"        # "textured" or "lfd"
    top_k: int = 20                      # candidates from Stage 1
    min_angle_evidence: int = 2          # Stage 3 threshold (high-precision)
    n_views: int = 8                     # views to use in Stage 3
    num_threads: int = 8                 # parallel API calls
    max_tokens: int = 300                # per-call token limit
    skip_stage2: bool = False            # ablation: bypass single-view screening
    composite_size: int = 224            # per-image size in composite

    # Embedding configuration
    embedding_file: str = "dinov2_giant_lfd_perview.npz"
    aggregation: str = "concat_pca"
    agg_n_views: int = 28


# ═══════════════════════════════════════════════════════════════════════
# Image Utilities
# ═══════════════════════════════════════════════════════════════════════

def image_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64 data URL for the OpenAI vision API."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "png" if fmt.upper() == "PNG" else "jpeg"
    return f"data:image/{mime};base64,{b64}"


def make_side_by_side(img_a: Image.Image, img_b: Image.Image,
                      size: int = 224) -> Image.Image:
    """Create a side-by-side composite of two images."""
    composite = Image.new("RGB", (size * 2, size))
    composite.paste(img_a.resize((size, size)), (0, 0))
    composite.paste(img_b.resize((size, size)), (size, 0))
    return composite


# ═══════════════════════════════════════════════════════════════════════
# Ground Truth
# ═══════════════════════════════════════════════════════════════════════

def load_ground_truth() -> Tuple[Dict, Dict, Set, List]:
    """Load clone manifest and build ground truth mappings.

    Returns:
        clone_to_source: clone_id -> source_uid
        clone_to_tier: clone_id -> tier string (T1..T5)
        source_uids: set of source model UIDs
        all_model_ids: ordered list (from embeddings)
    """
    with open(CLONE_MANIFEST) as f:
        clones = json.load(f)
    with open(SOURCE_MANIFEST) as f:
        sources = json.load(f)

    clone_to_source = {}
    clone_to_tier = {}
    source_uids = set(sources["uids"])

    for c in clones:
        clone_to_source[c["clone_id"]] = c["source_uid"]
        clone_to_tier[c["clone_id"]] = c["tier"]

    return clone_to_source, clone_to_tier, source_uids


def get_source_id(model_id: str, clone_to_source: Dict, source_uids: Set) -> str:
    """Return the source UID for any model (itself if it's a source)."""
    if model_id in source_uids:
        return model_id
    return clone_to_source.get(model_id, model_id)


# ═══════════════════════════════════════════════════════════════════════
# Embedding Retrieval (Stage 1)
# ═══════════════════════════════════════════════════════════════════════

def load_and_aggregate_embeddings(config: RescorerConfig):
    """Load per-view embeddings and aggregate them.

    Returns:
        agg_embeddings: (N, D) L2-normalized
        model_ids: list of N model ID strings
    """
    emb_path = EMB_DIR / config.embedding_file
    log.info(f"Loading embeddings from {emb_path.name}")
    data = np.load(emb_path, allow_pickle=True)
    model_ids = list(data["model_ids"])
    per_view = data["embeddings"]  # (N, 28, dim)

    N, V, D = per_view.shape
    log.info(f"  Shape: {N} models × {V} views × {D} dim")

    # Aggregate
    strategy = config.aggregation
    n_views = config.agg_n_views
    if n_views is not None:
        per_view = per_view[:, :n_views, :]

    if strategy == "single":
        agg = per_view[:, 0, :]
    elif strategy == "mean":
        agg = per_view.mean(axis=1)
    elif strategy == "max":
        agg = per_view.max(axis=1)
    elif strategy == "concat_pca":
        from sklearn.decomposition import PCA
        flat = per_view.reshape(N, -1)
        target_dim = min(768, flat.shape[1], N - 1)
        pca = PCA(n_components=target_dim)
        agg = pca.fit_transform(flat)
        log.info(f"  PCA: {flat.shape[1]} -> {target_dim} (explained var: "
                 f"{pca.explained_variance_ratio_.sum():.4f})")
    else:
        raise ValueError(f"Unknown aggregation: {strategy}")

    # L2 normalize
    norms = np.linalg.norm(agg, axis=1, keepdims=True)
    agg = agg / np.maximum(norms, 1e-8)

    log.info(f"  Aggregated shape: {agg.shape}")
    return agg, model_ids


def retrieve_candidates(query_idx: int, embeddings: np.ndarray,
                        top_k: int = 20) -> List[Tuple[int, float]]:
    """Stage 1: cosine similarity retrieval.

    Returns: list of (candidate_idx, similarity_score)
    """
    query_emb = embeddings[query_idx]
    sims = embeddings @ query_emb  # already L2-normalized
    sims[query_idx] = -1  # exclude self

    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [(int(idx), float(sims[idx])) for idx in top_indices]


# ═══════════════════════════════════════════════════════════════════════
# VLM Rescorer (Stages 2 & 3)
# ═══════════════════════════════════════════════════════════════════════

class GeneralizedRescorer:
    """Multi-view VLM rescorer using Gemini via local proxy."""

    # ── Prompt Templates ──────────────────────────────────────────
    STAGE2_PROMPT = """Compare these two 3D models rendered from the same viewpoint.
Left is Model A, right is Model B.

Are these likely the same 3D model (possibly with different scale, rotation,
color changes, or minor geometric modifications)? Or are they clearly
different objects?

Respond with ONLY one of:
- LIKELY_DUPLICATE: The models appear to be the same or very similar
- DIFFERENT: The models are clearly different objects
- UNCERTAIN: Cannot determine from this single view"""

    STAGE3_PROMPT_TEMPLATE = """These are two 3D models rendered from {angle}° azimuth.
Left is Model A, right is Model B.

Considering shape, proportions, and structural features visible at this angle:
Is there evidence that these are the same 3D model?

Respond with ONLY one of:
- EVIDENCE: Clear similarity evidence at this angle
- NO_EVIDENCE: No compelling similarity at this angle"""

    def __init__(self, config: RescorerConfig):
        self.config = config
        self.render_base = Path(config.render_dir) / config.render_mode
        self.stats = {
            "stage1_pairs": 0,
            "stage2_screened": 0,
            "stage2_passed": 0,
            "stage2_rejected": 0,
            "stage3_evaluated": 0,
            "stage3_confirmed": 0,
            "api_calls": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "errors": 0,
        }

        from openai import OpenAI
        self.client = OpenAI(
            base_url=config.proxy_base_url,
            api_key="not-needed",
        )
        log.info(f"Rescorer initialized: model={config.model_name}, "
                 f"render_mode={config.render_mode}, "
                 f"top_k={config.top_k}, threshold={config.min_angle_evidence}")

    def _vlm_call(self, prompt: str, image: Image.Image,
                  retries: int = 3) -> Optional[str]:
        """Make a vision-language model call via the proxy."""
        data_url = image_to_data_url(image)

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url",
                             "image_url": {"url": data_url}},
                        ],
                    }],
                    max_tokens=self.config.max_tokens,
                )

                self.stats["api_calls"] += 1
                if response.usage:
                    self.stats["total_tokens"] += response.usage.total_tokens
                    self.stats["prompt_tokens"] += response.usage.prompt_tokens
                    self.stats["completion_tokens"] += response.usage.completion_tokens

                content = response.choices[0].message.content
                if content:
                    return content.strip()
                else:
                    log.warning(f"Empty response on attempt {attempt+1}")

            except Exception as e:
                self.stats["errors"] += 1
                log.warning(f"VLM call failed (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

        return None

    def _load_view(self, model_id: str, view_idx: int) -> Optional[Image.Image]:
        """Load a specific rendered view for a model."""
        path = self.render_base / model_id / f"view_{view_idx:02d}.png"
        if not path.exists():
            return None
        return Image.open(path).convert("RGB")

    def stage2_screen(self, model_a_id: str,
                      model_b_id: str) -> str:
        """Stage 2: Single-view VLM screening using front view (view_00).

        Returns: "LIKELY_DUPLICATE", "DIFFERENT", or "UNCERTAIN"
        """
        self.stats["stage2_screened"] += 1
        img_a = self._load_view(model_a_id, 0)
        img_b = self._load_view(model_b_id, 0)

        if img_a is None or img_b is None:
            log.warning(f"Missing renders for {model_a_id} or {model_b_id}")
            return "UNCERTAIN"

        composite = make_side_by_side(img_a, img_b, self.config.composite_size)
        result = self._vlm_call(self.STAGE2_PROMPT, composite)

        if result is None:
            return "UNCERTAIN"

        result_upper = result.upper()
        if "LIKELY_DUPLICATE" in result_upper:
            self.stats["stage2_passed"] += 1
            return "LIKELY_DUPLICATE"
        elif "DIFFERENT" in result_upper:
            self.stats["stage2_rejected"] += 1
            return "DIFFERENT"
        else:
            self.stats["stage2_passed"] += 1  # uncertain -> pass to stage 3
            return "UNCERTAIN"

    def stage3_multiview_rescore(self, model_a_id: str,
                                 model_b_id: str) -> Dict:
        """Stage 3: Multi-view evidence accumulation.

        Examines n_views angles. A pair is confirmed as duplicate
        only if >= min_angle_evidence angles show similarity.

        Returns dict with is_duplicate, evidence_count, angle_results.
        """
        self.stats["stage3_evaluated"] += 1

        n_views = min(self.config.n_views, len(RING_VIEW_INDICES))
        view_indices = RING_VIEW_INDICES[:n_views]
        angles_to_use = RING_ANGLES[:n_views]

        evidence_count = 0
        angle_results = []

        for view_idx, angle in zip(view_indices, angles_to_use):
            img_a = self._load_view(model_a_id, view_idx)
            img_b = self._load_view(model_b_id, view_idx)

            if img_a is None or img_b is None:
                angle_results.append({
                    "angle": angle, "view_idx": view_idx,
                    "result": "SKIPPED", "evidence": False
                })
                continue

            composite = make_side_by_side(img_a, img_b, self.config.composite_size)
            prompt = self.STAGE3_PROMPT_TEMPLATE.format(angle=angle)
            result = self._vlm_call(prompt, composite)

            if result is None:
                angle_results.append({
                    "angle": angle, "view_idx": view_idx,
                    "result": "ERROR", "evidence": False
                })
                continue

            result_upper = result.upper()
            # "EVIDENCE" but not "NO_EVIDENCE"
            has_evidence = "EVIDENCE" in result_upper and "NO_EVIDENCE" not in result_upper
            if has_evidence:
                evidence_count += 1

            angle_results.append({
                "angle": angle, "view_idx": view_idx,
                "result": result, "evidence": has_evidence,
            })

        is_duplicate = evidence_count >= self.config.min_angle_evidence
        if is_duplicate:
            self.stats["stage3_confirmed"] += 1

        return {
            "is_duplicate": is_duplicate,
            "evidence_count": evidence_count,
            "total_views": n_views,
            "threshold": self.config.min_angle_evidence,
            "angle_results": angle_results,
        }

    def process_query(self, query_idx: int, query_id: str,
                      candidates: List[Tuple[int, float]],
                      model_ids: List[str]) -> Dict:
        """Process a single query through Stages 2 & 3.

        Args:
            query_idx: index in embedding matrix
            query_id: model ID string
            candidates: list of (cand_idx, sim_score) from Stage 1
            model_ids: full model ID list

        Returns: result dict with confirmed_duplicates and rejected.
        """
        result = {
            "query": query_id,
            "confirmed_duplicates": [],
            "rejected": [],
        }

        for cand_idx, sim_score in candidates:
            cand_id = model_ids[cand_idx]
            entry = {"candidate": cand_id, "sim_score": round(sim_score, 6)}

            # Stage 2 — single-view screening
            if not self.config.skip_stage2:
                screen = self.stage2_screen(query_id, cand_id)
                if screen == "DIFFERENT":
                    entry["rejected_at"] = "stage2"
                    entry["stage2_result"] = screen
                    result["rejected"].append(entry)
                    continue
                entry["stage2_result"] = screen
            else:
                entry["stage2_result"] = "SKIPPED"

            # Stage 3 — multi-view rescoring
            rescore = self.stage3_multiview_rescore(query_id, cand_id)
            entry["evidence_count"] = rescore["evidence_count"]
            entry["total_views"] = rescore["total_views"]
            entry["angle_details"] = rescore["angle_results"]

            if rescore["is_duplicate"]:
                result["confirmed_duplicates"].append(entry)
            else:
                entry["rejected_at"] = "stage3"
                result["rejected"].append(entry)

        return result

    def run_pipeline(self, query_indices: List[int],
                     embeddings: np.ndarray,
                     model_ids: List[str]) -> List[Dict]:
        """Run the full 3-stage pipeline for a list of query models.

        Args:
            query_indices: indices into the embedding/model_ids arrays
            embeddings: (N, D) L2-normalized embeddings
            model_ids: list of model ID strings

        Returns: list of per-query result dicts.
        """
        all_results = []
        total = len(query_indices)

        for i, q_idx in enumerate(query_indices):
            q_id = model_ids[q_idx]
            log.info(f"[{i+1}/{total}] Query: {q_id}")

            # Stage 1 — embedding retrieval
            candidates = retrieve_candidates(q_idx, embeddings, self.config.top_k)
            self.stats["stage1_pairs"] += len(candidates)

            # Stages 2 & 3
            result = self.process_query(q_idx, q_id, candidates, model_ids)
            n_dup = len(result["confirmed_duplicates"])
            n_rej = len(result["rejected"])
            log.info(f"  → {n_dup} confirmed, {n_rej} rejected")

            all_results.append(result)

        return all_results


# ═══════════════════════════════════════════════════════════════════════
# Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════

def evaluate_rescorer(results: List[Dict],
                      clone_to_source: Dict,
                      clone_to_tier: Dict,
                      source_uids: Set,
                      model_ids: List[str],
                      embeddings: np.ndarray,
                      top_k: int = 20) -> Dict:
    """Compute precision/recall with and without VLM rescoring.

    Compares:
    1. Embedding-only top-K retrieval
    2. Embedding + VLM rescorer (confirmed duplicates)

    Uses clone manifest as ground truth.
    """
    # Build ground truth: for each source, which IDs are duplicates
    source_to_clones = defaultdict(set)
    for clone_id, src in clone_to_source.items():
        source_to_clones[src].add(clone_id)
    # Also: source is "duplicate" of its own clones (and vice versa)
    for src in source_uids:
        source_to_clones[src].add(src)

    def get_relevant(model_id):
        """Get the set of model IDs that are duplicates of model_id."""
        src = get_source_id(model_id, clone_to_source, source_uids)
        return source_to_clones.get(src, set()) - {model_id}

    # ── Embedding-only metrics ──
    emb_tp = 0
    emb_fp = 0
    emb_total_relevant = 0
    emb_total_retrieved = 0

    # ── Rescorer metrics ──
    res_tp = 0
    res_fp = 0
    res_total_relevant = 0

    # ── Per-tier ──
    tier_metrics = {t: {"emb_tp": 0, "emb_fp": 0, "emb_relevant": 0,
                        "res_tp": 0, "res_fp": 0, "res_relevant": 0}
                    for t in ["T1", "T2", "T3", "T4", "T5"]}

    id_to_idx = {mid: i for i, mid in enumerate(model_ids)}

    for r in results:
        query_id = r["query"]
        relevant = get_relevant(query_id)
        if not relevant:
            continue

        emb_total_relevant += len(relevant)
        res_total_relevant += len(relevant)

        # Embedding-only: use top-K from embedding retrieval
        q_idx = id_to_idx[query_id]
        candidates = retrieve_candidates(q_idx, embeddings, top_k)
        for c_idx, _ in candidates:
            c_id = model_ids[c_idx]
            emb_total_retrieved += 1
            if c_id in relevant:
                emb_tp += 1
                # Per-tier
                tier = clone_to_tier.get(c_id)
                if tier and tier in tier_metrics:
                    tier_metrics[tier]["emb_tp"] += 1
            else:
                emb_fp += 1
                tier = clone_to_tier.get(c_id)
                if tier and tier in tier_metrics:
                    tier_metrics[tier]["emb_fp"] += 1

        # Count relevant per tier for recall
        for rel_id in relevant:
            tier = clone_to_tier.get(rel_id)
            if tier and tier in tier_metrics:
                tier_metrics[tier]["emb_relevant"] += 1
                tier_metrics[tier]["res_relevant"] += 1

        # Rescorer: confirmed duplicates
        for dup in r["confirmed_duplicates"]:
            c_id = dup["candidate"]
            if c_id in relevant:
                res_tp += 1
                tier = clone_to_tier.get(c_id)
                if tier and tier in tier_metrics:
                    tier_metrics[tier]["res_tp"] += 1
            else:
                res_fp += 1
                tier = clone_to_tier.get(c_id)
                if tier and tier in tier_metrics:
                    tier_metrics[tier]["res_fp"] += 1

    # Compute metrics
    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    eval_result = {
        "n_queries": len(results),
        "embedding_only": {
            "true_positives": emb_tp,
            "false_positives": emb_fp,
            "total_relevant": emb_total_relevant,
            "precision": safe_div(emb_tp, emb_tp + emb_fp),
            "recall": safe_div(emb_tp, emb_total_relevant),
            "f1": safe_div(2 * emb_tp, 2 * emb_tp + emb_fp + (emb_total_relevant - emb_tp)),
        },
        "with_rescorer": {
            "true_positives": res_tp,
            "false_positives": res_fp,
            "total_relevant": res_total_relevant,
            "precision": safe_div(res_tp, res_tp + res_fp),
            "recall": safe_div(res_tp, res_total_relevant),
            "f1": safe_div(2 * res_tp, 2 * res_tp + res_fp + (res_total_relevant - res_tp)),
        },
        "per_tier": {},
    }

    for tier, tm in tier_metrics.items():
        eval_result["per_tier"][tier] = {
            "embedding_precision": safe_div(tm["emb_tp"], tm["emb_tp"] + tm["emb_fp"]),
            "embedding_recall": safe_div(tm["emb_tp"], tm["emb_relevant"]),
            "rescorer_precision": safe_div(tm["res_tp"], tm["res_tp"] + tm["res_fp"]),
            "rescorer_recall": safe_div(tm["res_tp"], tm["res_relevant"]),
        }

    return eval_result


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Generalized Multi-View VLM Rescorer for 3D Near-Duplicate Detection"
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                    help="VLM model name via proxy (default: %(default)s)")
    p.add_argument("--render-mode", choices=["textured", "lfd"], default="textured",
                    help="Which renders to use for VLM input")
    p.add_argument("--embedding-file", default="dinov2_giant_lfd_perview.npz",
                    help="Embedding file from Task 03")
    p.add_argument("--aggregation", default="concat_pca",
                    choices=["single", "mean", "max", "concat_pca"])
    p.add_argument("--agg-n-views", type=int, default=28)
    p.add_argument("--top-k", type=int, default=20,
                    help="Stage 1: number of candidates per query")
    p.add_argument("--min-evidence", type=int, default=2,
                    help="Stage 3: minimum angles with evidence")
    p.add_argument("--n-views", type=int, default=8,
                    help="Stage 3: number of views to evaluate")
    p.add_argument("--n-queries", type=int, default=50,
                    help="Number of source models to query")
    p.add_argument("--skip-stage2", action="store_true",
                    help="Ablation: skip Stage 2 screening")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--run-tag", default=None,
                    help="Tag for output files (auto-generated if not set)")
    return p.parse_args()


def main():
    args = parse_args()

    # Build config
    config = RescorerConfig(
        model_name=args.model,
        render_dir=str(RENDER_DIR),
        render_mode=args.render_mode,
        embedding_file=args.embedding_file,
        aggregation=args.aggregation,
        agg_n_views=args.agg_n_views,
        top_k=args.top_k,
        min_angle_evidence=args.min_evidence,
        n_views=args.n_views,
        skip_stage2=args.skip_stage2,
    )

    # Generate run tag
    if args.run_tag:
        run_tag = args.run_tag
    else:
        parts = [config.render_mode, f"top{config.top_k}",
                 f"ev{config.min_angle_evidence}", f"v{config.n_views}"]
        if config.skip_stage2:
            parts.append("noS2")
        run_tag = "_".join(parts)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("GENERALIZED MULTI-VIEW VLM RESCORER")
    log.info("=" * 70)
    log.info(f"Run tag: {run_tag}")
    log.info(f"Config: {json.dumps(asdict(config), indent=2)}")

    # ── Load ground truth ──
    clone_to_source, clone_to_tier, source_uids = load_ground_truth()
    log.info(f"Ground truth: {len(source_uids)} sources, "
             f"{len(clone_to_source)} clones")

    # ── Load & aggregate embeddings ──
    embeddings, model_ids = load_and_aggregate_embeddings(config)

    # ── Select query models ──
    # Query only source models (not clones) for clean evaluation
    random.seed(args.seed)
    source_indices = [i for i, mid in enumerate(model_ids) if mid in source_uids]
    n_queries = min(args.n_queries, len(source_indices))
    query_indices = sorted(random.sample(source_indices, n_queries))
    log.info(f"Selected {n_queries} source models as queries")

    # ── Run pipeline ──
    t0 = time.time()
    rescorer = GeneralizedRescorer(config)
    results = rescorer.run_pipeline(query_indices, embeddings, model_ids)
    elapsed = time.time() - t0

    # ── Evaluate ──
    log.info("Evaluating rescorer impact...")
    evaluation = evaluate_rescorer(
        results, clone_to_source, clone_to_tier, source_uids,
        model_ids, embeddings, config.top_k,
    )

    # ── Summary ──
    stats = rescorer.stats
    log.info("=" * 70)
    log.info("RESULTS SUMMARY")
    log.info("=" * 70)
    log.info(f"Pipeline runtime: {elapsed:.1f}s")
    log.info(f"API calls: {stats['api_calls']}")
    log.info(f"Total tokens: {stats['total_tokens']:,}")
    log.info(f"  Prompt: {stats['prompt_tokens']:,}")
    log.info(f"  Completion: {stats['completion_tokens']:,}")
    log.info(f"Errors: {stats['errors']}")
    log.info("")
    log.info("Stage funnel:")
    log.info(f"  Stage 1 pairs: {stats['stage1_pairs']}")
    log.info(f"  Stage 2 screened: {stats['stage2_screened']}")
    log.info(f"  Stage 2 passed: {stats['stage2_passed']}")
    log.info(f"  Stage 2 rejected: {stats['stage2_rejected']}")
    log.info(f"  Stage 3 evaluated: {stats['stage3_evaluated']}")
    log.info(f"  Stage 3 confirmed: {stats['stage3_confirmed']}")
    log.info("")
    log.info("Embedding-only:")
    e = evaluation["embedding_only"]
    log.info(f"  Precision: {e['precision']:.4f}")
    log.info(f"  Recall:    {e['recall']:.4f}")
    log.info(f"  F1:        {e['f1']:.4f}")
    log.info("")
    log.info("With VLM rescorer:")
    r = evaluation["with_rescorer"]
    log.info(f"  Precision: {r['precision']:.4f}")
    log.info(f"  Recall:    {r['recall']:.4f}")
    log.info(f"  F1:        {r['f1']:.4f}")
    log.info("")
    log.info("Per-tier rescorer precision/recall:")
    for tier in ["T1", "T2", "T3", "T4", "T5"]:
        t = evaluation["per_tier"].get(tier, {})
        log.info(f"  {tier}: P={t.get('rescorer_precision',0):.4f}  "
                 f"R={t.get('rescorer_recall',0):.4f}")

    # ── Save outputs ──
    output = {
        "run_tag": run_tag,
        "config": asdict(config),
        "stats": stats,
        "evaluation": evaluation,
        "runtime_seconds": round(elapsed, 1),
        "pipeline_results": results,
    }

    out_path = output_dir / f"pipeline_{run_tag}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to: {out_path}")

    return output


if __name__ == "__main__":
    main()
