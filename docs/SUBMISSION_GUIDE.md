# TDS Submission Guide — "Detecting 3D Near-Duplicates at Scale"

## Pre-Submission Checklist

- [x] All 5 posts reviewed and finalized
- [x] YAML frontmatter added to all posts
- [x] Images optimized and under TDS size limits (all ≤2000px wide)
- [x] Code archive created with README and requirements.txt
- [x] No sensitive information (API keys, internal URLs, credentials)
- [x] Heading hierarchy verified (H1 title → H2 sections → H3 subsections)
- [x] No unnecessary HTML tags in Markdown

## Post Summary

| File | Title | Words | Read Time |
|------|-------|-------|-----------|
| `post-1.md` | Detecting 3D Near-Duplicates at Scale: A Multi-View Embedding Pipeline | 3,184 | 14 min |
| `post-2.md` | Multi-View Rendering for 3D Understanding: Camera Strategies That Actually Work | 4,536 | 20 min |
| `post-3.md` | Which Vision Model Sees 3D Shapes Best? A DINOv2, CLIP, and Geometry Bake-Off | 4,416 | 19 min |
| `post-4a.md` | 3D-DupBench: A Five-Tier Benchmark for Near-Duplicate 3D Model Detection | 4,460 | 19 min |
| `post-4b.md` | From Embeddings to Production: VLM Verification and Billion-Scale Search | 5,214 | 23 min |
| **Total** | | **21,810** | **~95 min** |

## Submission Process

### Step 1: Create a TDS Contributor Account

1. Go to https://contributor.insightmediagroup.io
2. Sign up / log in
3. Complete your author profile (use your real name, LinkedIn, and a professional headshot)
4. In your bio, mention: Principal ML Engineer at Roblox, specializing in 3D Computer Vision and Generative AI

### Step 2: Submit Post 1 First

1. Click "Write a Post" in the contributor portal
2. Use the WordPress block editor to paste content
3. **Strip the YAML frontmatter** — do not paste the `---` block into WordPress; use it as reference for the metadata fields in the submission form
4. For images: upload each image from `submission/images/` and insert inline at the correct position
5. For code blocks: use the "Code" block type (Prismatic formatting is built in)
6. Add tags: **3D Computer Vision**, **Machine Learning**, **Near-Duplicate Detection**
7. Click "Submit for Review"
8. Wait for response (~1 week)

### Step 3: Flag This as a Series

When submitting Post 1, include in the submission notes:

> This is Part 1 of a 5-part series titled "Detecting 3D Near-Duplicates at Scale."
> The series covers building a complete 3D near-duplicate detection pipeline:
> multi-view rendering, vision model comparison, benchmark construction, VLM
> verification, and billion-scale FAISS search. I plan to submit the remaining
> parts after this is reviewed.

TDS calls multi-part series "Online Books" (12–25 min per chapter, 5+ articles). This series fits that format perfectly.

### Step 4: After Post 1 Acceptance

- Submit Posts 2 and 3 (max 3 pending at a time)
- After acceptance of 2–3, submit 4a and 4b
- When submitting each subsequent post, reference the series and link to already-published parts

### Step 5: Cross-Link Published Posts

After each post is published:
- Go back to previously published posts and update the cross-reference links
- Replace the `../post-X/post.md` paths with actual TDS URLs
- The posts contain cross-references in the "What's Next" and "Series Recap" sections

## Important TDS Policies

### Content Requirements
- **AI writing:** Tools used "judiciously" are OK. Author's voice must drive the content. Don't mention AI assistance unless asked.
- **Images:** All must be original (yours are — they're generated from your data). Add alt text and captions.
- **Code:** Use Prismatic code blocks (built into their editor). Keep inline code examples short.
- **Length:** 12–25 min per post (3,000–6,500 words) for "Online Book" format. All posts are within range.
- **Simultaneous submissions:** Max 3 at a time.
- **Response time:** ~1 week for editorial review.

### What Editors Look For
- Clear, jargon-free explanations (you're writing for data scientists, not only 3D experts)
- Original research or analysis (the 3D-DupBench benchmark and ModelNet40 audit are strong hooks)
- Practical takeaways and reproducible code
- Good visual storytelling (figures that explain, not just decorate)

### WordPress Block Editor Tips
- Paste Markdown content — it will auto-convert most formatting
- For inline code, use backticks (they convert properly)
- For images, use the "Image" block and upload from the `images/` directory
- For math/equations, use the "LaTeX" block if available (or describe in text)
- Preview before submitting — check that code blocks render with syntax highlighting
- Tables in Markdown paste well but may need manual adjustment

## Payment Program

- TDS pays based on 30-day engaged views after publication
- Tiers: $100 (500+ views) → $7,500 (25,000+ views)
- First-time authors typically see 500–5,000 views on technical posts
- Series tend to accumulate views across all parts (cross-linking helps significantly)
- The ModelNet40 audit finding is likely to generate strong interest in the 3D/ML community

## Image Naming Convention

All images in `submission/images/` follow the pattern:
```
{post-id}_{original-filename}.png
```

For example:
- `post-1_pipeline_overview.png` — Pipeline diagram for Post 1
- `post-2_camera_positions.png` — Camera configuration visualization for Post 2
- `post-4a_modelnet40_duplicates.png` — ModelNet40 duplicate pairs for Post 4a

When uploading to WordPress, you can rename to shorter names if desired.

## Code Repository

The `submission/code/` directory contains all pipeline scripts. Consider:
- Hosting on GitHub as a public repository (e.g., `harish-wajjala/3d-dupbench`)
- Adding a link to the GitHub repo in each post's "Code & Data" section
- Including a Colab notebook for the simpler demonstrations (embedding extraction, visualization)

## File Manifest

```
submission/
├── SUBMISSION_GUIDE.md          ← This file
├── posts/
│   ├── post-1.md                ← Part 1: Pipeline Overview (14 min)
│   ├── post-2.md                ← Part 2: Multi-View Rendering (20 min)
│   ├── post-3.md                ← Part 3: Embedding Bake-Off (19 min)
│   ├── post-4a.md               ← Part 4: 3D-DupBench + ModelNet40 (19 min)
│   └── post-4b.md               ← Part 5: VLM + FAISS Production (23 min)
├── images/
│   ├── post-1_*.png             ← 8 images
│   ├── post-2_*.png             ← 14 images
│   ├── post-3_*.png             ← 9 images
│   ├── post-4a_*.png            ← 15 images
│   └── post-4b_*.png            ← 27 images
│                                  (73 images total, ~14 MB)
└── code/
    ├── README.md                ← Code documentation
    ├── requirements.txt         ← Python dependencies
    ├── download_lvis.py         ← Download Objaverse-LVIS
    ├── convert_to_obj.py        ← Format conversion
    ├── generate_clones_5tier.py ← 5-tier clone generation
    ├── validate_clones.py       ← Clone quality validation
    ├── render_multiview.py      ← Multi-view rendering (nvdiffrast)
    ├── compute_embeddings.py    ← DINOv2/CLIP embedding extraction
    ├── compute_geometry_baselines.py ← Geometry baselines
    ├── evaluate_bakeoff.py      ← Retrieval evaluation
    ├── generalized_rescorer.py  ← VLM verification (Gemini)
    ├── run_ablations.py         ← Ablation studies
    ├── offline_ablations.py     ← Offline ablation analysis
    ├── generate_plots.py        ← Figure generation (Posts 1–4a)
    └── generate_rescorer_plots.py ← Figure generation (Post 4b)
```
