#!/usr/bin/env python3
"""Generate article cover variants for review.

Usage:
    uv run python scripts/gen_covers.py

Output: output/covers/{slug}/variant_01/cover.jpg … variant_NN/cover.jpg
"""
from __future__ import annotations

import asyncio
import secrets
import time
from pathlib import Path

from mediakit.pipelines.article_cover import ArticleCoverPipeline

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_BASE = Path("output/covers")
N_VARIANTS = 1
BACKEND = "flux"   # "flux" = better quality | "sdxl" = faster (~8 GB VRAM)

# 1536×864 triggers ComfyUI TE VRAM offload — fixes sequential OOM on 24GB cards.
# Smart_crop → 1200×630 after generation.
GEN_WIDTH = 1536
GEN_HEIGHT = 864

MAX_RETRIES = 5
RETRY_DELAY_S = 30
# ─────────────────────────────────────────────────────────────────────────────

COVERS = [
    {
        "slug": "mountain-sunrise-landscape",
        "prompt": (
            "A photorealistic landscape photograph of a mountain range at sunrise. "
            "Golden light catches the snow-capped peaks, casting long shadows across an alpine meadow "
            "in the foreground. Clear blue sky, a few wispy clouds near the horizon. "
            "Shot with a 24mm wide-angle lens, natural color grading, no filters. "
            "Aspect ratio 1200x630 pixels, landscape orientation."
        ),
    },
    {
        "slug": "modern-city-street-night",
        "prompt": (
            "A photorealistic photograph of a modern city street at night. "
            "Neon reflections on wet pavement, a mix of pedestrians and light trails from passing cars. "
            "Glass-and-steel buildings in the background, warm streetlight glow in the foreground. "
            "Shot with a 35mm lens at f/2.8, natural bokeh, long-exposure style. "
            "Aspect ratio 1200x630 pixels, landscape orientation."
        ),
    },
    {
        "slug": "minimalist-product-studio",
        "prompt": (
            "A clean product photography setup: a single object on a pure white studio background "
            "with soft diffused lighting. Subtle shadow beneath the product, no harsh highlights. "
            "Minimal composition, professional commercial style. "
            "Aspect ratio 1200x630 pixels, landscape orientation."
        ),
    },
]


async def generate_one(
    pipeline: ArticleCoverPipeline,
    slug: str,
    prompt: str,
    variant_n: int,
    seed: int,
    total_done: list[int],
    total: int,
) -> None:
    out_dir = OUTPUT_BASE / slug / f"variant_{variant_n:02d}"
    cover_path = out_dir / "cover.jpg"

    if cover_path.exists():
        total_done[0] += 1
        print(f"  [{total_done[0]:2d}/{total}] {slug}/variant_{variant_n:02d}  SKIP (already exists)")
        return

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.monotonic()
            result = await pipeline.run(
                prompt=prompt,
                slug=slug,
                output_dir=out_dir,
                backend=BACKEND,
                seed=seed,
                gen_width=GEN_WIDTH,
                gen_height=GEN_HEIGHT,
            )
            elapsed = time.monotonic() - t0
            total_done[0] += 1
            print(
                f"  [{total_done[0]:2d}/{total}] {slug}/variant_{variant_n:02d}"
                f"  seed={seed}  {elapsed:.0f}s  → {result.outputs[0]}"
            )
            return
        except Exception as exc:
            if attempt < MAX_RETRIES:
                print(
                    f"  [!] {slug}/variant_{variant_n:02d} attempt {attempt}/{MAX_RETRIES} failed: "
                    f"{type(exc).__name__}: {exc}. Retry in {RETRY_DELAY_S}s..."
                )
                await asyncio.sleep(RETRY_DELAY_S)
            else:
                print(
                    f"  [✗] {slug}/variant_{variant_n:02d} failed after {MAX_RETRIES} attempts — skipping. "
                    f"({type(exc).__name__}: {exc})"
                )


async def main() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    pipeline = ArticleCoverPipeline()

    total = len(COVERS) * N_VARIANTS
    already_done = sum(
        1 for c in COVERS for n in range(1, N_VARIANTS + 1)
        if (OUTPUT_BASE / c["slug"] / f"variant_{n:02d}" / "cover.jpg").exists()
    )
    total_done = [already_done]

    print(f"Backend: {BACKEND}  |  {GEN_WIDTH}×{GEN_HEIGHT}")
    print(f"Covers: {len(COVERS)}  ×  variants: {N_VARIANTS}  =  {total} images")
    if already_done:
        print(f"Already done: {already_done} — skipping")
    print(f"Output: {OUTPUT_BASE.resolve()}\n")

    seeds_file = OUTPUT_BASE / "seeds.txt"
    if seeds_file.exists():
        all_seeds = [int(x) for x in seeds_file.read_text().splitlines()]
    else:
        all_seeds = [secrets.randbits(32) for _ in range(total)]
        seeds_file.write_text("\n".join(str(s) for s in all_seeds))

    t_start = time.monotonic()
    idx = 0

    for cover in COVERS:
        slug = cover["slug"]
        print(f"── {slug} ──")
        for i in range(1, N_VARIANTS + 1):
            seed = all_seeds[idx]
            idx += 1
            await generate_one(pipeline, slug, cover["prompt"], i, seed, total_done, total)

    elapsed_total = time.monotonic() - t_start
    print(f"\nDone in {elapsed_total / 60:.1f} min. Check: {OUTPUT_BASE.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
