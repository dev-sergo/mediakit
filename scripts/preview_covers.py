#!/usr/bin/env python3
"""Build a static HTML gallery for reviewing generated covers.

Usage:
    uv run python scripts/preview_covers.py && open output/covers/gallery.html

Works with whatever variants are already generated — safe to run mid-generation.
"""
from __future__ import annotations

import base64
from pathlib import Path

OUTPUT_BASE = Path("output/covers")
GALLERY = OUTPUT_BASE / "gallery.html"

SLUG_LABELS: dict[str, str] = {}  # optional slug → display label overrides


def img_src(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:image/jpeg;base64,{b64}"


def build_gallery() -> None:
    slugs = sorted(
        [d for d in OUTPUT_BASE.iterdir() if d.is_dir()],
        key=lambda d: list(SLUG_LABELS).index(d.name) if d.name in SLUG_LABELS else 99,
    )

    sections_html = ""
    total_found = 0

    for slug_dir in slugs:
        slug = slug_dir.name
        label = SLUG_LABELS.get(slug, slug)
        variants = sorted(slug_dir.glob("variant_*/cover.jpg"))
        if not variants:
            continue

        cards = ""
        for cover in variants:
            variant_n = cover.parent.name
            src = img_src(cover)
            cards += f"""
        <div class="card">
          <a href="{src}" target="_blank">
            <img src="{src}" alt="{variant_n}" loading="lazy">
          </a>
          <div class="meta">
            <strong>{variant_n}</strong>
            <code>{cover}</code>
            <button onclick="navigator.clipboard.writeText('{cover}').then(()=>this.textContent='✓ copied').catch(()=>0)">
              📋 copy path
            </button>
          </div>
        </div>"""
            total_found += 1

        sections_html += f"""
    <section>
      <h2>{label} <span class="slug">/ {slug}</span></h2>
      <div class="grid">{cards}
      </div>
    </section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Cover gallery — {total_found} images</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, sans-serif; background: #111; color: #eee; padding: 24px; }}
    h1 {{ font-size: 1.4rem; margin-bottom: 32px; color: #aaa; }}
    h2 {{ font-size: 1.1rem; margin: 40px 0 12px; }}
    .slug {{ color: #666; font-weight: 400; font-size: .9rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }}
    .card {{ background: #1e1e1e; border-radius: 8px; overflow: hidden; }}
    .card img {{ width: 100%; aspect-ratio: 1200/630; object-fit: cover; display: block; transition: opacity .15s; }}
    .card img:hover {{ opacity: .85; }}
    .meta {{ padding: 8px 10px; font-size: .75rem; color: #888; }}
    .meta strong {{ color: #ccc; display: block; margin-bottom: 4px; }}
    .meta code {{ display: block; word-break: break-all; margin-bottom: 6px; font-size: .7rem; }}
    button {{ background: #2a2a2a; border: 1px solid #444; color: #aaa; border-radius: 4px; padding: 3px 8px; cursor: pointer; font-size: .72rem; }}
    button:hover {{ background: #333; color: #fff; }}
    @media (max-width: 1200px) {{ .grid {{ grid-template-columns: repeat(3, 1fr); }} }}
    @media (max-width: 700px)  {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body>
  <h1>Cover gallery — {total_found} / 30 images generated</h1>
  {sections_html}
</body>
</html>"""

    GALLERY.write_text(html, encoding="utf-8")
    print(f"Gallery: {GALLERY.resolve()}  ({total_found} images)")
    print(f"Open:    open {GALLERY.resolve()}")


if __name__ == "__main__":
    build_gallery()
