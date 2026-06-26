# Changelog

All notable changes to mediakit are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Showcase gallery** in the README, plus a curated [`examples/`](examples/) set
  (article covers, video previews, and a `responsive_set` LQIP / WebP-ladder demo).
- **`seamless_video` pipeline** — arbitrary-length clips with the seam hidden:
  ≤ native window emits a single seam-free clip; longer requests split into
  overlapping segments stitched with a native ffmpeg `xfade` crossfade, and
  img2video-capable models continue motion from the previous segment's last frame.
- **CogVideoX-5B** added to `txt2video` / `img2video` (kijai ComfyUI-CogVideoXWrapper),
  with a ~6 s native window. Verified on RTX 3090 — img2video produces 49-frame 720×480
  clips; `CogVideoImageEncode` node uses `start_image` (kijai wrapper ≥1.x rename from
  `image`).
- **Lab experiments** — 5 parameter grids run on RTX 3090 (CFG scale, step count, sampler
  comparison, seed variance, SDXL steps). Results committed as contact-sheet images in
  [`docs/experiments/`](docs/experiments/) with per-grid findings. Confirmed defaults:
  CFG 7.5, 25 steps, `dpmpp_2m + karras`.
- Generated [`openapi.json`](openapi.json) (OpenAPI 3.1, 21 operations) committed at
  the repo root for offline API reference.

### Fixed
- CI lint job was failing on every push/PR: `mypy` targeted a non-existent
  `src/imgkit/` path (package is `mediakit`), and `ruff` check/format had not been
  applied. Corrected the path and cleared the pre-existing lint/format debt; no
  behaviour change (61 tests still pass).

### Changed
- Documentation counts synced with reality: 61 tests (12 unit + 49 integration),
  14 arq tasks, 21 routes, 13 CLI commands.
