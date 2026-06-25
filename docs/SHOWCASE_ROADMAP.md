# mediakit — Showcase Roadmap

> Forward-looking plan for turning mediakit into a flagship visual showcase.
> Each **session** below is self-contained — pick one, finish it, ship it.
> Status legend: `[ ]` todo · `[~]` in progress · `[x]` done.
> Tags: **(no-GPU)** doable on the Mac · **(GPU)** needs the RTX 3090 + ComfyUI running.

Created 2026-06-24.

---

## Why this exists

The code is mature; the gap is **visual proof**. The biggest quality levers for local
generation, in order of impact, are: **checkpoint/LoRA choice → pipeline/workflow design →
sampler params → prompt → post-processing**. mediakit already has the `lab/` infrastructure
(8 grids, 11 benchmark categories) to drive exactly this kind of systematic comparison —
the roadmap below is about *running it and publishing the results* as showcase content.

### Priority / impact at a glance

| Session | Effort | Needs GPU | Showcase impact |
|---|---|---|---|
| S0 — Doc polish | 15 min | no | Low (credibility) |
| S1 — Verify video feature | 1 session | **yes** | High (video is the most eye-catching output) |
| S2 — Image-pipeline gallery | 1 session | **yes** | **Highest** (cheap, direct proof) |
| S3 — Publish lab experiments | 1–2 sessions | **yes** | High (proves the "pipelines > prompts" thesis) |
| S4 — Infra polish | optional | no | Low–Medium (ease of evaluation) |

Recommended order: **S0 → S2 → S1 → S3 → S4.**

---

## S0 — Documentation polish (no-GPU, ~15 min)

Fixes from the 2026-06-24 audit. Pure credibility, no code logic.

- [x] `docs/DOCS.md:3` — test count `52 (12+40)` → **`61 (12 unit + 49 integration)`** (verified with `pytest -q`: 61 passed = 12 unit + 49 integration).
- [x] `docs/DOCS.md` — drift in the architecture tree: `13 arq tasks` → **14**, `20 routes` → **21**, commands recounted → **13** (11 media + serve + worker). (Implementation-status section was already correct at 14/21.)
- [x] `docs/DOCS.md` API section — `txt2video` route lists `model(ltxv|wan)`; added **`cogvideox`** to match the schema.
- [x] Decide on stray `output.wav` in repo root — not present and untracked; added `*.wav` to `.gitignore` as a guard (audio is reserved Phase-10).
- [x] (optional) Add `CHANGELOG.md` with a first entry: Gallery, `examples/`, `seamless_video`, CogVideoX. — CogVideoX entry kept honest (still pending GPU verification).
- [x] (optional) Add `SECURITY.md`. — Minimal policy: private GitHub reporting + deployment trust-boundary notes.

---

## S2 — Image-processing pipeline gallery (GPU) — *do this first for impact*

The single highest-ROI visual win. We have pipelines that produce striking before/after
output but show **none** of it. Run each, commit curated results to `examples/`, add to the
README Gallery.

- [x] `product_shot` — run on thai-ice-green-tea.jpeg; before/after pair committed to `examples/product_shot/`. Added to README Gallery.
- [x] `responsive_set` — **native, no-GPU**; done on the Mac. Curated `examples/responsive/lqip-blur-vs-full.jpg` (123-byte LQIP placeholder vs full image) + a WebP width-ladder table in the README. (Source covers are 1200 px, so the ladder tops out at 1024w; AVIF runs larger than WebP on these already-compressed JPEGs, so the table shows WebP only.)
- [ ] `photo_finalize` — marketplace-ready before/after (bg-remove → upscale → compress).
- [ ] `upscale` — a tight crop comparison (input vs 4×) so the detail gain is visible.
- [~] Add an **"Image pipelines"** subsection to the README Gallery with these. (Created, currently covers `responsive_set`; product_shot/photo_finalize/upscale entries pending GPU.)
- [ ] Keep `examples/` curated — pick the best 1–2 per pipeline, not bulk `output/`.

**Payoff:** turns "we have 8 pipelines" (told) into "look what they do" (shown). Directly
answers the original showcase gap.

---

## S1 — Verify & showcase the video feature (GPU)

The CogVideoX-5B + `seamless_video` feature is code-complete and tested with mocks, but
the workflow builders carry `VERIFY on GPU box` notes — not yet run on real hardware.

- [ ] Run `txt2video --model cogvideox` and `img2video --model cogvideox` on the 3090; confirm node arity against the installed kijai wrapper.
- [ ] Run `seamless_video` for a clip longer than one native window; confirm the crossfade/continuation hides the seam.
- [ ] Remove the `VERIFY on GPU box` comments from the workflow builders once confirmed.
- [ ] Generate 1–2 **higher-quality** showcase clips (the current `video_cat.gif` is a placeholder) → convert to optimized GIF (≤2.5 MB) → replace in `examples/video/`.
- [ ] Update `docs/DOCS.md` to drop "pending GPU-box verification".

**Payoff:** video is the most attention-grabbing output a local-GPU showcase can have;
seamless arbitrary-length clips are a genuine differentiator.

---

## S3 — Publish lab experiments (GPU) — *the "pipelines beat prompts" proof*

`lab/` already has the grids. Run them, summarize the findings, publish as `docs/experiments/`.
This both improves the project's own defaults *and* becomes showcase content demonstrating
that quality is engineered, not lucked into.

Existing grids (`lab/grids/`):
- [ ] `flux-vs-sdxl.yaml` — checkpoint comparison (the #1 quality lever). Publish a side-by-side.
- [ ] `qwen-vs-sdxl.yaml` — img-edit backend comparison.
- [ ] `model-compare.yaml` — broader model sweep.
- [ ] `sampler-matrix.yaml` + `steps-grid.yaml` + `cfg-grid.yaml` — parameter tuning; find the sweet spot per checkpoint.
- [ ] `lora-strength.yaml` — LoRA weight sweep.
- [ ] `variance.yaml` — same params, different seeds (shows output stability).

Then:
- [ ] Create `docs/experiments/README.md` — one short report per grid: setup, contact-sheet image, conclusion ("Flux wins on X, SDXL on Y; CFG 3.5 / 20 steps is the sweet spot for …").
- [ ] Commit a few **contact-sheet** images (montages) — not the full grid output.
- [ ] Fold the winning defaults back into the pipeline configs.
- [ ] Link `docs/experiments/` from the README.

**Payoff:** this is the content that validates your strategic thesis — and it's the kind of
thing a portfolio reviewer remembers.

---

## S4 — Infrastructure polish (no-GPU, optional)

Lower visual impact, raises "easy to evaluate" score.

- [x] Full-stack `docker-compose` (mediakit worker + Redis + ComfyUI) or a clear note that ComfyUI is external. — Added a header note in `docker-compose.yml`: only Redis is containerised; ComfyUI is external (needs host GPU).
- [x] GitHub Actions running the 61 tests on PR. — `.github/workflows/ci.yml` already runs unit+integration on push/PR; fixed its broken lint job (see commit `dba021e`).
- [x] Link the FastAPI `/docs` (Swagger) in the README, or commit an `openapi.json`. — Committed generated `openapi.json` (OpenAPI 3.1, 21 ops) at repo root; linked from README + DOCS.
- [x] A short "how to add an op/pipeline" already exists in CONTRIBUTING — cross-link it from DOCS. — Added a "Contributing" pointer at the end of DOCS.md.

---

## Out of scope

- New model families beyond CogVideoX/Flux/SDXL/Wan — only if a clear quality win on 3090.
- Audio backend (`backends/audio/` is a reserved Phase-10 placeholder) — belongs in sona-audio.
