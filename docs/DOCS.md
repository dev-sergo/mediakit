# mediakit — developer documentation

> Status: native ops, AI image ops, AI video ops, job queue, HTTP server, CLI, and pipelines are all implemented. 61 tests (12 unit + 49 integration). New pipelines: product_shot, photo_animate, txt_to_video_hq added and manually tested. CogVideoX-5B added to txt2video/img2video (kijai wrapper, pending GPU-box verification) and a `seamless_video` pipeline for arbitrary-length clips.

---

## What this is

A generic local media processing toolkit that runs entirely on a single GPU machine with ComfyUI.

**Three interfaces to one codebase:**

```
mediakit compress photo.jpg --format webp        # CLI
await ops.compress(CompressParams(...))         # Python import
POST /v1/ops/compress  (multipart)             # HTTP REST
```

**What is intentionally absent:**
- Telegram bot, FSM, aiogram
- Billing, credits, user accounts
- Postgres / Alembic (Redis only for the queue)
- Product-specific prompt templates (belong in the consumer)

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Consumers                                       │
│  any HTTP client      →  HTTP /v1/...            │
│  any Python script    →  import mediakit           │
│  dev / scripts        →  mediakit CLI              │
└───────────────┬─────────────────────────────────┘
                │
    ┌───────────┴───────────────────┐
    ▼                               ▼
FastAPI server              Typer CLI
/v1/ops/*   (sync → 200)    mediakit compress ...
/v1/jobs/*  (async → 202)   mediakit txt2img ...
/v1/pipelines/*
    │
    └──→ arq queue (Redis)
              │
              ▼
         arq Worker (max_jobs=1)
              │
              ▼
         ComfyUI :8188
```

### Execution rules

| Op type | Execution | HTTP response |
|---|---|---|
| Native (compress, resize, convert, variants, lqip) | In-process, Pillow | `200 OK` immediately |
| ComfyUI (txt2img, img_edit, bg_remove, upscale, txt2video, img2video) | arq worker, GPU concurrency=1 | `202 Accepted` + `job_id` |
| Pipeline | Combination in worker | `202 Accepted` + `job_id` |

**Invariant:** nothing that touches the GPU runs directly from a FastAPI handler. Always via the queue.

---

## Implementation status

### Native ops ✅

| Op | CLI | HTTP | Description |
|---|---|---|---|
| `compress` | `mediakit compress` | `POST /v1/ops/compress` | JPEG/WebP/AVIF/PNG, quality preset or 0-100, max_width |
| `resize` | `mediakit resize` | `POST /v1/ops/resize` | fit / fill / smart_crop / pad |
| `convert` | `mediakit convert` | `POST /v1/ops/convert` | format conversion |
| `variants` | `mediakit variants` | `POST /v1/ops/variants` | responsive set (640/768/1024/1280/1536, multi-format) |
| `lqip` | `mediakit lqip` | `POST /v1/ops/lqip` | base64 WebP data URL for `blurDataURL` |

**Smart-crop** uses OpenCV StaticSaliencyFineGrained + Haar face cascade. Falls back to center-crop if OpenCV is unavailable.

**Quality presets:** `low`=60, `medium`=75, `high`=85, `max`=95. Integer (0-100) also accepted.

### ComfyUI backend ✅

| Op | CLI | HTTP | Models |
|---|---|---|---|
| `bg_remove` | `mediakit bg-remove` | `POST /v1/ops/bg-remove` | BiRefNet-HR (and 6 variants: HR-matting, general, dynamic, lite-2K, matting, portrait) |
| `upscale` | `mediakit upscale` | `POST /v1/ops/upscale` | 4x_NMKD-Siax_200k.pth or RealESRGAN_x4.pth |

**ComfyUI client** (`backends/comfyui/client.py`): HTTP + WebSocket, image upload via `/upload/image`, result polling via `/ws` with fallback to `/history`, result download.

### AI image generation ✅

| Op | CLI | HTTP | Architecture |
|---|---|---|---|
| `txt2img` | `mediakit txt2img "..."` | `POST /v1/ops/txt2img` | SDXL or Flux 2 |
| `img_edit` | `mediakit img-edit photo.jpg "..."` | `POST /v1/ops/img-edit` | SDXL inpainting + BiRefNet mask, or Qwen Image Edit |

`img_edit` supports two backends:
- `--backend sdxl` (default) — SDXL inpainting + BiRefNet mask
- `--backend qwen` — Qwen Image Edit 2511 (instruction-based, 4 steps with Lightning LoRA)

`txt2img` supports two backends:
- `--backend sdxl` (default) — CheckpointLoaderSimple (RealVisXL etc.)
- `--backend flux` — Flux 2 Dev, better quality, different workflow

### Job queue + HTTP server ✅

- **arq worker** (`jobs/worker.py`): 14 tasks — 4 AI image ops + 2 video ops + 7 pipeline tasks + 1 cron cleanup. `max_jobs=1` (GPU concurrency).
- **FastAPI** (`server/app.py`): 21 routes, bearer-token auth (disabled if `API_TOKEN` is empty).
- **Job polling:** `GET /v1/jobs/{id}` → `{status, result, enqueue_time}`. Result lives in Redis for 1 hour.
- **Health check:** `GET /healthz` → checks ComfyUI (`/system_stats`) and Redis (ping).

### Pipelines ✅

| Pipeline | Steps | Purpose |
|---|---|---|
| `article_cover` | txt2img (sdxl\|flux) → smart_crop(1200×630) → compress | OG cover image + optional variants |
| `responsive_set` | compress → variants(webp+avif) → lqip | Next.js-ready responsive set |
| `photo_finalize` | bg_remove → upscale → compress → [variants] | Product photo finalization |
| `product_shot` | bg_remove → contact_shadow → composite(gradient_bg) → [upscale] → variants | E-commerce product photo on clean background |
| `photo_animate` | [bg_remove] → [upscale] → img2video | Animate a still photo into a short video clip |
| `txt_to_video_hq` | txt2img → img2video | Generate high-quality video from a text prompt |
| `seamless_video` | [single pass] **or** segment → continuation → crossfade | Arbitrary-length clip with the seam hidden |

**product_shot notes:**
- Shadow is a synthetic contact shadow (blurred oval at the object base) drawn from the BiRefNet alpha mask
- Background: radial gradient vignette with configurable color and strength
- Parameters: `bg_color`, `padding_pct`, `gradient_strength`, `shadow_opacity`, `shadow_blur`

**photo_animate notes:**
- Best prompt for products: `subtle camera zoom in, product still, bokeh background`
- Best negative: `hands, fingers, person, human, arm, water, pouring, liquid, motion`
- LTX-Video native window = 49 frames (~2 sec). Requesting >49 frames creates a visible seam at the join point — stay at ≤49 for seamless LTX results, or use CogVideoX (~6 s window) / the `seamless_video` pipeline for longer clips.

**txt_to_video_hq notes:**
- Same 49-frame seam limitation applies for vid_length
- `httpx` write/read timeout set to 120s (was 60s) — needed for large keyframe upload over slow network

**seamless_video notes:**
- Default model `cogvideox` (~6 s seam-free in one pass). For ≤ native window the
  pipeline emits a single clip and `seam_free: true` — nothing is stitched.
- Longer requests are split into overlapping segments. With img2video-capable models
  (ltxv, cogvideox) each segment continues from the *last frame* of the previous one,
  so motion genuinely carries over; a short crossfade over `overlap_frames` (default 8)
  removes any residual micro-jump. Wan has no img2video path, so it falls back to
  independent segments joined by crossfade only — the cut is hidden but motion is not
  continued (`continuation: false` in the result meta).
- Stitching is native: OpenCV reads the seed frame, ffmpeg `xfade` does the crossfade
  (`backends/native/video.py`). ffmpeg must be on PATH on the worker box.
- `fps` defaults to the model's native rate; overriding it changes playback speed, not
  the generated motion.

### AI video ops ✅

| Op | CLI | HTTP | Models |
|---|---|---|---|
| `txt2video` | `mediakit txt2video "..."` | `POST /v1/ops/txt2video` | LTX-Video (ltxv), Wan 2.1 (wan) or CogVideoX-5B (cogvideox) |
| `img2video` | `mediakit img2video --input photo.jpg "..."` | `POST /v1/ops/img2video` | LTX-Video or CogVideoX-5B-I2V (cogvideox) |

Video workflow builders are in `backends/comfyui/workflows/`. Video jobs use `video_timeout_s = 900` (15 min).

**Model native windows** (single seam-free pass — see `backends/comfyui/video_models.py`):

| Model | Native fps | Native window | ≈ seconds | img2video |
|---|---|---|---|---|
| `ltxv` | 24 | 49 frames (9+8n) | ~2 s | yes |
| `wan` | 16 | 81 frames (4n+1) | ~5 s | no |
| `cogvideox` | 8 | 49 frames (8n+1) | ~6 s | yes (CogVideoX-5B-I2V) |

CogVideoX-5B is trained at 8 fps, so its 49-frame window is ~6 s — roughly 3x LTX's
seam-free window. For clips up to a model's native window, **no stitching is needed**;
just pick the model. For longer clips use the `seamless_video` pipeline.

**CogVideoX workflow builders** (`txt2video_cogvideox.py`, `img2video_cogvideox.py`) target
the **kijai ComfyUI-CogVideoXWrapper** node set (`DownloadAndLoadCogVideoModel`,
`CogVideoTextEncode`, `CogVideoImageEncode`, `CogVideoSampler`, `CogVideoDecode`). Each
builder carries a `VERIFY on GPU box` block — node input/output arity must be confirmed
against the installed wrapper version (weights auto-download to `ComfyUI/models/CogVideo`).

### Audio backend (Phase 10 — reserved)

Placeholder: `src/mediakit/backends/audio/__init__.py`

Potential ops:

| Op | Model | VRAM |
|---|---|---|
| `tts` (text-to-speech) | XTTS-v2, Bark | ~4 GB |
| `music_gen` | MusicGen (Meta AudioCraft) | ~8 GB |
| `sfx_gen` | AudioLDM2, Stable Audio | ~6 GB |

Open questions:
- Check ComfyUI audio node compatibility vs. separate backend
- Estimate VRAM: a single 24GB GPU should fit audio + image simultaneously
- First op: `ops/tts.py`
- Add to arq worker (audio synthesis is slow, queue is mandatory)

### Lab / Experiment runner ✅

Implemented in `lab/`:

```
lab/
├── runner.py          ← runs YAML manifests
├── report.py          ← HTML report
├── configs/           ← experiments (YAML)
└── presets/           ← preset library by category
```

Usage:
```bash
uv run python lab/runner.py lab/presets/landscape/example.yaml
uv run python lab/runner.py lab/configs/example.yaml
```

---

## HTTP API — full route list

> A generated [`openapi.json`](../openapi.json) (OpenAPI 3.1, 21 operations) is committed
> at the repo root for offline reference; the live Swagger UI is at `/docs`.

```
GET   /healthz                              → {status, checks: {comfyui, redis}}
GET   /docs                                 → Swagger UI

# Sync ops (200 OK immediately)
POST  /v1/ops/compress                      multipart: file, format, quality, max_width
POST  /v1/ops/resize                        multipart: file, width, height, mode
POST  /v1/ops/convert                       multipart: file, format, quality
POST  /v1/ops/lqip                          multipart: file, size
POST  /v1/ops/variants                      multipart: file, sizes, formats, quality

# Async AI ops (202 → job_id)
POST  /v1/ops/txt2img                       form: prompt, negative, backend(sdxl|flux), width, height, steps, cfg, seed, checkpoint
POST  /v1/ops/img-edit                      multipart: file + form: prompt, negative, backend(sdxl|qwen), mask, lora_strength, steps, cfg, seed
POST  /v1/ops/bg-remove                     multipart: file + form: model, background, color
POST  /v1/ops/upscale                       multipart: file + form: model, scale
POST  /v1/ops/txt2video                     form: prompt, negative, model(ltxv|wan|cogvideox), width, height, length, fps, steps, cfg, seed
POST  /v1/ops/img2video                     multipart: file + form: prompt, negative, width, height, length, fps, steps, cfg, seed

# Job polling + output download
GET   /v1/jobs/{job_id}                     → {job_id, status, result, enqueue_time}
GET   /v1/jobs/{job_id}/output              → FileResponse (409 if not ready, 410 if cleaned up)

# Pipeline async (202 → job_id)
POST  /v1/pipelines/article-cover           form: prompt, slug, output_dir, negative, backend, steps, cfg, seed, cover_format, responsive
POST  /v1/pipelines/photo-finalize          multipart: file + form: output_dir, background_mode, upscale_scale
POST  /v1/pipelines/responsive-set          multipart: file + form: output_dir, sizes, formats, quality, generate_lqip
POST  /v1/pipelines/product-shot            multipart: file + form: output_dir, birefnet_model, bg_color, padding_pct, gradient_strength, shadow_opacity, shadow_blur, do_upscale, upscale_model, upscale_scale, formats, widths, quality
POST  /v1/pipelines/photo-animate           multipart: file + form: prompt, negative_prompt, output_dir, remove_bg, birefnet_model, do_upscale, upscale_model, upscale_scale, width, height, length, fps, steps, cfg, seed
POST  /v1/pipelines/txt-to-video-hq         form: prompt, negative_prompt, output_dir, img_backend, width, height, img_steps, img_cfg, img_seed, vid_length, vid_fps, vid_steps, vid_cfg, vid_seed
POST  /v1/pipelines/seamless-video          form: prompt, output_dir, model, total_frames, fps, width, height, overlap_frames, steps, cfg, seed + optional file (first-frame image)
```

**Auth:** `Authorization: Bearer <API_TOKEN>`. If `API_TOKEN` is empty, auth is disabled.

---

## CLI — full command list

```bash
# Native (no ComfyUI required)
mediakit compress  photo.jpg [--output out.webp] [--format webp] [--quality high|medium|low|max|0-100] [--max-width 1920]
mediakit resize    photo.jpg --width 1200 --height 630 [--mode fit|fill|smart_crop|pad]
mediakit convert   photo.jpg webp [--output out.webp] [--quality high]
mediakit variants  photo.jpg [--sizes 640,768,1024,1280,1536] [--formats webp,avif] [--quality high]
mediakit lqip      photo.jpg [--size 16] [--output placeholder.txt]

# AI image (requires running ComfyUI)
mediakit bg-remove --input photo.jpg [--model BiRefNet-HR|BiRefNet-portrait|...] [--bg transparent|color]
mediakit upscale   --input photo.jpg [--scale 2.0] [--model 4x_NMKD-Siax_200k.pth|RealESRGAN_x4.pth]
mediakit txt2img  --prompt "cyberpunk city sunset" [--backend sdxl|flux] [--width 1024] [--height 1024]
mediakit img-edit  --input photo.jpg --prompt "white studio background" [--backend sdxl|qwen]

# AI video (requires running ComfyUI + video models)
mediakit txt2video --prompt "night market, golden hour" [--model ltxv|wan] [--length 49] [--fps 24]
mediakit img2video --input photo.jpg --prompt "person looks to camera" [--length 49] [--fps 24]

# Server and worker
mediakit serve    # start FastAPI (uvicorn) — alias mediakit-server
mediakit worker   # start arq worker — alias mediakit-worker
```

---

## Environment variables

Copy `.env.example` to `.env`:

```bash
COMFYUI_URL=http://127.0.0.1:8188      # ComfyUI address
COMFYUI_TIMEOUT_S=300                  # max inference wait time
REDIS_URL=redis://localhost:6379/0
STORAGE_UPLOADS=./storage/uploads      # uploaded files from consumers
STORAGE_OUTPUTS=./storage/outputs      # op results
STORAGE_MAX_UPLOAD_MB=20
API_TOKEN=changeme                     # bearer token; empty = auth disabled
HOST=0.0.0.0
PORT=8000
WORKER_CONCURRENCY_GPU=1              # do not change — 1 GPU = 1 job
VIDEO_TIMEOUT_S=900                   # video job timeout (15 min)
LOG_LEVEL=INFO
LOG_FORMAT=json                        # json | console
SENTRY_DSN=                           # empty = Sentry disabled
COMFYUI_MODELS_DIR=                   # path to ComfyUI models for startup check
```

---

## Running

```bash
# 1. Dependencies
uv sync --extra dev

# 2. Redis
docker-compose up -d redis

# 3. ComfyUI — must be running on :8188 separately

# 4. Worker (separate terminal)
uv run mediakit-worker

# 5. HTTP server (optional)
uv run mediakit-server

# 6. Tests
uv run pytest tests/unit/ -v                     # 12 unit (native ops, no GPU)
uv run pytest tests/integration/ -v             # 49 integration (HTTP, mock Redis/ComfyUI)
uv run pytest tests/ -m "gpu" -v                # GPU-only (requires ComfyUI + models)
```

---

## Known issues / technical debt

| # | Issue | Priority |
|---|---|---|
| 1 | No job history beyond 1 hour in Redis | Medium — acceptable until Postgres is introduced |
| 2 | `storage/` module is an empty stub — files are written directly via `Path` in ops | Low |
| 3 | No formal Op interface or registry — ops are discovered via string names in worker.py | Low |
| 4 | No per-task timeout override in arq — all tasks share global `job_timeout` | Low |
| 5 | Audio backend (Phase 10) not started | Planned |
| 6 | No Timeline/composition abstraction for multi-track video+audio rendering | Planned |

---

## File structure

```
mediakit/
├── src/mediakit/
│   ├── ops/                    # atomic operations (async functions)
│   │   ├── compress.py         ✅ native
│   │   ├── resize.py           ✅ native
│   │   ├── convert.py          ✅ native
│   │   ├── variants.py         ✅ native
│   │   ├── lqip.py             ✅ native
│   │   ├── bg_remove.py        ✅ comfyui (BiRefNet)
│   │   ├── upscale.py          ✅ comfyui (ESRGAN)
│   │   ├── txt2img.py          ✅ comfyui (SDXL / Flux 2)
│   │   ├── img_edit.py         ✅ comfyui (SDXL inpainting / Qwen)
│   │   ├── txt2video.py        ✅ comfyui (LTX-Video / Wan 2.1 / CogVideoX-5B)
│   │   └── img2video.py        ✅ comfyui (LTX-Video / CogVideoX-5B-I2V)
│   ├── pipelines/              # named op compositions
│   │   ├── article_cover.py    ✅ txt2img → smart_crop → compress
│   │   ├── responsive_set.py   ✅ compress → variants → lqip
│   │   ├── photo_finalize.py   ✅ bg_remove → upscale → compress
│   │   ├── product_shot.py     ✅ bg_remove → contact_shadow → composite → [upscale] → variants
│   │   ├── photo_animate.py    ✅ [bg_remove] → [upscale] → img2video
│   │   ├── txt_to_video_hq.py  ✅ txt2img → img2video
│   │   └── seamless_video.py   ✅ single pass | segment → continuation → crossfade
│   ├── backends/
│   │   ├── comfyui/
│   │   │   ├── client.py       ✅ HTTP+WS client
│   │   │   ├── exceptions.py   ✅
│   │   │   ├── video_models.py ✅ per-model native window profiles
│   │   │   └── workflows/      ✅ programmatic workflow builders (+ cogvideox t2v/i2v)
│   │   ├── native/
│   │   │   ├── encoder.py      ✅ Pillow encode
│   │   │   ├── resizer.py      ✅ resize + OpenCV saliency
│   │   │   └── video.py        ✅ last-frame extract + ffmpeg crossfade concat
│   │   └── audio/              ⚠️ placeholder (Phase 10)
│   ├── jobs/
│   │   ├── queue.py            ✅ arq enqueue / poll
│   │   ├── worker.py           ✅ 14 arq tasks, max_jobs=1
│   │   └── status.py           ✅ JobStatus enum
│   ├── server/
│   │   ├── app.py              ✅ FastAPI, 21 routes
│   │   ├── deps.py             ✅ bearer auth
│   │   ├── utils.py            ✅ save_upload
│   │   └── routes/
│   │       ├── health.py       ✅ /healthz
│   │       ├── ops.py          ✅ sync ops
│   │       ├── jobs.py         ✅ async AI ops + polling
│   │       └── pipelines.py    ✅ pipeline endpoints
│   ├── cli/
│   │   ├── main.py             ✅ Typer app, 13 commands (11 media + serve + worker)
│   │   └── commands/
│   ├── schemas/
│   │   ├── ops.py              ✅ native op params/results
│   │   ├── ai_ops.py           ✅ AI image op params/results
│   │   └── video_ops.py        ✅ video op params/results
│   ├── models_registry/        ✅ startup model presence check
│   ├── storage/                ⚠️ empty stub
│   ├── config.py               ✅ Pydantic Settings
│   └── logging.py              ✅ structlog
├── tests/
│   ├── unit/                   ✅ 12 tests (native ops, no GPU)
│   └── integration/            ✅ 49 tests (HTTP layer, mocked Redis/ComfyUI)
├── lab/                        experiment runner + presets
├── scripts/                    utility scripts
├── docs/deploy/                systemd unit files
├── docker-compose.yml          ✅ Redis 7
└── pyproject.toml
```

Legend: ✅ done, ⚠️ stub or empty directory.

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for setup, linting/type-checking, and a
step-by-step **"Adding a new op"** guide, plus the architecture rules every op/pipeline
must follow (nothing GPU-touching runs outside the worker).
