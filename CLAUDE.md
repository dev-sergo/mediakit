# mediakit — project rules for AI assistant

> **Before starting any work: read `docs/DOCS.md`.**
> It contains current implementation status, architecture overview, known tech debt, and open questions.
> `CLAUDE.md` = rules only. `docs/DOCS.md` = project state.

## What this is
Generic local media processing toolkit. No domain logic, no Telegram bot, no billing, no user concept.
Designed to be used as a Python package, CLI, or HTTP service by any consumer.
Hardware target: single GPU machine (e.g. RTX 3090 24GB), Ubuntu 24.04, ComfyUI at `http://127.0.0.1:8188`.

## Core abstraction rules
- **Op** = atomic stateless operation (1 input → 1 result). Lives in `src/mediakit/ops/`.
- **Pipeline** = named composition of ops. Lives in `src/mediakit/pipelines/`.
- **Fast ops** (native: compress, resize, convert, variants, lqip) → execute in-process, sync HTTP 200.
- **Slow ops** (ComfyUI: txt2img, img_edit, bg_remove, upscale, txt2video, img2video) → always via arq queue, GPU concurrency=1. HTTP 202 + job_id.
- Nothing that touches GPU runs inside a FastAPI handler. Ever.
- ComfyUI interaction: HTTP+WS only, workflow.json patched by node title.

## Stack
- Python 3.11, uv, ruff, mypy --strict
- FastAPI + Pydantic v2 + Typer (CLI)
- arq + Redis (no Postgres, no Alembic in v1)
- Pillow + opencv-python-headless (native backend)
- structlog JSON, sentry-sdk

## Code rules
- Typed signatures everywhere. No `from x import *`. Async for I/O.
- Pydantic v2 syntax (`model_config = ConfigDict(...)`).
- No `print` — only structlog. No hardcoded secrets.
- English variable names and comments.
- No comments that describe *what* the code does — only *why* if non-obvious.

## What never goes here
- Telegram bot handlers, FSM, aiogram
- Credit/billing logic
- User authentication (only a shared bearer token for HTTP)
- Product-specific prompt templates (those live in the consumer)
- Postgres / Alembic
