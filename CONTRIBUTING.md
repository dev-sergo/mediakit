# Contributing

## Setup

```bash
git clone <repo>
cd mediakit
uv sync --extra dev
cp .env.example .env
```

## Running tests

```bash
# Unit tests — no GPU or Redis required
uv run pytest tests/unit/ -v

# Integration tests — mocked Redis and ComfyUI
uv run pytest tests/integration/ -v

# GPU tests — requires running ComfyUI with models
uv run pytest tests/ -m gpu -v
```

## Linting and type-checking

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/mediakit/
```

## Adding a new op

1. Create `src/mediakit/ops/<name>.py` — async function `async def <name>(params: XParams) -> XResult`.
2. Add params/result types to `src/mediakit/schemas/` (ops.py for native, ai_ops.py or video_ops.py for AI).
3. Register a worker task in `src/mediakit/jobs/worker.py` (for slow GPU ops).
4. Add HTTP route in `src/mediakit/server/routes/jobs.py` (async) or `ops.py` (sync).
5. Add CLI command in `src/mediakit/cli/commands/<name>.py` and register it in `cli/main.py`.
6. Export from `src/mediakit/ops/__init__.py`.
7. Add tests in `tests/unit/` (native) or `tests/integration/` (AI ops).

## Code style

- Typed signatures on everything. No `from x import *`.
- No `print` — use `structlog`.
- English variable names and comments.
- Comments only for non-obvious *why*, never for *what*.
- Pydantic v2 (`model_config = ConfigDict(...)`).

## Architecture rules

- **Native ops** (Pillow-based) run in-process → sync HTTP 200.
- **AI ops** (ComfyUI) always go through the arq queue → async HTTP 202 + job_id.
- Nothing that touches the GPU runs inside a FastAPI handler.
- `max_jobs=1` in the worker is intentional (single-GPU constraint) — do not change.
