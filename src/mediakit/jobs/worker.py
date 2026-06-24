"""arq worker entrypoint.

All slow (GPU/ComfyUI) ops run here. concurrency=1 because RTX 3090
can only run one inference job at a time.

Start with: mediakit-worker
or:         uv run mediakit-worker
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import arq
import arq.connections
import sentry_sdk
import structlog
from arq import cron
from arq.worker import Retry

from mediakit.config import settings
from mediakit.logging import configure_logging

log = structlog.get_logger(__name__)


# ─── Task functions (called by arq) ──────────────────────────────────────────


def _should_retry(ctx: dict[str, Any]) -> bool:
    return int(ctx.get("job_try", 1)) < 2


def _resolve_enum(cls: type[Any], value: str) -> Any:
    """Resolve a StrEnum by member name ('hr') or by value ('BiRefNet-HR')."""
    try:
        return cls[value]
    except KeyError:
        return cls(value)


def _is_oom(exc: Exception) -> bool:
    """Detect GPU out-of-memory errors from ComfyUI execution error messages."""
    msg = str(exc).lower()
    return "allocation on device" in msg or "out of memory" in msg


async def _emergency_free_vram() -> None:
    """Call ComfyUI /free directly without a full client (safe in exception handlers)."""
    import httpx

    try:
        async with httpx.AsyncClient(base_url=settings.comfyui_url, timeout=20.0) as c:
            await c.post("/free", json={"unload_models": True, "free_memory": True})
        log.info("worker.emergency_vram_freed")
    except Exception as e:
        log.warning("worker.emergency_free_failed", error=str(e))


async def task_txt2img(ctx: dict[str, Any], params_dict: dict[str, Any]) -> dict[str, Any]:
    from mediakit.backends.comfyui.exceptions import ComfyUIExecutionError, ComfyUITimeoutError
    from mediakit.ops.txt2img import txt2img
    from mediakit.schemas.ai_ops import Txt2ImgParams

    try:
        params = Txt2ImgParams.model_validate(params_dict)
        result = await txt2img(params)
        return {"output": str(result.output), "seed": result.seed}
    except (ComfyUITimeoutError, ComfyUIExecutionError) as exc:
        if isinstance(exc, ComfyUIExecutionError) and not _is_oom(exc):
            raise  # node errors (wrong names etc.) — don't retry
        if _should_retry(ctx):
            await _emergency_free_vram()
            raise Retry(defer=15) from None
        raise


async def task_img_edit(ctx: dict[str, Any], params_dict: dict[str, Any]) -> dict[str, Any]:
    from mediakit.backends.comfyui.exceptions import ComfyUIExecutionError, ComfyUITimeoutError
    from mediakit.ops.img_edit import img_edit
    from mediakit.schemas.ai_ops import ImgEditParams

    try:
        params = ImgEditParams.model_validate(params_dict)
        result = await img_edit(params)
        return {"output": str(result.output), "seed": result.seed}
    except (ComfyUITimeoutError, ComfyUIExecutionError) as exc:
        if isinstance(exc, ComfyUIExecutionError) and not _is_oom(exc):
            raise
        if _should_retry(ctx):
            await _emergency_free_vram()
            raise Retry(defer=15) from None
        raise


async def task_bg_remove(ctx: dict[str, Any], params_dict: dict[str, Any]) -> dict[str, Any]:
    from mediakit.backends.comfyui.exceptions import ComfyUIExecutionError, ComfyUITimeoutError
    from mediakit.ops.bg_remove import bg_remove
    from mediakit.schemas.ai_ops import BgRemoveParams

    try:
        params = BgRemoveParams.model_validate(params_dict)
        result = await bg_remove(params)
        return {"output": str(result.output)}
    except (ComfyUITimeoutError, ComfyUIExecutionError) as exc:
        if isinstance(exc, ComfyUIExecutionError) and not _is_oom(exc):
            raise
        if _should_retry(ctx):
            await _emergency_free_vram()
            raise Retry(defer=15) from None
        raise


async def task_upscale(ctx: dict[str, Any], params_dict: dict[str, Any]) -> dict[str, Any]:
    from mediakit.backends.comfyui.exceptions import ComfyUIExecutionError, ComfyUITimeoutError
    from mediakit.ops.upscale import upscale
    from mediakit.schemas.ai_ops import UpscaleParams

    try:
        params = UpscaleParams.model_validate(params_dict)
        result = await upscale(params)
        return {"output": str(result.output)}
    except (ComfyUITimeoutError, ComfyUIExecutionError) as exc:
        if isinstance(exc, ComfyUIExecutionError) and not _is_oom(exc):
            raise
        if _should_retry(ctx):
            await _emergency_free_vram()
            raise Retry(defer=15) from None
        raise


async def task_pipeline_article_cover(
    ctx: dict[str, Any], params_dict: dict[str, Any]
) -> dict[str, Any]:
    from mediakit.pipelines.article_cover import ArticleCoverPipeline

    params_dict["output_dir"] = Path(params_dict["output_dir"])
    result = await ArticleCoverPipeline().run(**params_dict)
    return {"outputs": [str(p) for p in result.outputs], **result.meta}


async def task_pipeline_photo_finalize(
    ctx: dict[str, Any], params_dict: dict[str, Any]
) -> dict[str, Any]:
    from mediakit.pipelines.photo_finalize import PhotoFinalizePipeline

    params_dict["input"] = Path(params_dict["input"])
    params_dict["output_dir"] = Path(params_dict["output_dir"])
    result = await PhotoFinalizePipeline().run(**params_dict)
    return {"outputs": [str(p) for p in result.outputs]}


async def task_pipeline_responsive_set(
    ctx: dict[str, Any], params_dict: dict[str, Any]
) -> dict[str, Any]:
    from mediakit.pipelines.responsive_set import ResponsiveSetPipeline

    params_dict["input"] = Path(params_dict["input"])
    if params_dict.get("output_dir"):
        params_dict["output_dir"] = Path(params_dict["output_dir"])
    result = await ResponsiveSetPipeline().run(**params_dict)
    return {"outputs": [str(p) for p in result.outputs]}


async def task_pipeline_product_shot(
    ctx: dict[str, Any], params_dict: dict[str, Any]
) -> dict[str, Any]:
    from mediakit.backends.comfyui.exceptions import ComfyUIExecutionError, ComfyUITimeoutError
    from mediakit.pipelines.product_shot import ProductShotPipeline
    from mediakit.schemas.ai_ops import BiRefNetModel, UpscaleModel

    try:
        params_dict["input"] = Path(params_dict["input"])
        if params_dict.get("output_dir"):
            params_dict["output_dir"] = Path(params_dict["output_dir"])
        if isinstance(params_dict.get("birefnet_model"), str):
            params_dict["birefnet_model"] = _resolve_enum(
                BiRefNetModel, params_dict["birefnet_model"]
            )
        if isinstance(params_dict.get("upscale_model"), str):
            params_dict["upscale_model"] = _resolve_enum(UpscaleModel, params_dict["upscale_model"])
        result = await ProductShotPipeline().run(**params_dict)
        return {"outputs": [str(p) for p in result.outputs], **result.meta}
    except (ComfyUITimeoutError, ComfyUIExecutionError) as exc:
        if isinstance(exc, ComfyUIExecutionError) and not _is_oom(exc):
            raise
        if _should_retry(ctx):
            await _emergency_free_vram()
            raise Retry(defer=15) from None
        raise


async def task_pipeline_txt_to_video_hq(
    ctx: dict[str, Any], params_dict: dict[str, Any]
) -> dict[str, Any]:
    from mediakit.backends.comfyui.exceptions import ComfyUIExecutionError, ComfyUITimeoutError
    from mediakit.pipelines.txt_to_video_hq import TxtToVideoHqPipeline

    try:
        params_dict["output_dir"] = Path(params_dict["output_dir"])
        result = await TxtToVideoHqPipeline().run(**params_dict)
        return {"outputs": [str(p) for p in result.outputs], **result.meta}
    except (ComfyUITimeoutError, ComfyUIExecutionError) as exc:
        if isinstance(exc, ComfyUIExecutionError) and not _is_oom(exc):
            raise
        if _should_retry(ctx):
            await _emergency_free_vram()
            raise Retry(defer=30) from None
        raise


async def task_pipeline_photo_animate(
    ctx: dict[str, Any], params_dict: dict[str, Any]
) -> dict[str, Any]:
    from mediakit.backends.comfyui.exceptions import ComfyUIExecutionError, ComfyUITimeoutError
    from mediakit.pipelines.photo_animate import PhotoAnimatePipeline
    from mediakit.schemas.ai_ops import BiRefNetModel, UpscaleModel

    try:
        params_dict["input"] = Path(params_dict["input"])
        if params_dict.get("output_dir"):
            params_dict["output_dir"] = Path(params_dict["output_dir"])
        if isinstance(params_dict.get("birefnet_model"), str):
            params_dict["birefnet_model"] = _resolve_enum(
                BiRefNetModel, params_dict["birefnet_model"]
            )
        if isinstance(params_dict.get("upscale_model"), str):
            params_dict["upscale_model"] = _resolve_enum(UpscaleModel, params_dict["upscale_model"])
        result = await PhotoAnimatePipeline().run(**params_dict)
        return {"outputs": [str(p) for p in result.outputs], **result.meta}
    except (ComfyUITimeoutError, ComfyUIExecutionError) as exc:
        if isinstance(exc, ComfyUIExecutionError) and not _is_oom(exc):
            raise
        if _should_retry(ctx):
            await _emergency_free_vram()
            raise Retry(defer=30) from None
        raise


async def task_pipeline_seamless_video(
    ctx: dict[str, Any], params_dict: dict[str, Any]
) -> dict[str, Any]:
    from mediakit.backends.comfyui.exceptions import ComfyUIExecutionError, ComfyUITimeoutError
    from mediakit.pipelines.seamless_video import SeamlessVideoPipeline

    try:
        params_dict["output_dir"] = Path(params_dict["output_dir"])
        if params_dict.get("input"):
            params_dict["input"] = Path(params_dict["input"])
        result = await SeamlessVideoPipeline().run(**params_dict)
        return {"outputs": [str(p) for p in result.outputs], **result.meta}
    except (ComfyUITimeoutError, ComfyUIExecutionError) as exc:
        if isinstance(exc, ComfyUIExecutionError) and not _is_oom(exc):
            raise
        if _should_retry(ctx):
            await _emergency_free_vram()
            raise Retry(defer=30) from None
        raise


async def task_txt2video(ctx: dict[str, Any], params_dict: dict[str, Any]) -> dict[str, Any]:
    from mediakit.backends.comfyui.exceptions import ComfyUIExecutionError, ComfyUITimeoutError
    from mediakit.ops.txt2video import txt2video
    from mediakit.schemas.video_ops import Txt2VideoParams

    try:
        params = Txt2VideoParams.model_validate(params_dict)
        result = await txt2video(params)
        return {"output": str(result.output), "seed": result.seed, "duration_s": result.duration_s}
    except (ComfyUITimeoutError, ComfyUIExecutionError) as exc:
        if isinstance(exc, ComfyUIExecutionError) and not _is_oom(exc):
            raise
        if _should_retry(ctx):
            await _emergency_free_vram()
            raise Retry(defer=30) from None
        raise


async def task_img2video(ctx: dict[str, Any], params_dict: dict[str, Any]) -> dict[str, Any]:
    from mediakit.backends.comfyui.exceptions import ComfyUIExecutionError, ComfyUITimeoutError
    from mediakit.ops.img2video import img2video
    from mediakit.schemas.video_ops import Img2VideoParams

    try:
        params_dict["input"] = Path(params_dict["input"])
        params = Img2VideoParams.model_validate(params_dict)
        result = await img2video(params)
        return {"output": str(result.output), "seed": result.seed, "duration_s": result.duration_s}
    except (ComfyUITimeoutError, ComfyUIExecutionError) as exc:
        if isinstance(exc, ComfyUIExecutionError) and not _is_oom(exc):
            raise
        if _should_retry(ctx):
            await _emergency_free_vram()
            raise Retry(defer=30) from None
        raise


async def task_cleanup_storage(ctx: dict[str, Any]) -> dict[str, Any]:
    cutoff = time.time() - 86400  # 24 hours
    deleted = 0
    for directory in [settings.storage_uploads, settings.storage_outputs]:
        if not directory.exists():
            continue
        for f in directory.rglob("*"):
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                deleted += 1
    log.info("mediakit_cleanup.done", deleted=deleted)
    return {"deleted": deleted}


# ─── Worker config ────────────────────────────────────────────────────────────


class WorkerSettings:
    functions = [
        task_txt2img,
        task_img_edit,
        task_bg_remove,
        task_upscale,
        task_txt2video,
        task_img2video,
        task_pipeline_article_cover,
        task_pipeline_photo_animate,
        task_pipeline_photo_finalize,
        task_pipeline_product_shot,
        task_pipeline_responsive_set,
        task_pipeline_txt_to_video_hq,
        task_pipeline_seamless_video,
        task_cleanup_storage,
    ]
    cron_jobs = [cron(task_cleanup_storage, hour={3}, minute={0})]
    redis_settings = arq.connections.RedisSettings.from_dsn(settings.redis_url)
    max_jobs = settings.worker_concurrency_gpu  # = 1: one GPU job at a time
    job_timeout = settings.video_timeout_s  # covers both image (300s) and video (900s)
    keep_result = 3600  # keep result in Redis for 1 hour


def main() -> None:
    configure_logging(settings.log_level, settings.log_format)
    if settings.sentry_dsn:
        sentry_sdk.init(dsn=settings.sentry_dsn)
    if settings.comfyui_models_dir is not None:
        from mediakit.models_registry import warn_missing_models

        warn_missing_models(settings.comfyui_models_dir)
    log.info("mediakit_worker.starting", max_jobs=WorkerSettings.max_jobs)
    arq.run_worker(WorkerSettings)  # type: ignore[arg-type]
