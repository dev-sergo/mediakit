import secrets
import shutil
import uuid

import structlog

from mediakit.backends.comfyui.client import ComfyUIClient
from mediakit.backends.comfyui.workflows.txt2video_cogvideox import (
    CogVideoXTxt2VideoParams,
    build_cogvideox_txt2video_workflow,
)
from mediakit.backends.comfyui.workflows.txt2video_ltxv import (
    LtxvTxt2VideoParams,
    build_ltxv_txt2video_workflow,
)
from mediakit.backends.comfyui.workflows.txt2video_wan import (
    WanTxt2VideoParams,
    build_wan_txt2video_workflow,
)
from mediakit.config import settings
from mediakit.schemas.video_ops import Txt2VideoParams, VideoResult

log = structlog.get_logger(__name__)


async def txt2video(params: Txt2VideoParams) -> VideoResult:
    seed = secrets.randbits(32) if params.seed == -1 else params.seed
    settings.ensure_storage_dirs()
    tmp_dir = settings.storage_outputs / "tmp" / uuid.uuid4().hex
    tmp_dir.mkdir(parents=True)

    timeout = float(settings.video_timeout_s)

    async with ComfyUIClient(
        settings.comfyui_url, output_dir=tmp_dir, timeout_seconds=timeout
    ) as comfy:
        if params.model == "cogvideox":
            workflow = build_cogvideox_txt2video_workflow(
                CogVideoXTxt2VideoParams(
                    positive_prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    width=params.width,
                    height=params.height,
                    length=params.length,
                    fps=params.fps,
                    steps=params.steps,
                    cfg=params.cfg,
                    seed=seed,
                )
            )
        elif params.model == "wan":
            workflow = build_wan_txt2video_workflow(
                WanTxt2VideoParams(
                    positive_prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    width=params.width,
                    height=params.height,
                    length=params.length,
                    fps=params.fps,
                    steps=params.steps,
                    cfg=params.cfg,
                    seed=seed,
                )
            )
        else:  # ltxv
            workflow = build_ltxv_txt2video_workflow(
                LtxvTxt2VideoParams(
                    positive_prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    width=params.width,
                    height=params.height,
                    length=params.length,
                    fps=params.fps,
                    steps=params.steps,
                    cfg=params.cfg,
                    seed=seed,
                )
            )

        prompt_id = await comfy.submit_workflow(workflow)
        raw_outputs = await comfy.wait_for_result(prompt_id)
        try:
            await comfy.soft_free_memory()  # clear activations, keep model warm
        except Exception:
            log.debug("txt2video.soft_free_skipped", exc_info=True)

    suffix = ".mp4"
    final = params.output or (settings.storage_outputs / f"video_{uuid.uuid4().hex}{suffix}")
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_outputs[0], final)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    duration_s = round(params.length / params.fps, 2)
    log.info(
        "txt2video.done",
        output=str(final),
        seed=seed,
        model=params.model,
        duration_s=duration_s,
    )
    return VideoResult(output=final, seed=seed, duration_s=duration_s)
