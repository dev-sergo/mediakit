import secrets
import shutil
import uuid

import structlog

from mediakit.backends.comfyui.client import ComfyUIClient
from mediakit.backends.comfyui.workflows.img2video_cogvideox import (
    CogVideoXImg2VideoParams,
    build_cogvideox_img2video_workflow,
)
from mediakit.backends.comfyui.workflows.img2video_ltxv import (
    LtxvImg2VideoParams,
    build_ltxv_img2video_workflow,
)
from mediakit.config import settings
from mediakit.schemas.video_ops import Img2VideoParams, VideoResult

log = structlog.get_logger(__name__)


async def img2video(params: Img2VideoParams) -> VideoResult:
    seed = secrets.randbits(32) if params.seed == -1 else params.seed
    settings.ensure_storage_dirs()
    tmp_dir = settings.storage_outputs / "tmp" / uuid.uuid4().hex
    tmp_dir.mkdir(parents=True)

    timeout = float(settings.video_timeout_s)

    async with ComfyUIClient(
        settings.comfyui_url, output_dir=tmp_dir, timeout_seconds=timeout
    ) as comfy:
        server_name = await comfy.upload_image(params.input)

        if params.model == "cogvideox":
            workflow = build_cogvideox_img2video_workflow(CogVideoXImg2VideoParams(
                positive_prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                image_filename=server_name,
                width=params.width,
                height=params.height,
                length=params.length,
                fps=params.fps,
                steps=params.steps,
                cfg=params.cfg,
                seed=seed,
            ))
        else:  # ltxv
            workflow = build_ltxv_img2video_workflow(LtxvImg2VideoParams(
                positive_prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                image_filename=server_name,
                width=params.width,
                height=params.height,
                length=params.length,
                fps=params.fps,
                steps=params.steps,
                cfg=params.cfg,
                seed=seed,
            ))

        prompt_id = await comfy.submit_workflow(workflow)
        raw_outputs = await comfy.wait_for_result(prompt_id)
        try:
            await comfy.soft_free_memory()  # clear activations, keep model warm
        except Exception:
            log.debug("img2video.soft_free_skipped", exc_info=True)

    final = params.output or (settings.storage_outputs / f"video_{uuid.uuid4().hex}.mp4")
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_outputs[0], final)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    duration_s = round(params.length / params.fps, 2)
    log.info(
        "img2video.done",
        input=str(params.input), output=str(final), seed=seed, duration_s=duration_s,
    )
    return VideoResult(output=final, seed=seed, duration_s=duration_s)
