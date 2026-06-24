import secrets
import shutil
import uuid

import structlog

from mediakit.backends.comfyui.client import ComfyUIClient
from mediakit.backends.comfyui.workflows.img_edit import (
    ImgEditWorkflowParams,
    build_img_edit_workflow,
)
from mediakit.backends.comfyui.workflows.img_edit_qwen import (
    ImgEditQwenWorkflowParams,
    build_img_edit_qwen_workflow,
)
from mediakit.backends.native.metadata import write_metadata
from mediakit.config import settings
from mediakit.schemas.ai_ops import ImgEditParams, ImgEditResult

log = structlog.get_logger(__name__)


async def img_edit(params: ImgEditParams) -> ImgEditResult:
    seed = secrets.randbits(32) if params.seed == -1 else params.seed
    settings.ensure_storage_dirs()
    tmp_dir = settings.storage_outputs / "tmp" / uuid.uuid4().hex
    tmp_dir.mkdir(parents=True)

    async with ComfyUIClient(
        settings.comfyui_url,
        output_dir=tmp_dir,
        timeout_seconds=float(settings.comfyui_timeout_s),
    ) as comfy:
        # Qwen needs ~10 GB, SDXL inpainting ~8 GB
        min_vram = 10_000 if params.backend == "qwen" else 8_000
        await comfy.ensure_vram(min_mb=min_vram)

        server_name = await comfy.upload_image(params.input)

        if params.backend == "qwen":
            workflow = build_img_edit_qwen_workflow(
                ImgEditQwenWorkflowParams(
                    positive_prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    image_filename=server_name,
                    lora_strength=params.lora_strength,
                    steps=params.steps if params.steps != 25 else 4,
                    cfg=params.cfg if params.cfg != 7.5 else 1.0,
                    seed=seed,
                    width=params.width,
                    height=params.height,
                )
            )
        else:
            workflow = build_img_edit_workflow(
                ImgEditWorkflowParams(
                    image_filename=server_name,
                    positive_prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    checkpoint=params.checkpoint,
                    width=params.width,
                    height=params.height,
                    steps=params.steps,
                    cfg=params.cfg,
                    seed=seed,
                    mask_target=params.mask_target,
                    mask_blur=params.mask_blur,
                )
            )

        prompt_id = await comfy.submit_workflow(workflow)
        raw_outputs = await comfy.wait_for_result(prompt_id)
        try:
            await comfy.soft_free_memory()  # clear activations, keep model warm
        except Exception:
            log.debug("img_edit.soft_free_skipped", exc_info=True)

    final = params.output or (settings.storage_outputs / f"img_edit_{uuid.uuid4().hex}.png")
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_outputs[0], final)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    write_metadata(
        final,
        seed=seed,
        steps=params.steps,
        cfg=params.cfg,
        checkpoint=params.checkpoint,
        backend=params.backend,
        prompt=params.prompt,
        mask_target=params.mask_target,
    )
    log.info(
        "img_edit.done",
        input=str(params.input),
        output=str(final),
        seed=seed,
        backend=params.backend,
    )
    return ImgEditResult(output=final, seed=seed)
