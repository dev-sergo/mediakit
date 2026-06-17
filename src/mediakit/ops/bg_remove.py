import shutil
import uuid

import structlog

from mediakit.backends.comfyui.client import ComfyUIClient
from mediakit.backends.comfyui.workflows.bg_remove import (
    BgRemoveWorkflowParams,
    build_bg_remove_workflow,
)
from mediakit.backends.native.metadata import write_metadata
from mediakit.config import settings
from mediakit.schemas.ai_ops import BgRemoveParams, BgRemoveResult

log = structlog.get_logger(__name__)


async def bg_remove(params: BgRemoveParams) -> BgRemoveResult:
    settings.ensure_storage_dirs()
    tmp_dir = settings.storage_outputs / "tmp" / uuid.uuid4().hex
    tmp_dir.mkdir(parents=True)

    async with ComfyUIClient(
        settings.comfyui_url,
        output_dir=tmp_dir,
        timeout_seconds=float(settings.comfyui_timeout_s),
    ) as comfy:
        server_name = await comfy.upload_image(params.input)
        workflow = build_bg_remove_workflow(
            BgRemoveWorkflowParams(
                image_filename=server_name,
                birefnet_model=params.model.value,
                background_mode=params.background_mode,
                background_color=params.background_color,
                mask_blur=params.mask_blur,
                mask_offset=params.mask_offset,
                refine_foreground=params.refine_foreground,
                width=params.width,
                height=params.height,
            )
        )
        prompt_id = await comfy.submit_workflow(workflow)
        raw_outputs = await comfy.wait_for_result(prompt_id)
        try:
            await comfy.soft_free_memory()  # clear activations, keep model warm
        except Exception:
            log.debug("bg_remove.soft_free_skipped", exc_info=True)

    ext = "png" if params.background_mode == "transparent" else "png"
    if params.output:
        final = params.output
    else:
        final = settings.storage_outputs / f"bg_remove_{uuid.uuid4().hex}.{ext}"

    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_outputs[0], final)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    write_metadata(final, model=params.model.value, background_mode=params.background_mode)
    log.info("bg_remove.done", input=str(params.input), output=str(final))
    return BgRemoveResult(output=final)
