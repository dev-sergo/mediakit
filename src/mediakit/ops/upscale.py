import shutil
import uuid

import structlog

from mediakit.backends.comfyui.client import ComfyUIClient
from mediakit.backends.comfyui.workflows.upscale import (
    UpscaleWorkflowParams,
    build_upscale_workflow,
)
from mediakit.backends.native.metadata import write_metadata
from mediakit.config import settings
from mediakit.schemas.ai_ops import UpscaleParams, UpscaleResult

log = structlog.get_logger(__name__)


async def upscale(params: UpscaleParams) -> UpscaleResult:
    settings.ensure_storage_dirs()
    tmp_dir = settings.storage_outputs / "tmp" / uuid.uuid4().hex
    tmp_dir.mkdir(parents=True)

    async with ComfyUIClient(
        settings.comfyui_url,
        output_dir=tmp_dir,
        timeout_seconds=float(settings.comfyui_timeout_s),
    ) as comfy:
        server_name = await comfy.upload_image(params.input)
        workflow = build_upscale_workflow(
            UpscaleWorkflowParams(
                image_filename=server_name,
                upscale_model=params.model.value,
                target_scale=params.scale,
            )
        )
        prompt_id = await comfy.submit_workflow(workflow)
        raw_outputs = await comfy.wait_for_result(prompt_id)
        try:
            await comfy.soft_free_memory()  # clear activations, keep model warm
        except Exception:
            log.debug("upscale.soft_free_skipped", exc_info=True)

    if params.output:
        final = params.output
    else:
        final = settings.storage_outputs / f"upscale_{uuid.uuid4().hex}.png"

    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_outputs[0], final)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    write_metadata(final, model=params.model.value, scale=params.scale)
    log.info("upscale.done", input=str(params.input), output=str(final), scale=params.scale)
    return UpscaleResult(output=final)
